"""Validation checks for the aggregated-prefill pricing (aggcore.py). Writes JSON + prints a summary.

Usage: python validate.py OUT_JSON
  (a) zero-chunk parity is recorded per row by run_agg.py (zero_chunk_gap); summarized by score_agg.py.
  L   linearity of Delta in chunk count (1, 2, 4 x 1024-token chunks) and chunk size (one request).
  K   does the mixed path's chunk cost depend on the decode KV read? (Delta at decode KV x1 vs /4)
  (b) pure prefill through the FPM path vs IPI's saved best-prefill rows, all cells; decode graph
      (nextn=3, used for aggregated steps) vs prefill graph (nextn=0) on the same layout.
  (c) hand roofline for one 1024-token chunk of Qwen3-Coder-480B (plain GQA) on 3 chips.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggcore as A
A.install(None)
import yaml
from agent_index.current_native import KEYS

OUT = Path(sys.argv[1])
pub = A.published()
res = {}
lay_of = lambda r: {k: int(r[k]) for k in KEYS}

def near(fr, s=100.):
    return min(fr, key=lambda r: abs(math.log(r['speed'] / s)))

# ---- L and K ----
L = []
for h in ['GB300 NVL72', 'B300 · 800G', 'MI355X · 400G', 'VR200 Rubin', 'Maia 300 rack', 'H100 SXM · FP8 · 400G', 'TPU7x Ironwood · FP8']:
    for m in A.MODELS:
        g = A.Agg(h, m); r = near(pub[h, m][0]); lay, b = lay_of(r), int(r['batch'])
        F0 = g.fpm(lay, b, 0)
        d = {n: g.chunks(lay, b, n) - F0 for n in (1, 2, 4)}
        single = {q: g.fpm(lay, b, q, 1) - F0 for q in (128, 256, 512, 2048, 4096)}
        # K: emulate KV-read-once in the decode part by dividing decode KV tokens by 4
        _, _, eng, _ = g.N.graph('decode', lay); eng = A.native_engine(eng)
        def est(kvdiv, n):
            s = dict(num_decode_requests=4 * b, sum_decode_kv_tokens=4 * b * (A.CONTEXT + 1) // kvdiv)
            if n: s.update(num_prefill_requests=n, sum_prefill_tokens=n * A.CHUNK, sum_prefill_kv_tokens=n * A.CONTEXT)
            return eng.estimate_forward_pass_time_ms([{'scheduled_requests': s}] * lay['dp'])
        dK = est(4, 1) - est(4, 0)
        # pure-prefill single chunk on the decode graph (no decode work)
        pure = eng.estimate_forward_pass_time_ms([{'scheduled_requests': dict(num_prefill_requests=1, sum_prefill_tokens=A.CHUNK, sum_prefill_kv_tokens=A.CONTEXT)}] * lay['dp'])
        L.append(dict(hardware=h, model=m, layout=r.get('layout', str(lay)), batch=b, speed=r['speed'], dec_ms=r['raw_ms'],
                      fpm0=F0, d1=d[1], d2_per=d[2] / 2, d4_per=d[4] / 4, single=single, pure_chunk=pure,
                      d1_kv_div4=dK, k_rank=A.AL * A.RATIO * b))
        chips = lay['tp'] * lay['dp'] * lay['pp']
        L[-1].update(P=pub[h, m][1], P_marg_1chunk=A.CHUNK * lay['dp'] * 1000 / (d[1] * chips),
                     P_marg_q128=128 * lay['dp'] * 1000 / (single[128] * chips) if single[128] > 0 else float('inf'))
        x = L[-1]
        print(f"L {h[:14]:14s} {m[:6]:6s} B={b:4d} dec={r['raw_ms']:7.2f} pure={pure:7.2f} d1={d[1]:7.2f} d2/2={d[2]/2:7.2f} d4/4={d[4]/4:7.2f} "
              f"q128={single[128]:6.2f} q512={single[512]:6.2f} q2048={single[2048]:7.2f} | d1(kv/4)={dK:7.2f} ({dK/d[1]-1:+.1%}) | P={x['P']:.0f} marg(1024)={x['P_marg_1chunk']:.0f} marg(128)={x['P_marg_q128']:.0f}", flush=True)
res['linearity_and_kv'] = L

# ---- (b) prefill ----
B = []
kv = A.load_parts(A.ROOT / 'results/kvonce', with_prefill_row=True)
for (h, m), (_, P, row) in sorted(kv.items()):
    if not row:
        continue
    g = A.Agg(h, m); lay, b = lay_of(row), int(row['batch'])
    _, _, pe, _ = g.N.graph('prefill', lay); pe = A.native_engine(pe)
    s = [{'scheduled_requests': dict(num_prefill_requests=b, sum_prefill_tokens=b * A.CHUNK, sum_prefill_kv_tokens=b * A.CONTEXT)}] * lay['dp']
    ms = pe.estimate_forward_pass_time_ms(s) + row['hop_ms']
    try:
        _, _, de, _ = g.N.graph('decode', lay); de = A.native_engine(de)
        msd = de.estimate_forward_pass_time_ms(s) + row['hop_ms']
    except Exception as e:  # some prefill layouts are not valid decode graphs
        msd = None
    Ppub = pub[h, m][1]
    B.append(dict(hardware=h, model=m, layout={**lay, 'batch': b}, saved_ms=row['raw_ms'], fpm_ms=ms,
                  rel_err=ms / row['raw_ms'] - 1, decode_graph_ms=msd,
                  decode_over_prefill_graph=msd / ms if msd else None, P_kvonce=P, P_published=Ppub))
res['prefill'] = B
errs = [abs(x['rel_err']) for x in B]
ratios = [x['decode_over_prefill_graph'] for x in B if x['decode_over_prefill_graph']]
print(f"(b) prefill cells={len(B)} max|fpm/saved-1|={max(errs):.2e} (Trainium MoE repricing not in FPM path: "
      f"{max(abs(x['rel_err']) for x in B if not x['hardware'].startswith('Trainium')):.2e} excl. Trainium); "
      f"decode-graph/prefill-graph chunk cost: min {min(ratios):.3f} median {sorted(ratios)[len(ratios)//2]:.3f} max {max(ratios):.3f}; "
      f"P kvonce==published: {max(abs(x['P_kvonce']/x['P_published']-1) for x in B):.1e}")
for x in B:
    if abs(x['rel_err']) > 1e-9:
        print('   (b) gap', x['hardware'], x['model'], f"{x['rel_err']:+.3%}")

# ---- (c) hand roofline, Qwen3-Coder-480B ----
cfg = json.loads((A.REF / 'models/Qwen-480B/config.json').read_text())
Hd, nq, nkv, hd, Lyr = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['num_hidden_layers']
E, k_act, Ie = cfg['num_experts'], cfg['num_experts_per_tok'], cfg['moe_intermediate_size']
Q, S = A.CHUNK, A.CONTEXT
attn_proj = Hd * (nq + 2 * nkv) * hd + nq * hd * Hd
moe = k_act * 3 * Hd * Ie + Hd * E
C = []
cat = A.catalog()
for h in ['GB300 NVL72', 'H100 SXM · FP8 · 400G', 'Maia 300 rack']:
    m = 'Qwen-480B'; g = A.Agg(h, m); r = near(pub[h, m][0]); lay, b = lay_of(r), int(r['batch'])
    gpu = yaml.safe_load((A.REF / 'hardware' / (cat[h]['profile'] + '.yaml')).read_text())['gpu']
    tp = lay['tp']; quant = cat[h]['quant']
    moe_flops_rate = gpu.get(f'{quant}_tc_flops') or gpu.get('fp8_tc_flops') or gpu['bfloat16_tc_flops']
    bf16 = gpu['bfloat16_tc_flops']; bw = gpu['mem_bw'] * gpu.get('mem_bw_empirical_scaling_factor', 1)
    # per chip; every DP rank carries one chunk, so MoE work per chip = (rank FLOPs)/tp as well
    f_attn = 4 * nq * hd * Q * (S + Q / 2) * Lyr / tp
    f_proj = 2 * attn_proj * Q * Lyr / tp
    f_moe = 2 * moe * Q * Lyr / tp
    kv_bytes = Lyr * 2 * nkv * hd * S * 1 / min(tp, nkv) / lay['dcp']   # fp8 prefix KV, read once per chunk
    t_attn = max(f_attn / bf16, kv_bytes / bw) * 1e3
    t_lin = (f_proj / bf16 + f_moe / moe_flops_rate) * 1e3
    F0 = g.fpm(lay, b, 0); d1 = g.chunks(lay, b, 1) - F0
    _, _, eng, _ = g.N.graph('decode', lay); eng = A.native_engine(eng)
    pure = eng.estimate_forward_pass_time_ms([{'scheduled_requests': dict(num_prefill_requests=1, sum_prefill_tokens=Q, sum_prefill_kv_tokens=S)}] * lay['dp'])
    C.append(dict(hardware=h, layout={**lay, 'batch': b}, quant=quant, attn_flops_chip=f_attn, proj_flops_chip=f_proj,
                  moe_flops_chip=f_moe, prefix_kv_bytes_chip=kv_bytes, t_attn_ms=t_attn, t_linear_ms=t_lin,
                  hand_ms=t_attn + t_lin, delta1_ms=d1, pure_chunk_ms=pure, dec_ms=r['raw_ms']))
    print(f"(c) {h[:14]:14s} {lay} B={b}: attn {f_attn/1e12:.1f} TF -> {t_attn:.2f} ms (KV {kv_bytes/1e9:.2f} GB -> {kv_bytes/bw*1e3:.2f} ms); "
          f"linear {(f_proj+f_moe)/1e12:.1f} TF -> {t_lin:.2f} ms; hand {t_attn+t_lin:.2f} ms | AISim pure chunk {pure:.2f}, Delta(1 chunk on decode) {d1:.2f}", flush=True)
res['hand_qwen'] = C
OUT.write_text(json.dumps(res, indent=1) + '\n')
