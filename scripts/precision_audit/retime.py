"""Re-time saved IPI decode frontier points under alternative PRECISION configs.

Native where AISim supports the dtype (leaf-spec rewrites evaluated through the pinned
engine), analytic where it does not (dispatch payload dtype, AR dtype, attention math
dtype, FP4 KV). Capacity (batch, memory admission) is held at the saved values.

Switches (a config is a comma list):
  g8      non-expert GEMMs / MLA BMMs / DSA projections -> FP8 W8A8 (native rewrite).
          Router and LM head stay BF16 (FP8 checkpoints keep them high precision).
  e4a8    FP8 experts -> W4A8 (MoEQuantMode.w4afp8: 0.5 B/param, FP8 MMA) (native).
          Only changes chips whose catalog quant is fp8 (TPU7x, H100, H200).
  w4a16   all experts -> W4A16 MXFP4 (0.5 B/param, BF16 MMA) (native; Kimi checkpoint semantics)
  e8      all experts -> FP8 W8A8 (native; e.g. what Hopper/TPU get today)
  nccl8   NCCL collectives (DCP, MAI TP all-reduce) payload FP8 (native rewrite)
  ar8     CustomAllReduce payload FP8: t*0.5 (analytic; engine ignores AR dtype)
  disp8   MoE pre-dispatch payload FP8+1/128 scales: alpha+(t-alpha)*0.516 (analytic)
  disp4   MoE pre-dispatch payload NVFP4 (+1/16 fp8 scale): alpha+(t-alpha)*0.281 (analytic)
  att16   decode attention MMA at BF16 (policy-literal): math*fp8/bf16 (analytic on SOL)
  kv4     NVFP4 KV cache: attention memory*0.5625, math unchanged (analytic on SOL; not DSA)
  dispS   SOTA dispatch: disp4 on Blackwell/Rubin NVIDIA profiles, disp8 elsewhere
  c1      verify reads KV once: generation_attention (GQA/MLA) memory/4 (as sweep_kv_once)
"""
import sys, json, copy, math
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'kernel_audit'))
from common import *
from agent_index.interconnect import NativeLeaves
from agent_index import index as IDX

ATT_KINDS = ('GenerationAttention', 'GenerationMla')
_ctx = {}
def ctx(hw, model, l):
    key = (hw, model, *(l[k] for k in KEYS))
    if key not in _ctx:
        n = native(hw, model)
        c, g, e, hop = n.graph('decode', l)
        nl = NativeLeaves(c, g, 'decode', False)
        _ctx[key] = (nl, hop)
    return _ctx[key]

def rewrite(leaves, sw, hwq):
    out = copy.deepcopy(leaves)
    for leaf in out:
        k, v = next(iter(leaf.items()))
        nm = v['name']
        if 'g8' in sw:
            if k == 'Gemm' and 'router' not in nm and 'logits' not in nm and v.get('quant_mode') == 'bfloat16':
                v['quant_mode'] = 'fp8'
            if k == 'MlaBmm' and v.get('quant_mode') == 'bfloat16':
                v['quant_mode'] = 'fp8'
            if k == 'DsaGeneration':
                v['gemm_quant_mode'] = 'fp8'
                if v.get('attn_projection_quant_modes'):
                    v['attn_projection_quant_modes'] = {a: 'fp8' for a in v['attn_projection_quant_modes']}
        if k == 'Moe':
            if 'e4a8' in sw and v['quant_mode'] == 'fp8':
                v['quant_mode'] = 'w4afp8'
            if 'w4a16' in sw:
                v['quant_mode'] = 'w4a16_mxfp4'
            if 'e8' in sw:
                v['quant_mode'] = 'fp8'
        if 'nccl8' in sw and k == 'Nccl' and v.get('dtype') == 'half':
            v['dtype'] = 'fp8'
    return out

def _eval(nl, leaves, batch):
    rows = nl.handle.evaluate_ops_json(json.dumps(leaves), is_context=False, batch_size=batch*4, s=262145, prefix=0)
    d = {r[0]: r[1] for r in rows}
    return [d[next(iter(x.values()))['name']] for x in leaves]

def _sol(nl, leaves, idxs, batch):
    if not idxs: return {}
    L = [leaves[i] for i in idxs]
    rows = nl.handle.evaluate_ops_sol_json(json.dumps(L), is_context=False, batch_size=batch*4, s=262145, prefix=0)
    d = {r[0]: r for r in rows}
    return {i: d[next(iter(leaves[i].values()))['name']] for i in idxs}

_alpha = {}
def alpha(nl, leaf):
    key = (id(nl), json.dumps(leaf, sort_keys=True))
    if key not in _alpha:
        s = copy.deepcopy(leaf); v = next(iter(s.values())); v['attention_dp_size'] = 1; v['name'] = 'alpha_' + v['name']
        _alpha[key] = nl.handle.evaluate_ops_json(json.dumps([s]), is_context=False, batch_size=4, s=262145, prefix=0)[0][1]
    return _alpha[key]

def retime(hw, model, l, batch, sw, detail=False):
    """Return (ms, per-leaf list) for one point under switch set sw."""
    nl, hop = ctx(hw, model, l)
    y = hwyaml(hw)['gpu']
    leaves = rewrite(nl.tree.leaves, sw, CAT[hw]['quant'])
    t = _eval(nl, leaves, batch)
    att = [i for i, x in enumerate(leaves) if next(iter(x)) in ATT_KINDS]
    need = att if any(s in sw for s in ('c1', 'att16', 'kv4')) else []
    sol = _sol(nl, leaves, need, batch)
    r16 = y['fp8_tc_flops'] / y['bfloat16_tc_flops']
    for i, leaf in enumerate(leaves):
        k, v = next(iter(leaf.items()))
        nm = v['name'].split('_', 2)[2]
        if i in sol:
            _, lat, m, mm = sol[i]
            if abs(lat - t[i]) > 1e-6 * max(1e-9, t[i]):
                raise RuntimeError(f'SOL/native mismatch {nm} {lat} {t[i]}')
            if 'att16' in sw: m *= r16
            if 'kv4' in sw: mm *= 0.5625
            if 'c1' in sw and nm == 'generation_attention': mm /= 4
            t[i] = max(m, mm)
        if k == 'CustomAllReduce' and 'ar8' in sw:
            t[i] *= 0.5
        if k == 'Nccl' and 'ar8' in sw and 'nccl8' not in sw and v.get('operation') == 'all_reduce':
            t[i] *= 0.5
        if k == 'MoeDispatch' and v.get('pre_dispatch') and ('disp8' in sw or 'disp4' in sw or 'dispS' in sw):
            nv4 = CAT[hw]['profile'].startswith(('gb', 'b2', 'b3', 'vr'))
            r = 0.281 if ('disp4' in sw or ('dispS' in sw and nv4)) else 0.516
            a = alpha(nl, leaf)
            t[i] = a + max(0., t[i] - a) * r
    ms = nl.tree.fold(t) + hop
    if detail:
        return ms, [(next(iter(x.values()))['name'].split('_', 2)[2], next(iter(x)), tt) for x, tt in zip(leaves, t)]
    return ms

def run(hws, models, configs):
    res = {}
    for hw in hws:
        for m in models:
            pts = []
            for p in CURVES[hw][m]:
                l = {k: p[k] for k in KEYS}
                ms = {name: retime(hw, m, l, p['batch'], set(sw.split('+')) if sw else set()) for name, sw in configs.items()}
                if abs(ms['base'] - p['raw_ms']) > 1e-6 * p['raw_ms']:
                    print('WARN baseline mismatch', hw, m, lay(p), p['batch'], ms['base'], p['raw_ms'], file=sys.stderr)
                pts.append(dict(capacity=p['capacity'], ms=ms))
            res[(hw, m)] = pts
            print('done', hw, m, len(pts), file=sys.stderr, flush=True)
    return res

def area(res, hw, m, name):
    P = [dict(speed=3000 / q['ms'][name], capacity=q['capacity']) for q in res[(hw, m)]]
    return IDX.area(IDX.envelope(P), 50, 400, True)
