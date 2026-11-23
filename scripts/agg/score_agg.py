"""Score aggregated (chunked-prefill) runs against a disaggregated baseline.

Usage: python -I score_agg.py AGG_RUN [--base published|RUN_DIR] [--variant A|T|A0|T0] [--json OUT]

Per speed s the deployment picks the better mode:  B(s) = max(B_disagg(s), c_agg(s)),
  B_disagg(s) = c(s) / (1 + r s c(s)/P)   (IPI log400 formula on the base decode frontier + best prefill)
  c_agg(s)    = aggregated-frontier capacity at speed >= s (no prefill chips)
integrated exactly over ln s on [50, 400] (breakpoints at both frontiers' speeds plus the crossing
points inside each segment) and normalized per model to GB300 NVL72 computed the same way.
Index = mean of the 4 models. Variants: A = 1024-token chunks (design), T = best chunk policy per
row, suffix 0 = memory without the extra in-flight prefill sequence.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggcore as A
sys.path.insert(0, str(A.REPO / 'src')); sys.path.insert(0, str(A.REPO / 'studies'))
from agent_index.index import capacity
from log400_release import combined_area

LOW, HIGH, G = 50., 400., 'GB300 NVL72'
r = A.RATIO


def best_area(disagg, P, agg, mode='max'):
    """Exact integral over ln s of max(c_d/(1+r s c_d/P), c_agg). mode: 'max' | 'disagg' | 'agg'."""
    edges = sorted({LOW, HIGH, *(p['speed'] for p in disagg + agg if LOW < p['speed'] < HIGH)})
    tot, win = 0., 0.
    for a, b in zip(edges, edges[1:]):
        mid = (a + b) / 2
        cd = capacity(disagg, mid) if P > 0 else 0.
        ca = capacity(agg, mid)
        al = r * cd / P if cd else 0.
        F = lambda x, y: cd * math.log1p((y - x) / (x * (1 + al * y))) if cd else 0.
        f = lambda s: cd / (1 + al * s) if cd else 0.
        if mode == 'disagg' or (mode == 'max' and ca <= f(b)):
            tot += F(a, b)
        elif mode == 'agg' or ca >= f(a):
            tot += ca * math.log(b / a); win += math.log(b / a) if mode == 'max' else 0.
        else:
            s = (cd / ca - 1) / al
            tot += F(a, s) + ca * math.log(b / s); win += math.log(b / s)
    return tot, win / math.log(HIGH / LOW)


def load_agg(d, variant):
    out = {}
    for f in (Path(d) / 'parts').glob('*.json'):
        p = json.loads(f.read_text())
        out[p['hardware'], p['model']] = p
    return out


def main():
    argv = sys.argv[1:]
    run = Path(argv[0])
    base_spec = argv[argv.index('--base') + 1] if '--base' in argv else 'published'
    variant = argv[argv.index('--variant') + 1] if '--variant' in argv else 'A'
    pub = A.published()
    base = A.baseline(base_spec)
    parts = load_agg(run, variant)
    hws = list(A.catalog())
    models = A.MODELS
    # sanity: disagg-only integral equals IPI's combined_area on the same inputs
    err = max(abs(best_area(*base[k], [], 'disagg')[0] / combined_area(*base[k], r) - 1)
              for k in base if combined_area(*base[k], r) > 0)
    def per(h, src, kind):
        vals, wins = [], []
        for m in models:
            fr, P = src[h, m]; frG, PG = src[G, m]
            ag = parts[h, m][f'frontier_{variant}'] if kind != 'disagg' and (h, m) in parts else []
            agG = parts[G, m][f'frontier_{variant}'] if kind != 'disagg' else []
            a, w = best_area(fr, P, ag, kind); aG, _ = best_area(frG, PG, agG, kind)
            vals.append(a / aG); wins.append(w)
        return sum(vals) / len(vals), vals, wins
    rows = []
    for h in hws:
        if not all((h, m) in parts for m in models):
            continue
        ip, _, _ = per(h, pub, 'disagg')
        ib, vb, _ = per(h, base, 'disagg')
        ia, va, _ = per(h, base, 'agg')
        ix, vx, wx = per(h, base, 'max')
        absr = []
        for m in models:
            fr, P = base[h, m]
            absr.append(best_area(fr, P, parts[h, m][f'frontier_{variant}'], 'max')[0] / best_area(fr, P, [], 'disagg')[0])
        rest = [combined_area(parts[h, m]['decode_frontier_restricted'], 0, 0.) / combined_area(base[h, m][0], 0, 0.) for m in models]
        dec_only = sum(combined_area(base[h, m][0], 0, 0.) / combined_area(base[G, m][0], 0, 0.) for m in models) / 4
        rows.append(dict(hardware=h, published=ip, base_disagg=ib, agg_only=ia, best=ix,
                         change_vs_base=ix / ib - 1, change_vs_published=ix / ip - 1, decode_only_bound=dec_only,
                         per_model_best_over_base=[x / y for x, y in zip(vx, vb)], agg_band_share=wx,
                         abs_area_best_over_disagg=absr, restricted_decode_area_ratio=rest,
                         max_zero_chunk_gap=max(parts[h, m]['max_zero_chunk_gap'] for m in models),
                         base_parity=max(parts[h, m]['base_parity_max_rel'] for m in models)))
    print(f"run={run.name} base={base_spec if base_spec == 'published' else Path(base_spec).name} variant={variant} "
          f"cells={len(parts)}  disagg-integral vs combined_area max rel err {err:.1e}")
    print(f"{'hardware':26s} {'publ':>6s} {'base':>6s} {'agg':>6s} {'best':>6s} {'chg':>6s} {'vsPub':>6s} {'decOnly':>7s}  "
          f"best/base per model (GLM Kimi Qwen MAI) | agg-wins share of log band")
    for x in sorted(rows, key=lambda x: -x['best']):
        print(f"{x['hardware'][:26]:26s} {x['published']:6.3f} {x['base_disagg']:6.3f} {x['agg_only']:6.3f} {x['best']:6.3f} "
              f"{x['change_vs_base']:+6.1%} {x['change_vs_published']:+6.1%} {x['decode_only_bound']:7.3f}  "
              + ' '.join(f'{v:4.2f}' for v in x['per_model_best_over_base']) + ' | '
              + ' '.join(f'{v:4.2f}' for v in x['agg_band_share']))
    print('\nAbsolute (un-normalized) area best/disagg per model, GLM Kimi Qwen MAI; restricted-search decode-only area / base:')
    for x in sorted(rows, key=lambda x: -x['best']):
        print(f"{x['hardware'][:26]:26s} " + ' '.join(f'{v:5.3f}' for v in x['abs_area_best_over_disagg'])
              + f"  mean {sum(x['abs_area_best_over_disagg'])/4:5.3f} | " + ' '.join(f'{v:5.3f}' for v in x['restricted_decode_area_ratio']))
    print('max zero-chunk gap', max(x['max_zero_chunk_gap'] for x in rows), ' max base parity', max(x['base_parity'] for x in rows))
    if '--json' in argv:
        Path(argv[argv.index('--json') + 1]).write_text(json.dumps(dict(variant=variant, base=base_spec, run=str(run), rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
