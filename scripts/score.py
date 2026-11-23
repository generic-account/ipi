"""Score a sweep (possibly partial/targeted) against a baseline, producing full 23-chip index.

Usage: python3 -I score.py REPO RUN_DIR [--base BASE_DIR] [--chips-only]
  BASE_DIR omitted -> published 15-Oct frozen curves (+ 20-Oct 'VR200 Rubin', + saved Trainium2/3 study runs). Parts in RUN_DIR override base per (hw, model).
  Prints old (base) vs new combined index and per-model ratio changes, for chips with all 4 models.
"""
import json, sys
from pathlib import Path
repo, run = Path(sys.argv[1]), Path(sys.argv[2])
base = Path(sys.argv[sys.argv.index('--base') + 1]) if '--base' in sys.argv else None
sys.path.insert(0, str(repo / 'studies'))
from log400_release import combined_area
ref = json.loads((repo / 'reference/2026-10-15/curves_and_inputs.json').read_text())
r, models, G = ref['ratio'], ref['models'], 'GB300 NVL72'

def load_parts(d):
    out = {}
    for f in (d / 'parts').glob('*.json'):
        p = json.loads(f.read_text())
        P = p['best_prefill']['input_tokens_chip_second'] if p['best_prefill'] else 0.
        out[p['hardware'], p['model']] = (p['frontier'], P)
    return out

frozen = {(h, m): (ref['curves'][h][m], x) for h in ref['curves'] for m in models
          for x in [next(q['tokens_chip_second'] for q in ref['prefill_rates'] if q['hardware'] == h and q['model'] == m)]}
# Published Rubin revision (20 Oct) as baseline for the overlay key 'VR200 Rubin'
_rub = json.loads((repo / 'results/rubin-update/curves_and_inputs.json').read_text())
for m in models:
    frozen['VR200 Rubin', m] = (_rub['curves']['VR200'][m],
        next(q['tokens_chip_second'] for q in _rub['prefill_rates'] if q['hardware'] == 'VR200' and q['model'] == m))
# Published Trainium2/3 research-extension baselines (Trainium2 64_link_limited = 0.128,
# Trainium3 64_detailed_tiers = 0.349): saved study runs docs/research/trainium/runs/*.json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trainium_ext import saved_baseline
frozen.update(saved_baseline())
old = dict(frozen)
if base: old.update(load_parts(base))
new = dict(old); swept = load_parts(run); new.update(swept)

def idx(src, h):
    per = [combined_area(*src[h, m], r) / combined_area(*src[G, m], r) for m in models]
    return sum(per) / len(per), per

hws = list(ref['curves']) + sorted({h for (h, m) in frozen if h not in ref['curves']})
if '--chips-only' in sys.argv:
    hws = [h for h in hws if any((h, m) in swept for m in models)]
missing_g = [m for m in models if (G, m) not in old]
print(f"run={run.name} base={'published' if not base else base.name} swept_cells={len(swept)}")
print(f"{'hardware':26s} {'base':>6s} {'new':>6s} {'chg':>6s}   per-model new/base (GLM Kimi Qwen MAI)")
for h in hws:
    (so, po), (sn, pn) = idx(old, h), idx(new, h)
    n_sw = sum((h, m) in swept for m in models)
    flag = '' if n_sw == len(models) else f'  [{n_sw}/4 swept; rest from base]'
    print(f"{h[:26]:26s} {so:6.3f} {sn:6.3f} {sn/so-1:+6.0%}   " + ' '.join(f'{b/a:5.2f}' if a else '  n/a' for a, b in zip(po, pn)) + flag)
g = [combined_area(*new[G, m], r) / combined_area(*old[G, m], r) for m in models]
print('GB300 absolute area new/base per model:', ' '.join(f'{x:.2f}' for x in g))
