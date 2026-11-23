"""Re-integrate a run's curves at given input/output ratios r (vs the same run at r0=1.5133).
Usage: python3 -I ratio_at.py REPO RUN_DIR|published r1 [r2 ...]"""
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo / 'studies'))
from log400_release import combined_area
ref = json.loads((repo / 'reference/2026-10-15/curves_and_inputs.json').read_text()); r0, models = ref['ratio'], ref['models']
rub = json.loads((repo / 'results/rubin-update/curves_and_inputs.json').read_text())
src = {(h, m): (ref['curves'][h][m], next(q['tokens_chip_second'] for q in ref['prefill_rates'] if q['hardware'] == h and q['model'] == m)) for h in ref['curves'] for m in models}
for m in models:
    src['VR200 Rubin', m] = (rub['curves']['VR200'][m], next(q['tokens_chip_second'] for q in rub['prefill_rates'] if q['hardware'] == 'VR200' and q['model'] == m))
if sys.argv[2] != 'published':
    for f in Path(sys.argv[2], 'parts').glob('*.json'):
        p = json.loads(f.read_text()); src[p['hardware'], p['model']] = (p['frontier'], p['best_prefill']['input_tokens_chip_second'])
rs = [r0] + [float(x) for x in sys.argv[3:]]
G = 'GB300 NVL72'
def idx(h, r): return sum(combined_area(*src[h, m], r) / combined_area(*src[G, m], r) for m in models) / 4
chips = ['MI455X', 'VR200 Rubin', 'TPU8i · corrected · hops', 'GB200 NVL72', 'TPU7x Ironwood · FP8', 'Maia 200 rack', 'MI355X · 400G', 'B300 · 800G',
         'H200 SXM · FP8 · 400G', 'H100 SXM · FP8 · 400G', 'Trainium3', 'Trainium2', 'B200 · 800G', 'Maia 300 rack', 'Jalapeño', 'Meta MTIA 400', 'TPU8t · FP4 equivalent', 'Ascend 950DT']
print(f"{'chip':24s} " + ' '.join(f'r={r:<6.2f}' for r in rs) + '   change at each r vs r0')
for h in chips:
    if not all((h, m) in src for m in models): continue
    v = [idx(h, r) for r in rs]
    print(f"{h[:24]:24s} " + ' '.join(f'{x:8.3f}' for x in v) + '   ' + ' '.join(f'{x/v[0]-1:+5.0%}' for x in v[1:]))
