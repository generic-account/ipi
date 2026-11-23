"""Heatmap: % change in IPI from each fix applied alone (vs published), chips x fixes.
Usage: venv python make_heatmap.py REPO OUT_PREFIX"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

repo, outp = Path(sys.argv[1]), sys.argv[2]; ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'studies'))
from log400_release import combined_area
ref = json.loads((repo / 'reference/2026-10-15/curves_and_inputs.json').read_text()); R0, models = ref['ratio'], ref['models']
rub = json.loads((repo / 'results/rubin-update/curves_and_inputs.json').read_text())
pub = {(h, m): (ref['curves'][h][m], next(q['tokens_chip_second'] for q in ref['prefill_rates'] if q['hardware'] == h and q['model'] == m)) for h in ref['curves'] for m in models}
for m in models:
    pub['VR200 Rubin', m] = (rub['curves']['VR200'][m], next(q['tokens_chip_second'] for q in rub['prefill_rates'] if q['hardware'] == 'VR200' and q['model'] == m))
# Trainium published baselines: saved study runs (same as score.py)
import importlib.util
spec = importlib.util.spec_from_file_location('trainium_ext', ROOT / 'scripts/trainium_ext.py'); te = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(te); tb = te.saved_baseline()
    for (h, m), v in tb.items(): pub[h, m] = v
except Exception as e:
    print('trainium baseline unavailable:', e)
def load(run):
    t = dict(pub)
    for f in (ROOT / 'results' / run / 'parts').glob('*.json'):
        p = json.loads(f.read_text()); t[p['hardware'], p['model']] = (p['frontier'], p['best_prefill']['input_tokens_chip_second'])
    return t
G = 'GB300 NVL72'
def idx(t, h, r=R0):
    return sum(combined_area(*t[h, m], r) / combined_area(*t[G, m], r) for m in models) / len(models)
COLS = [('1. KV read\nonce', 'kvonce', R0), ('1b. Related\nre-reads (on 1)', ('kfix-verified', 'kvonce'), R0),
        ('2. Parallelism\nsearch', 'free', R0), ('3. Pipelined PP', 'pp', R0), ('2+3', 'free+pp', R0),
        ('4. 4-bit experts', 'e4a8', R0), ('5. Cache misses\n(measured)', None, 8.56), ('6. All-to-all\ndispatch', 'kfix-C3', R0),
        ('7a. All-BF16\nattention', 'attn-bf16', R0), ('7b. All-FP8\nattention', 'attn-fp8', R0)]
CHIPS = [('MI455X', 'AMD MI455X'), ('VR200 Rubin', 'NVIDIA VR200'), ('Maia 300 rack', 'Microsoft Maia 300'), ('Jalapeño', 'OpenAI Jalapeño'),
         ('Meta MTIA 400', 'Meta MTIA 400'), ('TPU8i · corrected · hops', 'Google TPU v8i'), ('GB200 NVL72', 'NVIDIA GB200'),
         ('TPU7x Ironwood · FP8', 'Google TPU v7'), ('TPU8t · FP4 equivalent', 'Google TPU v8t'), ('Maia 200 rack', 'Microsoft Maia 200'),
         ('MI355X · 400G', 'AMD MI355X'), ('B300 · 800G', 'NVIDIA B300'), ('B200 · 800G', 'NVIDIA B200'), ('Ascend 950DT', 'Huawei Ascend 950DT'),
         ('Trainium3', 'AWS Trainium3'), ('Trainium2', 'AWS Trainium2'), ('H200 SXM · FP8 · 400G', 'NVIDIA H200'), ('H100 SXM · FP8 · 400G', 'NVIDIA H100')]
cache = {}
def table(run):
    if run not in cache: cache[run] = load(run)
    return cache[run]
M = []
for key, label in CHIPS:
    base = idx(pub, key); row = []
    for _, run, r in COLS:
        try:
            if run is None: v = idx(pub, key, r) / base - 1
            elif isinstance(run, tuple): v = idx(table(run[0]), key) / idx(table(run[1]), key) - 1
            else: v = idx(table(run), key) / base - 1
        except KeyError:
            v = float('nan')
        row.append(v * 100)
    M.append(row)
json.dump(dict(chips=[c[1] for c in CHIPS], fixes=[c[0].replace('\n', ' ') for c in COLS], pct=M), open(outp + '.json', 'w'), indent=1)

SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
cmap = LinearSegmentedColormap.from_list('div', ['#c33d3c', '#e98c8b', '#f3f2ee', '#86b6ef', '#1f5fae'])  # red (down) - gray - blue (up)
fig, ax = plt.subplots(figsize=(12.5, 8.2), facecolor=SURF); ax.set_facecolor(SURF)
import numpy as np
A = np.array(M, dtype=float)
norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
im = ax.imshow(np.clip(A, -40, 40), cmap=cmap, norm=norm, aspect='auto')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        v = A[i, j]
        txt = '–' if np.isnan(v) else (f'{v:+.0f}%' if abs(v) >= 0.5 else '0%')
        ax.text(j, i, txt, ha='center', va='center', fontsize=8.5, color=INK if np.isnan(v) or abs(v) < 28 else '#ffffff')
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[0] for c in COLS], fontsize=8.5, color=INK)
ax.xaxis.tick_top()
ax.set_yticks(range(len(CHIPS))); ax.set_yticklabels([c[1] for c in CHIPS], fontsize=9, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(COLS)), minor=True); ax.set_yticks(np.arange(-.5, len(CHIPS)), minor=True)
ax.grid(which='minor', color=SURF, linewidth=2); ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label('% change in IPI vs published (clipped at ±40%)', color=INK2, fontsize=8.5)
cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.canvas.draw()
_bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
fig.text(0.5, (_bb.y1 + 1.0) / 2, 'Inference Performance Index', ha='center', va='center',
         fontsize=15, fontweight='bold', color=INK)
fig.savefig(outp + '.png', dpi=200, facecolor=SURF); fig.savefig(outp + '.svg', facecolor=SURF)
for (k, l), r in zip(CHIPS, M): print(f'{l:22s} ' + ' '.join(f'{x:+5.0f}' for x in r))
