"""Final outputs: (1) published vs central-estimate bar chart with scenario-range whiskers,
(2) full matrix CSV. Central = all fixes + BF16 attention (stated policy) + measured cache misses (r = 8.56).
Range = min/max over {BF16, FP8 attention} x {no misses (r = 1.5133), measured misses (r = 8.56)}.
All fixes = KV read once + related re-reads + all-to-all dispatch (kfix C1-C7), W4A8 experts, parallelism search,
pipelined PP. Usage: venv python make_final.py REPO OUT_PREFIX"""
import csv, json, sys, importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo, outp = Path(sys.argv[1]), sys.argv[2]; ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'studies'))
from log400_release import combined_area
ref = json.loads((repo / 'reference/2026-10-15/curves_and_inputs.json').read_text()); R0, models = ref['ratio'], ref['models']
RM = 8.56
rub = json.loads((repo / 'results/rubin-update/curves_and_inputs.json').read_text())
pub = {(h, m): (ref['curves'][h][m], next(q['tokens_chip_second'] for q in ref['prefill_rates'] if q['hardware'] == h and q['model'] == m)) for h in ref['curves'] for m in models}
for m in models:
    pub['VR200 Rubin', m] = (rub['curves']['VR200'][m], next(q['tokens_chip_second'] for q in rub['prefill_rates'] if q['hardware'] == 'VR200' and q['model'] == m))
spec = importlib.util.spec_from_file_location('te', ROOT / 'scripts/trainium_ext.py'); te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
pub.update(te.saved_baseline())
def load(run):
    t = dict(pub)
    for f in (ROOT / 'results' / run / 'parts').glob('*.json'):
        p = json.loads(f.read_text()); t[p['hardware'], p['model']] = (p['frontier'], p['best_prefill']['input_tokens_chip_second'])
    return t
G = 'GB300 NVL72'
def idx(t, h, r=R0): return sum(combined_area(*t[h, m], r) / combined_area(*t[G, m], r) for m in models) / len(models)
CHIPS = [('MI455X', 'AMD MI455X'), ('VR200 Rubin', 'NVIDIA VR200'), ('Maia 300 rack', 'Microsoft Maia 300'), ('Jalapeño', 'OpenAI Jalapeño'),
         ('Meta MTIA 400', 'Meta MTIA 400'), ('GB300 NVL72', 'NVIDIA GB300'), ('TPU8i · corrected · hops', 'Google TPU v8i'),
         ('GB200 NVL72', 'NVIDIA GB200'), ('TPU7x Ironwood · FP8', 'Google TPU v7'), ('TPU8t · FP4 equivalent', 'Google TPU v8t'),
         ('Maia 200 rack', 'Microsoft Maia 200'), ('MI355X · 400G', 'AMD MI355X'), ('B300 · 800G', 'NVIDIA B300'), ('B200 · 800G', 'NVIDIA B200'),
         ('Ascend 950DT', 'Huawei Ascend 950DT'), ('Trainium3', 'AWS Trainium3'), ('Trainium2', 'AWS Trainium2'),
         ('H200 SXM · FP8 · 400G', 'NVIDIA H200'), ('H100 SXM · FP8 · 400G', 'NVIDIA H100')]
runs = {k: load(k) for k in ('kvonce', 'kfix-verified', 'free', 'pp', 'free+pp', 'e4a8', 'kfix-C3', 'attn-bf16', 'attn-fp8',
                             'allfix', 'allfix-bf16', 'allfix-fp8')}
COLS = [('published', lambda h: idx(pub, h)),
        ('1_kv_once', lambda h: idx(runs['kvonce'], h)),
        ('1b_kv_once_plus_related_rereads', lambda h: idx(runs['kfix-verified'], h)),
        ('2_parallelism_search', lambda h: idx(runs['free'], h)),
        ('3_pipelined_pp', lambda h: idx(runs['pp'], h)),
        ('2plus3', lambda h: idx(runs['free+pp'], h)),
        ('4_w4a8_experts', lambda h: idx(runs['e4a8'], h)),
        ('5_cache_misses_r8.56', lambda h: idx(pub, h, RM)),
        ('6_all_to_all_dispatch', lambda h: idx(runs['kfix-C3'], h)),
        ('7a_all_bf16_attention', lambda h: idx(runs['attn-bf16'], h)),
        ('7b_all_fp8_attention', lambda h: idx(runs['attn-fp8'], h)),
        ('allfix_orig_attn_no_misses', lambda h: idx(runs['allfix'], h)),
        ('allfix_orig_attn_misses', lambda h: idx(runs['allfix'], h, RM)),
        ('allfix_bf16_no_misses', lambda h: idx(runs['allfix-bf16'], h)),
        ('allfix_bf16_misses_CENTRAL', lambda h: idx(runs['allfix-bf16'], h, RM)),
        ('allfix_fp8_no_misses', lambda h: idx(runs['allfix-fp8'], h)),
        ('allfix_fp8_misses', lambda h: idx(runs['allfix-fp8'], h, RM))]
rows = []
for key, label in CHIPS:
    v = {name: f(key) for name, f in COLS}
    scen = [v['allfix_bf16_no_misses'], v['allfix_bf16_misses_CENTRAL'], v['allfix_fp8_no_misses'], v['allfix_fp8_misses']]
    v['range_min'], v['range_max'] = min(scen), max(scen)
    rows.append((label, key, v))
with open(outp + '_matrix.csv', 'w', newline='') as f:
    w = csv.writer(f); names = [c[0] for c in COLS] + ['range_min', 'range_max']
    w.writerow(['chip', 'profile'] + names)
    for label, key, v in rows: w.writerow([label, key] + [f'{v[n]:.4f}' for n in names])

# ---- chart: published vs central with range whiskers
rows_c = sorted(rows, key=lambda r: r[2]['published'])
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = ['#2a78d6', '#eb6834']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows_c) + 1.6), facecolor=SURF); ax.set_facecolor(SURF)
h, gap = 0.36, 0.03
for i, (label, key, v) in enumerate(rows_c):
    yp, yc = i + (h + gap) / 2, i - (h + gap) / 2
    ax.barh(yp, v['published'], height=h, color=COL[0], edgecolor=SURF, linewidth=0.8, zorder=2)
    ax.barh(yc, v['allfix_bf16_misses_CENTRAL'], height=h, color=COL[1], edgecolor=SURF, linewidth=0.8, zorder=2)
    ax.plot([v['range_min'], v['range_max']], [yc, yc], color=INK, lw=1.2, zorder=4, solid_capstyle='butt')
    for x in (v['range_min'], v['range_max']):
        ax.plot([x, x], [yc - h * 0.32, yc + h * 0.32], color=INK, lw=1.2, zorder=4)
    fmt = lambda x: f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
    bb = dict(facecolor=SURF, edgecolor='none', pad=0.6)
    ax.text(v['published'] + 0.03, yp, fmt(v['published']), va='center', ha='left', fontsize=7.5, color=INK, fontweight='bold', zorder=5, bbox=bb)
    c = v['allfix_bf16_misses_CENTRAL']
    inward = v['range_min'] < c - 0.005   # whisker reaches back into the bar
    if c >= 0.15 and not inward:   # inside the bar end, so it is never read as the whisker's end
        ax.text(c - 0.025, yc, fmt(c), va='center', ha='right', fontsize=7.5, color=INK, fontweight='bold', zorder=5)
    else:                          # short bar, or whisker points into the bar: just right of the bar
        ax.text(max(v['range_max'], c) + 0.03, yc, fmt(c), va='center', ha='left', fontsize=7.5, color=INK,
                fontweight='bold', zorder=5, bbox=bb)
ax.set_yticks(range(len(rows_c))); ax.set_yticklabels([r[0] for r in rows_c], color=INK)
ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.text(1.0, len(rows_c) - 0.45, 'GB300 = 1', fontsize=8, color=INK2, ha='left', va='bottom')
ax.set_xlabel('Inference Performance Index (GB300 NVL72 = 1)', color=INK2)
ax.grid(axis='x', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
xmax = max(max(r[2]['published'], r[2]['range_max']) for r in rows_c)
ax.set_xlim(0, xmax * 1.1); ax.set_ylim(-0.6, len(rows_c) - 0.25)
from matplotlib.lines import Line2D
ax.legend([plt.Rectangle((0, 0), 1, 1, color=COL[0]), plt.Rectangle((0, 0), 1, 1, color=COL[1]), Line2D([0], [0], color=INK, lw=1.2)],
          ['Published IPI', 'Corrected (central estimate)', 'Range across scenarios'], loc='lower right', frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.canvas.draw()
_bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
fig.text(0.5, (_bb.y1 + 1.0) / 2, 'Inference Performance Index', ha='center', va='center', fontsize=15, fontweight='bold', color=INK)
fig.savefig(outp + '.png', dpi=200, facecolor=SURF); fig.savefig(outp + '.svg', facecolor=SURF)
for label, key, v in rows:
    print(f"{label:22s} pub {v['published']:6.3f}  central {v['allfix_bf16_misses_CENTRAL']:6.3f}  range {v['range_min']:6.3f}-{v['range_max']:6.3f}")
