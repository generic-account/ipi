"""Baseline chart + 6 scenario charts with identical chip order and x-axis.
Scenarios: {original (BF16 prefill / FP8 decode), all-BF16, all-FP8 attention} x {no cache misses (r=1.5133), measured (r=8.56)},
each = all fixes (kfix C1-C7, W4A8 experts, parallelism search, pipelined PP) on that attention policy.
Usage: venv python make_scenarios.py REPO OUT_DIR"""
import json, sys, importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo, outd = Path(sys.argv[1]), Path(sys.argv[2]); ROOT = Path(__file__).resolve().parents[1]; outd.mkdir(exist_ok=True)
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
SCEN = [('1_orig_attn_no_misses', 'allfix', R0, 'All fixes  ·  Original attention (BF16 prefill, FP8 decode)  ·  No cache misses'),
        ('2_orig_attn_misses', 'allfix', RM, 'All fixes  ·  Original attention (BF16 prefill, FP8 decode)  ·  Measured cache misses'),
        ('3_bf16_no_misses', 'allfix-bf16', R0, 'All fixes  ·  All-BF16 attention  ·  No cache misses'),
        ('4_bf16_misses', 'allfix-bf16', RM, 'All fixes  ·  All-BF16 attention  ·  Measured cache misses (central estimate)'),
        ('5_fp8_no_misses', 'allfix-fp8', R0, 'All fixes  ·  All-FP8 attention  ·  No cache misses'),
        ('6_fp8_misses', 'allfix-fp8', RM, 'All fixes  ·  All-FP8 attention  ·  Measured cache misses')]
tables = {r: load(r) for r in {s[1] for s in SCEN}}
P = {k: idx(pub, k) for k, _ in CHIPS}
V = {s[0]: {k: idx(tables[s[1]], k, s[2]) for k, _ in CHIPS} for s in SCEN}
order = sorted(CHIPS, key=lambda c: P[c[0]])  # ascending -> largest at top
XMAX = max(max(P.values()), *(max(v.values()) for v in V.values())) * 1.12
json.dump(dict(published={l: P[k] for k, l in CHIPS}, **{s[0]: {l: V[s[0]][k] for k, l in CHIPS} for s in SCEN}),
          open(outd / 'scenarios.json', 'w'), indent=1)

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fmt = lambda x: f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
BB = dict(facecolor=SURF, edgecolor='none', pad=0.6)

def chart(name, series, legend, subtitle):
    nser = len(series)
    fig, ax = plt.subplots(figsize=(10, 0.66 * len(order) + 1.6), facecolor=SURF); ax.set_facecolor(SURF)
    h = 0.6 if nser == 1 else 0.38; gap = 0.03
    for i, (k, label) in enumerate(order):
        for j, (vals, col) in enumerate(series):
            y = i if nser == 1 else i + (h + gap) / 2 - j * (h + gap)
            v = vals[k]
            ax.barh(y, v, height=h, color=col, edgecolor=SURF, linewidth=0.8, zorder=2)
            ax.text(v + XMAX * 0.008, y, fmt(v), va='center', ha='left', fontsize=7.5, color=INK, fontweight='bold', zorder=5, bbox=BB)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order], color=INK)
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(1.0, len(order) - 0.45, 'GB300 = 1', fontsize=8, color=INK2, ha='left', va='bottom')
    ax.set_xlabel('Inference Performance Index (GB300 NVL72 = 1)', color=INK2)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.6, len(order) - 0.25)
    if legend:
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in series], legend, loc='lower right', frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    mid = (bb.y1 + 1.0) / 2
    fig.text(0.5, mid + 0.009, 'Inference Performance Index', ha='center', va='bottom', fontsize=15, fontweight='bold', color=INK)
    fig.text(0.5, mid - 0.004, subtitle, ha='center', va='top', fontsize=11.5, color=INK2)
    fig.savefig(outd / (name + '.png'), dpi=200, facecolor=SURF); fig.savefig(outd / (name + '.svg'), facecolor=SURF)
    plt.close(fig)

chart('0_baseline_published', [(P, BLUE)], ['Published IPI'], 'Published IPI (no fixes)')
for name, run, r, lab in SCEN:
    chart(name, [(P, BLUE), (V[name], ORANGE)], ['Published IPI', 'Corrected'], lab)
print('chip                   pub    ' + '  '.join(s[0][:14] for s in SCEN))
for k, l in CHIPS:
    print(f'{l:22s} {P[k]:6.3f} ' + ' '.join(f'{V[s[0]][k]:14.3f}' for s in SCEN))
