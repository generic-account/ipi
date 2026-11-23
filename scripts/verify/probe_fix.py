"""Single-point probe: per-op decode latency for one saved frontier point, native vs
kernel_audit/sweep_kernel_fixes.py corrections. Light (one layout, one batch); no sweep.

Usage: ipi-repo/.venv-current/bin/python scripts/verify/probe_fix.py REPO 'HW' MODEL IDX C1,C2,...
"""
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve(); hw, model, idx, fx = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))
from agent_index.current_native import Native, KEYS, doctor
ref = repo / 'reference/2026-10-15'
cat = json.loads((ref / 'catalog.json').read_text())
pt = json.loads((ref / 'curves_and_inputs.json').read_text())['curves'][hw][model][idx]
layout = {k: pt[k] for k in KEYS}; B = pt['batch']
doctor(ref)
def ops():
    n = Native(ref, cat[hw] | {'name': hw}, model)
    c, g, e, h = n.graph('decode', layout)
    out = {}
    for o in e.static_phase_diagnostics(batch_size=B, context_length=262144, prefix=0, prefill=False):
        out[o['name']] = out.get(o['name'], 0.) + o['latency_ms']
    return out
base = ops()
import importlib.util
here = Path(__file__).resolve().parents[1]
sys.argv = ['x', str(repo), '--smoke', '--fix', fx]
spec = importlib.util.spec_from_file_location('kfix', here / 'kernel_audit' / 'sweep_kernel_fixes.py')
spec.loader.exec_module(importlib.util.module_from_spec(spec))
fixed = ops()
print(hw, model, idx, layout, 'batch', B, 'fixes', fx)
print(f"{'op':48s} {'native_ms':>10s} {'fixed_ms':>10s}")
for k in sorted(base, key=lambda k: -base[k]):
    if base[k] > 1e-3 or abs(base[k] - fixed[k]) > 1e-6:
        print(f"{k:48s} {base[k]:10.3f} {fixed[k]:10.3f}" + ('  *' if abs(base[k] - fixed[k]) > 1e-6 else ''))
print(f"{'TOTAL':48s} {sum(base.values()):10.3f} {sum(fixed.values()):10.3f}")
