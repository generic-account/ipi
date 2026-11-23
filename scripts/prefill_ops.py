"""List prefill ops with math/memory split for the saved best-prefill layout (to find attention ops)."""
import json, sys, logging
from pathlib import Path
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo/'src'))
from agent_index.current_native import Native, KEYS, doctor
ref = repo/'reference/2026-10-15'; doctor(ref); logging.getLogger().setLevel(logging.ERROR)
cat = json.loads((ref/'catalog.json').read_text()); pw = {(p['hardware'], p['model']): p for p in json.loads((ref/'prefill_winners.json').read_text())}
hw, m = sys.argv[2], sys.argv[3]; p = pw[hw, m]
n = Native(ref, cat[hw] | {'name': hw}, m); c, g, e, hop = n.graph('prefill', {k: p[k] for k in KEYS})
ops = e.static_phase_diagnostics(batch_size=p['batch'], context_length=262144+1024, prefix=262144, prefill=True)
tot = sum(o['latency_ms'] for o in ops); print(hw, m, {k: p[k] for k in KEYS}, 'batch', p['batch'], 'total', round(tot, 2), 'ms')
for o in sorted(ops, key=lambda o: -o['latency_ms'])[:8]:
    s = (o.get('details') or {}).get('sol') or {}
    print(f"  {o['name']:40s} {o['latency_ms']:9.3f}  math {s.get('math_ms')}  mem {s.get('memory_ms')}")
