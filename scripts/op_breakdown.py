"""Per-op decode latency for a saved frontier point; test whether attention scales like 4x KV reads."""
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo/'src'))
from agent_index.current_native import Native, KEYS, doctor
from agent_index import simulation
ref = repo/'reference/2026-10-15'
cat = json.loads((ref/"catalog.json").read_text())
curves = json.loads((ref/'curves_and_inputs.json').read_text())['curves']
hwname, model, idx = sys.argv[2], sys.argv[3], int(sys.argv[4])
pt = curves[hwname][model][idx]
layout = {k: pt[k] for k in KEYS}
doctor(ref)
hw = cat[hwname] | {'name': hwname}
n = Native(ref, hw, model)
c, g, engine, hop = n.graph('decode', layout)
def ops(b):
    return engine.static_phase_diagnostics(batch_size=b, context_length=262144, prefix=0, prefill=False)
B = pt['batch']
o1, o2 = ops(B), ops(2*B)
print(hwname, model, pt.get('layout', {k: pt[k] for k in KEYS}), 'batch', B, 'speed', round(pt['speed'],1), 'saved ms', round(pt['raw_ms'],3), 'hop', hop)
print('total ms @B', round(sum(o['latency_ms'] for o in o1),3), '@2B', round(sum(o['latency_ms'] for o in o2),3))
print('keys:', list(o1[0].keys()))
agg = {}
for a, b in zip(o1, o2):
    nm = a.get('name') or a.get('op')
    agg.setdefault(nm, [0, 0]); agg[nm][0] += a['latency_ms']; agg[nm][1] += b['latency_ms']
for nm, (x, y) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"{nm:45s} {x:9.3f} {y:9.3f}  marginal/seq {(y-x)/B*1000:8.2f} us")
for o in o1:
    if o['name'] in ('generation_attention','draft_attention'):
        print(o['name'], json.dumps(o['details'])[:1500]); 
mem = n.memory('decode', layout, B); print('memory', {k: v for k, v in mem.items()})
