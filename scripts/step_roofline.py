"""Bound on cross-op compute/memory overlap: per-op-sum SOL vs whole-step roofline, at frontier points.
KV fix (C1) applied to generation_attention memory (non-GLM). Comm ops (no sol) counted separately.
per_op  = sum_i max(m_i, b_i) + comm        (IPI-style, after C1)
step    = max(sum m_i, sum b_i) + comm       (ideal overlap of compute and HBM traffic, comm unchanged)
"""
import json, sys, logging
from pathlib import Path
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo/'src'))
from agent_index.current_native import Native, KEYS, doctor
ref = repo/'reference/2026-10-15'; doctor(ref); logging.getLogger().setLevel(logging.ERROR)
cat = json.loads((ref/'catalog.json').read_text())
fr = {}
for f in Path(sys.argv[2], 'parts').glob('*.json'):
    p = json.loads(f.read_text()); fr[p['hardware'], p['model']] = p['frontier']
for hw in sys.argv[3:]:
    for m in ('Kimi-K3', 'Qwen-480B', 'MAI2-informed-h128-w512-sh4352', 'GLM-5.2'):
        n = Native(ref, cat[hw] | {'name': hw}, m); out = []
        for s in (50, 100, 200):
            pts = [p for p in fr[hw, m] if p['speed'] >= s]
            if not pts: out.append('   none'); continue
            p = max(pts, key=lambda q: q['capacity'])
            c, g, e, hop = n.graph('decode', {k: p[k] for k in KEYS})
            ops = e.static_phase_diagnostics(batch_size=p['batch'], context_length=262144, prefix=0, prefill=False)
            M = B = per = comm = 0.
            for o in ops:
                sol = (o.get('details') or {}).get('sol')
                if sol and sol.get('math_ms') is not None and sol.get('memory_ms') is not None:
                    mm, bb = sol['math_ms'], sol['memory_ms']
                    if o['name'] == 'generation_attention' and m != 'GLM-5.2': bb /= 4
                    M += mm; B += bb; per += max(mm, bb)
                else:
                    comm += o['latency_ms']
            out.append(f"{(per+comm)/(max(M,B)+comm):6.2f}x")
        print(f"{hw[:22]:22s} {m[:6]:6s} per-op/step @50,100,200 tok/s: " + ' '.join(out))
