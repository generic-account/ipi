"""Shared helpers for the kernel-accounting audit (read-only use of the IPI repo)."""
import json, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'ipi-repo'
sys.path.insert(0, str(REPO/'src'))
from agent_index.current_native import Native, KEYS, doctor
REF = REPO/'reference/2026-10-15'
_OV = ROOT / 'overlay_ref'
if _OV.exists():  # superset of frozen ref (adds VR200 Rubin, Trainium2/3); identical symlinked inputs
    REF = _OV
CAT = json.loads((REF/'catalog.json').read_text())
CURVES = json.loads((REF/'curves_and_inputs.json').read_text())['curves']
_doc = False
from functools import lru_cache
@lru_cache(None)
def hwyaml(hwname):
    import yaml
    return yaml.safe_load((REF/'hardware'/(CAT[hwname]['profile']+'.yaml')).read_text())
def native(hwname, model):
    global _doc
    if not _doc:
        doctor(REF); _doc = True
    return Native(REF, CAT[hwname] | {'name': hwname}, model)
def point(hwname, model, idx):
    pt = CURVES[hwname][model][idx]
    return pt, {k: pt[k] for k in KEYS}
def diag(n, layout, batch, phase='decode'):
    c, g, engine, hop = n.graph(phase, layout)
    pre = phase == 'prefill'
    return engine.static_phase_diagnostics(batch_size=batch,
        context_length=262144 + (1024 if pre else 0), prefix=262144 if pre else 0, prefill=pre)
def specs(n, layout, phase='decode'):
    c, g, engine, hop = n.graph(phase, layout)
    ops = g.generation_ops if phase == 'decode' else g.context_ops
    out = []
    for o in ops:
        s = json.loads(o._spec_json()); k, v = next(iter(s.items()))
        out.append((k, v))
    return out
def lay(p):
    return p.get('layout') or ' '.join(f"{k.upper()}{p[k]}" for k in KEYS)
HWS = ['GB300 NVL72','VR200','B300 · 800G','TPU7x Ironwood · FP8','MI455X','H200 SXM · FP8 · 400G']
MODELS = ['GLM-5.2','Kimi-K3','Qwen-480B','MAI2-informed-h128-w512-sh4352']
def reps(h, m, targets=(60,120,250)):
    c = CURVES[h][m]; out = []
    for t in targets:
        cand = [(abs(math.log(p['speed']/t)), i) for i, p in enumerate(c) if 50 <= p['speed'] <= 400]
        if cand:
            d, i = min(cand)
            if i not in out: out.append(i)
    return out
