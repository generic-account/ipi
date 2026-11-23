"""Fresh IPI sweep with spec-decode verify attention reading the KV cache once per step.

AISim prices decode ops at batch*(nextn+1) sequences each at full 262k context, so
`generation_attention` reads the whole KV cache once per verify position (4x with nextn=3).
Fused verify kernels (FlashInfer/FlashMLA with q_len=4) read it once. This keeps the
4x attention math but divides the attention memory term by 4:
    latency = max(math_ms, memory_ms / 4)

Usage: python sweep_kv_once.py REPO OUT [--include-glm] [--hardware H ...] [--model M ...]
The repo is not modified; patches are applied in-process only.
"""
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(); out = sys.argv[2]
include_glm = '--include-glm' in sys.argv
extra = [a for a in sys.argv[3:] if a != '--include-glm']
sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))

from agent_index import current_native

PATCH_OPS = {'generation_attention'}
VERIFY = 4  # nextn + 1

class _Engine:
    def __init__(self, inner, model):
        self._inner, self._model = inner, model
    def __getattr__(self, k):
        return getattr(self._inner, k)
    def static_phase_diagnostics(self, **kw):
        ops = self._inner.static_phase_diagnostics(**kw)
        if kw.get('prefill') or (self._model == 'GLM-5.2' and not include_glm):
            return ops
        for o in ops:
            if o['name'] in PATCH_OPS:
                sol = o['details']['sol']
                o['latency_ms'] = max(sol['math_ms'], sol['memory_ms'] / VERIFY)
        return ops

_graph = current_native.Native.graph
def graph(self, phase, layout):
    c, g, engine, hop = _graph(self, phase, layout)
    if not isinstance(engine, _Engine):
        engine = _Engine(engine, self.model)
        self.cache[(phase, *(layout[k] for k in current_native.KEYS))] = (c, g, engine, hop)
    return c, g, engine, hop
current_native.Native.graph = graph

import runpy
sys.argv = [str(repo / 'reproduce.py'), 'sweep', '--output', out, *extra]
runpy.run_path(str(repo / 'reproduce.py'), run_name='__main__')
