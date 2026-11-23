"""In-process IPI sweep patch: precision and fusion/overlap switches (repo never modified).

Usage:
  python sweep_precision.py REPO OUT --prec g8,dispS,e4a8[,c1,att16,kv4,nccl8,ar8,w4a16,disp8,disp4]
                                     [--fuse none|ovl|tbo|moefuse|arnorm|asynctp|dcpovl|launch2|launch5]
                                     [reproduce.py sweep args, e.g. --hardware H --model M]

Decode timing is replaced by an exact native leaf re-timing (NativeLeaves, Overlap re-folded as
max of branch sums) with the switches in retime.py / fusion.py applied. 'e4a8' also changes memory
admission (graph built with moe_quant_mode=w4afp8 on fp8-expert profiles), so capacity is re-searched.
Other precision switches are timing-only (weights for g8 are a few % of HBM; flagged in the report).
Prefill is unchanged except 'e4a8' (graph-level) - use prefill_prec.py for prefill precision.
"""
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(); out = sys.argv[2]
args = sys.argv[3:]
prec, fuse, rest = set(), 'none', []
i = 0
while i < len(args):
    if args[i] == '--prec': prec = set(x for x in args[i+1].split(',') if x); i += 2
    elif args[i] == '--fuse': fuse = args[i+1]; i += 2
    else: rest.append(args[i]); i += 1
print('PRECISION switches:', sorted(prec), 'fuse:', fuse, flush=True)
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))
from agent_index import current_native
import retime, fusion
from agent_index.interconnect import NativeLeaves

_g = current_native.Native.graph
def graph(self, phase, layout):
    if 'e4a8' in prec and self.hw.get('quant') == 'fp8' and not getattr(self, '_e4a8', False):
        self.hw = dict(self.hw, quant='w4afp8'); self._e4a8 = True; self.cache.clear()
    return _g(self, phase, layout)
current_native.Native.graph = graph

_leaf = {}
def _ctx_for(native):
    def ctx(hw, model, l):
        key = (hw, model, *(l[k] for k in current_native.KEYS))
        if key not in _leaf:
            c, g, e, hop = native.graph('decode', l)
            _leaf[key] = (NativeLeaves(c, g, 'decode', False), hop)
        return _leaf[key]
    return ctx

_ev = current_native.Native.evaluate
def evaluate(self, phase, layout, batch):
    r = _ev(self, phase, layout, batch)
    if phase != 'decode' or (not prec - {'e4a8'} and fuse == 'none'):
        return r
    retime.ctx = fusion.ctx = _ctx_for(self)
    ms = fusion.cycle(self.hw['name'], self.model, layout, batch, fuse, frozenset(prec - {'e4a8'}))
    r.update(raw_ms=ms, speed=3000 / ms)
    return r
current_native.Native.evaluate = evaluate

if __name__ == '__main__' and out != '--smoke':
    import runpy
    sys.argv = [str(repo / 'reproduce.py'), 'sweep', '--output', out, *rest]
    runpy.run_path(str(repo / 'reproduce.py'), run_name='__main__')
