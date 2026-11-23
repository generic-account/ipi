"""IPI sweep with kernel-accounting corrections (in-process patch; repo not modified).

Same mechanism as scripts/sweep_kv_once.py: Native.graph is wrapped so that
engine.static_phase_diagnostics returns corrected per-op latencies. Corrections
(select with --fix C1,C2,...; default C1,C2,C3,C4,C5):

 C1 generation_attention (GQA/MLA, not DSA): max(math, memory/4)   [KV read once per request]
 C2 draft_attention (Kimi DSpark): max(math, memory/3)             [one parallel draft forward]
 C3 MoeDispatch: alpha + (t-alpha)*r, r = dp/(dp-1)*P_hit*(0.516 dispatch fp8 | 1.0 combine bf16),
    P_hit = 1-(1-1/ep)^topk, alpha = native time with attention_dp_size=1   [a2a with per-rank dedup]
 C4 DsaGeneration: t - 0.75*idx_ms, idx_ms = 4B*count*full_frac*ceil(s/dcp)*132/mem_bw  [indexer keys once]
 C5 DsaGeneration with dcp>1: t - [t(dcp) - t(dcp=1)] at 1 request, s=2049  [projection weights at local heads]
 C6 generation_kda_verify: t*2/5                                  [state 1R+1W instead of 1R+4W]
 C7 GLM dcp_topk_{pack,all_gather,merge}: t*21/78                 [shared-indexer layers reuse top-k]

Overlap ops (GLM MoE) are re-folded from native leaves as max(branch sums).
Usage: python sweep_kernel_fixes.py REPO OUT [--fix C1,C2,...] [reproduce.py sweep args]
"""
import copy, json, math, sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(); out = sys.argv[2]
fixes = {'C1', 'C2', 'C3', 'C4', 'C5'}
rest = sys.argv[3:]
if '--fix' in rest:
    i = rest.index('--fix'); fixes = set(rest[i+1].split(',')); rest = rest[:i] + rest[i+2:]
sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))
from agent_index import current_native
from agent_index.interconnect import NativeLeaves

def _yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())['gpu']

class _Engine:
    def __init__(self, inner, config, graph, model):
        self._inner, self._c, self._g, self._model = inner, config, graph, model
        self._nl = NativeLeaves(config, graph, 'decode', False)
        self._gpu = _yaml(Path(config['systems_paths'][0]) / (config['system'] + '.yaml'))
        self._memo = {}
    def __getattr__(self, k):
        return getattr(self._inner, k)
    def _native1(self, spec, batch, s):
        r = self._nl.handle.evaluate_ops_json(json.dumps([spec]), is_context=False, batch_size=4*batch, s=s, prefix=0)
        return r[0][1]
    def _alpha(self, kind, v):
        key = ('a', v['name'])
        if key not in self._memo:
            w = copy.deepcopy(v); w['attention_dp_size'] = 1; w['name'] = 'alpha_' + v['name']
            self._memo[key] = self._native1({kind: w}, 1, 262145)
        return self._memo[key]
    def _c5(self, kind, v):
        key = ('c5', v['name'])
        if key not in self._memo:
            a = copy.deepcopy(v); a['name'] = 'c5a'; b = copy.deepcopy(v); b['name'] = 'c5b'; b['dcp_size'] = 1
            self._memo[key] = max(0., self._native1({kind: a}, 1, 2049) - self._native1({kind: b}, 1, 2049))
        return self._memo[key]
    def _fix_leaf(self, kind, v, t, sol, batch):
        nm = v['name']
        if 'C1' in fixes and nm == 'generation_attention' and kind in ('GenerationAttention', 'GenerationMla') and sol:
            return None  # handled at op level (merged duplicates share one sol)
        if 'C2' in fixes and nm == 'draft_attention' and sol:
            return max(sol['math_ms'], sol['memory_ms'] / 3)
        if 'C3' in fixes and kind == 'MoeDispatch' and v['attention_dp_size'] > 1:
            dp, ep, k = v['attention_dp_size'], v['moe_ep_size'], v['topk']
            r = min(1., dp/(dp-1) * (1-(1-1/ep)**k) * ((1+4/128)/2 if v['pre_dispatch'] else 1.))
            a = self._alpha(kind, v); return a + max(0., t-a) * r
        if kind == 'DsaGeneration':
            if 'C4' in fixes:
                t -= 0.75 * 4*batch*v['scale_factor']*v['full_frac']*math.ceil(262145/v.get('dcp_size', 1))*132/self._gpu['mem_bw']*1e3
            if 'C5' in fixes and v.get('dcp_size', 1) > 1:
                t -= self._c5(kind, v)
            return max(t, 0.)
        if 'C6' in fixes and nm == 'generation_kda_verify':
            return t * 2/5
        if 'C7' in fixes and self._model == 'GLM-5.2' and 'dcp_topk' in nm:
            return t * 21/78
        return t
    def static_phase_diagnostics(self, **kw):
        ops = self._inner.static_phase_diagnostics(**kw)
        if kw.get('prefill'):
            return ops
        batch = kw['batch_size']
        sols = {o['name']: (o['details'].get('sol') or {}) for o in ops}
        leaves = self._nl.tree.leaves
        t = self._nl.times(batch)
        # per top-level op: corrected latency (overlap re-folded)
        def fold(node):
            kind, val = node
            if kind == 'leaf':
                k, v = next(iter(leaves[val].items())); v = dict(v, name=v['name'].split('_', 2)[2])
                r = self._fix_leaf(k, v, t[val], sols.get(v['name']), batch)
                return t[val] if r is None else r
            xs = [fold(x) for x in val]
            return max(xs) if kind == 'max' else math.fsum(xs)
        top = self._nl.tree.tree[1]
        newt = {}
        for node in top:
            if node[0] == 'leaf':
                nm = next(iter(leaves[node[1]].values()))['name'].split('_', 2)[2]
            else:  # overlap: find its op name from the original order
                nm = None
            newt.setdefault(nm, []).append(fold(node))
        it = iter([x for x in newt.get(None, [])])
        for o in ops:
            nm = o['name']
            if nm in newt and nm is not None:
                o['latency_ms'] = math.fsum(newt[nm])
            elif nm.endswith('_overlap'):
                o['latency_ms'] = next(it)
        if 'C1' in fixes:
            for o in ops:
                s = sols.get(o['name'])
                if o['name'] == 'generation_attention' and s and self._model != 'GLM-5.2':
                    o['latency_ms'] = max(s['math_ms'], s['memory_ms'] / 4)
        return ops

_graph = current_native.Native.graph
def graph(self, phase, layout):
    c, g, engine, hop = _graph(self, phase, layout)
    if phase == 'decode' and not isinstance(engine, _Engine):
        engine = _Engine(engine, c, g, self.model)
        self.cache[(phase, *(layout[k] for k in current_native.KEYS))] = (c, g, engine, hop)
    return c, g, engine, hop
current_native.Native.graph = graph

if __name__ == '__main__' and out != '--smoke':
    import runpy
    sys.argv = [str(repo / 'reproduce.py'), 'sweep', '--output', out, *rest]
    runpy.run_path(str(repo / 'reproduce.py'), run_name='__main__')
