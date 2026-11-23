"""Stackable in-process patches for IPI sweeps. Repo is never modified.

Usage:
  python harness.py REPO OUT --switch NAME[=VAL] [--switch ...] [reproduce.py sweep args]

Switches (combine freely):
  kfix=C1,C2,...   kernel-accounting fixes from kernel_audit/sweep_kernel_fixes.py
  budget=N         cap every profile's chip_budget at N (e.g. 64)
  pp_pipelined     PP stages each hold a microbatch in flight: requests = batch*dp*pp
                   (memory already reserves whole-model KV on every stage; cycle unchanged)
  comm_overlap     ideal compute/comm overlap: decode cycle = max(sum comm ops, sum other ops)
  ep_imbalance=F   multiply MoE expert GEMM + dispatch ops by F when EP >= 32
  tpu_bisection    multiply MoE dispatch ops on TPU profiles by a torus bisection factor:
                   EP<=16:1, 32:2, 64:3, 128:6 (estimate for 4x4x4 / 4x4x8 slices)
  attn_prec=bf16|fp8  attention arithmetic precision: bf16 = decode attention at BF16 (stated policy);
                   fp8 = prefill attention at FP8 (all-FP8). GLM DSA excluded (matches precision_audit att16)
  w4a8_experts     4-bit expert storage with FP8 compute on fp8-expert profiles (H100/H200/TPU7x/Trainium2)
  derate=M,B       divide math by M and memory by B fractions (e.g. 0.9,0.85): latency=max(math/M, mem/B)
"""
import sys, math
from pathlib import Path

repo = Path(sys.argv[1]).resolve(); out = sys.argv[2]
args = sys.argv[3:]
sw = {}
rest = []
i = 0
while i < len(args):
    if args[i] == '--switch':
        k, _, v = args[i+1].partition('='); sw[k] = v or True; i += 2
    else:
        rest.append(args[i]); i += 1
print('HARNESS switches:', sw, flush=True)
sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))
from agent_index import current_native
import os
if os.environ.get('IPI_NO_FASTYAML') != '1':
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import fastyaml  # noqa: F401  (speed only)

# 0. Trainium2/3 research-extension profiles (scripts/trainium_ext.py): repriced MoE leaves + MX
# scale memory, only for those profiles. Installed FIRST so kfix/prec/op switches wrap it.
if os.environ.get('IPI_NO_TRAINIUM') != '1':
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import trainium_ext; trainium_ext.install(sw)

# 1. kernel fixes (patches Native.graph first)
if 'kfix' in sw:
    import importlib.util
    here = Path(__file__).resolve().parent
    saved = sys.argv
    sys.argv = ['x', str(repo), '--smoke', '--fix', sw['kfix']]
    spec = importlib.util.spec_from_file_location('kfix', here / 'kernel_audit' / 'sweep_kernel_fixes.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.argv = saved

# 1a. precision / fusion re-timing (precision_audit/sweep_precision.py). Replaces decode timing
# with exact native leaf re-timing; has its own 'c1' (KV once). Mutually exclusive with kfix.
if 'prec' in sw or 'fuse' in sw:
    if 'kfix' in sw:
        raise SystemExit('prec/fuse and kfix both replace decode timing; use prec=c1,... instead')
    import importlib.util
    here = Path(__file__).resolve().parent
    saved = sys.argv
    sys.argv = ['x', str(repo), '--smoke', '--prec', sw.get('prec', '') if sw.get('prec') is not True else '',
                '--fuse', sw.get('fuse', 'none')]
    spec = importlib.util.spec_from_file_location('sweep_precision', here / 'precision_audit' / 'sweep_precision.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.argv = saved

# 1b. free layouts: drop the node-local / serial-PP-only-on-HGX / PP1-elsewhere policy.
# Every stage size up to chip_budget x every PP depth, on every profile, searched
# independently for decode and prefill. AISim prices groups by size natively
# (<= gpus_per_node: intra_node_bw; <= gpus_per_rack: inter_node_bw; else inter_rack_bw).
if 'free_layouts' in sw:
    from agent_index import simulation as _sim, current_models as _cm
    PPS = tuple(int(x) for x in (sw['free_layouts'] if sw['free_layouts'] is not True else '1,2,4,8').split(','))
    _mk = _sim.make_config
    def make_config(hardware, model, tp, dp, dcp, etp, ep, pp, systems_path=None):
        if tp * dp * pp > hardware['chip_budget']:
            raise ValueError('Instance exceeds the chip search budget')
        c = _mk(dict(hardware, restricted=False), model, tp, dp, dcp, etp, ep, 1, systems_path=systems_path)
        c['pp'] = pp
        return c
    _sim.make_config = make_config
    def _free(gen):
        def f(hw, model):
            seen = set()
            for p in gen(dict(hw, restricted=False), model):
                stage = p['tp'] * p['dp']
                # IPI's hop overlay (TPU8i) raises on PP>1 (topology.py:76): keep PP1 there.
                for pp in ((1,) if hw.get('hop_overlay') else PPS):
                    q = dict(p, pp=pp); k = tuple(sorted(q.items()))
                    if stage * pp <= hw['chip_budget'] and k not in seen:
                        seen.add(k); yield q
        return f
    _sim.enumerate_layouts = _free(_sim.enumerate_layouts)
    _cm.gqa_layouts = _free(_cm.gqa_layouts)
    _cm.prefill_layouts = _free(_cm.prefill_layouts)

# 1c. W4A8 experts (4-bit expert storage, FP8 compute) on profiles whose catalog quant is fp8
# (H100, H200, TPU7x, Trainium2). Same mechanism as precision_audit's 'e4a8', but compatible with kfix:
# the graph (and memory admission) is rebuilt natively with moe_quant_mode=w4afp8.
if 'w4a8_experts' in sw:
    if 'e4a8' in str(sw.get('prec', '')):
        raise SystemExit('use either w4a8_experts or prec=e4a8, not both')
    _g_w4 = current_native.Native.graph
    def graph(self, phase, layout):
        if self.hw.get('quant') == 'fp8' and not getattr(self, '_w4a8', False):
            self.hw = dict(self.hw, quant='w4afp8'); self._w4a8 = True; self.cache.clear()
            self.__dict__.pop('_hcache', None)
        return _g_w4(self, phase, layout)
    current_native.Native.graph = graph

# 2. chip budget
if 'budget' in sw:
    N = int(sw['budget'])
    _init = current_native.Native.__init__
    def __init__(self, reference, hardware, model):
        hardware = dict(hardware, chip_budget=min(N, hardware['chip_budget']))
        _init(self, reference, hardware, model)
    current_native.Native.__init__ = __init__

COMM = ('dispatch', 'all_to_all', 'all_gather', 'reduce_scatter', '_ar', 'allreduce', 'all_reduce', 'p2p', 'alltoall')
def is_comm(n): return any(s in n for s in COMM)
def is_moe(n): return n.endswith('_moe') or 'moe_pre_dispatch' in n or 'moe_post_dispatch' in n or n.endswith('moe_overlap')

op_switches = any(k in sw for k in ('comm_overlap', 'ep_imbalance', 'tpu_bisection', 'derate', 'attn_prec'))
if op_switches:
    class _Wrap:
        def __init__(self, inner, layout, hw, c=None, g=None, model=None):
            self._inner, self._lay, self._hw, self._c, self._g, self._nl = inner, layout, hw, c, g, None
            self._model, self._r = model, None
        def _r16(self):
            if self._r is None:
                import yaml
                y = yaml.safe_load(open(Path(self._c['systems_paths'][0]) / (self._c['system'] + '.yaml')))['gpu']
                self._r = y['fp8_tc_flops'] / y['bfloat16_tc_flops']
            return self._r
        def _overlap_parts(self, batch, leaf_scale):
            # Overlap ops (e.g. GLM generation_moe_overlap = max(router+dispatch+moe+combine, shared))
            # hide comm leaves from op-name matching. Re-fold their native leaf tree:
            # returns per-overlap-op list of (orig_ms, scaled_ms, noncomm_ms, comm_ms_after_scale).
            if self._nl is None:
                from agent_index.interconnect import NativeLeaves
                self._nl = NativeLeaves(self._c, self._g, 'decode', False)
            nl = self._nl; t = nl.times(batch); info = nl.tree.info
            def fold(node, tt):
                kind, val = node
                if kind == 'leaf': return tt[val]
                xs = [fold(x, tt) for x in val]
                return max(xs) if kind == 'max' else math.fsum(xs)
            ts = [x * leaf_scale(nl.tree.leaves[i]) for i, x in enumerate(t)]
            tx = [0. if info[i]['communication'] else x for i, x in enumerate(ts)]
            return [(fold(n, t), fold(n, ts), fold(n, tx)) for n in nl.tree.tree[1] if n[0] != 'leaf']
        def __getattr__(self, k): return getattr(self._inner, k)
        def static_phase_diagnostics(self, **kw):
            ops = self._inner.static_phase_diagnostics(**kw)
            ep = self._lay['ep']
            if 'derate' in sw:
                M, B = (float(x) for x in sw['derate'].split(','))
                for o in ops:
                    s = (o.get('details') or {}).get('sol')
                    if s and s.get('math_ms') is not None and s.get('memory_ms') is not None and o['latency_ms'] > 0:
                        o['latency_ms'] = o['latency_ms'] * max(s['math_ms']/M, s['memory_ms']/B) / max(s['math_ms'], s['memory_ms'], 1e-30) if max(s['math_ms'], s['memory_ms']) > 0 else o['latency_ms']
            if 'attn_prec' in sw and self._model != 'GLM-5.2':
                # Attention arithmetic precision, mirroring precision_audit 'att16' scope: dense/MLA attention
                # ops (decode generation_attention + Kimi draft_attention; prefill context_attention); GLM's
                # DSA op excluded. r = fp8/bf16 peak of this chip's profile. Memory term (after any kfix) kept.
                r = self._r16()
                for o in ops:
                    s_ = (o.get('details') or {}).get('sol')
                    if not s_ or s_.get('math_ms') is None:
                        continue
                    m_ = s_['math_ms']
                    if sw['attn_prec'] == 'bf16' and not kw.get('prefill') and o['name'] in ('generation_attention', 'draft_attention'):
                        o['latency_ms'] = max(o['latency_ms'], m_ * r)          # decode math FP8 -> BF16
                    elif sw['attn_prec'] == 'fp8' and kw.get('prefill') and o['name'] == 'context_attention':
                        mem_ = s_.get('memory_ms') or 0.
                        o['latency_ms'] = max(m_ / r, mem_)                      # prefill math BF16 -> FP8
            if kw.get('prefill'):
                return ops
            if 'tpu_bisection' in sw and 'tpu' in self._hw.get('profile', '').lower():
                f = 1 if ep <= 16 else 2 if ep <= 32 else 3 if ep <= 64 else 6
                for o in ops:
                    if ('dispatch' in o['name'] or 'all_to_all' in o['name']) and not o['name'].endswith('_overlap'):
                        o['latency_ms'] *= f
                ovl = [o for o in ops if o['name'].endswith('_overlap')]
                if ovl and f != 1:
                    sc = lambda leaf: f if next(iter(leaf)) == 'MoeDispatch' else 1.
                    parts = self._overlap_parts(kw['batch_size'], sc)
                    assert len(parts) == len(ovl), 'overlap op/tree mismatch'
                    for o, (orig, scaled, _) in zip(ovl, parts):
                        o['latency_ms'] = o['latency_ms'] - orig + scaled  # keep upstream edits, add bisection delta
            if 'ep_imbalance' in sw and ep >= 32:
                F = float(sw['ep_imbalance'])
                for o in ops:
                    if is_moe(o['name']):
                        o['latency_ms'] *= F
            if 'comm_overlap' in sw:
                ovl = [o for o in ops if o['name'].endswith('_overlap')]
                split = {}
                if ovl:
                    parts = self._overlap_parts(kw['batch_size'], lambda leaf: 1.)
                    assert len(parts) == len(ovl), 'overlap op/tree mismatch'
                    for o, (orig, _, noncomm) in zip(ovl, parts):
                        cur = o['latency_ms']; nc = min(cur, noncomm * (cur / orig if orig > 0 else 1.))
                        split[id(o)] = (nc, cur - nc)   # (non-comm critical path, comm share) of overlap op
                C = sum(o['latency_ms'] for o in ops if is_comm(o['name']) and id(o) not in split) + sum(v[1] for v in split.values())
                X = sum(o['latency_ms'] for o in ops if not is_comm(o['name']) and id(o) not in split) + sum(v[0] for v in split.values())
                scale = max(0., C - X) / C if C > 0 else 0.
                for o in ops:
                    if id(o) in split:
                        o['latency_ms'] = split[id(o)][0] + split[id(o)][1] * scale
                    elif is_comm(o['name']):
                        o['latency_ms'] *= scale
            return ops
    _graph = current_native.Native.graph
    def graph(self, phase, layout):
        c, g, engine, hop = _graph(self, phase, layout)
        hc = self.__dict__.setdefault('_hcache', {})
        key = (phase, *(layout[k] for k in current_native.KEYS))
        if key not in hc:
            hc[key] = _Wrap(engine, layout, self.hw, c, g, self.model)
        return c, g, hc[key], hop
    current_native.Native.graph = graph

# 3. pipelined PP (row-level)
if 'pp_pipelined' in sw:
    _eval = current_native.Native.evaluate
    def evaluate(self, phase, layout, batch):
        r = _eval(self, phase, layout, batch)
        pp = layout['pp']
        if pp > 1:
            r['requests'] *= pp; r['capacity'] = r['requests'] / r['chips']
            if phase == 'prefill':
                r['input_tokens_chip_second'] *= pp
        return r
    current_native.Native.evaluate = evaluate

import runpy
sys.argv = [str(repo / 'reproduce.py'), 'sweep', '--output', out, *rest]
# Overlay reference (scripts/make_overlay.py): frozen 15-Oct inputs + extra published profiles
# (VR200 Rubin revision, ...). verify() still checks the real frozen reference.
OVERLAY = Path(os.environ.get('IPI_OVERLAY') or Path(__file__).resolve().parents[1] / 'overlay_ref')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('ipi_reproduce', str(repo / 'reproduce.py'))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
g = _mod.__dict__  # live module globals (runpy.run_path returns a copy)
if OVERLAY.exists() and os.environ.get('IPI_NO_OVERLAY') != '1':
    _orig_ref, _orig_verify = g['REF'], g['verify']
    def _verify():
        g['REF'] = _orig_ref
        try: _orig_verify()
        finally: g['REF'] = OVERLAY
    g['verify'] = _verify; g['REF'] = OVERLAY
g['main']()
