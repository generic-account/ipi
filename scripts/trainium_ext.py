"""Trainium2/3 published research-extension profiles for the sweep harness (repo never modified).

Published values (ipi-repo/docs/research/trainium/overall.csv):
  Trainium2 = 0.128 GB300e  <- scenario 64_link_limited   (studies/trainium/run.py CASES[2])
  Trainium3 = 0.349 GB300e  <- scenario 64_detailed_tiers (studies/trainium/run.py CASES[3])

The study differs from the main native path in three ways; all are reproduced here:
  1. own hardware YAMLs (generated below with the study's exact main() code);
  2. MoE leaf pricing: every native `Moe` leaf time is replaced by the study's moe_time(...)
     (MXFP4 weights + scale bytes, MXFP8 activations, FP8 compute; for fp8 experts it equals
     native), inside the native overlap tree (max of branch sums);
  3. MX scale memory: + routed-expert weights/16 and fits <= 0.9*capacity (Trainium.memory).
Collective startup is 0 in both published scenarios, so it is not modeled.

Memory: the study prices memory on the base trainium{gen}.yaml and timing on the case YAML.
The two differ only in `node:` (link bandwidths / group sizes); memory admission is bitwise
identical on all 2815 (phase, layout, batch) probes per generation (nccl_mem is 0 for every
group size), so the case YAML is used for both.

Composition (one leaf-time hook, `adjust`, called by every re-timer):
  * NativeLeaves.times is patched for Trainium systems -> covers our _TrnEngine (base path,
    prefill, op switches) and kfix's _Engine (which re-times from NativeLeaves.times).
  * precision_audit/retime._eval is patched (retime evaluates rewritten leaves directly);
    retime.CAT / retime.hwyaml learn the Trainium entries (they read the frozen catalog).
  * _TrnEngine sits INNERMOST under Native.graph (install() runs before kfix/prec/op
    switches), so every other wrapper sees Trainium-corrected ops; Moe op `sol` is replaced
    with the Trainium math/memory split so `derate` scales the right roofline side.
  Raises (never silent): MoE quant modes the study formula cannot price (prec=w4a16), and
  any leaf/native parity break.

CLI:  python trainium_ext.py write     -> extra_hardware/*.yaml + catalog_extra.json
"""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'ipi-repo'
EXTRA = ROOT / 'extra_hardware'
STUDY = REPO / 'studies/trainium/run.py'
SAVED = REPO / 'docs/research/trainium/runs'
GIB = 2**30

# Published hardware name -> (generation, scenario)
PUBLISHED = {'Trainium2': (2, '64_link_limited'), 'Trainium3': (3, '64_detailed_tiers')}
PROFILES = {f'trainium{g}_{s}': g for g, s in PUBLISHED.values()}
NOTES = {
    'Trainium2': ('16-chip ring groups inside a 64-chip domain; link-limited one-way bandwidth '
                  '128/64 GiB/s local/wider (study scenario 64_link_limited)'),
    'Trainium3': ('4-chip local groups inside a 64-chip domain; 128 GB/s local, 160 GB/s rack '
                  '(study scenario 64_detailed_tiers)'),
}

_study = None
def study():
    """The study module itself (SPECS, CASES, moe_time) - imported read-only, main() not run."""
    global _study
    if _study is None:
        spec = importlib.util.spec_from_file_location('trainium_study_run', STUDY)
        _study = importlib.util.module_from_spec(spec); spec.loader.exec_module(_study)
    return _study


# ---------------------------------------------------------------- inputs (YAML + catalog)
def profiles():
    """Exact copy of the YAML construction in studies/trainium/run.py main()."""
    S = study(); out = {}
    for gen, s in S.SPECS.items():
        profile = dict(data_dir='data/estimate_only', gpu=dict(bfloat16_tc_flops=s['bf16'], fp8_tc_flops=s['fp8'],
            mem_bw=s['bandwidth'], mem_capacity=s['capacity']*GIB, mem_bw_empirical_scaling_factor=1, mem_empirical_constant_latency=0, sm_version=103),
            misc=dict(other_mem=4*GIB, nccl_mem={n: 0 for n in (1, 2, 4, 8, 16, 32, 64, 72, 128, 144, 256)}),
            node=dict(intra_node_bw=1e12, inter_node_bw=1e12, inter_rack_bw=1e12, num_gpus_per_node=256, num_gpus_per_rack=256, p2p_latency=0))
        if gen == 3: profile['gpu']['fp4_tc_flops'] = s['fp8']
        out[f'trainium{gen}'] = profile
        for name, case in S.CASES[gen].items():
            _, local, near, far, _ = case
            changed = profile | dict(node=dict(intra_node_bw=near, inter_node_bw=far, inter_rack_bw=far,
                num_gpus_per_node=local, num_gpus_per_rack=case[0], p2p_latency=0))
            out[f'trainium{gen}_{name}'] = changed
    return out


def catalog_entries():
    S = study(); cat = {}
    for h, (gen, scen) in PUBLISHED.items():
        case = S.CASES[gen][scen]
        assert case[4] == 0, 'published scenarios have zero collective startup'
        cat[h] = dict(name=f'AWS Trainium{gen} ({scen.replace("_", " ")}; 22 Oct 2026 research extension)',
            profile=f'trainium{gen}_{scen}', chip_budget=case[0], restricted=False,
            quant=S.SPECS[gen]['quant'], hop_overlay=False, local_scale_up_group_chips=case[1],
            topology_note=NOTES[h],
            source_notes=('studies/trainium/run.py: MoE leaves repriced (MXFP4+scales / MXFP8 acts / FP8 compute), '
                          'MX scale memory on non-fp8 experts; harness hook scripts/trainium_ext.py. Published '
                          + {'Trainium2': '0.128', 'Trainium3': '0.349'}[h] + ' GB300e.'))
    return cat


def write(dest=EXTRA):
    import yaml
    dest.mkdir(exist_ok=True)
    P = profiles()
    for prof in PROFILES:
        text = yaml.safe_dump(P[prof])
        ref = REPO / 'docs/research/trainium/hardware' / f'{prof}.yaml'
        if ref.exists() and ref.read_text() != text:
            raise RuntimeError(f'generated {prof}.yaml differs from the saved study YAML {ref}')
        (dest / f'{prof}.yaml').write_text(text)
    cpath = dest / 'catalog_extra.json'
    cat = json.loads(cpath.read_text()) if cpath.exists() else {}
    cat.update(catalog_entries())
    tmp = cpath.with_suffix('.json.tmp'); tmp.write_text(json.dumps(cat, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(cpath)
    return sorted(PROFILES), sorted(cat)


def saved_baseline():
    """{(hardware, model): (frontier, prefill_rate)} from the saved study runs (score.py)."""
    out = {}
    for h, (gen, scen) in PUBLISHED.items():
        for f in sorted(SAVED.glob(f'trn{gen}-*.json')):
            r = json.loads(f.read_text()); c = r['cases'][scen]
            out[h, r['model']] = (c['frontier'], c['prefill']['input_tokens_chip_second'] if c['prefill'] else 0.)
    return out


# ---------------------------------------------------------------- leaf-time hook
def _moe_parts(v, x, spec):
    """(math_ms, memory_ms) split of study moe_time(native=False); checked against moe_time."""
    h, i = v['hidden_size'], v['inter_size']; g = 3 if v['is_gated'] else 2
    ep, tp = v['moe_ep_size'], v['moe_tp_size']
    tokens = x*v['attention_dp_size']*v['topk']
    flop = (tokens*h*i*g*2//ep)//tp
    t = tokens//ep
    weights = (h*i*g//tp)*min(v['num_experts']//ep, t)
    if v['quant_mode'] == 'fp8':
        memory = weights + t*h*2 + (t*i*g//tp)
    else:
        memory = weights*.53125 + t*h*(1.03125+2) + (t*i*g//tp)*1.03125
    sf = 1000*v['scale_factor']
    return flop/spec['fp8']*sf, memory/spec['bandwidth']*sf

PRICEABLE = {'fp8', 'w4afp8', 'w4a8_mxfp4_mxfp8'}   # FP8-MMA expert modes the study formula covers

def adjust(leaves, times, phase, batch, gen, sols=None):
    """Replace every Moe leaf time with the study's moe_time. `times` aligned with `leaves`
    (NativeLeaves.tree.leaves or a rewritten copy). Returns a new list; fills sols[j]=(math,mem)."""
    S = study(); spec = S.SPECS[gen]
    x = batch*(1024 if phase == 'prefill' else 4)
    out = list(times)
    for j, leaf in enumerate(leaves):
        kind, v = next(iter(leaf.items()))
        if kind != 'Moe':
            continue
        if v['quant_mode'] not in PRICEABLE:
            raise RuntimeError(f"Trainium{gen}: study MoE pricing is undefined for expert quant "
                               f"'{v['quant_mode']}' (e.g. prec=w4a16, BF16 MMA); switch not supported on Trainium")
        orig = S.moe_time(v, x, spec, True)
        if abs(orig - times[j]) > 1e-10*max(times[j], 1e-12):
            raise RuntimeError(f'Trainium{gen}: native Moe leaf no longer matches the study native formula '
                               f'({v["name"]}, {v["quant_mode"]}): {times[j]} vs {orig}')
        new = S.moe_time(v, x, spec)
        m, b = _moe_parts(v, x, spec)
        if abs(max(m, b) - new) > 1e-12*new:
            raise RuntimeError('Trainium moe split mismatch')
        out[j] = new
        if sols is not None:
            sols[j] = (m, b)
    return out


# ---------------------------------------------------------------- patches
class _TrnEngine:
    """Innermost engine wrapper: Moe leaves repriced, overlap trees re-folded per top-level op."""
    _trn_wrapped = True
    def __init__(self, inner, c, g, phase, gen):
        from agent_index.interconnect import NativeLeaves
        self._inner, self._phase, self._gen = inner, phase, gen
        self._nl = NativeLeaves(c, g, phase, False)
        self._max_parity = 0.
    def __getattr__(self, k):
        return getattr(self._inner, k)
    def static_phase_diagnostics(self, **kw):
        ops = self._inner.static_phase_diagnostics(**kw)
        if bool(kw.get('prefill')) != (self._phase == 'prefill'):
            raise RuntimeError('Trainium engine phase mismatch')
        nl, batch = self._nl, kw['batch_size']
        raw = nl._raw_times(batch)
        tot = math.fsum(o['latency_ms'] for o in ops); f = nl.tree.fold(raw)
        err = abs(f - tot)/max(tot, 1e-30); self._max_parity = max(self._max_parity, err)
        if err > 1e-9:
            raise RuntimeError(f'Trainium leaf-tree/native parity failure {err}')
        sols = {}
        t = adjust(nl.tree.leaves, raw, self._phase, batch, self._gen, sols)
        top = nl.tree.tree[1]
        def fold(node):
            k, v = node
            if k == 'leaf': return t[v]
            xs = [fold(n) for n in v]
            return max(xs, default=0.) if k == 'max' else math.fsum(xs)
        def leafs(node):
            return [node[1]] if node[0] == 'leaf' else [i for n in node[1] for i in leafs(n)]
        def nm(node):
            return next(iter(nl.tree.leaves[node[1]].values()))['name'].split('_', 2)[2] if node[0] == 'leaf' else None
        # Native diagnostics merge same-named top-level ops (e.g. MAI2 global+SWA
        # 'generation_attention'); overlaps appear in order as '*_overlap' ops.
        groups, ovl = {}, []
        for n in top:
            if n[0] == 'leaf': groups.setdefault(nm(n), []).append(n)
            else: ovl.append(n)
        onames = [o['name'] for o in ops]
        if (set(groups) != {x for x in onames if not x.endswith('_overlap')} or len(set(onames)) != len(onames)
                or len(ovl) != sum(x.endswith('_overlap') for x in onames)):
            raise RuntimeError(f"Trainium: native ops do not align with the leaf tree top level ({self._phase}): "
                               f"{[nm(n) or 'OVL' for n in top]} vs {onames}")
        it = iter(ovl)
        for o in ops:
            nodes = [next(it)] if o['name'].endswith('_overlap') else groups[o['name']]
            idx = [i for n in nodes for i in leafs(n)]
            if not any(i in sols for i in idx):
                continue
            o['latency_ms'] = math.fsum(fold(n) for n in nodes)
            s = (o.get('details') or {}).get('sol')
            if s and s.get('math_ms') is not None and all(n[0] == 'leaf' and n[1] in sols for n in nodes):
                m = math.fsum(sols[n[1]][0] for n in nodes); b = math.fsum(sols[n[1]][1] for n in nodes)
                o['details'] = dict(o['details'], sol=dict(s, latency_ms=o['latency_ms'], math_ms=m, memory_ms=b))
        new = math.fsum(o['latency_ms'] for o in ops)
        if abs(new - nl.tree.fold(t)) > 1e-9*new:
            raise RuntimeError('Trainium re-fold mismatch')
        return ops


def _patch_retime(retime):
    if getattr(retime, '_trn_patched', False):
        return
    import yaml
    _ev = retime._eval
    def _eval(nl, leaves, batch):
        t = _ev(nl, leaves, batch)
        gen = getattr(nl, '_trn_gen', None)
        return adjust(leaves, t, nl.phase, batch, gen) if gen else t
    retime._eval = _eval
    # catalog_extra.json, not catalog_entries(): no study import for non-Trainium prec runs
    extra = EXTRA / 'catalog_extra.json'
    cat = {k: v for k, v in (json.loads(extra.read_text()) if extra.exists() else {}).items() if k in PUBLISHED}
    retime.CAT.update(cat)   # retime/common read the frozen 15-Oct catalog
    _hw = retime.hwyaml
    def hwyaml(h):
        if h in cat:
            return yaml.safe_load((EXTRA / (cat[h]['profile'] + '.yaml')).read_text())
        return _hw(h)
    retime.hwyaml = hwyaml
    retime._trn_patched = True


_installed = False
def install(sw):
    """Patch Native/NativeLeaves for the Trainium profiles. Call BEFORE kfix/prec/op switches."""
    global _installed
    if _installed:
        return
    _installed = True
    from agent_index import current_native
    from agent_index.interconnect import NativeLeaves
    Native = current_native.Native

    _nl_init, _nl_times = NativeLeaves.__init__, NativeLeaves.times
    def nl_init(self, config, graph, phase, hop_overlay):
        _nl_init(self, config, graph, phase, hop_overlay)
        self._trn_gen = PROFILES.get(config['system'])
    def nl_times(self, batch):
        t = _nl_times(self, batch)
        return adjust(self.tree.leaves, t, self.phase, batch, self._trn_gen) if getattr(self, '_trn_gen', None) else t
    NativeLeaves.__init__, NativeLeaves.times, NativeLeaves._raw_times = nl_init, nl_times, _nl_times

    _graph = Native.graph
    def graph(self, phase, layout):
        c, g, engine, hop = _graph(self, phase, layout)
        gen = PROFILES.get(self.hw.get('profile'))
        if gen and not getattr(engine, '_trn_wrapped', False):
            if self.hw.get('hop_overlay'):
                raise RuntimeError('Trainium profiles are hop_overlay=false')
            engine = _TrnEngine(engine, c, g, phase, gen)
            self.cache[(phase, *(layout[k] for k in current_native.KEYS))] = (c, g, engine, hop)
        return c, g, engine, hop
    Native.graph = graph

    _memory = Native.memory
    def memory(self, phase, layout, batch):
        m = _memory(self, phase, layout, batch)
        gen = PROFILES.get(self.hw.get('profile'))
        if not gen or 'memory_gib' not in m:
            return m
        c, g, _, _ = self.graph(phase, layout)
        if c['moe_quant_mode'] == 'fp8':     # study: gen 2 (fp8 experts) has no MX scales
            return m
        # Study Trainium.memory: 1 scale byte / 32 params = +1/16 of native 0.5 B/param experts.
        # Study only ran PP1; for PP>1 (free_layouts) count one stage's share.
        routed = sum(o.get_weights() for o in g.context_ops if 'Moe' in json.loads(o._spec_json()))
        extra = routed/16/GIB/c['pp']
        m = dict(m, mx_scale_gib=extra, weights_gib=m['weights_gib']+extra, memory_gib=m['memory_gib']+extra)
        m['fits'] = m['fits'] and m['memory_gib'] <= study().SPECS[gen]['capacity']*.9
        return m
    Native.memory = memory

    if 'prec' in sw or 'fuse' in sw:
        here = str(Path(__file__).resolve().parent / 'precision_audit')
        if here not in sys.path:
            sys.path.insert(0, here)
        import retime            # same module object sweep_precision/fusion import later
        _patch_retime(retime)


if __name__ == '__main__':
    if sys.argv[1:] == ['write']:
        print(write())
    else:
        print(__doc__)
