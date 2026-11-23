"""Fusion / overlap overrides on IPI decode leaf times (computable from op names + kinds).
Each override maps the per-leaf time list t (aligned with NativeLeaves tree leaves) to a cycle time.

  ovl     ideal compute/comm overlap: cycle = max(fold(noncomm), sum(comm))      [harness comm_overlap]
  tbo     two-batch overlap: two microbatches of B/2; cycle = max(2*X(B/2), 2*C(B/2))
          (weights re-read per microbatch, comm hidden behind the other microbatch)
  moefuse fused dispatch+grouped-GEMM+combine megakernel: per MoE block t -= min(D, M)
  arnorm  TP all-reduce fused with residual-add+RMSNorm: drop norm/residual Elementwise ops when TP>1
  asynctp GEMM/collective overlap (async TP): t -= min(sum TP-AR, sum non-expert GEMM)
  dcpovl  DCP collectives overlapped with attention compute: t -= min(sum DCP comm, attention)
  launchN add N us per kernel call (scale_factor calls of every nonzero leaf): SOL charges 0
"""
from retime import *
COMMK = {'CustomAllReduce', 'Nccl', 'MoeDispatch', 'P2P'}

def leaves_info(nl):
    out = []
    for x in nl.tree.leaves:
        k, v = next(iter(x.items()))
        out.append((v['name'].split('_', 2)[2], k, v))
    return out

def cycle(hw, model, l, B, ov, base_sw=frozenset()):
    nl, hop = ctx(hw, model, l)
    info = leaves_info(nl)
    ms, det = retime(hw, model, l, B, set(base_sw), detail=True)
    t = [d[2] for d in det]
    fold = nl.tree.fold
    iscomm = [k in COMMK for _, k, _ in info]
    if ov == 'none':
        return fold(t) + hop
    if ov == 'ovl':
        X = fold([0 if c else x for x, c in zip(t, iscomm)]); C = sum(x for x, c in zip(t, iscomm) if c)
        return max(X, C) + hop
    if ov == 'tbo':
        if B < 2: return fold(t) + hop
        _, d2 = retime(hw, model, l, B // 2, set(base_sw), detail=True)
        t2 = [d[2] for d in d2]
        X = 2 * fold([0 if c else x for x, c in zip(t2, iscomm)]); C = 2 * sum(x for x, c in zip(t2, iscomm) if c)
        return min(fold(t), max(X, C)) + hop
    tt = list(t)
    if ov == 'moefuse':
        blocks = {}
        for i, (nm, k, v) in enumerate(info):
            if k in ('MoeDispatch', 'Moe'):
                pre = nm.replace('_moe_pre_dispatch', '').replace('_moe_post_dispatch', '').replace('_moe', '')
                blocks.setdefault(pre, {'D': [], 'M': []})['D' if k == 'MoeDispatch' else 'M'].append(i)
        for b in blocks.values():
            D = sum(tt[i] for i in b['D']); M = sum(tt[i] for i in b['M'])
            s = min(D, M)
            for i in b['D']:
                if D > 0: tt[i] -= s * tt[i] / D
        return fold(tt) + hop
    if ov == 'arnorm':
        if l['tp'] > 1:
            for i, (nm, k, v) in enumerate(info):
                if k == 'Elementwise' and any(s in nm for s in ('norm', '_res', 'residual')) and 'latent_norm' not in nm:
                    tt[i] = 0.
        return fold(tt) + hop
    if ov == 'asynctp':
        ar = [i for i, (nm, k, v) in enumerate(info) if k == 'CustomAllReduce' or (k == 'Nccl' and v.get('operation') == 'all_reduce')]
        gm = [i for i, (nm, k, v) in enumerate(info) if k in ('Gemm', 'MlaBmm') and 'router' not in nm]
        A = sum(tt[i] for i in ar); G = sum(tt[i] for i in gm); s = min(A, G)
        for i in ar:
            if A > 0: tt[i] -= s * tt[i] / A
        return fold(tt) + hop
    if ov == 'dcpovl':
        dc = [i for i, (nm, k, v) in enumerate(info) if k == 'Nccl' and 'dcp' in nm]
        at = [i for i, (nm, k, v) in enumerate(info) if k in ('GenerationAttention', 'GenerationMla', 'DsaGeneration') and nm == 'generation_attention']
        D = sum(tt[i] for i in dc); A = sum(tt[i] for i in at); s = min(D, A)
        for i in dc:
            if D > 0: tt[i] -= s * tt[i] / D
        return fold(tt) + hop
    if ov.startswith('launch'):
        L = float(ov[6:]) / 1000.
        extra = sum(v.get('scale_factor', 1) * L for (nm, k, v), x in zip(info, tt) if x > 0)
        return fold(tt) + hop + extra
    raise KeyError(ov)
