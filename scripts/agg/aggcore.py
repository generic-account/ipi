"""Aggregated (chunked-prefill-in-decode) pricing on top of IPI's native engine.

Imported in-process; the IPI repo, harness.py, pool.py and score.py are untouched.

Patch stack (same order as scripts/harness.py):
  fastyaml (speed only) -> trainium_ext.install -> optional kfix (e.g. C1, KV read once)
so Native.evaluate('decode', ...) is exactly the harness' decode time for that switch set.

Mixed steps are priced with IPI's own engine object (perf.best_available(c), SOL op_level),
through its direct-FPM entry point estimate_forward_pass_time_ms(list of per-DP-rank FPMs):
  decode part : num_decode_requests = batch*(nextn+1), sum_decode_kv_tokens = that*262145
                (the native FPM path does NOT expand nextn itself; with this convention the
                 zero-chunk estimate equals static_phase_diagnostics to round-off, see validate.py)
  prefill part: num_prefill_requests = n, sum_prefill_tokens = n*1024,
                sum_prefill_kv_tokens = n*262144  (past KV per chunk; this convention equals
                static prefill diagnostics with prefix=262144 exactly)
Every DP rank carries the same work (symmetric, as IPI's decode).
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'ipi-repo'
REF = ROOT / 'overlay_ref'
SCRIPTS = ROOT / 'scripts'
CONTEXT = 262144
CHUNK = 1024
VERIFY = 4            # nextn + 1
AL = 3                # accepted tokens per step
RATIO = json.loads((REPO / 'reference/2026-10-15/curves_and_inputs.json').read_text())['ratio']
KEYS = ('tp', 'dp', 'dcp', 'etp', 'ep', 'pp')

_installed = None


def install(kfix=None):
    """Install the harness-equivalent patch stack once per process."""
    global _installed
    if _installed is not None:
        if _installed != (kfix or ''):
            raise RuntimeError('patch stack already installed with different kfix')
        return
    sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(SCRIPTS))
    if os.environ.get('IPI_NO_FASTYAML') != '1':
        import fastyaml  # noqa: F401
    import trainium_ext
    sw = {'kfix': kfix} if kfix else {}
    trainium_ext.install(sw)
    if kfix:
        import importlib.util
        saved = sys.argv
        sys.argv = ['x', str(REPO), '--smoke', '--fix', kfix]
        spec = importlib.util.spec_from_file_location('kfix', SCRIPTS / 'kernel_audit' / 'sweep_kernel_fixes.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        sys.argv = saved
    from agent_index import current_native, simulation
    current_native.doctor(REF)          # pins AISim version exactly as reproduce.py does
    import logging; logging.getLogger().setLevel(logging.ERROR)
    # Memory hook: extra activation tokens for the prefill chunk(s) in a mixed step.
    _sdk = simulation._sdk
    perf, est, build, apply, get = _sdk()
    def est2(*a, **kw):
        kw['max_num_tokens'] = kw['max_num_tokens'] + _EXTRA_TOKENS[0]
        return est(*a, **kw)
    tup = (perf, est2, build, apply, get)
    simulation._sdk = lambda: tup
    _installed = kfix or ''


_EXTRA_TOKENS = [0]


def catalog():
    return json.loads((REF / 'catalog.json').read_text())


def native_engine(e):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
    while not isinstance(e, RustForwardPassPerfModel):
        e = e._inner
    return e


class Agg:
    def __init__(self, hw, model):
        from agent_index.current_native import Native
        self.hw, self.model = hw, model
        self.N = Native(REF, catalog()[hw] | {'name': hw}, model)
        self._f = {}

    # ---- decode (IPI, with whatever patch stack is installed) ----
    def decode(self, lay, batch):
        _EXTRA_TOKENS[0] = 0
        return self.N.evaluate('decode', lay, batch)

    # ---- mixed step via the native FPM path ----
    def fpm(self, lay, batch, prefill_tokens, prefill_reqs=None):
        """Native mixed-step ms for one iteration: every DP rank has `batch` decode requests
        (verify width 4) and `prefill_tokens` new prefill tokens over a 262144 prefix."""
        key = (tuple(lay[k] for k in KEYS), batch, prefill_tokens, prefill_reqs)
        if key not in self._f:
            _, _, eng, _ = self.N.graph('decode', lay)
            eng = native_engine(eng)
            s = {}
            if batch:
                s.update(num_decode_requests=VERIFY * batch, sum_decode_kv_tokens=VERIFY * batch * (CONTEXT + 1))
            if prefill_tokens:
                n = prefill_reqs or max(1, math.ceil(prefill_tokens / CHUNK))
                s.update(num_prefill_requests=n, sum_prefill_tokens=int(prefill_tokens),
                         sum_prefill_kv_tokens=n * CONTEXT)
            self._f[key] = eng.estimate_forward_pass_time_ms([{'scheduled_requests': s}] * lay['dp'])
        return self._f[key]

    def chunks(self, lay, batch, n):
        """F(n): mixed step with n whole 1024-token chunks (separate requests' chunks)."""
        return self.fpm(lay, batch, n * CHUNK, n) if n else self.fpm(lay, batch, 0)

    def delta(self, lay, batch, x):
        """Time-averaged extra ms per step to carry x chunks/step/rank (policy A):
        a fraction (x - floor x) of steps carries ceil(x) chunks, the rest floor(x)."""
        lo = math.floor(x); f = x - lo
        F0 = self.chunks(lay, batch, 0)
        a = self.chunks(lay, batch, lo)
        b = self.chunks(lay, batch, lo + 1) if f > 0 else a
        return (1 - f) * a + f * b - F0

    def delta_direct(self, lay, batch, tokens):
        """Policy B: every step carries one partial chunk of `tokens` (rounded) tokens."""
        t = max(1, round(tokens))
        return self.fpm(lay, batch, t, 1 if t <= CHUNK else None) - self.fpm(lay, batch, 0)

    def delta_tuned(self, lay, batch, tokens):
        """Best chunk policy for this row: min of policy A (1024-token chunks, interpolated),
        policy B (one k-token chunk every step) and single q-token chunks every k/q steps for
        q in (128, 256, 512) with k < q. All candidates need <= ceil(k/1024)*1024 activation
        tokens, so fits_agg() admission covers each of them."""
        F0 = self.fpm(lay, batch, 0)
        c = {'A': self.delta(lay, batch, tokens / CHUNK), 'B': self.delta_direct(lay, batch, tokens)}
        for q in (128, 256, 512):
            if q > tokens:
                c[f'q{q}'] = tokens / q * (self.fpm(lay, batch, q, 1) - F0)
        best = min(c, key=c.get)
        return c[best], best

    # ---- memory ----
    def fits_agg(self, lay, batch, extra_seq=1):
        """Decode admission plus the prefill chunk: +ceil(k/1024)*1024 activation tokens and
        `extra_seq` additional resident 262k sequences per DP rank (the request being prefilled)."""
        k = AL * RATIO * batch
        _EXTRA_TOKENS[0] = math.ceil(k / CHUNK) * CHUNK
        try:
            m = self.N.memory('decode', lay, batch + extra_seq)
        finally:
            _EXTRA_TOKENS[0] = 0
        return bool(m['fits'])

    def max_batch_agg(self, lay, extra_seq=1, hi_hint=None):
        lo, hi = 0, 1
        while hi < 65536 and self.fits_agg(lay, hi, extra_seq):
            lo, hi = hi, hi * 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.fits_agg(lay, mid, extra_seq): lo = mid
            else: hi = mid
        return lo

    def first_batch(self, lay):
        return math.ceil(lay['ep'] / (4 * lay['dp'] * 7)) if self.model.startswith('MAI2') else 1

    def agg_row(self, lay, batch, dec=None):
        """Aggregated row for one decode layout/batch. dec: IPI decode row (else evaluated)."""
        d = dec or self.decode(lay, batch)
        k = AL * RATIO * batch                     # new prefill tokens per step per DP rank
        x = k / CHUNK
        dA = self.delta(lay, batch, x)
        dB = self.delta_direct(lay, batch, k)
        dT, pol = self.delta_tuned(lay, batch, k)
        tA = d['raw_ms'] + dA; tB = d['raw_ms'] + dB; tT = d['raw_ms'] + dT
        F0 = self.fpm(lay, batch, 0)
        return dict(**{q: lay[q] for q in KEYS}, batch=batch, chips=d['chips'], requests=d['requests'],
                    capacity=d['capacity'], dec_ms=d['raw_ms'], hop_ms=d['hop_ms'], fpm0_ms=F0,
                    prefill_tokens_rank_step=k, chunks_rank_step=x,
                    delta_A_ms=dA, delta_B_ms=dB, delta_T_ms=dT, policy_T=pol,
                    agg_ms_A=tA, agg_ms_B=tB, agg_ms_T=tT,
                    speed=AL * 1000 / tA, speed_B=AL * 1000 / tB, speed_T=AL * 1000 / tT,
                    dec_speed=d['speed'])


# ---- baseline frontiers (same sources as scripts/score.py) ----
def published():
    """{(hw, model): (frontier, prefill tok/s/chip)} for the published baseline: frozen 15-Oct
    curves, 20-Oct 'VR200 Rubin', saved Trainium2/3 study runs (score.py's baseline)."""
    ref = json.loads((REPO / 'reference/2026-10-15/curves_and_inputs.json').read_text())
    out = {(h, m): (ref['curves'][h][m], next(q['tokens_chip_second'] for q in ref['prefill_rates']
                                              if q['hardware'] == h and q['model'] == m))
           for h in ref['curves'] for m in ref['models']}
    rub = json.loads((REPO / 'results/rubin-update/curves_and_inputs.json').read_text())
    for m in ref['models']:
        out['VR200 Rubin', m] = (rub['curves']['VR200'][m], next(q['tokens_chip_second'] for q in rub['prefill_rates']
                                                                if q['hardware'] == 'VR200' and q['model'] == m))
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from trainium_ext import saved_baseline
    out.update(saved_baseline())
    return out


def load_parts(d, with_prefill_row=False):
    out = {}
    for f in (Path(d) / 'parts').glob('*.json'):
        p = json.loads(f.read_text())
        P = p['best_prefill']['input_tokens_chip_second'] if p['best_prefill'] else 0.
        out[p['hardware'], p['model']] = (p['frontier'], P, p['best_prefill']) if with_prefill_row else (p['frontier'], P)
    return out


def baseline(spec):
    return published() if spec in (None, 'published') else load_parts(spec)


MODELS = json.loads((REPO / 'reference/2026-10-15/curves_and_inputs.json').read_text())['models']
