"""Aggregated-serving driver: re-prices saved decode-frontier layouts with chunked prefill.

Usage:
  python run_agg.py OUT [--kfix C1] [--base published|RUN_DIR] [--layouts-from RUN_DIR ...] [--all-layouts]
                        [--hardware H ...] [--model M ...] [--workers N]

For each (hardware, model): layouts = union of the frontier layouts in --base and every
--layouts-from run; with --all-layouts also IPI's whole decode layout family (Native.layouts). For each layout, every batch from IPI's first
admitted batch up to the aggregated memory limit is evaluated:
  T_dec  = Native.evaluate('decode') under the installed patch stack (kfix optional; trainium_ext
           always, as in harness.py)  -> IPI decode time incl. hop penalty
  Delta  = F(prefill k) - F(0) from the native mixed-step FPM estimator (aggcore.py), with
           k = 3 * r * batch new tokens per step per DP rank
  T_agg  = T_dec + Delta;  speed = 3000/T_agg;  capacity = batch*dp/chips (no prefill chips)
Rows record 'fits1' (admission with one extra resident 262k sequence per DP rank = request being
prefilled, + chunk activations) and are admitted at 'fits0' (no extra sequence) by construction.
Writes OUT/parts/NN-<model>.json (resumable). With --workers and no --hardware, shards by hardware
into at most N subprocesses.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggcore as A

argv = sys.argv[1:]
out = Path(argv[0]).resolve()
def opt(name, default=None, many=False):
    vals = [argv[i + 1] for i, a in enumerate(argv) if a == name]
    return vals if many else (vals[-1] if vals else default)
kfix = opt('--kfix'); base_spec = opt('--base', 'published')
extra_layout_runs = opt('--layouts-from', many=True)
hws = opt('--hardware', many=True); models = opt('--model', many=True) or A.MODELS
workers = int(opt('--workers', 0))
all_layouts = '--all-layouts' in argv
cat = A.catalog(); CAT = list(cat)

if workers and not hws:
    out.mkdir(parents=True, exist_ok=True)
    base_args = [a for a in argv[1:]]
    i = 0; clean = []
    while i < len(base_args):
        if base_args[i] == '--workers': i += 2; continue
        if base_args[i] == '--hardware': raise SystemExit('--workers shards all hardware')
        clean.append(base_args[i]); i += 1
    queue = sorted(CAT, key=lambda h: -cat[h]['chip_budget'])
    running = []
    py = str(A.REPO / '.venv-current/bin/python')
    while queue or running:
        for p, h, f in running[:]:
            if p.poll() is not None:
                running.remove((p, h, f)); f.close()
                print(time.strftime('%H:%M:%S'), 'END', h, p.returncode, flush=True)
        while queue and len(running) < workers:
            h = queue.pop(0)
            f = open(out / f'{CAT.index(h):02d}.log', 'w')
            p = subprocess.Popen([py, __file__, str(out), *clean, '--hardware', h], stdout=f, stderr=subprocess.STDOUT)
            running.append((p, h, f)); print(time.strftime('%H:%M:%S'), 'START', h, flush=True)
        time.sleep(2)
    (out / 'DONE').touch()
    sys.exit(0)

A.install(kfix)
from agent_index.index import envelope
from agent_index.current_native import KEYS

base = A.baseline(base_spec)
extra = [A.load_parts(d) for d in extra_layout_runs]
(out / 'parts').mkdir(parents=True, exist_ok=True)
meta = dict(all_layouts=all_layouts, kfix=kfix, base=base_spec, layouts_from=extra_layout_runs, ratio=A.RATIO, chunk=A.CHUNK)
(out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n')


def lay_of(r):
    return {k: int(r[k]) for k in KEYS}


for h in (hws or CAT):
    for m in models:
        path = out / 'parts' / f'{CAT.index(h):02d}-{m}.json'
        if path.exists():
            print('Resume', h, m, flush=True); continue
        t0 = time.time()
        g = A.Agg(h, m)
        fr, P = base[h, m]
        layouts = {}
        for src in [fr] + [e[h, m][0] for e in extra if (h, m) in e]:
            for r in src:
                layouts.setdefault(tuple(lay_of(r).values()), lay_of(r))
        if all_layouts:   # IPI's full published decode layout family for this chip/model
            for lay in g.N.layouts('decode'):
                layouts.setdefault(tuple(lay[k] for k in KEYS), {k: lay[k] for k in KEYS})
        # parity of the installed decode timing vs the base frontier
        parity = max((abs(g.decode(lay_of(r), int(r['batch']))['raw_ms'] / r['raw_ms'] - 1) for r in fr), default=0.)
        rows, laud = [], []
        for key, lay in layouts.items():
            try:
                mb0 = g.max_batch_agg(lay, 0); mb1 = g.max_batch_agg(lay, 1)
            except ValueError as e:
                laud.append(dict(**lay, status='error', reason=str(e))); continue
            first = g.first_batch(lay)
            laud.append(dict(**lay, first=first, max_batch_agg0=mb0, max_batch_agg1=mb1))
            for b in range(first, mb0 + 1):
                try:
                    d = g.decode(lay, b)
                    r = g.agg_row(lay, b, d)
                except ValueError as e:
                    laud[-1].update(status='error', reason=f'batch {b}: {e}'); break
                _, _, eng, _ = g.N.graph('decode', lay)
                nat = sum(o['latency_ms'] for o in A.native_engine(eng).static_phase_diagnostics(
                    batch_size=b, context_length=A.CONTEXT, prefix=0, prefill=False)) + d['hop_ms']
                r.update(dec_native_ms=nat, zero_chunk_gap=(r['fpm0_ms'] + d['hop_ms']) / nat - 1, fits1=b <= mb1)
                rows.append(r)
        def front(speed_key, need1):
            pts = [dict(r, speed=r[speed_key]) for r in rows if r['fits1'] or not need1]
            return envelope(pts)
        part = dict(hardware=h, model=m, meta=meta, base_prefill=P, base_parity_max_rel=parity,
                    layouts=laud, n_rows=len(rows),
                    max_zero_chunk_gap=max((abs(r['zero_chunk_gap']) for r in rows), default=0.),
                    frontier_A=front('speed', True), frontier_T=front('speed_T', True),
                    frontier_A0=front('speed', False), frontier_T0=front('speed_T', False),
                    decode_frontier_restricted=front('dec_speed', False),
                    seconds=time.time() - t0)
        # keep only rows on some frontier or at a base-frontier (layout, batch): parts stay small
        keep = {(tuple(r[k] for k in KEYS), r['batch']) for f in ('frontier_A', 'frontier_T', 'frontier_A0', 'frontier_T0')
                for r in part[f]} | {(tuple(lay_of(r).values()), int(r['batch'])) for r in fr}
        part['rows'] = [r for r in rows if (tuple(r[k] for k in KEYS), r['batch']) in keep]
        path.write_text(json.dumps(part) + '\n')
        print('Completed', h, m, len(layouts), 'layouts', len(rows), 'rows', f'parity={parity:.1e}',
              f'gap={part["max_zero_chunk_gap"]:.1e}', f'{part["seconds"]:.0f}s', flush=True)
