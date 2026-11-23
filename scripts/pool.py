"""Parallel sweep pool: shards each job by hardware into subprocesses, bounded by CPU and RAM.

Usage: python3 pool.py JOBFILE [--workers N]
JOBFILE lines: NAME|SWITCHES|HARDWARE(; separated, empty=all)
Each (job, hardware) shard runs harness.py with --hardware H (all four models) into results/NAME.
Shards resume from existing parts. When every shard of a full-hardware job finishes, a final
unfiltered pass merges parts into the complete index (results/NAME/index.csv).
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # workspace root (contains scripts/, ipi-repo/, results/)
PY = str(ROOT / 'ipi-repo/.venv-current/bin/python')
_OVC = Path(os.environ['IPI_OVERLAY']) / 'catalog.json' if os.environ.get('IPI_OVERLAY') else ROOT / 'overlay_ref/catalog.json'
CAT_D = json.loads((_OVC if _OVC.exists() else (ROOT / 'ipi-repo/reference/2026-10-15/catalog.json')).read_text()); CAT = list(CAT_D)
workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else max(1, (os.cpu_count() or 2) - 1)
MIN_FREE_MB = 600

def free_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) // 1024
    return 10**6

jobs = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    name, switches, hw = (line.split('|') + ['', ''])[:3]
    hws = [h for h in hw.split(';') if h] or CAT
    jobs.append(dict(name=name, switches=switches.split(), hws=hws, full=not hw.strip()))

def log(msg):
    with open(ROOT / 'results/pool.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')

queue = [(j, h) for j in jobs for h in j['hws']]
# longest-first heuristic: big-budget chips first
queue.sort(key=lambda t: -CAT_D[t[1]]['chip_budget'])
running, done = [], {j['name']: 0 for j in jobs}
while queue or running:
    for p, j, h in running[:]:
        if p.poll() is not None:
            running.remove((p, j, h)); done[j['name']] += 1
            log(f"END {j['name']} | {h} rc={p.returncode}")
            if done[j['name']] == len(j['hws']):
                out = ROOT / 'results' / j['name']
                if j['full']:
                    with open(ROOT / f"results/{j['name']}.merge.log", 'w') as f:
                        rc = subprocess.call([PY, str(ROOT / 'scripts/harness.py'), str(ROOT / 'ipi-repo'), str(out), *j['switches']], stdout=f, stderr=subprocess.STDOUT)
                    log(f"MERGED {j['name']} rc={rc}")
                (out / 'DONE').touch()
    while queue and len(running) < workers and free_mb() > MIN_FREE_MB:
        j, h = queue.pop(0)
        out = ROOT / 'results' / j['name']; out.mkdir(parents=True, exist_ok=True)
        logf = open(ROOT / f"results/{j['name']}.{CAT.index(h):02d}.log", 'w')
        p = subprocess.Popen([PY, str(ROOT / 'scripts/harness.py'), str(ROOT / 'ipi-repo'), str(out), *j['switches'], '--hardware', h],
                             stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, j, h)); log(f"START {j['name']} | {h}")
        time.sleep(2)
    time.sleep(3)
log('POOL DONE')
