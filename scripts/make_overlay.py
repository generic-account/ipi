"""Build overlay reference = 15-Oct reference (symlinks) + extra published profiles appended.

Extra profiles (catalog keys appended AFTER the 23 frozen ones, so part filenames don't collide):
  'VR200 Rubin'  : 20-Oct Rubin revision YAML (19.2 TB/s HBM, 1.5 TB/s outbound NVLink),
                   otherwise identical to the frozen VR200 catalog entry. Published IPI 2.177.
  'Trainium2/3'  : research-extension profiles (scripts/trainium_ext.py write) from
                   extra_hardware/*.yaml + extra_hardware/catalog_extra.json, appended after
                   'VR200 Rubin' in catalog_extra.json order (parts 24-, 25-).

Safe while sweeps read overlay_ref: the new tree is built in a sibling temp dir and swapped in
with an atomic renameat2(RENAME_EXCHANGE); the old tree is then removed. Never rmtree's the
live directory. Usage: python make_overlay.py [TARGET_DIR]   (default overlay_ref)

manifest.json / runtime.json / models are symlinked unchanged, so run signatures match the
frozen-reference runs and overlay parts merge into existing results/<run>/parts.
"""
import ctypes, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'ipi-repo/reference/2026-10-15'
OV = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'overlay_ref'
EXTRA_HW = ROOT / 'extra_hardware'   # extra YAMLs written by other builders (e.g. Trainium)


def build(dst):
    dst.mkdir()
    for p in REF.iterdir():
        if p.name not in ('catalog.json', 'hardware'):
            os.symlink(p, dst / p.name)
    (dst / 'hardware').mkdir()
    for p in (REF / 'hardware').iterdir():
        os.symlink(p, dst / 'hardware' / p.name)
    shutil.copyfile(ROOT / 'ipi-repo/reference/2026-10-20-rubin/hardware/vr200_deployment.yaml',
                    dst / 'hardware' / 'vr200_rubin_oct20.yaml')
    EXTRA_HW.mkdir(exist_ok=True)
    for p in sorted(EXTRA_HW.glob('*.yaml')):
        os.symlink(p, dst / 'hardware' / p.name)
    cat = json.loads((REF / 'catalog.json').read_text())
    cat['VR200 Rubin'] = dict(cat['VR200'], profile='vr200_rubin_oct20',
                              name=cat['VR200']['name'] + ' (20 Oct 2026 Rubin revision)')
    extra = EXTRA_HW / 'catalog_extra.json'
    if extra.exists():
        cat.update(json.loads(extra.read_text()))
    for h, e in cat.items():
        if not (dst / 'hardware' / (e['profile'] + '.yaml')).exists():
            raise FileNotFoundError(f'{h}: profile {e["profile"]}.yaml missing from overlay')
    (dst / 'catalog.json').write_text(json.dumps(cat, indent=2, ensure_ascii=False))
    return cat


def exchange(a, b):
    """Atomically swap two existing paths (Linux renameat2 RENAME_EXCHANGE)."""
    libc = ctypes.CDLL(None, use_errno=True)
    AT_FDCWD, RENAME_EXCHANGE = -100, 2
    if libc.renameat2(AT_FDCWD, os.fsencode(a), AT_FDCWD, os.fsencode(b), RENAME_EXCHANGE) != 0:
        e = ctypes.get_errno(); raise OSError(e, os.strerror(e), str(a))


tmp = OV.with_name(OV.name + f'.tmp-{os.getpid()}')
cat = build(tmp)
if OV.exists():
    exchange(tmp, OV)          # OV now holds the new tree, tmp the old one
    shutil.rmtree(tmp)         # old tree: no longer reachable via OV
else:
    os.rename(tmp, OV)
frozen = json.loads((REF / 'catalog.json').read_text())
print('overlay', OV, 'entries:', len(cat), 'extra:', [k for k in cat if k not in frozen])
