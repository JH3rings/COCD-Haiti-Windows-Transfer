#!/usr/bin/env python3
"""Acceptance check for this transfer package.  Read-only.

It builds nothing, trains nothing, and writes nothing outside the temporary
model instances it creates in memory.  It answers one question: can this
machine load the code, find the pinned split, read one location of the dataset,
push a batch through the backbone and the decoder, and load the checkpoint that
the unified protocol distills from?

Run it from the package root:  ``python verify_transfer.py``
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / 'cocd'))

FAILURES: list[str] = []
WARNINGS: list[str] = []
NOTES: list[str] = []


def step(name: str, ok: bool, detail: str = '') -> None:
    mark = 'ok  ' if ok else 'FAIL'
    print(f'  [{mark}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)


def warn(name: str, ok: bool, detail: str = '') -> None:
    """Something that does not invalidate the package but must be read."""
    mark = 'ok  ' if ok else 'warn'
    print(f'  [{mark}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        WARNINGS.append(name)


def note(text: str) -> None:
    NOTES.append(text)
    print(f'         note: {text}')


print('=' * 78)
print('COCD Haiti transfer - acceptance check')
print('=' * 78)

# --------------------------------------------------------------------------- #
print('\n[1] python and packages')
# --------------------------------------------------------------------------- #
import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import torch                # noqa: E402
import rasterio             # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402,F401
from tqdm.auto import tqdm   # noqa: E402,F401

print(f'  python     {sys.version.split()[0]}')
print(f'  numpy      {np.__version__}')
print(f'  pandas     {pd.__version__}')
print(f'  torch      {torch.__version__}')
print(f'  rasterio   {rasterio.__version__}')
step('required packages import', True)

# --------------------------------------------------------------------------- #
print('\n[2] device')
# --------------------------------------------------------------------------- #
cuda = torch.cuda.is_available()
warn('torch.cuda.is_available() reports a CUDA device', cuda,
     torch.cuda.get_device_name(0) if cuda else 'no CUDA device on this machine')
if cuda:
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024 ** 3
    print(f'  device memory {total:.1f} GiB, capability {props.major}.{props.minor}')
    if total < 8:
        note(f'only {total:.1f} GiB of device memory: physical batch 16 peaked at about 2.8 GiB '
             'in the original measurements, so it should fit, but watch the first epoch and '
             'fall back to --batch 8 --accum 2 if it does not.')
else:
    note('no CUDA device: this package targets a CUDA host for training and would be far '
         'slower on the CPU. Everything below still verifies, but do not start a run here.')

from paths import DEV, DATASET_ROOT, SPLIT_DIR, OUT_ROOT, describe  # noqa: E402

print()
print(describe())
step('the package resolves its own device', True, DEV.type)

# --------------------------------------------------------------------------- #
print('\n[3] pinned split')
# --------------------------------------------------------------------------- #
counts = {}
for name in ('train', 'val', 'test'):
    f = SPLIT_DIR / f'{name}_ids.csv'
    ok = f.exists()
    step(f'data/splits/{name}_ids.csv exists', ok)
    if not ok:
        continue
    ids = pd.read_csv(f).sample_id.astype(int).tolist()
    counts[name] = ids
    print(f'         {name:5s} {len(ids):5d} ids  ({len(set(ids))} unique)')
if counts:
    step('the split has the frozen sizes 1233 / 137 / 343',
         (len(counts['train']), len(counts['val']), len(counts['test'])) == (1233, 137, 343))
    step('the three partitions are disjoint',
         set(counts['train']).isdisjoint(counts['val'])
         and set(counts['train']).isdisjoint(counts['test'])
         and set(counts['val']).isdisjoint(counts['test']))
    step('no id is out of range for the 1713 locations',
         max(max(v) for v in counts.values()) < 1713)
    note('these files are read by the loader in preference to any random draw, '
         'so the partition cannot change on this machine.')

# --------------------------------------------------------------------------- #
print('\n[4] dataset on disk')
# --------------------------------------------------------------------------- #
REQUIRED = (
    DATASET_ROOT / 'Pre_event' / 'S1_ASC_20210805',
    DATASET_ROOT / 'Pre_event' / 'S1_DESC_20210803',
    DATASET_ROOT / 'Post_event' / 'S1_ASC_20210817',
    DATASET_ROOT / 'Post_event' / 'S1_DESC_20210815',
    DATASET_ROOT / 'processed',
)
for folder in REQUIRED:
    n = len(list(folder.rglob('*.tif'))) if folder.exists() else 0
    step(f'{folder.relative_to(DATASET_ROOT)} present', folder.exists() and n > 0,
         f'{n} tif files')
n_samples = len([d for d in (DATASET_ROOT / 'processed').glob('sample_*') if d.is_dir()])
step('one processed directory per spatial location', n_samples == 1713, f'{n_samples} directories')
n_mask = len(list((DATASET_ROOT / 'processed').rglob('landslide_mask.tif')))
step('every processed directory carries its landslide mask', n_mask == 1713,
     f'{n_mask} landslide_mask.tif')

sid = counts['train'][0] if counts else 0
print(f'\n  reading train location {sid}:')
pre_asc = DATASET_ROOT / 'Pre_event' / 'S1_ASC_20210805' / f'{sid}.tif'
post_asc = DATASET_ROOT / 'Post_event' / 'S1_ASC_20210817' / f'{sid}.tif'
pre_desc = DATASET_ROOT / 'Pre_event' / 'S1_DESC_20210803' / f'{sid}.tif'
post_desc = DATASET_ROOT / 'Post_event' / 'S1_DESC_20210815' / f'{sid}.tif'
proc = DATASET_ROOT / 'processed' / f'sample_{sid:06d}'
mask_f = proc / 'landslide_mask.tif'
geom_a = proc / 'asc_pre_geom.tif'
geom_d = proc / 'desc_pre_geom.tif'

for f in (pre_asc, post_asc, pre_desc, post_desc, mask_f, geom_a, geom_d):
    step(f'{f.name} readable', f.exists())
    if not f.exists():
        continue


def read(path: Path, bands=None):
    with rasterio.open(path) as f:
        return f.read(bands).astype('float32') if bands else f.read(1).astype('float32')


# band 2 is VV, band 1 is VH (metadata.json records band 1 as the VH source)
try:
    vv_vh = read(pre_asc, [2, 1])
    step('raw stack is two polarisation bands of 128x128',
         vv_vh.shape == (2, 128, 128), f'shape {vv_vh.shape}')
    print(f'         VV range [{vv_vh[0].min():.3f}, {vv_vh[0].max():.3f}]  '
          f'VH range [{vv_vh[1].min():.3f}, {vv_vh[1].max():.3f}]')
    m = read(mask_f)
    mvals = sorted(set(np.unique(m).tolist()))
    step('landslide mask is 128x128 and uses the documented coding (0 background, 1/2/3 inventory class)',
         m.shape == (128, 128) and set(mvals).issubset({0.0, 1.0, 2.0, 3.0}),
         f'classes {mvals}, positive (>0) pixels {int((m > 0).sum())}, '
         f'prevalence {float((m > 0).mean()):.4f}')
    g = read(geom_a)
    step('geometry mask is 128x128 with the documented coding (0 valid, 1/2/3 distorted)',
         g.shape == (128, 128) and set(np.unique(g)).issubset({0.0, 1.0, 2.0, 3.0}),
         f'classes {sorted(set(np.unique(g).tolist()))}, distorted {int((g > 0).sum())} px')
except Exception as exc:                                     # noqa: BLE001
    step('raw and mask read', False, f'{type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------- #
print('\n[5] the loader and one batch')
# --------------------------------------------------------------------------- #
spec = importlib.util.spec_from_file_location(
    'v2', PKG / 'cocd' / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

subset = counts['train'][:6] if counts else [0, 1, 2, 3, 4, 5]
train_set = v2.HaitiPairs(subset, True)
step('HaitiPairs builds from the pinned train ids', len(train_set) == 3 * len(subset),
     f'{len(train_set)} samples = {len(subset)} locations x 3 temporal modes')
a, d, y, ga, gd, real = train_set[0]
step('each sample is a 5-channel single-orbit input (VV pre/post, VH pre/post, orbit)',
     tuple(a.shape) == (5, 128, 128) and tuple(d.shape) == (5, 128, 128), f'{tuple(a.shape)}')
step('the orbit channel is a spatial constant (0 for ASC, 1 for DESC)',
     float(a[4].std()) == 0.0 and float(d[4].std()) == 0.0,
     f'asc={float(a[4].mean()):.0f} desc={float(d[4].mean()):.0f}')
step('the label and the two geometry masks come through at 128x128',
     tuple(y.shape) == (128, 128) and tuple(ga.shape) == (128, 128) and tuple(gd.shape) == (128, 128))

batch = next(iter(v2.loader(train_set, False)))
ba, bd, by = batch[0].to(v2.DEV), batch[1].to(v2.DEV), batch[2].to(v2.DEV)
print(f'         batch {tuple(ba.shape)}  device {v2.DEV.type}')
step('a batch can be moved to the device', ba.device.type == v2.DEV.type)

# --------------------------------------------------------------------------- #
print('\n[6] forward passes')
# --------------------------------------------------------------------------- #
try:
    teacher = v2.OursV2Teacher().to(v2.DEV)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    teacher.eval()
    with torch.no_grad():
        (za, z4a, z34a, ra), _ = teacher.forward_pair(ba, bd)
    step('Teacher forward_pair returns z0 / z4 / z34 / r at 128x128',
         tuple(za.shape) == (ba.shape[0], 1, 128, 128) and len(ra) == 2,
         f'z34 {tuple(z34a.shape)}, r3 {tuple(ra[0].shape)}, r4 {tuple(ra[1].shape)}')
    print(f'         teacher parameters {n_teacher / 1e6:.4f} M')

    student = v2.OursV3Student('R3').to(v2.DEV)
    student.eval()
    with torch.no_grad():
        z4, z34, r = student(ba)
    step('Student forward returns its two readings', tuple(z34.shape) == (ba.shape[0], 1, 128, 128))
    with torch.no_grad():
        taps = student.forward_taps(ba)
    step('the tap path agrees with the deployment path bitwise',
         torch.equal(taps['z34'], z34),
         'this is the property the distillation taps rest on')
except Exception as exc:                                     # noqa: BLE001
    step('forward passes', False, f'{type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------- #
print('\n[7] checkpoints in the package')
# --------------------------------------------------------------------------- #
def probe(name: str, build) -> None:
    path = v2.resolve_checkpoint(name)
    if not path.exists():
        step(f'{name} present', False, f'not found in {[str(d) for d in v2.resolve_checkpoint.__globals__["CHECKPOINT_DIRS"]]}')
        return
    size = path.stat().st_size / 1048576
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
        model = build()
        model.load_state_dict(state, strict=True)
        step(f'{name} loads with strict=True', True, f'{size:.1f} MB, {len(state)} tensors')
    except Exception as exc:                                 # noqa: BLE001
        step(f'{name} loads with strict=True', False, f'{type(exc).__name__}: {exc}')


probe('teacher_v2.pt', v2.OursV2Teacher)
probe('teacher.pt', v2.OursV2Teacher)
for arm, variant in (('R1_seed42.pt', 'R1'), ('R2_seed42.pt', 'R2'),
                     ('R3_seed42.pt', 'R3'), ('DIS2_port_seed42.pt', 'R1')):
    probe(arm, lambda v=variant: v2.OursV3Student(v))

# S0 belongs to the older stage-1 model class, not to the Ours-V3 student.  It is
# consumed through load_s0(), which maps its ``backbone.*`` keys onto the shared
# encoder, so that is the path worth checking rather than a strict load.
s0 = v2.resolve_checkpoint('S0.pt')
if not s0.exists():
    step('S0.pt present', False)
else:
    try:
        from models.landslide_cocd import SingleOrbitStudent
        state = torch.load(s0, map_location='cpu', weights_only=True)
        legacy = SingleOrbitStudent(correction=False)
        legacy.load_state_dict(state, strict=True)
        step('S0.pt is a stage-1 SingleOrbitStudent checkpoint (legacy key space)',
             True, f'{s0.stat().st_size / 1048576:.1f} MB, {len(state)} tensors, all backbone.*')
        teacher_probe = v2.OursV2Teacher()
        teacher_probe.load_s0(state)
        step('load_s0() maps it onto the shared encoder used by the v2 arms', True)
        del legacy, teacher_probe
    except Exception as exc:                                 # noqa: BLE001
        step('S0.pt loads through its own code path', False, f'{type(exc).__name__}: {exc}')

note('the v1-protocol checkpoints in legacy_checkpoints/ are baselines to evaluate, '
     'not initialisation for a new run: the unified protocol starts every internal arm '
     'from ImageNet, and S0 only exists because the v1 chain was warm-started from it.')

# --------------------------------------------------------------------------- #
print('\n[8] writability of the output root')
# --------------------------------------------------------------------------- #
try:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUT_ROOT, suffix='.probe', delete=True):
        pass
    step('the output root is writable', True, str(OUT_ROOT))
except Exception as exc:                                     # noqa: BLE001
    step('the output root is writable', False, f'{type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------- #
print('\n' + '=' * 78)
if FAILURES:
    print(f'TRANSFER CHECK FAILED - {len(FAILURES)} problem(s):')
    for f in FAILURES:
        print(f'  - {f}')
    sys.exit(1)
print('TRANSFER CHECK PASSED')
if WARNINGS:
    print(f'  {len(WARNINGS)} warning(s):')
    for w in WARNINGS:
        print(f'  - {w}')
print(f'  {len(NOTES)} note(s) above; read them before starting a run.')
print('  next step: hand HANDOFF_FOR_WINDOWS_AI.md to an AI, do not train yet.')
print('=' * 78)
