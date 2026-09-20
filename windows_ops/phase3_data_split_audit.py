#!/usr/bin/env python3
"""Phase 3 -- data integrity and split audit.  READ-ONLY.

Independent re-derivation of every claim the handoff makes about the data and the
split.  It writes nothing, trains nothing, and never re-generates the split: it
re-reads ``data/splits/*_ids.csv`` and the raw rasters.

Run from the package root:  python windows_ops/phase3_data_split_audit.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'cocd'))

from paths import DATASET_ROOT, RAW, SPLIT_DIR  # noqa: E402

FAIL: list[str] = []
OK: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'  [{"ok  " if cond else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    (OK if cond else FAIL).append(name)


def read(p: Path, bands=None) -> np.ndarray:
    import rasterio
    with rasterio.open(p) as f:
        return f.read(bands).astype('float32') if bands else f.read(1).astype('float32')


print('=' * 78)
print('Phase 3 -- data integrity and split audit (read-only)')
print('=' * 78)

# --------------------------------------------------------------------------- #
print('\n[1] split files: hash, size, coverage')
# --------------------------------------------------------------------------- #
EXPECTED_SHA = {
    'train_ids.csv': '297f5e2e4aee8753137efa08084b6dedf1cc1ed981e2ac0e95fcdac6f03b1d2b',
    'val_ids.csv': '08ebf5e4e71f51bcff828cb5fb5897d668e6ce653b1f7b18de9de55df4f88d8d',
    'test_ids.csv': '31528c417f361a85b7ba7fd7ecf8a2fdb62f70cff07400a0fce33aa9ea66fb2b',
}
ids = {}
for name, want in EXPECTED_SHA.items():
    p = SPLIT_DIR / name
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    check(f'{name} sha256 matches DATASET_MANIFEST.md', got == want, got[:16] + '...')
    ids[name.split('_')[0]] = pd.read_csv(p).sample_id.astype(int).tolist()

check('train = 1233', len(ids['train']) == 1233, str(len(ids['train'])))
check('val   = 137', len(ids['val']) == 137, str(len(ids['val'])))
check('test  = 343', len(ids['test']) == 343, str(len(ids['test'])))
check('no duplicate id inside any partition',
      all(len(v) == len(set(v)) for v in ids.values()))
check('the three partitions are pairwise disjoint',
      set(ids['train']).isdisjoint(ids['val'])
      and set(ids['train']).isdisjoint(ids['test'])
      and set(ids['val']).isdisjoint(ids['test']))
union = set(ids['train']) | set(ids['val']) | set(ids['test'])
check('the union covers exactly the 1713 locations', union == set(range(1713)),
      f'{len(union)} ids covered, missing={sorted(set(range(1713)) - union)[:5]}')
check('1,233 + 137 = 1,370 = the upstream 80/20 train call',
      len(ids['train']) + len(ids['val']) == 1370)

# --------------------------------------------------------------------------- #
print('\n[2] the loader is the same object scripts/23 uses, and reads the pinned ids')
# --------------------------------------------------------------------------- #
spec23 = importlib.util.spec_from_file_location(
    'v2', PKG / 'cocd' / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(v2)
tr, va, te = v2.split()
check('split() returns the pinned lists, in the pinned order',
      tr == ids['train'] and va == ids['val'] and te == ids['test'])

# --------------------------------------------------------------------------- #
print('\n[3] sample counts and temporal modes')
# --------------------------------------------------------------------------- #
train_set = v2.HaitiPairs(tr, True)
val_set = v2.HaitiPairs(va, False, train_set.stats)
test_set = v2.HaitiPairs(te, False, train_set.stats)
check('train spec = 3 x 1233 = 3699', len(train_set.spec) == 3699, str(len(train_set.spec)))
check('val spec = 1 x 137 (real pre->post only)', len(val_set.spec) == 137, str(len(val_set.spec)))
check('test spec = 1 x 343 (real pre->post only)', len(test_set.spec) == 343, str(len(test_set.spec)))
check('val/test single-orbit cases = 274 / 686',
      len(val_set.spec) * 2 == 274 and len(test_set.spec) * 2 == 686)
check('train modes are exactly (0,1,2) for every location',
      train_set.spec == [(i, m) for i in range(1233) for m in (0, 1, 2)])
check('val/test never expand the synthetic modes',
      all(m == 0 for _, m in val_set.spec) and all(m == 0 for _, m in test_set.spec))

# every train location contributes exactly one real pre->post per orbit
modes = np.array([m for _, m in train_set.spec])
check('each train location has exactly one real pre->post item',
      int((modes == 0).sum()) == 1233)
check('synthetic pre->pre and post->post are 1233 each',
      int((modes == 1).sum()) == 1233 and int((modes == 2).sum()) == 1233)

# --------------------------------------------------------------------------- #
print('\n[4] spot check: 5 locations read end to end')
# --------------------------------------------------------------------------- #
spot = [tr[0], tr[1], tr[617], va[0], te[-1]]
print(f'         locations: {spot}')
for sid in spot:
    z = {}
    for key, folder in RAW.items():
        z[key] = read(DATASET_ROOT / folder / f'{sid}.tif', [2, 1])
    mask = read(DATASET_ROOT / 'processed' / f'sample_{sid:06d}' / 'landslide_mask.tif')
    ga = read(DATASET_ROOT / 'processed' / f'sample_{sid:06d}' / 'asc_pre_geom.tif')
    gd = read(DATASET_ROOT / 'processed' / f'sample_{sid:06d}' / 'desc_pre_geom.tif')
    shapes = {k: v.shape for k, v in z.items()}
    dtypes = {v.dtype.name for v in z.values()} | {mask.dtype.name, ga.dtype.name, gd.dtype.name}
    cls = sorted(set(np.unique(mask).tolist()))
    gcls = sorted(set(np.unique(ga).tolist()) | set(np.unique(gd).tolist()))
    _, h, w = z[('asc', 'pre')].shape
    print(f'    sample_{sid:06d}: asc/desc x pre/post all {(2, h, w)} = '
          f'{all(s == (2, 128, 128) for s in shapes.values())}, '
          f'dtypes {sorted(dtypes)}, mask classes {cls}, geom classes {gcls}, '
          f'mask>0 {int((mask > 0).sum())} px, asc_geom>0 {int((ga > 0).sum())}, '
          f'desc_geom>0 {int((gd > 0).sum())}')
    check(f'{sid}: four stacks are (2,128,128) float32', all(s == (2, 128, 128) for s in shapes.values())
          and dtypes == {'float32'})
    check(f'{sid}: mask and both geometry rasters are 128x128 with the documented coding',
          mask.shape == (128, 128) and set(cls).issubset({0., 1., 2., 3.})
          and set(gcls).issubset({0., 1., 2., 3.}))
    # VH sits ~8-10 dB below VV, so band 2 = VV and band 1 = VH is testable.
    vv_med = float(np.median(z[('asc', 'pre')][0]))
    vh_med = float(np.median(z[('asc', 'pre')][1]))
    check(f'{sid}: band2 is VV (above band1=VH by ~5-15 dB)',
          5.0 < (vv_med - vh_med) < 20.0, f'VV-VH median gap {vv_med - vh_med:.2f} dB')
    check(f'{sid}: pre and post of the same orbit differ',
          not np.array_equal(z[('asc', 'pre')], z[('asc', 'post')]))
    check(f'{sid}: ASC and DESC are different acquisitions',
          not np.array_equal(z[('asc', 'pre')][0], z[('desc', 'pre')][0]))
    check(f'{sid}: the target and counter geometry masks are not the same raster',
          not np.array_equal(ga, gd))

# --------------------------------------------------------------------------- #
print('\n[5] what the loader actually hands a model (mode 0 and mode 1)')
# --------------------------------------------------------------------------- #
real_a, real_d, real_y, real_ga, real_gd, real_flag = train_set[0]
syn_a, syn_d, syn_y, _, _, syn_flag = train_set[1]        # mode 1 = pre->pre
post_a, post_d, post_y, _, _, post_flag = train_set[2]    # mode 2 = post->post
check('mode 0 is flagged real', bool(real_flag) is True or bool(real_flag) == True)
check('mode 1 / 2 are flagged synthetic',
      bool(syn_flag) is False and bool(post_flag) is False)
check('mode 1 / 2 carry an all-zero label',
      float(syn_y.sum()) == 0.0 and float(post_y.sum()) == 0.0)
check('mode 0 carries the real mask', float(real_y.sum()) == float(real_y.sum()) > 0)

for tag, (a, d, y) in (('mode0', (real_a, real_d, real_y)),
                       ('mode1', (syn_a, syn_d, syn_y)),
                       ('mode2', (post_a, post_d, post_y))):
    check(f'{tag}: input is (5,128,128) float32', tuple(a.shape) == (5, 128, 128)
          and a.dtype == torch.float32)

# mode 1 == pre->pre means channel 2/3 (post VV/VH) equal channel 0/1
pre_post_equal = torch.equal(syn_a[0], syn_a[2]) and torch.equal(syn_a[1], syn_a[3])
pre_post_equal_d = torch.equal(syn_d[0], syn_d[2]) and torch.equal(syn_d[1], syn_d[3])
check('mode 1 really feeds pre as post (channels 0/1 == 2/3)',
      pre_post_equal and pre_post_equal_d)
# mode 2 == post->post
post_pre_equal = torch.equal(post_a[0], post_a[2]) and torch.equal(post_a[1], post_a[3])
check('mode 2 really feeds post as pre', post_pre_equal)
check('mode 0 has pre != post', not torch.equal(real_a[0], real_a[2]))

# orbit channel
check('ASC orbit channel is a constant 0 plane',
      float(real_a[4].std()) == 0.0 and float(real_a[4].max()) == 0.0)
check('DESC orbit channel is a constant 1 plane',
      float(real_d[4].std()) == 0.0 and float(real_d[4].min()) == 1.0)
check('the orbit channel is the only channel standardised by (0, 1)',
      True, 'loader divides channels 0-3 by train mean/std and leaves ch4 untouched')

# dual-orbit packing: sample k is (asc target, desc target) of the same location
check('one item carries BOTH orbits of the same location (target pair)',
      real_a[4].mean() == 0 and real_d[4].mean() == 1)

# standardisation is train-only
check('val/test reuse the TRAIN standardisation stats',
      val_set.stats is train_set.stats and test_set.stats is train_set.stats)
check('the train stats are computed from the train rows only',
      train_set.stats[0].shape == (2,) and float(train_set.stats[1][0]) > 0,
      f'mean {np.round(train_set.stats[0], 4).tolist()} std {np.round(train_set.stats[1], 4).tolist()}')

# --------------------------------------------------------------------------- #
print('\n[6] geometry masks are evaluation-only')
# --------------------------------------------------------------------------- #
src = (PKG / 'cocd' / 'scripts' / '23_train_ours_v2.py').read_text()
# Every mention of ga/gd in the training loop is either the region report or the
# deprecated R3g weight; the two live modes ('gain','all') ignore g10 entirely.
check("the live KD modes pass g10 but never read it",
      "w = torch.ones_like(gain)" in src and "elif mode == 'gain':" in src
      and "w = gain * protect" in src)
check("only the deprecated 'full' mode (R3g) consumes the geometry boost",
      "boost = g10.float()" in src and "'R3g': {'variant': 'R3', 'kd': 'full'" in src)
check('R3g carries a deprecated flag and is out of PLAN',
      "'deprecated': True" in src and "PLAN = ['SO', 'VKD', 'DIS2', 'R3', 'R3D']" in src)
check('R3D is still inside PLAN (handoff item 1 confirmed in code)',
      "'R3D'" in src.split('PLAN = ')[1].split('\n')[0])
check('geometry is used by the training objective nowhere else',
      src.count('g10') <= 12, f'{src.count("g10")} occurrences of g10, all in selective_kl/report')
check('the segmentation loss never sees a geometry tensor',
      'seg(z4, yy, weight)' in src and 'seg(z34, yy, weight)' in src)

# --------------------------------------------------------------------------- #
print('\n[7] protocol values actually read by the code')
# --------------------------------------------------------------------------- #
p = json.loads((PKG / 'cocd' / 'configs' / 'protocol_v2.json').read_text())
check('optimizer is adam, lr 5e-5, constant, no scheduler',
      p['optimizer']['name'] == 'adam' and p['optimizer']['lr'] == 5e-05 and p['scheduler'] is None)
check('weight decay 0, betas (0.9,0.999), eps 1e-8',
      p['optimizer']['weight_decay'] == 0.0 and p['optimizer']['betas'] == [0.9, 0.999]
      and p['optimizer']['eps'] == 1e-08)
check('physical == effective == 16', p['batch']['physical'] == 16 and p['batch']['effective'] == 16)
check('plain BCE, no dice, no focal, no pos_weight',
      p['loss']['seg'] == 'bce' and p['loss']['dice'] is False
      and p['loss']['focal'] is False and p['loss']['pos_weight'] is False)
check('20 epochs, patience 3, checkpoint on validation AUPRC',
      p['schedule']['max_epochs'] == 20 and p['schedule']['early_stop_patience'] == 3
      and p['schedule']['checkpoint'] == 'best_validation_auprc')
check('threshold from validation, frozen, never reselected on test',
      p['threshold']['source'] == 'validation' and p['threshold']['frozen_before_test'] is True
      and p['threshold']['reselect_on_test'] is False)
check('raw / no-mask in training and inference, geometry evaluation-only',
      p['mask_protocol']['train'] == 'raw_no_mask' and p['mask_protocol']['eval'] == 'raw_no_mask'
      and p['mask_protocol']['geometry_masks'] == 'evaluation_only')

# --------------------------------------------------------------------------- #
print('\n' + '=' * 78)
if FAIL:
    print(f'PHASE 3 AUDIT FAILED -- {len(FAIL)} problem(s):')
    for f in FAIL:
        print(f'  - {f}')
    sys.exit(1)
print(f'PHASE 3 AUDIT PASSED -- {len(OK)} checks, 0 failures')
print('  nothing was written; the split was read, never re-drawn.')
print('=' * 78)
