#!/usr/bin/env python3
"""Phase 4 -- legacy / shipped-baseline smoke test.  VALIDATION ONLY.

Loads every shipped checkpoint through the code path that consumes it, runs the
forward pass, and re-reads the *validation* partition so the recorded numbers can
be checked against CHECKPOINTS.md.  Nothing is trained, nothing is written, and
the test partition is never touched.

Run from the package root:  python windows_ops/phase4_legacy_smoke.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'cocd'))

spec = importlib.util.spec_from_file_location(
    'v2', PKG / 'cocd' / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

DEV = v2.DEV
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'  [{"ok  " if cond else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        FAIL.append(name)


print('=' * 78)
print('Phase 4 -- legacy baseline smoke test (validation only, no training)')
print('=' * 78)
print(f'\ndevice {DEV.type}  ({torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"})')

tr, va, te = v2.split()
train_set = v2.HaitiPairs(tr, True)
val_set = v2.HaitiPairs(va, False, train_set.stats)
print(f'splits: train {len(tr)}  val {len(va)}  test {len(te)}   (test NOT evaluated here)')
print(f'val specs {len(val_set.spec)} -> {len(val_set.spec) * 2} single-orbit cases')

# --------------------------------------------------------------------------- #
print('\n--- A. the v2 teacher (checkpoints/teacher_v2.pt) ---')
# --------------------------------------------------------------------------- #
teacher = v2.load_teacher(tag='_v2')
qv = v2.predict_teacher_heads(teacher, val_set)
got = {h: v2.metrics(qv[h], qv['y'])['auprc'] for h in ('z0', 'z4', 'z34')}
want = {'z0': 0.6304307068862383, 'z4': 0.6512060162024425, 'z34': 0.6670806978004765}
print(f'         tap_path_max_abs_diff = {qv["tap_path_max_abs_diff"]:.3e} '
      f'(deployment path vs tap path)')
# Tolerance 5e-5, not exact equality: the recorded values were produced on MPS
# with a different sklearn build, so the last two or three decimals of
# average_precision_score can move.  Everything at or above 1e-5 is agreement.
for h in ('z0', 'z4', 'z34'):
    check(f'teacher_v2 validation {h} AUPRC reproduces CHECKPOINTS.md',
          abs(got[h] - want[h]) < 5e-5, f'{got[h]:.6f} vs {want[h]:.6f} '
          f'(delta {got[h] - want[h]:+.2e})')
check('the tap path is bitwise the deployment path on CUDA',
      qv['tap_path_max_abs_diff'] == 0.0, f'{qv["tap_path_max_abs_diff"]:.1e}')
check('on the v2 teacher, dual > self on validation', got['z34'] > got['z0'],
      f'z34 {got["z34"]:.5f} > z0 {got["z0"]:.5f}  (+{got["z34"] - got["z0"]:.5f})')

# per-region ordering: the handoff says G10 gain should be the largest
yf = qv['y'].ravel().astype(bool); gt, gc = qv['gt'] > 0, qv['gc'] > 0
gain = {}
for region, m in (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc)):
    sel = m.ravel()
    a0 = v2.metrics(qv['z0'].ravel()[sel], yf[sel])['auprc'] if yf[sel].any() else float('nan')
    a34 = v2.metrics(qv['z34'].ravel()[sel], yf[sel])['auprc'] if yf[sel].any() else float('nan')
    gain[region] = a34 - a0
print('         dual - self AUPRC per region: ' +
      '  '.join(f'{k}={v:+.4f}' for k, v in gain.items()))
check('the dual gain is positive in every region', all(v > 0 for v in gain.values()))
del teacher
torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
print('\n--- B. the v1 teacher (legacy_checkpoints/v1_protocol/teacher.pt) ---')
# --------------------------------------------------------------------------- #
t1 = v2.OursV2Teacher().to(DEV)
t1.load_state_dict(torch.load(v2.resolve_checkpoint('teacher.pt'), map_location=DEV, weights_only=True))
t1.eval()
q1 = v2.predict_teacher_heads(t1, val_set)
g1 = v2.metrics(q1['z34'], q1['y'])['auprc']
check('v1 teacher checkpoint loads and runs its forward path', True, f'z34 val AUPRC {g1:.5f}')
check('v1 teacher tracks its recorded 0.89626 dual validation AUPRC (max over the curve)',
      g1 > 0.88, f'{g1:.5f} -- the recorded value is the max over the 50-epoch curve')
print(f'         v1 teacher: z0 {v2.metrics(q1["z0"], q1["y"])["auprc"]:.5f}  '
      f'z4 {v2.metrics(q1["z4"], q1["y"])["auprc"]:.5f}  z34 {g1:.5f}')
del t1, q1
torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
print('\n--- C. the v1 student arms ---')
# --------------------------------------------------------------------------- #
arms = [('R1_seed42.pt', 'R1'), ('R2_seed42.pt', 'R2'), ('R3_seed42.pt', 'R3'),
        ('DIS2_port_seed42.pt', 'R1')]
recorded = {'R1_seed42.pt': 0.89040, 'R2_seed42.pt': 0.89090,
            'R3_seed42.pt': 0.89400, 'DIS2_port_seed42.pt': 0.89320}
for name, variant in arms:
    m = v2.OursV3Student(variant).to(DEV)
    m.load_state_dict(torch.load(v2.resolve_checkpoint(name), map_location=DEV, weights_only=True))
    q = v2.predict_student(m, val_set)
    a = v2.metrics(q['p'], q['y'])['auprc']
    check(f'{name} loads, runs, and reproduces its recorded validation AUPRC',
          abs(a - recorded[name]) < 2e-4,
          f'{a:.5f} vs recorded {recorded[name]:.5f} '
          f'(delta {a - recorded[name]:+.2e}; recorded = max over the curve)')
    del m, q
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
print('\n--- D. the stage-1 S0 checkpoint and the SO / N0 code path ---')
# --------------------------------------------------------------------------- #
from models.landslide_cocd import SingleOrbitStudent  # noqa: E402

s0 = torch.load(v2.resolve_checkpoint('S0.pt'), map_location=DEV, weights_only=True)
legacy = SingleOrbitStudent(correction=False).to(DEV)
legacy.load_state_dict(s0, strict=True)
legacy.eval()
a, d, *_ = next(iter(v2.loader(val_set)))
with torch.no_grad():
    base, corr, r = legacy(a.to(DEV))
check('S0.pt loads into its own stage-1 class and runs', base.shape == (a.shape[0], 1, 128, 128),
      f'stage-1 output {tuple(base.shape)}')
del legacy, s0
torch.cuda.empty_cache()

# SO == OursV3Student('R0'), the N0 / single-orbit baseline, no teacher needed
so = v2.OursV3Student('R0').to(DEV)
so.eval()
with torch.no_grad():
    t = so.forward_taps(a.to(DEV))
check('SO / N0 (OursV3Student R0) runs and returns a single reading',
      torch.equal(t['z0'], t['z34']), 'R0 has no correction, so z0 == z34 by construction')
check('SO needs no teacher checkpoint at all',
      v2.CONFIGS['SO']['kd'] is None and v2.CONFIGS['SO']['init'] == 'scratch')
del so, t
torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
print('\n--- E. the configs the handoff says are runnable ---')
# --------------------------------------------------------------------------- #
for arm in ('SO', 'VKD', 'DIS2', 'R3', 'R3D'):
    check(f'CONFIGS[{arm!r}] exists', arm in v2.CONFIGS,
          f"variant={v2.CONFIGS[arm]['variant']} kd={v2.CONFIGS[arm]['kd']} "
          f"init={v2.CONFIGS[arm]['init']}")
check('R3g exists but is flagged deprecated', v2.CONFIGS['R3g'].get('deprecated') is True)
check('PLAN still carries R3D (HANDOFF item 1, unresolved in code)',
      'R3D' in v2.PLAN, f'PLAN = {v2.PLAN} -- drive a sweep with --stages until this is settled')
check('the v1 protocol branch is still reachable and differs from v2',
      v2.protocol_values('v1')['optimizer'] == 'adamw' and v2.protocol_values('v1')['epochs'] == 50
      and v2.protocol_values('v1')['batch'] == 2)
check('the v2 protocol branch is the default and matches protocol_v2.json',
      v2.PROTOCOL == 'v2' and v2.P['optimizer'] == 'adam' and v2.P['lr'] == 5e-05
      and v2.P['batch'] == 16 and v2.P['epochs'] == 20 and v2.P['patience'] == 3
      and v2.P['pos_weight'] is False and v2.LOSS_MODE == 'bce')

# --------------------------------------------------------------------------- #
print('\n--- F. the v1/v2 scale gap at the shared ImageNet initialisation ---')
# --------------------------------------------------------------------------- #
# The handoff records 0.742 for a fresh v2 student on the validation partition.
# This is one batch of the validation partition, so it is an order-of-magnitude
# read, not a reproduction of that number.
weight = v2.posweight(train_set)
fresh = v2.OursV3Student('R1').to(DEV)
fresh.train()
a, d, y, *_ = next(iter(v2.loader(val_set)))
xv, yv = torch.cat((a.to(DEV), d.to(DEV))), torch.cat((y.to(DEV), y.to(DEV)))
with torch.no_grad():
    _, z34, _ = fresh(xv)
    v2_val = float(v2.seg(z34, yv, weight))
    r3 = torch.sigmoid(fresh.forward_taps(xv)['z34'])[:, 0]
check('a fresh v2 student starts near the 0.74 BCE scale, not the v1 0.13',
      0.70 < v2_val < 0.85, f'one validation batch seg = {v2_val:.4f} '
      f'(handoff recorded 0.742 for the whole partition)')
check('and it is not predicting a constant background at initialisation',
      0.0 < float(r3.std()) and 0.0 < float(r3.mean()) < 1.0,
      f'initial sigmoid mean {float(r3.mean()):.4f} std {float(r3.std()):.4f}')
del fresh
torch.cuda.empty_cache()

print('\n' + '=' * 78)
if FAIL:
    print(f'PHASE 4 SMOKE FAILED -- {len(FAIL)} problem(s):')
    for f in FAIL:
        print(f'  - {f}')
    sys.exit(1)
print('PHASE 4 SMOKE PASSED -- every shipped code path loads and runs on CUDA,')
print('  and the recorded validation numbers reproduce.')
print('  the test partition was NOT evaluated.')
print('=' * 78)
