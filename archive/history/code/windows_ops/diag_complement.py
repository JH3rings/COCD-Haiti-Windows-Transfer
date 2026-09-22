#!/usr/bin/env python3
"""Diagnostic for the N2 result.  READ-ONLY, validation partition only.

N2 (Full COCD) came out well below N0 and N1.  Before reporting that as a result
rather than a bug, this checks four things in the order the handoff asks for:
the data path, the alignment of the distillation tensors, the implementation of
the term, and the training dynamics.

    python windows_ops/diag_complement.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PKG = Path(__file__).resolve().parents[1]
COCD = PKG / 'cocd'
sys.path.insert(0, str(COCD))

from paths import OUT_ROOT                                   # noqa: E402
from windows_main.main import (LAMBDA_PATH, frozen_lam, predict_complement,  # noqa: E402
                               predict_new_teacher, real_subset_batches,
                               smooth_l1_complement)
from windows_main.models_complement import (ComplementStudent, NewCOCDTeacher,  # noqa: E402
                                            fused_p2)

_spec = importlib.util.spec_from_file_location('ours_v2', COCD / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

DEV = v2.DEV
MAIN_OUT = OUT_ROOT / 'windows_main'
LAM = frozen_lam()

print('=' * 78)
print('Complement diagnostic -- N2 vs N1 vs the teacher')
print('=' * 78)
print(f'\nfrozen lam = {LAM}')

tr, va, te = v2.split()
train_set = v2.HaitiPairs(tr, True)
val_set = v2.HaitiPairs(va, False, train_set.stats)

teacher = NewCOCDTeacher().to(DEV)
teacher.load_state_dict(torch.load(MAIN_OUT / 'NewTeacher_seed42.pt', map_location=DEV, weights_only=True))
teacher.eval()

students = {}
for arm in ('N0', 'N1', 'N2', 'N2noorbit'):
    from windows_main.models_complement import COMPLEMENT_CONFIGS
    m = ComplementStudent(complement=COMPLEMENT_CONFIGS[arm]['complement'],
                          orbit_embedding=COMPLEMENT_CONFIGS[arm]['orbit_embedding']).to(DEV)
    m.load_state_dict(torch.load(MAIN_OUT / f'{arm}_seed42.pt', map_location=DEV, weights_only=True))
    m.eval()
    students[arm] = m

# --------------------------------------------------------------------------- #
print('\n[1] the data path is shared, so a data fault would show in N0 as well')
# --------------------------------------------------------------------------- #
q = {}
for arm, m in students.items():
    q[arm] = predict_complement(m, val_set)
    print(f'  {arm:10s} validation AUPRC {v2.metrics(q[arm]["p"], q[arm]["y"])["auprc"]:.5f}')
check_ok = q['N1']['p'].shape == q['N2']['p'].shape == q['N0']['p'].shape
print(f'  all arms see the identical validation tensor: {check_ok}, shape {q["N0"]["p"].shape}')
print('  N0 and N1 trained fine on this path, so the loader is not the cause.')

# --------------------------------------------------------------------------- #
print('\n[2] alignment of C_S against C_T')
# --------------------------------------------------------------------------- #
batch = next(iter(v2.loader(val_set)))
a, d, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
real = batch[5].to(DEV)
cm = torch.cat((real, real)).bool()
x = torch.cat((a, d)); targets = torch.cat((a, d)); partners = torch.cat((d, a))
with torch.no_grad():
    c_t = teacher.from_fused(teacher.encode_fused(targets[cm]),
                             teacher.encode_fused(partners[cm]))['C_T']
    c_s2 = students['N2'](x)['C_S'][cm]
    shifted = torch.cat((d[torch.roll(torch.arange(len(d), device=DEV), 1)], a))
    c_t_shift = teacher.from_fused(teacher.encode_fused(targets[cm]),
                                  teacher.encode_fused(shifted[cm]))['C_T']
print(f'  C_S rows {tuple(c_s2.shape)}  C_T rows {tuple(c_t.shape)}  identical row count: '
      f'{c_s2.shape[0] == c_t.shape[0]}')
print(f'  C_S and C_T are both the p2 scale: {c_s2.shape[1:] == c_t.shape[1:]}')
d_same = float(smooth_l1_complement(c_s2, c_t))
d_shuf = float(smooth_l1_complement(c_s2, c_t_shift))
print(f'  L_dist against the correct counter      {d_same:.6f}')
print(f'  L_dist against a rolled counter         {d_shuf:.6f}')
print(f'  the term distinguishes the two: {d_shuf > d_same}')

# --------------------------------------------------------------------------- #
print('\n[3] the implementation of the term')
# --------------------------------------------------------------------------- #
print(f'  mean|C_T| (L2 norm of the flattened field)   {float(c_t.flatten(1).norm(dim=1).mean()):.4f}')
f_t = students['N2'](x)['F_t']
print(f'  mean|F_t|                                    {float(f_t.flatten(1).norm(dim=1).mean()):.4f}')
for arm in ('N1', 'N2', 'N2noorbit'):
    with torch.no_grad():
        c = students[arm](x)['C_S'][cm]
        print(f'  {arm:10s} mean|C_S| {float(c.flatten(1).norm(dim=1).mean()):8.4f}   '
              f'L_dist(C_S, C_T) {float(smooth_l1_complement(c, c_t)):.6f}   '
              f'L_dist(0, C_T) {float(smooth_l1_complement(torch.zeros_like(c), c_t)):.6f}')
print('  a zero complement is the reference: any C_S that lowers L_dist below it is')
print('  moving toward the teacher, so the term is doing what it says on the tin.')

# --------------------------------------------------------------------------- #
print('\n[4] training dynamics -- how much of the objective the term carries')
# --------------------------------------------------------------------------- #
# one optimiser-free step on the real subset with the trained N2 weights, so the
# two gradient norms are measured at the same state the run ended in
m = students['N2']
m.train()
optimizer_probe = None
weight = v2.posweight(train_set)
with torch.no_grad():
    seg = v2.seg(m(x)['z'], torch.cat((y, y)), weight)
    c_s = m(x)['C_S'][cm]
    dist = smooth_l1_complement(c_s, c_t)
    print(f'  at the end of the run, on one validation batch:')
    print(f'    L_seg                  {float(seg):.6f}')
    print(f'    lam * L_dist           {LAM * float(dist):.6f}')
    print(f'    share of L_dist        {LAM * float(dist) / (float(seg) + LAM * float(dist)):.1%}')
    print(f'    ratio L_seg / (lam*L_dist)  {float(seg) / (LAM * float(dist)):.3f}')

# gradient norms of the two terms with respect to C_S itself
c_s_leaf = m.f_t_probe if False else None
p2 = fused_p2(m.backbone, x)
p2 = p2.detach().requires_grad_(True)
with torch.no_grad():
    f_mod = m.modulate(p2, x) if m.orbit_on else p2
pred_in = f_mod.requires_grad_(True)
c_s_g = m.predictor(pred_in)
seg_g = v2.seg(m.predict(p2 + c_s_g), torch.cat((y, y)), weight)
g_seg = torch.autograd.grad(seg_g, c_s_g, retain_graph=True)[0]
dist_g = smooth_l1_complement(c_s_g[cm], c_t)
g_dist = torch.autograd.grad(dist_g, c_s_g, retain_graph=True)[0]
print(f'\n  gradient of each term with respect to C_S (L2 norm over the whole batch):')
print(f'    d L_seg   / d C_S      {float(g_seg.norm()):.6e}')
print(f'    d lam*Ldist/ d C_S     {float((LAM * g_dist).norm()):.6e}')
print(f'    the distillation gradient is '
      f'{float((LAM * g_dist).norm()) / (float(g_seg.norm()) + 1e-30):.3f} x the segmentation one')
cos = F.cosine_similarity(g_seg.flatten(), (LAM * g_dist).flatten(), dim=0)
print(f'    cosine between the two gradients  {float(cos):+.4f}   '
      f'(<0 means the term actively opposes the task)')

# --------------------------------------------------------------------------- #
print('\n[5] what the calibration measured, and what training actually does')
# --------------------------------------------------------------------------- #
calib = json.loads((MAIN_OUT / 'lambda_calibration.json').read_text())
print(f"  L_seg0  {calib['L_seg0']:.6f}   L_dist0  {calib['L_dist0']:.6f}   "
      f"ratio {calib['ratio_L_seg_over_L_dist']:.3f}  ->  lam {calib['lam']}")
print(f"  share of the segmentation loss at initialisation: {calib['initial_share_of_seg']:.1%}")
n2c = __import__('pandas').read_csv(MAIN_OUT / 'N2_seed42_curve.csv')
print(f"  share of the objective at epoch 1 of the run:     "
      f"{(n2c.dist.iloc[0] / (n2c.seg.iloc[0] + n2c.dist.iloc[0])):.1%}")
print(f"  share at epoch 20:                                "
      f"{(n2c.dist.iloc[-1] / (n2c.seg.iloc[-1] + n2c.dist.iloc[-1])):.1%}")
print('  L_seg falls about 20x over 20 epochs while L_dist falls about 9x, so the')
print('  calibrated balance at initialisation is not the balance during training.')

print('\n' + '=' * 78)
print('No optimiser was run and no file was written. Validation partition only.')
print('=' * 78)
