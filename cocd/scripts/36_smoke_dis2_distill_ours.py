#!/usr/bin/env python3
"""CPU-only structural and numerical verification of the distillation port.

Runs before any training.  Everything here is an assertion about the code, not
about the data: the tap path must reproduce the deployment path exactly, the
DIS2 ports must match an independently written reference, the distilled arms
must add no parameters to their control, and the student must start exactly on
the Teacher's self path.  Nothing touches MPS.

The frozen Teacher is loaded from disk when it is present, so the operator
inheritance and the driver-gradient properties are exercised at the real
initialisation rather than at a random one.
"""
import inspect
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
from losses.distill import (dis2_logit_kl, dis2_multilevel_kd, gain_protect, l2_kd_loss,
                            orthogonality_loss, pool_tokens, selective_multilevel_kd,
                            _weighted_norm_mse)
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student

torch.manual_seed(0)
N = 0


def ok(condition, label):
    global N
    N += 1
    if not condition:
        raise AssertionError(f'FAILED  {label}')
    print(f'  ok  {label}')


def grad_mass(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


print('building the Teacher and one student per variant (CPU)')
teacher = OursV2Teacher().eval()
ckpt = A / 'experiments' / 'ours_v2' / 'teacher.pt'
if ckpt.exists():
    teacher.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    print(f'  loaded the frozen Teacher from {ckpt.name}')
else:
    # No checkpoint: give the correction operators a non-trivial output so the
    # inheritance properties below are still exercised.
    with torch.no_grad():
        for op in (teacher.a3, teacher.a4):
            op.net[-1].weight.normal_(0, 1e-2); op.net[-1].bias.normal_(0, 1e-2)
    print('  teacher.pt absent; operators perturbed instead')
students = {v: OursV3Student(v).eval() for v in ('R0', 'R1', 'R2', 'R3')}
for s in students.values():
    s.load_teacher_self(teacher)
a = torch.randn(2, 5, 128, 128)
d = torch.randn(2, 5, 128, 128)

print('\n[1] the student starts exactly on the Teacher self path')
with torch.no_grad():
    t_plain, t_taps = teacher.forward_pair(a, d), teacher.forward_pair_taps(a, d)
    for i, key in enumerate(('z0', 'z4', 'z34')):
        ok(torch.equal(t_taps[0][key], t_plain[0][i]) and torch.equal(t_taps[1][key], t_plain[1][i]),
           f'teacher forward_pair_taps {key} is bit-identical to forward_pair')
    for v, s in students.items():
        ok(torch.allclose(s(a)[1], t_plain[0][0], atol=1e-6, rtol=1e-5), f'{v} initialises on the Teacher self path')

print('\n[2] the tap path reproduces the deployment path')
with torch.no_grad():
    ok(len(t_taps[0]['levels']) == 4, 'teacher exposes four pyramid levels')
    ok(t_taps[0]['pen'].shape[1] == 64 and t_taps[0]['logits'].shape[1] == 1,
       'penultimate map is 64 channels and logits are single channel')
    for v, s in students.items():
        z4, z34, _ = s(a)
        taps = s.forward_taps(a)
        ok(torch.equal(taps['z4'], z4) and torch.equal(taps['z34'], z34),
           f'{v} forward_taps z4/z34 is bit-identical to forward')
        ok(taps['z34'].shape == (2, 1, 128, 128), f'{v} tap logits land on the deployed resolution')
        ok([t.shape for t in taps['r']] == [taps['levels'][2].shape, taps['levels'][1].shape],
           f'{v} correction taps share the scale of the states they correct (r3 on P3, r4 on P4)')

print('\n[3] taps are the same arithmetic as decode')
with torch.no_grad():
    ft = teacher.backbone.encode(a)
    c2, p3, p4 = teacher.decoder.pyramid(ft)
    p2, pen, logits = teacher.decoder.taps(c2, p3, p4)
    ok(torch.allclose(teacher.decoder.decode(c2, p3, p4),
                      F.interpolate(logits, (128, 128), mode='bilinear', align_corners=False)),
       'decode equals interpolated taps logits')
    c2b, p5b, p4b, p3b = teacher.decoder.pyramid4(ft)
    ok(torch.equal(p3b, p3) and torch.equal(p4b, p4) and c2b is c2, 'pyramid is pyramid4 minus P5')

print('\n[4] DIS2 ports match an independent reference on random tensors')
x = torch.randn(3, 8, 16, 16)
y = torch.randn(3, 8, 16, 16)
ref = (F.normalize(x, p=2, dim=1, eps=1e-6) - F.normalize(y, p=2, dim=1, eps=1e-6)).pow(2).mean()
ok(torch.allclose(l2_kd_loss(x, y), ref, atol=1e-7), 'l2_kd_loss equals a hand-written channel-normalised MSE')
ok(torch.allclose(l2_kd_loss(x, y, spatial=True), l2_kd_loss(x, y), atol=1e-7),
   "l2_kd_loss spatial=True equals spatial=False (the snapshot's flag is a no-op)")
t1, t2 = torch.randn(4, 8), torch.randn(4, 8)
ref_orth = ((F.normalize(t1, dim=-1) * F.normalize(t2, dim=-1)).sum(-1) ** 2).mean()
ok(torch.allclose(orthogonality_loss(t1, t2), ref_orth, atol=1e-7), 'orthogonality_loss equals squared mean cosine')
ok(pool_tokens(x).shape == (3, 8), 'pool_tokens returns one token per sample')
zs = torch.randn(2, 1, 32, 32)
ok(torch.allclose(dis2_logit_kl(zs, zs), torch.zeros(()), atol=1e-6), 'T=2 logits KL vanishes for an identical student')
ok(dis2_logit_kl(torch.zeros_like(zs), torch.randn(2, 1, 32, 32)) > 0, 'T=2 logits KL is positive for a mismatched student')

print('\n[5] the DIS2 block and the selective block are zero on identical taps')
stu, tea = students['R1'].forward_taps(a), t_taps[0]
tgt = (torch.rand(2, 128, 128) > .7).float()
w0 = torch.rand(2, 128, 128)
ok(torch.allclose(dis2_multilevel_kd(stu, stu)[0], torch.zeros(()), atol=1e-6),
   'dis2_multilevel_kd is zero when student taps equal the target taps')
val, parts = dis2_multilevel_kd(stu, tea)
ok(torch.isfinite(val) and set(parts) == {'feat', 'pen', 'logit', 'div'},
   'dis2_multilevel_kd returns a finite total and names all four DIS2 terms')
ref_div = sum(orthogonality_loss(pool_tokens(r), pool_tokens(p))
              for r, p in zip(stu['r'], stu['levels'][1:3][::-1]))
ok(torch.allclose(parts['div'], ref_div, atol=1e-7),
   'the diversity term is the DIS2 cosine penalty between each correction and the state it corrects')
ok(torch.allclose(parts['div'], torch.zeros(()), atol=1e-7),
   'a zero-initialised correction has zero diversity, so the term is a regulariser and never a matching target')
ok(torch.allclose(selective_multilevel_kd(stu, stu, w0, tgt)[0], torch.zeros(()), atol=1e-6),
   'selective_multilevel_kd is zero when student taps equal the target taps')
val_s, parts_s = selective_multilevel_kd(stu, tea, w0, tgt)
ok(torch.isfinite(val_s) and set(parts_s) == {'feat', 'eff', 'pen', 'logit'},
   'selective_multilevel_kd scores the effect alongside the three DIS2 taps')

print('\n[6] the selection and the aggregation behave as specified')
# Synthetic logits, because the gate is data-dependent: on random inputs the
# Teacher's self and dual tracks agree, so nothing is selected.  The unit test
# fixes the three BCE values instead.
one = torch.ones(2, 1, 128, 128)
w_better = gain_protect(torch.full_like(one, 2.), torch.full_like(one, 6.), torch.full_like(one, 2.), one[:, 0])
ok(float(w_better.mean()) > 0.5, 'a dual track that is clearly better than the self track is selected')
w_worse = gain_protect(torch.full_like(one, 2.), torch.full_like(one, 6.), torch.full_like(one, 8.), one[:, 0])
ok(float(w_worse.mean()) == 0, 'nothing is selected where the student already beats the dual track')
ok(not w_better.requires_grad and float(w_better.min()) >= 0 and float(w_better.max()) <= 1,
   'gain_protect is detached and stays in [0, 1]')
ok(torch.allclose(_weighted_norm_mse(stu['levels'][1], tea['levels'][1], w0),
                  _weighted_norm_mse(stu['levels'][1], tea['levels'][1], w0 * 10), atol=1e-5),
   'a per-level term is a weighted average, invariant to the scale of w')
ok(torch.allclose(selective_multilevel_kd(stu, tea, torch.zeros_like(w0), tgt)[0], torch.zeros(()), atol=1e-6),
   'an all-zero selection makes the selective loss exactly zero, not undefined')
ok(not torch.allclose(selective_multilevel_kd(stu, tea, w0, tgt)[0],
                      selective_multilevel_kd(stu, tea, torch.ones_like(w0), tgt)[0]),
   'w actually changes the selective loss')
ok(not torch.allclose(selective_multilevel_kd(stu, tea, w0, tgt)[0], dis2_multilevel_kd(stu, tea)[0]),
   'the selective rule and the DIS2 rule are different numbers')

print('\n[7] gradients open the right block and never reach the teacher')
w = torch.ones_like(tgt)
ok(all(p.grad is None for p in teacher.parameters()), 'the teacher has no gradient before any backward')
for v, net, frozen in [('R1', 'q3', None), ('R2', 'd3', 'op3'), ('R3', 'd3', 'op3')]:
    s = students[v]
    st = s.forward_taps(a)
    (.5 * F.binary_cross_entropy_with_logits(st['z4'][:, 0], tgt) + selective_multilevel_kd(st, tea, w, tgt)[0]).backward()
    ok(grad_mass(getattr(s, net)) > 0, f'{v}: the distillation reaches the learnable entry point {net}')
    ok(s.backbone.features[0][0].weight.grad is not None, f'{v}: the distillation reaches the backbone')
    ok(all(p.grad is None for p in teacher.parameters()), f'{v}: no teacher parameter receives gradient')
    if frozen:
        ok(grad_mass(getattr(s, frozen)) == 0,
           f'{v}: the inherited operator {frozen} is exactly frozen while its driver still emits zero')

s3 = students['R3']
with torch.no_grad():
    for mod in (s3.d3, s3.d4):
        mod.net[-1].weight.normal_(0, 1e-3); mod.net[-1].bias.normal_(0, 1e-3)
s3.zero_grad()
st3 = s3.forward_taps(a)
(.5 * F.binary_cross_entropy_with_logits(st3['z4'][:, 0], tgt) + selective_multilevel_kd(st3, tea, w, tgt)[0]).backward()
ok(grad_mass(s3.op3) > 0, 'R3: opening the driver lets the inherited operator train again')

print('\n[8] the distilled arms add no parameters to their control')
counts = {v: sum(p.numel() for p in m.parameters()) for v, m in students.items()}
heads = sum(p.numel() for p in students['R1'].q3.parameters()) + sum(p.numel() for p in students['R1'].q4.parameters())
ok(counts['R1'] == counts['R0'] + heads, 'R1 is R0 plus the two free residual heads')
extra = {k for k in students['R1'].state_dict() if k not in students['R0'].state_dict()}
ok(all(k.startswith(('q3.', 'q4.')) for k in extra) and len(extra) == 12, 'R1 adds the residual heads and nothing else')

print('\n[9] deployment purity is unchanged')
for v, mod in students.items():
    names = [p.name for p in inspect.signature(type(mod).forward).parameters.values()
             if p.name != 'self' and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    ok(names == ['target'], f'{v} forward accepts the target track only')
    ok('geom' not in inspect.getsource(type(mod).forward).lower(), f'{v} forward never mentions geometry')

print(f'\nALL {N} ASSERTIONS PASSED')
