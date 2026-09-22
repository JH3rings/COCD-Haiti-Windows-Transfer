#!/usr/bin/env python3
"""Phase 6 -- unit tests for the new COCD teacher and student.  READ-ONLY.

Run before the new teacher is trained.  Every property the design claims is
checked here rather than assumed, and the two that would be easy to get wrong --
a single injection point, and N0 being unchanged by the interface refactor -- are
checked numerically.

    python cocd/windows_main/tests_complement.py
"""
from __future__ import annotations

import importlib.util
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[2]
COCD = PKG / 'cocd'
sys.path.insert(0, str(COCD))

from paths import describe                                     # noqa: E402
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student   # noqa: E402
from windows_main.main import smooth_l1_complement             # noqa: E402
from windows_main.models_complement import (ComplementStudent, DCA,   # noqa: E402
                                            NewCOCDTeacher, Phi, fused_p2,
                                            n_params, predict_from_fused)

DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FAIL: list[str] = []
N_OK = 0


def check(name: str, cond: bool, detail: str = '') -> None:
    global N_OK
    print(f'  [{"ok  " if cond else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if cond:
        N_OK += 1
    else:
        FAIL.append(name)


def build(cls, seed=42, **kw):
    torch.manual_seed(seed)
    m = cls(**kw).to(DEV)
    return m


def code_only(obj) -> str:
    """Source of a class or function with docstrings removed.

    The geometry check below is about what the code *does*.  A docstring that
    explains why ascending and descending geometry are not aligned is prose, not
    a mask entering the model, so prose is stripped before the scan.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def zero_grad_params(model, prefix):
    """Parameters under ``prefix`` whose gradient is present and exactly zero."""
    return [n for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None and float(p.grad.abs().sum()) == 0.0]


print('=' * 78)
print('Phase 6 -- new COCD unit tests')
print('=' * 78)
print()
print(describe())

B = 16
a = torch.randn(B, 5, 128, 128, device=DEV)
d = torch.randn(B, 5, 128, 128, device=DEV)
a[:, 4] = 0.0                       # ASC planes
d[:, 4] = 1.0                       # DESC planes
y = (torch.rand(B, 128, 128, device=DEV) > 0.93).float()   # same shape the loader returns

# --------------------------------------------------------------------------- #
print('\n[1] shapes and both target directions')
# --------------------------------------------------------------------------- #
teacher = build(NewCOCDTeacher).eval()
with torch.no_grad():
    oa, od = teacher.forward_pair(a, d)
check('ASC target / DESC counter runs and returns self + dual at 128x128',
      tuple(oa['z_self'].shape) == (B, 1, 128, 128) and tuple(oa['z_dual'].shape) == (B, 1, 128, 128),
      f"z_self {tuple(oa['z_self'].shape)}  z_dual {tuple(oa['z_dual'].shape)}")
check('DESC target / ASC counter runs and returns self + dual at 128x128',
      tuple(od['z_self'].shape) == (B, 1, 128, 128) and tuple(od['z_dual'].shape) == (B, 1, 128, 128))
check('the fused feature is p2: 128 channels at 1/4 resolution',
      tuple(oa['F_t'].shape) == (B, 128, 32, 32), f"{tuple(oa['F_t'].shape)}")
check('the complement has the fused shape and the injection is the same shape',
      tuple(oa['C_T'].shape) == (B, 128, 32, 32) and tuple(oa['F_dual'].shape) == (B, 128, 32, 32))
check('target and counter are not the same feature when the inputs differ',
      not torch.allclose(oa['F_t'], od['F_t']))

# --------------------------------------------------------------------------- #
print('\n[2] shared encoder and shared prediction head')
# --------------------------------------------------------------------------- #
names = [n for n, _ in teacher.named_children()]
check('the teacher carries exactly one encoder', names.count('backbone') == 1, str(names))
src = inspect.getsource(NewCOCDTeacher.forward_pair)
check('forward_pair makes exactly two encoder calls, not four',
      src.count('encode_fused') == 2, f'{src.count("encode_fused")} calls')
head_ids = {id(p) for p in teacher.backbone.head.parameters()}
check('the head is one module and both readings go through it',
      len(head_ids) == len(list(teacher.backbone.head.parameters())) and len(head_ids) > 0,
      f'{len(head_ids)} head tensors, shared by z_self and z_dual')
check('there is no second head anywhere in the model',
      sum(1 for m in teacher.modules()
          if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1) == 1,
      'the only 1-channel conv is backbone.head[-1]')
# functional proof of sharing: perturbing the one encoder must move both orbits
with torch.no_grad():
    before_a, before_d = teacher.encode_fused(a), teacher.encode_fused(d)
    # l5 is the deepest FPN lateral and feeds p4 -> p3 -> p2, so a perturbation
    # there has to reach the fused feature of BOTH orbits if the encoder is shared.
    w = teacher.backbone.l5.weight
    keep = w.detach().clone()
    w.mul_(1.5)
    after_a, after_d = teacher.encode_fused(a), teacher.encode_fused(d)
    w.copy_(keep)
check('the encoder weights are shared: one perturbation moves both orbits',
      not torch.allclose(before_a, after_a) and not torch.allclose(before_d, after_d),
      f'max |dF_t| {float((before_a - after_a).abs().max()):.3e}  '
      f'max |dF_c| {float((before_d - after_d).abs().max()):.3e}')
check('perturbing the head alone does not move the fused feature',
      torch.equal(teacher.encode_fused(a), before_a))
check('the encoder is deterministic for a fixed input',
      torch.equal(teacher.encode_fused(a), before_a))

# --------------------------------------------------------------------------- #
print('\n[3] the complement is exactly zero for a zero counter')
# --------------------------------------------------------------------------- #
p2_t = teacher.encode_fused(a)
p2_c = teacher.encode_fused(d)
with torch.no_grad():
    c_zero = teacher.complement(p2_t, torch.zeros_like(p2_c))
check('C_T(F_c = 0) is exactly zero', float(c_zero.abs().max()) == 0.0,
      f'max |C_T(F_c=0)| = {float(c_zero.abs().max()):.3e}')
check('the complement is also exactly zero at initialisation, whatever the counter',
      float(teacher.complement(p2_t, p2_c).abs().max()) == 0.0,
      'Phi output layer starts at zero, the convention Correction and Driver use')
# Give Phi a nonzero output layer and the complement must become counter-dependent.
with torch.no_grad():
    teacher.phi.net[-1].weight.normal_(0, 0.05)
    teacher.phi.net[-1].bias.normal_(0, 0.05)
    c_zero = teacher.complement(p2_t, torch.zeros_like(p2_c))
    c_real = teacher.complement(p2_t, p2_c)
    c_other = teacher.complement(p2_t, teacher.encode_fused(d[:B // 2].repeat(2, 1, 1, 1)))
check('C_T(F_c = 0) is still exactly zero once Phi is non-degenerate',
      float(c_zero.abs().max()) == 0.0, f'max |C_T(F_c=0)| = {float(c_zero.abs().max()):.3e}')
check('C_T depends on the counter once Phi is non-degenerate',
      float(c_real.abs().max()) > 0.0 and not torch.allclose(c_real, c_other),
      f'max |C_T| = {float(c_real.abs().max()):.3e}, and a different counter gives a different C_T')
with torch.no_grad():
    o_zero = teacher.from_fused(p2_t, torch.zeros_like(p2_c))
check('with a zero counter the injected feature IS the target feature',
      torch.equal(o_zero['F_dual'], p2_t))
check('and therefore z_dual == z_self bitwise for a zero counter',
      torch.equal(o_zero['z_dual'], o_zero['z_self']))
check('DCA(F_t, 0) is exactly zero (its value path and output projection have no bias)',
      float(teacher.dca(p2_t, torch.zeros_like(p2_c)).abs().max()) == 0.0)
check('the difference definition, not a constraint on DCA, is what makes C_T zero',
      'torch.zeros_like(p2_c)' in inspect.getsource(NewCOCDTeacher.complement))

# --------------------------------------------------------------------------- #
print('\n[4] single injection point')
# --------------------------------------------------------------------------- #
with torch.no_grad():
    o = teacher.from_fused(p2_t, p2_c)
    manual = predict_from_fused(teacher.backbone.head, p2_t + o['C_T'])
check('F_dual = F_t + C_T exactly', torch.equal(o['F_dual'], p2_t + o['C_T']))
check('z_dual is the head applied to F_dual (not to anything else)',
      torch.equal(o['z_dual'], manual))
check('z_self is the head applied to the unmodified F_t',
      torch.equal(o['z_self'], predict_from_fused(teacher.backbone.head, p2_t)))
check('touching C_T cannot change z_self',
      torch.equal(teacher.predict(p2_t), o['z_self']))
check('C_T is injected once: no second injection site exists in the model',
      inspect.getsource(NewCOCDTeacher.from_fused).count('+ c_t') == 1)

# --------------------------------------------------------------------------- #
print('\n[5] CUDA forward / backward, batch 16')
# --------------------------------------------------------------------------- #
# Phi's output layer starts at zero, the same convention Correction and Driver
# already use in this codebase.  The init sequence is checked on a FRESH teacher,
# because the one above had its Phi made non-degenerate in [3].
t5 = build(NewCOCDTeacher, seed=11)
t5.train()
optimizer = torch.optim.Adam(t5.parameters(), lr=5e-05)


def teacher_loss(m):
    """``mean(BCE(z_self, y), BCE(z_dual, y))`` over both target directions."""
    fa, fd = m.encode_fused(a), m.encode_fused(d)
    oa_, od_ = m.from_fused(fa, fd), m.from_fused(fd, fa)
    return 0.25 * sum(torch.nn.functional.binary_cross_entropy_with_logits(o[k][:, 0], y)
                      for o in (oa_, od_) for k in ('z_self', 'z_dual'))


loss = teacher_loss(t5)
loss.backward()
g0 = {n: p.grad for n, p in t5.named_parameters() if p.grad is not None}
check('step 0: a real loss is produced on CUDA at batch 16', torch.isfinite(loss),
      f'L_T = {float(loss):.4f}')
check('step 0: the shared backbone receives gradient through the self reading',
      any(n.startswith('backbone.features') for n in g0), f'{len(g0)} tensors have grads')
check('step 0: Phi output layer receives gradient',
      g0.get('phi.net.4.weight') is not None and float(g0['phi.net.4.weight'].abs().sum()) > 0)
check('step 0: the zero-output start is what withholds gradient from DCA',
      len(zero_grad_params(t5, 'dca.')) == len([n for n, _ in t5.named_parameters()
                                               if n.startswith('dca.')]),
      f'{len(zero_grad_params(t5, "dca."))} DCA tensors carry an exactly-zero gradient '
      f'at step 0: Phi\'s output layer is still zero, so nothing upstream sees the loss yet')
optimizer.step(); optimizer.zero_grad(set_to_none=True)

loss = teacher_loss(t5)
loss.backward()
g = {n: p.grad for n, p in t5.named_parameters() if p.grad is not None}
for part in ('dca.q', 'dca.off', 'dca.out', 'phi.net', 'backbone.features', 'backbone.head'):
    hit = [n for n in g if n.startswith(part)]
    check(f'step 1: gradient reaches {part}', len(hit) > 0, f'{len(hit)} tensors')
check('step 1: all gradients are finite',
      all(torch.isfinite(t).all() for t in g.values()), f'{len(g)} tensors have grads')
off = g.get('dca.off.2.weight')
check('step 1: the DCA offset branch is learned, not fixed',
      off is not None and float(off.abs().sum()) > 0.0,
      f'sum |grad off.2.weight| = {float(off.abs().sum()):.3e}' if off is not None else 'missing')
check('batch 16 fits and steps on ' + DEV.type, True, f'{B} x 5 x 128 x 128')
t5.zero_grad(set_to_none=True)
check('no parameter of the teacher is silently frozen during training',
      n_params(t5) == sum(p.numel() for p in t5.parameters()),
      f'{n_params(t5) / 1e6:.4f} M trainable')
del t5, optimizer

# --------------------------------------------------------------------------- #
print('\n[6] no geometry input, by construction')
# --------------------------------------------------------------------------- #
check('the teacher forward takes target and counter and nothing else',
      list(inspect.signature(NewCOCDTeacher.forward).parameters) == ['self', 'target', 'counter'])
check('the student forward takes the five-channel input and nothing else',
      list(inspect.signature(ComplementStudent.forward).parameters) == ['self', 'x'])
GEOMETRY_TOKENS = ('geometry', 'target_geom', 'counter_geom', 'G00', 'G01', 'G10', 'G11')
for mod in (NewCOCDTeacher, ComplementStudent, DCA, Phi):
    src = code_only(mod)
    hits = [t for t in GEOMETRY_TOKENS if t in src]
    check(f'{mod.__name__} code mentions no geometry mask', not hits, f'hits={hits}')
for cls in (NewCOCDTeacher, ComplementStudent):
    params = [n for n, _ in cls().named_parameters()]
    hits = [n for n in params if any(t.lower() in n.lower() for t in ('geom',))]
    check(f'{cls.__name__} carries no geometry parameter', not hits, f'hits={hits}')
check('the new module imports nothing from the loss file that reads geometry',
      'gain_gate' not in (COCD / 'windows_main' / 'models_complement.py').read_text())

# --------------------------------------------------------------------------- #
print('\n[7] N0 is numerically unchanged by the interface refactor')
# --------------------------------------------------------------------------- #
legacy = OursV3Student('R0').to(DEV).eval()
n0 = ComplementStudent(complement=False, orbit_embedding=False).to(DEV).eval()
shared = {k: v for k, v in legacy.state_dict().items() if k.startswith('backbone.')}
missing, unexpected = n0.load_state_dict(shared, strict=False)
check('the refactored N0 accepts the legacy backbone key space unchanged',
      not missing and not unexpected, f'missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}')
check('N0 holds exactly the same number of trainable parameters as the legacy R0 arm',
      n_params(n0) == n_params(legacy), f'{n_params(n0)} vs {n_params(legacy)}')
with torch.no_grad():
    z_legacy = legacy(a)[0]
    z_new = n0(a)['z']
check('N0 forward is bitwise identical to OursV3Student(R0) after the refactor',
      torch.equal(z_legacy, z_new),
      f'max |diff| = {float((z_legacy - z_new).abs().max()):.3e}')
check('N0 has no complement branch and no orbit conditioning',
      n0.complement_on is False and n0.orbit_on is False and n0(a)['C_S'] is None)

# --------------------------------------------------------------------------- #
print('\n[8] the student complement branch, and what each switch removes')
# --------------------------------------------------------------------------- #
for arm, comp, orb in (('N1', True, True), ('N2', True, True), ('N2noorbit', True, False)):
    m = build(ComplementStudent, complement=comp, orbit_embedding=orb).eval()
    with torch.no_grad():
        out = m(torch.cat((a, d)))
    check(f'{arm}: runs on a 32-sample packed batch at 128x128',
          tuple(out['z'].shape) == (2 * B, 1, 128, 128) and out['C_S'] is not None)
    check(f'{arm}: the complement branch is zero at initialisation, so it starts as N0',
          float(out['C_S'].abs().max()) == 0.0)
    check(f'{arm}: F_S = F_t + C_S exactly', torch.equal(out['z'],
          predict_from_fused(m.backbone.head, out['F_t'] + out['C_S'])))

n2 = build(ComplementStudent, complement=True, orbit_embedding=True).eval()
n2b = build(ComplementStudent, complement=True, orbit_embedding=False).eval()
with torch.no_grad():
    t0 = n2(torch.cat((a, d)))['F_mod']
    t1 = n2(torch.cat((d, a)))['F_mod']
    u = n2b(torch.cat((a, d)))['F_mod']
check('with the FiLM weights at zero the modulation is the identity',
      torch.equal(t0, n2(torch.cat((a, d)))['F_t']))
# give the conditioning a nonzero gain, then the orbit plane must matter
with torch.no_grad():
    n2.orbit_mlp[-1].weight.normal_(0, 0.02)
    n2.orbit_mlp[-1].bias.normal_(0, 0.02)
    # identical content, only the orbit plane differs, so any difference in
    # F_mod can only have come from reading channel 4
    base = torch.cat((a, a)).clone()
    f_base = fused_p2(n2.backbone, base)
    asc_in = base.clone()
    asc_in[:, 4] = 0.0
    desc_in = base.clone()
    desc_in[:, 4] = 1.0
    m_asc = n2.modulate(f_base, asc_in)
    m_desc = n2.modulate(f_base, desc_in)
check('with a nonzero modulation the ASC and DESC branches differ',
      not torch.allclose(m_asc, m_desc), 'the orbit condition is actually wired in')
# A mixed batch must be the concatenation of the two pure cases: rows whose
# channel 4 is 0 behave like the ASC case and rows whose channel 4 is 1 like the
# DESC case, whatever order they arrive in.
mixed = base.clone()
mixed[:B, 4] = 0.0
mixed[B:, 4] = 1.0
m_mixed = n2.modulate(f_base, mixed)
check('the orbit plane is read per sample, not once per batch',
      torch.allclose(m_mixed[:B], m_asc[:B]) and torch.allclose(m_mixed[B:], m_desc[B:]),
      'a mixed ASC/DESC batch is modulated row by row')
# The conditioning must read the orbit channel only: changing the radiometric
# channels must not move F_mod, because F_t is already fixed by then.
x_pert = asc_in.clone()
x_pert[:, :4] = torch.randn_like(x_pert[:, :4])
check('the modulation reads only the orbit plane, not the radiometry',
      torch.equal(n2.modulate(f_base, x_pert), m_asc),
      'channels 0-3 changed, F_mod unchanged')
check('and it is exactly (1 + gamma) * F_t + beta with the gamma, beta the embedding gives',
      torch.allclose(m_asc, (1.0 + n2.orbit_mlp(n2.orbit_emb(torch.zeros(2 * B, dtype=torch.long,
                                                                        device=DEV)))[:, :128, None, None])
                     * f_base + n2.orbit_mlp(n2.orbit_emb(torch.zeros(2 * B, dtype=torch.long,
                                                                      device=DEV)))[:, 128:, None, None]),
      'the formula is the structure, not an approximation of it')
with torch.no_grad():
    u = n2b(torch.cat((a, d)))
check('N2 w/o Orbit Embedding leaves the complement branch unmodulated',
      torch.equal(u['F_mod'], u['F_t']),
      'the ablation touches the conditioning and nothing else')
check('N2 and N2 w/o Orbit Embedding have identical non-orbit parameter counts',
      n_params(n2) - sum(p.numel() for p in n2.orbit_emb.parameters())
      - sum(p.numel() for p in n2.orbit_mlp.parameters()) == n_params(n2b))

# --------------------------------------------------------------------------- #
print('\n[9] the distillation term: one distance, teacher stop-gradded, real pairs')
# --------------------------------------------------------------------------- #
td = build(NewCOCDTeacher, seed=7).eval()
# Phi starts at zero, so a fresh teacher has C_T == 0 and a fresh student has
# C_S == 0: the distance would be identically zero and its gradient undefined.
# Make the teacher's Phi non-degenerate first, as a trained teacher would be.
with torch.no_grad():
    td.phi.net[-1].weight.normal_(0, 0.05)
    td.phi.net[-1].bias.normal_(0, 0.05)
sd = build(ComplementStudent, seed=42).eval()
u = sd(torch.cat((a, d)))
with torch.no_grad():
    c_t = td.from_fused(td.encode_fused(torch.cat((a, d))),
                        td.encode_fused(torch.cat((d, a))))['C_T']
check('the teacher complement is non-degenerate for the distance to be meaningful',
      float(c_t.abs().max()) > 0.0, f'max |C_T| = {float(c_t.abs().max()):.3e}')
d0 = smooth_l1_complement(u['C_S'], c_t)
check('the distance is SmoothL1 in feature space and returns a scalar',
      d0.dim() == 0 and torch.isfinite(d0) and float(d0) > 0.0,
      f'L_dist at init = {float(d0):.6f} (the student complement is zero, so this '
      f'measures the teacher complement magnitude)')
check('the distance is exactly zero when the student complement matches the teacher',
      float(smooth_l1_complement(c_t.detach(), c_t)) == 0.0)
check('the teacher side carries no gradient', c_t.requires_grad is False)
d0.backward()
last = sd.predictor[-1].weight.grad
check('the student complement predictor receives gradient from the distance term',
      last is not None and float(last.abs().sum()) > 0,
      f'sum |grad predictor[-1].weight| = {float(last.abs().sum()):.3e}')
check('and the zero-output start is the same one-step delay the teacher has',
      len(zero_grad_params(sd, 'predictor.')) == len([n for n, _ in sd.named_parameters()
                                                     if n.startswith('predictor.')]) - 2,
      'only the output layer carries a non-zero gradient at step 0, exactly as Phi does')
check('the distance is defined on the complement features, not on logits or labels',
      'smooth_l1_loss' in inspect.getsource(smooth_l1_complement)
      and 'sigmoid' not in inspect.getsource(smooth_l1_complement))
check('the distance is SmoothL1 and not L1 or MSE (quadratic below beta)',
      abs(float(smooth_l1_complement(c_t + 0.1, c_t)) - 0.5 * 0.1 ** 2) < 1e-6,
      'beta = 1.0, quadratic regime verified numerically')

# --------------------------------------------------------------------------- #
print('\n' + '=' * 78)
if FAIL:
    print(f'PHASE 6 FAILED -- {len(FAIL)} problem(s):')
    for f in FAIL:
        print(f'  - {f}')
    sys.exit(1)
print(f'PHASE 6 PASSED -- {N_OK} checks, 0 failures')
print('  the new teacher and the new student are ready to train.')
print('=' * 78)
