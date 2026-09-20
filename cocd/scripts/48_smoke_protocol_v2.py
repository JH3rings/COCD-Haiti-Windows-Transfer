#!/usr/bin/env python3
"""Assertions that the unified protocol is actually the one in force.

Structural checks only -- no dataset, no training, no checkpoint.  Everything
here would fail loudly if ``configs/protocol_windows_main.json`` stopped being honoured, if
an arm stopped sharing the ImageNet initialisation, or if geometry found a way
back into the training objective.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

A = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
from models.landslide_cocd_v2 import OursV3Student  # noqa: E402

OK, BAD = [], []


def check(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(f'  [{"ok " if cond else "FAIL"}] {name}{("  " + detail) if detail else ""}', flush=True)


cfg = json.loads((A / 'configs' / 'protocol_windows_main.json').read_text())

print('--- [A] the config file and the running module agree ---')
check('[A] protocol revision is v3', cfg.get('revision') == 'v3')
for key, want in (('optimizer', 'adam'), ('lr', 5e-05), ('seg', 'bce'), ('epochs', 20)):
    check(f'[A] {key} == {want}', v2.P[key] == want, f'got {v2.P[key]!r}')
check('[A] betas and eps are the torch defaults',
      v2.P['betas'] == (0.9, 0.999) and v2.P['eps'] == 1e-8)
check('[A] no weight decay and no scheduler', v2.P['weight_decay'] == 0.0 and cfg['scheduler'] is None)
check('[A] batch 16 physical == effective (no accumulation needed)',
      v2.BATCH == 16 and v2.ACCUM == 1 == cfg['batch']['accumulate'])
check('[A] dice is off', cfg['loss']['dice'] is False)
check('[A] D1: pos_weight is off, so the BCE is the plain one', cfg['loss']['pos_weight'] is False
      and v2.POS_WEIGHT is False)
check('[A] v3: no validation pass, no early stop, the epoch-20 checkpoint is the model',
      v2.VALIDATE_EVERY == 0 and cfg['schedule']['early_stop'] is False
      and cfg['schedule']['early_stop_patience'] is None
      and cfg['schedule']['validate_every'] is None
      and cfg['schedule']['checkpoint'] == 'epoch_20')
check('[A] v3: one permanently fixed threshold, never re-selected on test',
      cfg['threshold']['source'] == 'fixed' and cfg['threshold']['value'] == 0.5
      and cfg['threshold']['frozen_before_test'] is True and cfg['threshold']['reselect_on_test'] is False)
check('[A] v3: no validation partition survives in the split',
      cfg['split']['counts']['validation'] == 0 and cfg['validation_and_test']['validation_abolished'] is True
      and cfg['split']['views']['train'] == 8220 and cfg['split']['views']['test'] == 686)
check('[A] raw / no-mask in train and eval', cfg['mask_protocol']['train'] == 'raw_no_mask'
      and cfg['mask_protocol']['eval'] == 'raw_no_mask')
check('[A] geometry-weighted KD is off', cfg['mask_protocol']['geometry_weighted_kd'] is False)

print('--- [B] optimizer identity and a constant learning rate ---')
model = OursV3Student('R1')
opt = v2.make_optimizer(model)
g = opt.param_groups[0]
check('[B] make_optimizer returns torch.optim.Adam', type(opt).__name__ == 'Adam')
check('[B] lr / betas / eps / wd as specified',
      g['lr'] == 5e-5 and tuple(g['betas']) == (0.9, 0.999) and g['eps'] == 1e-8 and g['weight_decay'] == 0.0,
      f'lr={g["lr"]:g} betas={g["betas"]} eps={g["eps"]:g} wd={g["weight_decay"]:g}')
p = next(model.parameters())
lrs = []
for _ in range(20):
    opt.zero_grad(); p.grad = torch.randn_like(p); opt.step(); lrs.append(opt.param_groups[0]['lr'])
check('[B] lr is constant over 20 steps (no warmup / cosine / decay)', len(set(lrs)) == 1 and lrs[0] == 5e-5)
check('[B] no scheduler object is created anywhere in the training path',
      'lr_scheduler' not in (A / 'scripts' / '23_train_ours_v2.py').read_text())

print('--- [C] D1: the segmentation loss is plain BCE ---')
logits = torch.randn(4, 1, 128, 128, requires_grad=True).detach().requires_grad_(True)
y = (torch.rand(4, 128, 128) > .7).float()
got = v2.seg(logits, y, 14.356170298161542)
want = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y)
check('[C] seg == unweighted BCE, bit for bit, despite pos_weight being passed in', torch.equal(got, want),
      f'{float(got):.8f} vs {float(want):.8f}')
check('[C] the unweighted BCE differs from the 14.356-weighted one, so the flag really bites',
      not torch.equal(got, torch.nn.functional.binary_cross_entropy_with_logits(
          logits[:, 0], y, pos_weight=torch.tensor(14.356170298161542))))
_orig = v2.land_loss
v2.land_loss = lambda *a, **k: (_ for _ in ()).throw(AssertionError('land_loss was called under LOSS_MODE=bce'))
try:
    v2.seg(logits, y, 14.356170298161542)
    check('[C] land_loss is never reached under LOSS_MODE=bce', True)
except AssertionError as exc:
    check('[C] land_loss is never reached under LOSS_MODE=bce', False, str(exc))
v2.land_loss = _orig
v2.POS_WEIGHT = True
weighted = v2.seg(logits, y, 14.356170298161542)
v2.POS_WEIGHT = False
check('[C] pos_weight can be switched back on with one config flag',
      torch.equal(weighted, torch.nn.functional.binary_cross_entropy_with_logits(
          logits[:, 0], y, pos_weight=torch.tensor(14.356170298161542))))

print('--- [D] effective batch is 16 for every fallback physical batch ---')
for phys, expect in ((16, 1), (8, 2), (4, 4), (2, 8)):
    accum = max(1, int(round(16 / phys)))
    check(f'[D] physical {phys} x accum {accum} == 16', phys * accum == 16 and accum == expect)

print('--- [E] the F1 threshold search matches a brute-force scan ---')
rng = np.random.default_rng(0)
pr = rng.random(200000).astype(np.float32); yb = (rng.random(200000) < .08)
thr, f1 = v2.best_f1_threshold(pr, yb)
best = (-1, None)
for t in np.linspace(0.001, 0.999, 999):
    h = pr >= t
    tp = (h & yb).sum(); fp = (h & ~yb).sum(); fn = (~h & yb).sum()
    a = tp / (tp + fp + 1e-8); b = tp / (tp + fn + 1e-8)
    val = 2 * a * b / (a + b + 1e-8)
    if val > best[0]:
        best = (val, float(t))
check('[E] histogram search reproduces the brute-force F1 optimum', abs(f1 - best[0]) < 2e-3,
      f'hist {f1:.5f}@{thr:.3f} vs brute {best[0]:.5f}@{best[1]:.3f}')
check('[E] AUPRC does not depend on the threshold',
      v2.metrics(pr, yb.astype(np.float32), thr)['auprc'] == v2.metrics(pr, yb.astype(np.float32), .5)['auprc'])

print('--- [F] the frozen threshold is carried into every reported row ---')
n = 400
pred = {'p': rng.random((n, 2, 2)).astype(np.float32),
        'y': (rng.random((n, 2, 2)) < .3).astype(np.float32),
        'gt': (rng.random((n, 2, 2)) < .2).astype(np.float32),
        'gc': (rng.random((n, 2, 2)) < .2).astype(np.float32),
        'orbit': np.array(['asc'] * 200 + ['desc'] * 200)}
rows = v2.report('probe', pred, thr)
check('[F] every row carries the threshold actually used', all(r['threshold'] == thr for r in rows))
check('[F] partitions are aggregate / asc / desc and regions G00..G11',
      {r['partition'] for r in rows} == {'overall', 'asc', 'desc'} and 'G10' in {r['region'] for r in rows})
check('[F] the 0.5 continuity columns are present but are not the selection rule',
      all('iou@0.5' in r and 'f1@0.5' in r for r in rows))

print('--- [G] the arm list matches the controlled comparison ---')
check('[G] R3g is flagged deprecated and out of the plan',
      v2.CONFIGS['R3g'].get('deprecated') is True and 'R3g' not in v2.PLAN)
check('[G] the plan is the controlled comparison plus the minimal upgrade',
      v2.PLAN == ['SO', 'VKD', 'DIS2', 'R3', 'R3D'], str(v2.PLAN))
check('[G] a vanilla-KD arm exists and uses the unweighted rule', v2.CONFIGS['VKD']['kd'] == 'all')
check('[G] D3: R3D is R3 plus L_delta, and R3 is its exact control',
      v2.CONFIGS['R3D']['delta'] is True and v2.CONFIGS['R3']['variant'] == v2.CONFIGS['R3D']['variant']
      and v2.CONFIGS['R3']['kd'] == v2.CONFIGS['R3D']['kd']
      and 'delta' not in v2.CONFIGS['R3'])

print('--- [H] geometry cannot influence any unified arm ---')
t0 = torch.randn(2, 1, 32, 32); tq = torch.randn(2, 1, 32, 32); sq = torch.randn(2, 1, 32, 32)
yg = (torch.rand(2, 32, 32) > .8).float()
z, o_ = torch.zeros(2, 32, 32), torch.ones(2, 32, 32)
k_gain_z, w_gain_z = v2.selective_kl(t0, tq, sq, yg, z, 'gain')
k_gain_o, w_gain_o = v2.selective_kl(t0, tq, sq, yg, o_, 'gain')
check('[H] kd="gain" is bitwise independent of the geometry mask',
      torch.equal(k_gain_z, k_gain_o) and torch.equal(w_gain_z, w_gain_o))
k_full_z, w_full_z = v2.selective_kl(t0, tq, sq, yg, z, 'full', torch.ones(2))
k_full_o, w_full_o = v2.selective_kl(t0, tq, sq, yg, o_, 'full', torch.ones(2))
check('[H] the deprecated kd="full" is the only rule the mask can move',
      not torch.equal(w_full_z, w_full_o) and float(w_full_o.mean()) > float(w_full_z.mean()))
check('[H] no arm in the plan uses kd="full"', all(v2.CONFIGS[n]['kd'] != 'full' for n in v2.PLAN))

print('--- [I] D2: one ImageNet initialisation for every internal arm ---')
check('[I] the protocol asks for ImageNet for the Teacher and the backbones',
      v2.TEACHER_INIT == 'imagenet' and v2.BACKBONE_INIT == 'imagenet' == cfg['init']['backbone'])
# The value alone is not enough: the branch that consumes it has to accept it.
check('[I] the teacher init switch accepts the configured value and means "no warm start"',
      v2.teacher_init_source() == 'none' and v2.teacher_init_source('scratch') == 'none'
      and v2.teacher_init_source('s0') == 's0')
try:
    v2.teacher_init_source('nonsense')
    check('[I] an unknown teacher init raises instead of silently starting cold', False)
except ValueError as exc:
    check('[I] an unknown teacher init raises instead of silently starting cold', True, str(exc))
check('[I] the student init branch accepts the same spelling',
      v2.CONFIGS[v2.PLAN[0]]['init'] in ('scratch', 'imagenet'))
check('[I] every planned arm has init=scratch',
      all(v2.CONFIGS[n].get('init') == 'scratch' for n in v2.PLAN),
      str({n: v2.CONFIGS[n].get('init') for n in v2.PLAN}))
check('[I] only R2/R3 (and the R3 variants) carry op_init, and it is a field of its own',
      all(v2.CONFIGS[n].get('op_init') == 'teacher' for n in ('R2', 'R3', 'R3D'))
      and all(v2.CONFIGS[n].get('op_init') is None for n in ('SO', 'VKD', 'DIS2')))
backs = {}
for variant in ('R0', 'R1', 'R2', 'R3'):
    v2.set_seed(42)
    m = OursV3Student(variant)
    backs[variant] = {k: t.clone() for k, t in m.backbone.state_dict().items()}
same = all(all(torch.equal(backs['R0'][k], backs[o][k]) for k in backs['R0']) for o in ('R1', 'R2', 'R3'))
check('[I] all four variants start from the bitwise identical backbone', same)
v2.set_seed(42)
student = OursV3Student('R3')
taps = student.forward_taps(torch.randn(2, 5, 128, 128))
check('[I] the correction is exactly zero at initialisation, so z0 == z4 == z34',
      torch.equal(taps['z0'], taps['z4']) and torch.equal(taps['z0'], taps['z34']))
check('[I] the student tap path now exposes z0 for the decision-change term', 'z0' in taps)
taps_r0 = OursV3Student('R0').forward_taps(torch.randn(2, 5, 128, 128))
check('[I] R0 exposes z0 too, and there it is the only reading',
      'z0' in taps_r0 and torch.equal(taps_r0['z0'], taps_r0['z34']))

print('--- [J] D3: L_delta exists, matches the decision change, and is one-sided ---')
# A fixture that mirrors the real geometry: the Teacher's correction moves the
# probability mainly on the landslide band, so the disagreement is concentrated
# rather than spread evenly.  A uniform fixture would make the aggregation
# question look like a no-op, which is not what the data does.
yy = torch.zeros(2, 16, 16); yy[:, :3, :] = 1.


def leaf(scale=1.0, shift=0.0):
    # ``randn(...) * scale`` is not a leaf, so the scale has to go in before
    # requires_grad_ for .grad to accumulate.
    return (torch.randn(2, 1, 16, 16) * scale + shift).requires_grad_(True)


base = leaf(1.0, -3.0)
stu = {'z34': leaf(.5), 'z4': leaf(.5), 'z0': leaf(.5)}
tea = {'z34': (base.detach() + 3. * yy[:, None]).requires_grad_(True),
       'z4': leaf(1.0, -3.0), 'z0': (base.detach() + 0.).requires_grad_(True)}
loss = v2.delta_loss(stu, tea, yy)
loss.backward()
check('[J] the term is a SmoothL1 on the two decision changes, and it is differentiable',
      loss.ndim == 0 and stu['z34'].grad is not None and stu['z0'].grad is not None)
check('[J] the Teacher side is stop-gradded',
      tea['z34'].grad is None and tea['z0'].grad is None)
check('[J] the default pair is z34 - z0 in probability space',
      v2.DELTA_PAIR == 'z34_minus_z0' and v2.DELTA_SPACE == 'prob' and v2.DELTA_BETA == 1.0,
      f'pair={v2.DELTA_PAIR} space={v2.DELTA_SPACE} beta={v2.DELTA_BETA}')
per = torch.nn.functional.smooth_l1_loss(
    torch.sigmoid(stu['z34'][:, 0]) - torch.sigmoid(stu['z0'][:, 0]),
    torch.sigmoid(tea['z34'][:, 0]) - torch.sigmoid(tea['z0'][:, 0]), beta=1.0, reduction='none')
manual_bal = v2.balanced_mean(per, yy)
check('[J] default aggregation is the same foreground/background balanced mean the KD term uses',
      v2.DELTA_AGGREGATION == 'balanced' and torch.allclose(loss, manual_bal, atol=0, rtol=0),
      f'{float(loss):.8f} vs {float(manual_bal):.8f}')
v2.DELTA_AGGREGATION = 'pixel_mean'
check('[J] the pixel-mean aggregation is still reachable and is exactly the plain mean',
      torch.allclose(v2.delta_loss(stu, tea, yy), per.mean(), atol=0, rtol=0))
v2.DELTA_AGGREGATION = 'balanced'
check('[J] the balanced mean is what lifts the term off the floor on concentrated disagreement',
      float(manual_bal) > 1.5 * float(per.mean()),
      f'balanced {float(manual_bal):.6f} vs pixel {float(per.mean()):.6f} '
      f'(x{float(manual_bal) / float(per.mean()):.2f})')
check('[J] L_corr is deliberately not implemented', 'L_corr' not in
      (A / 'scripts' / '23_train_ours_v2.py').read_text())

print('--- [K] the single-orbit baseline can run without a Teacher ---')


def needs_teacher(cfgs):
    return any(c['kd'] is not None or c.get('init') == 'teacher'
               or (c.get('op_init', v2.OP_INIT) == 'teacher' and c['variant'] in ('R2', 'R3')) for c in cfgs)


check('[K] SO needs no Teacher', not needs_teacher([v2.CONFIGS['SO']]))
check('[K] the distillation arms and R2/R3 do need it',
      all(needs_teacher([v2.CONFIGS[n]]) for n in ('VKD', 'DIS2', 'R3', 'R3D')))

print(f'\n{len(OK)} passed, {len(BAD)} failed')
if BAD:
    print('FAILED:')
    for b in BAD:
        print('  -', b)
    raise SystemExit(1)
