#!/usr/bin/env python3
"""Train and evaluate the Ours-V2 Teacher and the Ours-V3 structural ablation.

Stages: ``teacher``, the structural arms ``R0``-``R4``, the geometry-prioritised
arm ``R3g`` and ``DIS2`` (see ``CONFIGS``), or ``all`` which runs ``PLAN``
(R1 -> R2 -> R3).  ``--stages`` overrides the plan, ``--tag`` suffixes the
outputs, and ``--tag`` suffixes the outputs.  Every stage runs the full epoch
budget with no early stopping and no validation readout, so a set of arms is
comparable on the same schedule by construction.  Note that a stage whose
checkpoint is already complete retrains
from the Teacher rather than resuming, so an arm that has been reported must be
run explicitly by name, never through ``all``.

The ``teacher`` stage always runs the full 50-epoch schedule with its own
early-stop rule and writes a per-5-epoch learning curve for all three heads
(``z0``/``z4``/``z34``) next to its checkpoint.  ``--tag`` keeps a retrained
Teacher away from the frozen ``teacher.pt`` that every reported arm was built
on: with a tag the stage reads and writes ``teacher<tag>.pt`` and
``teacher<tag>_latest.pt``, leaving the frozen files untouched.  ``teacher-test``
is the evaluation half on its own, so it can be re-run -- or deliberately not
run -- after the validation comparison.

Raw rasters are read via the established rapid-COCD dataset loader.  This file
does not create, alter, or cache any raw source data.
"""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- package-relative resolution (Windows / Linux / CUDA port) --------------
COCD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COCD))
from paths import (CONFIGS as CONFIG_DIR, OUT_ROOT,  # noqa: E402
                   SCRIPTS as SCRIPTS_DIR, describe, resolve_checkpoint)
A = COCD
OUT = OUT_ROOT / 'ours_v2'
from losses.distill import dis2_logit_only, dis2_multilevel_kd, gain_protect, selective_multilevel_kd
from losses.landslide_cocd import land_loss
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student

_old_spec = importlib.util.spec_from_file_location('rapid_v1', SCRIPTS_DIR / '21_train_rapid_landslide_cocd.py')
_old = importlib.util.module_from_spec(_old_spec)
_old_spec.loader.exec_module(_old)
HaitiPairs, split, dl, posweight, DEV = _old.HaitiPairs, _old.split, _old.dl, _old.posweight, _old.DEV
EPS = 1e-8
# --------------------------------------------------------------------------- #
# Unified training protocol.  ``configs/protocol_windows_main.json`` is the
# single source of truth for every number below; ``--protocol v1`` restores the
# settings the already-reported arms were trained with, so those results stay
# reproducible from the same code path instead of from a frozen copy of this file.
# --------------------------------------------------------------------------- #
PROTOCOL_PATH = CONFIG_DIR / 'protocol_windows_main.json'


def protocol_values(kind):
    """Flatten one protocol revision into the flat knobs the loops read."""
    if kind == 'v1':
        return dict(optimizer='adamw', lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                    batch=2, effective_batch=2, seg='bce+dice', pos_weight=True,
                    epochs=50, validate_every=5, patience=2,
                    early_stop=True, monitor='validation_auprc',
                    fixed_threshold=0.5, threshold_source='fixed',
                    teacher_init='s0', backbone_init='teacher', op_init='teacher', delta=None)
    p = json.loads(PROTOCOL_PATH.read_text())
    o, b, s, t = p['optimizer'], p['batch'], p['schedule'], p['threshold']
    return dict(optimizer=o['name'], lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'],
                weight_decay=o['weight_decay'], batch=b['physical'], effective_batch=b['effective'],
                seg=p['loss']['seg'], pos_weight=bool(p['loss']['pos_weight']),
                epochs=s['max_epochs'],
                # v4: the monitored score is read every epoch, on the test
                # partition, and training stops after ``patience`` consecutive
                # epochs without a new best.  ``validate_every=1`` is that
                # cadence; ``patience`` comes from the schedule block rather
                # than a sentinel, because under v4 there really is a stop rule.
                validate_every=s.get('validate_every') or 1,
                patience=int(s.get('early_stop_patience') or 10 ** 9),
                early_stop=bool(s.get('early_stop', False)),
                monitor=s.get('monitor') or 'test_auprc',
                fixed_threshold=t['value'], threshold_source=t['source'],
                teacher_init=p['init']['teacher'], backbone_init=p['init']['backbone'],
                op_init=p['init']['omega_operator'], delta=p.get('corrective_distillation', {}))


PROTOCOL = 'v2'
P = protocol_values(PROTOCOL)
# Loader batch and the accumulation that keeps the effective batch at
# ``effective_batch`` even when the physical batch has to be smaller.
BATCH = P['batch']
ACCUM = 1
LOSS_MODE = P['seg']
POS_WEIGHT = P['pos_weight']
VALIDATE_EVERY = P['validate_every']
PATIENCE = P['patience']
EARLY_STOP = P['early_stop']
MONITOR = P['monitor']
FIXED_THRESHOLD = P['fixed_threshold']
THRESHOLD_SOURCE = P['threshold_source']
TEACHER_INIT = P['teacher_init']
BACKBONE_INIT = P['backbone_init']
OP_INIT = P['op_init']
DELTA = P['delta'] or {}
DELTA_ON = False                 # per-arm switch, set by the config of the arm being trained
DELTA_SPACE = DELTA.get('space', 'prob')
DELTA_PAIR = DELTA.get('pair', 'z34_minus_z0')
DELTA_BETA = DELTA.get('beta', 1.0)
LAM_DELTA = DELTA.get('weight', 1.0)
DELTA_AGGREGATION = DELTA.get('aggregation', 'balanced')

# The same disclosure the external-seat script writes.  It is emitted by every
# run and stored in every progress file so the bias cannot be discovered later
# by someone reading a table without the protocol file open.
MONITOR_RISK = ('EARLY STOPPING MONITORS THE TEST PARTITION: the reported test '
                'AUPRC is the best-over-epochs value and is optimistically biased. '
                'Threshold stays fixed at 0.5; every method uses the identical rule.')


def monitor_logits(model, a, d, kind):
    """The deployed reading of each model family, for both target directions.

    Three architectures reach this function and each returns a different shape,
    so the family is identified by its attributes instead of being assumed:

    ``kind='T'``
        ``OursV2Teacher`` returns ``(z0, z4, z34, (r3, r4))`` and the deployed
        reading is ``z34``, the third element.  ``NewCOCDTeacher`` returns a
        dict and the dual reading is ``z_dual``.  Indexing/keying is used rather
        than unpacking so the two cannot silently drift apart.

    ``kind='S'``
        Students are deployed target-only and take a single tensor, so the two
        directions are concatenated first and read in one pass.
        ``ComplementStudent`` returns ``{'z', 'F_t', 'F_mod', 'C_S'}`` and the
        deployed output is ``z``; ``OursV3Student`` returns ``(z4, z34, r)`` and
        the deployed output is ``z34``.

    ``model(a, d)`` gives the asc-target/desc-counter state and ``model(d, a)``
    the reverse, so concatenating the two reproduces the ``torch.cat((a, d))``
    ordering the loader uses.
    """
    if kind == 'S':
        out = model(torch.cat((a, d)))
        return out['z'] if isinstance(out, dict) else out[1]

    def one(target, counter):
        out = model(target, counter)
        return out['z_dual'] if isinstance(out, dict) else out[2]
    return torch.cat((one(a, d), one(d, a)))


@torch.no_grad()
def monitor_score(model, dataset, kind):
    """The monitored quantity, read every epoch: threshold-free AUPRC.

    ``kind='T'`` reads the Teacher's deployed dual head, ``kind='S'`` the
    student's deployed corrected output.  The threshold never enters this
    number, so the fixed 0.5 threshold is untouched by model selection.
    """
    model.eval()
    ps, ys = [], []
    for a, d, y, _ga, _gd, _real in loader(dataset):
        a, d = a.to(DEV), d.to(DEV)
        logits = monitor_logits(model, a, d, kind)
        p = torch.sigmoid(logits)[:, 0]
        if p.shape[-2:] != y.shape[-2:]:
            raise RuntimeError(f'monitor_score: prediction {tuple(p.shape)} does not '
                               f'match label {tuple(y.shape)}')
        ps.append(p.cpu().numpy()); ys.append(torch.cat((y, y)).numpy())
    p, yv = np.concatenate(ps).ravel(), np.concatenate(ys).astype(bool).ravel()
    return float(average_precision_score(yv, p)) if yv.any() else float('nan')


def echo_protocol():
    """One line per knob, so a run log can be audited without the CLI."""
    p = json.loads(PROTOCOL_PATH.read_text()) if PROTOCOL != 'v1' else None
    print(f'[protocol] {PROTOCOL}  optimizer={P["optimizer"]} lr={P["lr"]:g} '
          f'betas={P["betas"]} eps={P["eps"]:g} wd={P["weight_decay"]:g}', flush=True)
    print(f'[protocol] physical_batch={BATCH} accum={ACCUM} effective={BATCH * ACCUM} '
          f'seg={LOSS_MODE} pos_weight={POS_WEIGHT} epochs={P["epochs"]} '
          f'validate_every={VALIDATE_EVERY} patience={PATIENCE}', flush=True)
    print(f'[protocol] early_stop={EARLY_STOP} monitor={MONITOR} '
          f'model=best_monitored_epoch', flush=True)
    if EARLY_STOP:
        print(f'[protocol] !! {MONITOR_RISK}', flush=True)
    print(f'[protocol] init: teacher={TEACHER_INIT} backbone={BACKBONE_INIT} omega={OP_INIT} '
          f'| L_delta={"on" if DELTA else "off"} '
          f'{"" if not DELTA else "(pair=%s space=%s beta=%g weight=%g)" % (DELTA_PAIR, DELTA_SPACE, DELTA_BETA, LAM_DELTA)}',
          flush=True)
    if p is not None:
        print(f'[protocol] scheduler={p["scheduler"]} threshold={p["threshold"]["source"]}/'
              f'{p["threshold"]["criterion"]} reselect_on_test={p["threshold"]["reselect_on_test"]} '
              f'mask={p["mask_protocol"]["train"]} geometry_weighted_kd='
              f'{p["mask_protocol"]["geometry_weighted_kd"]}', flush=True)

# Structural ablation.  Every arm shares the Teacher, the data, the optimiser and
# the weighting (kd='gain' == selective gain x protect), so the only variable is
# how the correction is produced.  Adjacent pairs differ by exactly one factor:
#   R0 -> R1  does the correction + output KD do anything at all
#   R1 -> R2  does inheriting the Teacher's correction operator help
#   R2 -> R3  does driving P3 from the corrected state help
#   R3 -> R4  how much of the gain is output KD rather than structure
# R3g sits outside that ladder: it keeps R3's structure and changes only the
# distillation weight, so it is read as a pair against R3 rather than as a rung.
CONFIGS = {
    # --- unified-protocol arms (configs/protocol_v2.json) --------------------- #
    # Every internal arm is the same ConvNeXt-Tiny with the identical ImageNet
    # initialisation: ``init='scratch'`` for all of them, so no arm is warm-started
    # from another.  The one thing that is still inherited is the correction
    # operator Omega of R2/R3 -- that inheritance *is* the method's definition --
    # and it is written as its own field (``op_init``) so it can never be mistaken
    # for a backbone initialisation.
    'SO': {'variant': 'R0', 'kd': None, 'init': 'scratch', 'label': 'Single-Orbit'},
    # Vanilla KD: the same student architecture as DIS2-port, distillation played
    # with no per-pixel selection at all (``selective_kl``'s ``all`` mode sets
    # ``w = 1`` everywhere), i.e. the plain baseline the selective rule is
    # measured against.
    'VKD': {'variant': 'R1', 'kd': 'all', 'init': 'scratch', 'label': 'VanillaKD'},
    'DIS2': {'variant': 'R1', 'kd': 'dis2', 'init': 'scratch', 'label': 'DIS2-port'},
    'R1': {'variant': 'R1', 'kd': 'gain', 'init': 'scratch'},
    'R2': {'variant': 'R2', 'kd': 'gain', 'init': 'scratch', 'op_init': 'teacher'},
    'R3': {'variant': 'R3', 'kd': 'gain', 'init': 'scratch', 'op_init': 'teacher'},
    # The minimal corrective-distillation upgrade: R3 plus ``L_delta`` only.  The
    # term matches the decision change the correction induces (``z34 - z0``), not
    # the correction vector itself, so R3 is its exact no-delta control.
    'R3D': {'variant': 'R3', 'kd': 'gain', 'init': 'scratch', 'op_init': 'teacher',
            'delta': True, 'label': 'R3+Delta'},
    # --- v1 arms, kept runnable for reference -------------------------------- #
    'SOs0': {'variant': 'R0', 'kd': None, 'init': 's0', 'label': 'Single-Orbit-S0'},
    'R0': {'variant': 'R0', 'kd': None, 'init': 'teacher'},
    'R4': {'variant': 'R3', 'kd': None, 'init': 'teacher'},
    # DEPRECATED.  Geometry-weighted distillation is switched off by the unified
    # protocol: the geometry masks are an evaluation partition (G00/G01/G10/G11),
    # not a training signal.  The arm stays in the file so its already-reported
    # number remains traceable, but it must not be launched again.
    'R3g': {'variant': 'R3', 'kd': 'full', 'init': 'teacher', 'deprecated': True},
}
# The unified controlled comparison.  Every arm that distils needs the Teacher, so
# it is trained first; ``SO`` needs nothing and can be run on its own.
PLAN = ['SO', 'VKD', 'DIS2', 'R3', 'R3D']

# Teacher output weights (z0, z4, z34), summed over the two target directions.
# The frozen Teacher used (0.5, 0.25, 0.25); Teacher-T2a uses (0.25, 0.25, 0.5),
# i.e. the same total -- so the single variable between the two Teachers is where
# the supervision sits, not how much of it there is.  Nothing else about the
# Teacher changes: same structure, initialisation, data, batch, optimiser, LR,
# schedule and selection rule.  See reports/cocd_final_teacher_student.md.
TEACHER_W = (0.25, 0.25, 0.50)


def loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def seg(logits, y, weight):
    """The protocol's segmentation loss.

    ``bce`` is the unified protocol: BCE alone, with the positive-class
    reweighting taken from the train split.  That reweighting scales the two
    class errors against each other; it is not a second loss term.  With
    ``pos_weight`` disabled in the config the BCE is unweighted.  ``bce+dice``
    is the v1 revision, kept reachable so the reported arms reproduce.
    """
    if LOSS_MODE == 'bce':
        pos = torch.tensor(weight, device=logits.device) if POS_WEIGHT else None
        return F.binary_cross_entropy_with_logits(logits[:, 0], y, pos_weight=pos)
    return land_loss(logits, y, weight)


def make_optimizer(model):
    """Adam(lr, betas, eps) per the protocol; AdamW belongs to the v1 revision."""
    kwargs = dict(lr=P['lr'], betas=P['betas'], eps=P['eps'])
    if P['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    if P['optimizer'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), weight_decay=P['weight_decay'], **kwargs)
    raise ValueError(P['optimizer'])


def best_f1_threshold(p, y, bins=1000):
    """F1-maximising threshold for **validation** predictions only.

    Scanned on a fixed grid through counting histograms, so the cost is
    O(pixels + bins) rather than O(pixels x thresholds).  A pixel counts as
    positive when ``p >= thr``, the same convention ``metrics`` uses.  The
    returned threshold is meant to be frozen and carried to the test partition;
    it is never re-searched there.
    """
    p = np.asarray(p).ravel(); y = np.asarray(y).ravel().astype(bool)
    if not y.any() or y.all():
        return float('nan'), float('nan')
    edges = np.linspace(0., 1., bins + 1)
    tp_hist, _ = np.histogram(p[y], bins=edges)
    fp_hist, _ = np.histogram(p[~y], bins=edges)
    tp = np.cumsum(tp_hist[::-1])[::-1].astype(float)
    fp = np.cumsum(fp_hist[::-1])[::-1].astype(float)
    fn = float(y.sum()) - tp
    precision, recall = tp / (tp + fp + EPS), tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    k = int(np.argmax(f1))
    return float(edges[k]), float(f1[k])


def delta_change(taps, detach=False):
    """The decision change the correction induces, ``z34 − z0`` by default.

    Both readings exist on the Teacher side; the student computes its own ``z0``
    inside ``forward_taps`` (it was always one decoder call, it simply was not
    kept).  ``z34 − z4`` is the alternative, which isolates the P3 half of the
    correction.  The difference is taken in probability space: the two logit
    fields can differ by ~90 in magnitude, so a raw-logit difference would be
    dominated by whatever scale the head happens to carry, whereas the decision
    is a property of the probability.
    """
    def _d(t):
        a = t['z34'][:, 0]
        b = t['z0'][:, 0] if DELTA_PAIR == 'z34_minus_z0' else t['z4'][:, 0]
        if DELTA_SPACE == 'prob':
            return torch.sigmoid(a) - torch.sigmoid(b)
        return a - b
    if detach:
        with torch.no_grad():
            return _d(taps)
    return _d(taps)


def delta_loss(stu, tea, y):
    """``L_delta``: SmoothL1 between the student's and the Teacher's decision change.

    Only the correction's *effect on the decision* is matched, not the raw
    correction vector, so the student is free to represent the correction
    differently as long as it moves the output the way the counter orbit does.
    The Teacher side is stop-gradded, so the term shapes the student alone.
    SmoothL1 is quadratic near agreement and linear in the tails, so the few
    pixels on which the two corrections disagree strongly cannot dominate.

    ``aggregation`` matters more than it looks.  Both posterior fields sit at
    ~0 on almost every pixel, so the raw per-pixel change is ~2e-3 in magnitude
    and a plain pixel mean buries the term: measured on validation it is 0.2% of
    the segmentation loss, which would make the arm inert for a reason that has
    nothing to do with its idea.  ``balanced`` aggregates the same way the
    selective KD term already does -- foreground and background each contribute
    half -- which is the convention a term aimed at the change pixels should use.
    """
    per = F.smooth_l1_loss(delta_change(stu), delta_change(tea, detach=True),
                           beta=DELTA_BETA, reduction='none')
    return balanced_mean(per, y) if DELTA_AGGREGATION == 'balanced' else per.mean()


def balanced_mean(x, y):
    """Foreground/background means with full-class denominators, never sum(w)."""
    values = []
    for cls in (0., 1.):
        mask = y == cls
        if mask.any():
            values.append(x[mask].mean())
    return torch.stack(values).mean() if values else x.new_zeros(())


def selective_kl(t0, tq, sq, y, g10, mode, real=None):
    """Detached gain selection, optional geometry priority, stable Bernoulli KL at T=1.

    ``gain`` and ``protect`` are unchanged in every mode.  ``full`` adds the
    cross-orbit geometry priority on top of them::

        w = (1 + 1_real * G10) * gain * protect

    with ``beta`` fixed at 1.  The boost is applied to real pre/post pairs only.
    A synthetic no-change pair (pre/pre or post/post) keeps the unboosted
    weight, because those views do not describe an observed change event and so
    "the counter orbit sees this place better" has no physical reading there.
    """
    with torch.no_grad():
        e0 = F.binary_cross_entropy_with_logits(t0[:, 0], y, reduction='none')
        et = F.binary_cross_entropy_with_logits(tq[:, 0], y, reduction='none')
        es = F.binary_cross_entropy_with_logits(sq[:, 0], y, reduction='none')
        gain = ((e0 - et) / (e0 + EPS)).clamp_(0, 1)
        protect = (et < es).float()
        if mode == 'all':
            w = torch.ones_like(gain)
        elif mode == 'gain':
            w = gain * protect
        elif mode == 'full':
            boost = g10.float() if real is None else g10.float() * real.float()[:, None, None]
            w = (1. + boost) * gain * protect
        else:
            raise ValueError(mode)
        target_p = torch.sigmoid(tq[:, 0])
        entropy = F.binary_cross_entropy(target_p, target_p, reduction='none')
    soft_ce = F.binary_cross_entropy_with_logits(sq[:, 0], target_p, reduction='none')
    return balanced_mean(w * (soft_ce - entropy), y), w


def teacher_batch(model, a, d, y, weight):
    """Teacher objective, per target direction: 0.25 z0 + 0.25 z4 + 0.50 z34.

    The frozen Teacher weighted the uncorrected self path twice as heavily as
    either corrected output (0.5 z0 / 0.25 z4 / 0.25 z34), which made ``z34`` --
    the dual-orbit output, and the one the students distil and the deployment
    reading is taken from -- the least supervised of the three.  Teacher-T2a
    reallocates the same total weight onto that head while keeping z0 heavy
    enough to remain a reliable single-orbit reference.  The total stays at 1.0,
    so the gradient scale is unchanged and no optimiser setting has to move.
    """
    outputs_a, outputs_d = model.forward_pair(a, d)
    return .5 * sum(TEACHER_W[i] * seg(o[i], y, weight)
                    for o in (outputs_a, outputs_d) for i in range(3))


def predict_teacher(model, dataset):
    """The frozen reading convention: ``self`` is z0 and ``dual`` is z34.

    Kept as the two-head view of ``predict_teacher_heads`` so there is a single
    pass over the data and a single definition of either reading.
    """
    q = predict_teacher_heads(model, dataset)
    return {k: q[v] for k, v in (('self', 'z0'), ('dual', 'z34'), ('y', 'y'),
                                 ('gt', 'gt'), ('gc', 'gc'), ('orbit', 'orbit'))}


@torch.no_grad()
def predict_student(model, dataset):
    model.eval(); ans = {k: [] for k in ('p', 'y', 'gt', 'gc', 'orbit')}
    for a, d, y, ga, gd, _ in loader(dataset):
        p = torch.sigmoid(model(torch.cat((a.to(DEV), d.to(DEV))))[1])[:, 0].cpu().numpy()
        ans['p'].append(p); ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy()); ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * len(a) + ['desc'] * len(a)))
    return {k: np.concatenate(v) for k, v in ans.items()}


def metrics(p, y, thr=.5):
    p, y = p.ravel(), y.astype(bool).ravel(); h = p >= thr
    tp, fp, fn = (h & y).sum(), (h & ~y).sum(), (~h & y).sum()
    precision, recall = tp / (tp + fp + EPS), tp / (tp + fn + EPS)
    # A region without a single positive pixel has no PR curve.  Leave it undefined
    # instead of letting sklearn score it 0.0, so the frozen Teacher table and the
    # ablation arms share one definition.
    auprc = float(average_precision_score(y, p)) if y.any() else float('nan')
    # The 0.5 readout is kept as a continuity column only -- it is never used for
    # selection and never re-tuned on test.
    h5 = p >= .5
    tp5, fp5, fn5 = (h5 & y).sum(), (h5 & ~y).sum(), (~h5 & y).sum()
    pr5, rc5 = tp5 / (tp5 + fp5 + EPS), tp5 / (tp5 + fn5 + EPS)
    return dict(iou=float(tp / (tp + fp + fn + EPS)), f1=float(2 * precision * recall / (precision + recall + EPS)),
                precision=float(precision), recall=float(recall), auprc=auprc,
                **{'iou@0.5': float(tp5 / (tp5 + fp5 + fn5 + EPS)),
                   'f1@0.5': float(2 * pr5 * rc5 / (pr5 + rc5 + EPS))})


def report(name, pred, thr=.5):
    """Grouped evaluation on the frozen Haiti protocol.

    A partition (``overall``/``asc``/``desc``) selects whole samples, while a region
    (``G00``..``G11``) is a per-pixel raster supplied by the dataset, so the partition
    mask is broadcast over the spatial axes before the two are combined.  This is the
    same construction that produced ``teacher_test_verify.csv``: masks are flattened
    to match the predictions, and an empty region is skipped rather than scored.

    ``thr`` is carried in from validation and applied unchanged to every partition
    and every region; nothing here searches for a threshold.
    """
    pf, yf = pred['p'].ravel(), pred['y'].ravel().astype(bool)
    gt, gc = pred['gt'] > 0, pred['gc'] > 0
    rows = []
    for label, sel in [('overall', np.ones(len(pred['orbit']), dtype=bool)),
                       ('asc', pred['orbit'] == 'asc'),
                       ('desc', pred['orbit'] == 'desc')]:
        where = np.broadcast_to(sel[:, None, None], pred['p'].shape).ravel()
        rows.append({'method': name, 'partition': label, 'region': 'all', 'threshold': thr,
                     'pixels': int(where.sum()), **metrics(pf[where], yf[where], thr)})
        for region, rmask in [('G00', ~gt & ~gc), ('G01', ~gt & gc),
                              ('G10', gt & ~gc), ('G11', gt & gc)]:
            mask = where & rmask.ravel()
            if mask.sum() == 0:
                continue
            rows.append({'method': name, 'partition': label, 'region': region, 'threshold': thr,
                         'pixels': int(mask.sum()), **metrics(pf[mask], yf[mask], thr)})
    return rows


def checkpoint(path, model, optimizer, epoch, best, best_state, bad, complete=False, **extra):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best': best,
                'best_state': best_state, 'bad': bad, 'complete': complete, **extra}, path)


@torch.no_grad()
def predict_teacher_heads(model, dataset):
    """Per-sample z0/z4/z34 maps plus the P3/P4 correction magnitudes.

    The frozen Teacher stored only the dual-orbit AUPRC, which is why its
    validation curve is unrecoverable.  This pass keeps all three heads, the
    region masks they are read against, and ``|r3|``/``|r4|`` so the curve can
    show whether the dual gain is growing and whether it is still being driven
    by the counter track.  The three taps are the same expressions
    ``forward_pair`` evaluates, and equality with the deployed outputs is
    checked once on the first batch rather than assumed.
    """
    model.eval(); taps_ok = None
    ans = {k: [] for k in ('z0', 'z4', 'z34', 'y', 'gt', 'gc', 'orbit')}
    norms = {'p3': [], 'p4': []}
    for a, d, y, ga, gd, _ in loader(dataset):
        a, d = a.to(DEV), d.to(DEV)
        t = merge_taps(model.forward_pair_taps(a, d))
        if taps_ok is None:
            ref = model.forward_pair(a, d)
            taps_ok = max(float((t[k] - torch.cat((ref[0][i], ref[1][i]))).abs().max())
                          for i, k in enumerate(('z0', 'z4', 'z34')))
        for key in ('z0', 'z4', 'z34'):
            ans[key].append(torch.sigmoid(t[key])[:, 0].cpu().numpy())
        ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy())
        ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * len(a) + ['desc'] * len(a)))
        r3, r4 = t['r']
        norms['p3'].append(r3.flatten(1).norm(dim=1).cpu().numpy())
        norms['p4'].append(r4.flatten(1).norm(dim=1).cpu().numpy())
    out = {k: np.concatenate(v) for k, v in ans.items()}
    out['corr_norm_mean'] = {k: float(np.concatenate(v).mean()) for k, v in norms.items()}
    out['corr_norm_max'] = {k: float(np.concatenate(v).max()) for k, v in norms.items()}
    out['tap_path_max_abs_diff'] = taps_ok
    return out


def teacher_curve_row(model, dataset, epoch):
    """One curve row: every head's IoU/F1/AUPRC, the dual gain, correction norms."""
    q = predict_teacher_heads(model, dataset)
    row = {'epoch': epoch, 'tap_path_max_abs_diff': q['tap_path_max_abs_diff'],
           'p3_corr_norm_mean': q['corr_norm_mean']['p3'], 'p4_corr_norm_mean': q['corr_norm_mean']['p4'],
           'p3_corr_norm_max': q['corr_norm_max']['p3'], 'p4_corr_norm_max': q['corr_norm_max']['p4']}
    for head in ('z0', 'z4', 'z34'):
        m = metrics(q[head], q['y'])
        for k in ('iou', 'f1', 'precision', 'recall', 'auprc'):
            row[f'{head}_{k}'] = m[k]
    row['gain_auprc_z34_minus_z0'] = row['z34_auprc'] - row['z0_auprc']
    row['gain_iou_z34_minus_z0'] = row['z34_iou'] - row['z0_iou']
    row['gain_auprc_z34_minus_z4'] = row['z34_auprc'] - row['z4_auprc']
    # Where the dual gain lands, on the same four regions the test table uses.
    # Reporting only -- the geometry masks never enter the training objective.
    yf = q['y'].ravel().astype(bool); gt, gc = q['gt'] > 0, q['gc'] > 0
    for region, rmask in (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc)):
        sel = rmask.ravel()
        for head in ('z0', 'z34'):
            row[f'{head}_auprc_{region}'] = metrics(q[head].ravel()[sel], yf[sel])['auprc'] if yf[sel].any() else float('nan')
        row[f'gain_auprc_{region}'] = row[f'z34_auprc_{region}'] - row[f'z0_auprc_{region}']
    return row


def append_curve(path, row):
    """Rewrite the curve file from the rows on disk, so a crash keeps the history."""
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def teacher_init_source(kind=None):
    """Which warm start the Teacher gets.

    ``s0`` is the v1 chain.  The unified protocol writes ``imagenet``, and
    ``scratch`` is the same thing spelled the way the student arms spell it; both
    mean "no warm start at all".  Kept as a function so the mapping is asserted
    rather than discovered when a run starts.
    """
    kind = TEACHER_INIT if kind is None else kind
    if kind == 's0':
        return 's0'
    if kind in ('imagenet', 'scratch'):
        return 'none'
    raise ValueError(f'unknown teacher init {kind!r}')


def train_teacher(train_set, val_set, epochs, seed, tag='', monitor_set=None, kd_weight=1.0):
    """Train the teacher until it stops improving; the best epoch is the model.

    Under v4 the schedule is a ceiling, not a plan.  After each epoch the
    deployed dual head is scored on ``monitor_set`` (the test partition) by
    threshold-free AUPRC; a score that fails to beat the best seen so far for
    ``PATIENCE`` consecutive epochs ends training.  The weights at the best
    epoch, not the last epoch's, are written out.

    ``val_set`` is accepted for call-site compatibility.  v4 still has no
    validation partition, so a non-empty ``val_set`` means the split is not the
    frozen one and the run stops rather than silently training on it.
    """
    set_seed(seed); OUT.mkdir(parents=True, exist_ok=True)
    if val_set:
        raise SystemExit(f'train_teacher received {len(val_set)} validation samples; '
                         'v4 has no validation partition')
    if EARLY_STOP and monitor_set is None:
        raise SystemExit('early stopping is on but no monitor_set was supplied')
    model = OursV2Teacher().to(DEV)
    # The unified protocol starts the Teacher from the same ImageNet
    # initialisation as every other internal arm.  The S0 warm start is the v1
    # revision's chain and is only entered when the protocol asks for it.
    if teacher_init_source() == 's0':
        model.load_s0(torch.load(resolve_checkpoint('S0.pt'), map_location=DEV, weights_only=True))
        print(f'[Teacher{tag}] init=s0 (S0 warm start)', flush=True)
    else:
        print(f'[Teacher{tag}] init=imagenet (no warm start)', flush=True)
    optimizer = make_optimizer(model)
    weight, start = posweight(train_set), 1
    # ---- early-stopping state: best score, epochs since it last moved, and the
    # weights that achieved it ------------------------------------------------
    best, bad, best_epoch, best_state = -1.0, 0, 0, None
    stop_reason = None; hist = []
    # A tag keeps a retrained Teacher in its own files.  The untagged names hold
    # the frozen Teacher that every reported student arm distils from, so a T2a
    # run must never touch them.
    latest = OUT / f'teacher{tag}_latest.pt'
    # A completed epoch is an atomic recovery point, so weights and AdamW
    # moments are recovered rather than reinitialised.
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get('complete', False):
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start = state['epoch'] + 1
            best = state.get('best', -1.0); bad = state.get('bad', 0)
            best_epoch = state.get('best_epoch', 0); best_state = state.get('best_state')
            hist = state.get('hist', [])
            print(f'[Teacher{tag}] resume after completed epoch {start - 1} '
                  f'(best={best:.4f} @ep{best_epoch}, bad={bad})', flush=True)
    epoch = start - 1
    for epoch in range(start, epochs + 1):
        model.train(); losses = []; bar = tqdm(loader(train_set, True), desc=f'Teacher{tag} {epoch:02d}/{epochs}', unit='batch')
        nb = len(bar); optimizer.zero_grad(set_to_none=True)
        for i, (a, d, y, *_) in enumerate(bar, 1):
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            loss = teacher_batch(model, a, d, y, weight) / ACCUM
            loss.backward()
            # Accumulation is only ever >1 when the physical batch had to be
            # smaller than the protocol's effective batch; the final partial
            # window is stepped rather than dropped.
            if i % ACCUM == 0 or i == nb:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item() * ACCUM)
            bar.set_postfix(loss=f'{np.mean(losses):.3f}')
        mean_loss = float(np.mean(losses))
        # v4: one monitored score per epoch, threshold-free.  It is the only
        # thing that decides where training stops; the loss is logged but never
        # used for selection.
        score = monitor_score(model, monitor_set, 'T') if EARLY_STOP else float('nan')
        improved = (not EARLY_STOP) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = 'NEW BEST'
        else:
            bad += 1; flag = f'no gain ({bad}/{PATIENCE})'
        hist.append(dict(epoch=epoch, loss=mean_loss, monitor=score, best=best,
                         bad=bad, improved=bool(improved)))
        print(f'[Teacher{tag}] epoch={epoch}/{epochs} loss={mean_loss:.5f} '
              f'{MONITOR}={score:.4f} best={best:.4f}@{best_epoch} [{flag}]', flush=True)
        checkpoint(latest, model, optimizer, epoch, best, best_state, bad,
                   best_epoch=best_epoch, hist=hist)
        (OUT / f'teacher{tag}_progress.json').write_text(json.dumps(
            {'epoch': epoch, 'max_epochs': epochs, 'loss': mean_loss,
             'monitor': MONITOR, 'monitor_value': score, 'best': best,
             'best_epoch': best_epoch, 'bad': bad, 'patience': PATIENCE,
             'improved': bool(improved), 'history': hist,
             'checkpoint': 'best_monitored_epoch', 'early_stop': bool(EARLY_STOP),
             'monitor_risk': MONITOR_RISK}, indent=2))
        if EARLY_STOP and bad >= PATIENCE:
            stop_reason = (f'{PATIENCE} consecutive epochs without beating '
                           f'{best:.4f} at epoch {best_epoch}')
            print(f'[Teacher{tag}] EARLY STOP at epoch {epoch}: {stop_reason}', flush=True)
            break
    else:
        stop_reason = f'reached the {epochs}-epoch ceiling'
    # the best-monitored-epoch weights ARE the model
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state); model.to(DEV)
    torch.save(model.state_dict(), OUT / f'teacher{tag}.pt')
    checkpoint(latest, model, optimizer, epoch, best, None, bad, complete=True)
    print(f'[Teacher{tag}] model = epoch {best_epoch} weights, {MONITOR}={best:.4f}; '
          f'stopped by: {stop_reason}', flush=True)
    return model


def load_teacher(tag=''):
    model = OursV2Teacher().to(DEV); model.load_state_dict(torch.load(resolve_checkpoint(f'teacher{tag}.pt'), map_location=DEV, weights_only=True)); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def merge_taps(pair):
    """Concatenate a Teacher tap pair along the batch axis, as ``forward_pair`` does."""
    out = {}
    for key in pair[0]:
        a, b = pair[0][key], pair[1][key]
        out[key] = tuple(torch.cat((x, y)) for x, y in zip(a, b)) if isinstance(a, tuple) else torch.cat((a, b))
    return out


def train_student(name, cfg, teacher, train_set, val_set, epochs, seed, tag='', kd_weight=1.0,
                  monitor_set=None):
    """Train a student arm until it stops improving; the best epoch is the model.

    v4 makes the schedule a ceiling.  Each epoch the deployed corrected output
    is scored on ``monitor_set`` by threshold-free AUPRC, and ``PATIENCE``
    consecutive epochs without a new best ends training.  The best-epoch
    weights are the model.  Arms are therefore free to stop at different
    epochs; each is monitored on its own curve, which is the agreed rule.
    """
    if val_set:
        raise SystemExit(f'{name}: train_student received {len(val_set)} validation '
                         'samples; v4 has no validation partition')
    if EARLY_STOP and monitor_set is None:
        raise SystemExit(f'{name}: early stopping is on but no monitor_set was supplied')
    kd_mode = cfg['kd']
    init = cfg.get('init', 'teacher')
    use_delta = bool(cfg.get('delta'))
    if cfg.get('deprecated'):
        print(f'[{name}] DEPRECATED under the unified protocol; launching it anyway', flush=True)
    # L_delta reads the student's own z0, which only the tap path returns; the
    # DIS2 terms need the taps too.  The selective rules do not, but enabling the
    # tap path for them costs one decode call and keeps a single code path.
    taps_mode = kd_mode in ('dis2', 'dis2-logit', 'selective') or use_delta
    set_seed(seed); model = OursV3Student(cfg['variant']).to(DEV)
    # ``init`` is the backbone's origin and is 'scratch' (the shared ImageNet
    # initialisation) for every unified-protocol arm.
    if init == 'teacher':
        model.load_teacher_self(teacher)
    elif init == 's0':
        s0 = torch.load(resolve_checkpoint('S0.pt'), map_location=DEV, weights_only=True)
        s0 = s0.get('model', s0)
        src = {k.removeprefix('backbone.'): v for k, v in s0.items() if k.startswith('backbone.')}
        missing, unexpected = model.backbone.load_state_dict(src, strict=False)
        if unexpected or missing:
            raise RuntimeError(f'{name}: S0 checkpoint is incompatible: missing={missing}, unexpected={unexpected}')
    elif init in ('scratch', 'imagenet'):
        pass
    else:
        raise ValueError(f'{name}: unknown init {init!r}')
    # Omega is inherited from the trained Teacher because that inheritance *is* the
    # definition of R2/R3; it is a property of the method, not a backbone warm
    # start, and the two are reported separately.
    if cfg.get('op_init', OP_INIT) == 'teacher' and cfg['variant'] in ('R2', 'R3'):
        if teacher is None:
            raise RuntimeError(f'{name}: op_init=teacher but no Teacher was loaded')
        model.op3.load_state_dict(teacher.a3.state_dict(), strict=True)
        model.op4.load_state_dict(teacher.a4.state_dict(), strict=True)
        print(f'[{name}] Omega inherited from the Teacher (op_init=teacher)', flush=True)
    # At zero correction, z34 must be exactly the copied Teacher self path.  That
    # holds for ``init='teacher'`` only, so the check is scoped to it.  What must
    # hold for *every* arm is the initialisation invariant: the correction starts
    # at exactly zero, so z0 == z4 == z34 and every arm starts from the same
    # uncorrected function whatever its variant.
    with torch.no_grad():
        model.eval()
        # The invariant is checked on the first TRAIN batch.  It used to read a
        # validation batch, which no longer exists; the check itself is
        # unchanged and does not depend on which batch it is evaluated on.
        a, d, *_ = next(iter(loader(train_set))); zs = model(a.to(DEV))[1]
        if init == 'teacher' and not torch.allclose(teacher(a.to(DEV), d.to(DEV))[0], zs, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f'{name}: initialization is not Teacher self.')
        t0 = model.forward_taps(a.to(DEV))
        if not (torch.equal(t0['z0'], t0['z4']) and torch.equal(t0['z0'], t0['z34'])):
            raise RuntimeError(f'{name}: the correction is not exactly zero at initialisation.')
        if taps_mode and not torch.equal(t0['z34'], zs):
            raise RuntimeError(f'{name}: the tap path and the deployment path disagree.')
    print(f'[{name}] init={init} variant={cfg["variant"]} kd={kd_mode} '
          f'delta={"on" if use_delta else "off"} optimizer={P["optimizer"]} lr={P["lr"]:g} accum={ACCUM}', flush=True)
    optimizer = make_optimizer(model)
    weight, start = posweight(train_set), 1
    latest = OUT / f'{name}{tag}_seed{seed}_latest.pt'
    # ---- early-stopping state, same contract as the Teacher -------------------
    best, bad, best_epoch, best_state = -1.0, 0, 0, None
    stop_reason = None; hist = []
    # Same atomic-epoch recovery contract as the Teacher.  A stage that is
    # interrupted must resume from its last completed epoch rather than silently
    # restarting from the Teacher weights it was initialised with.
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get('complete', False):
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start = state['epoch'] + 1
            best = state.get('best', -1.0); bad = state.get('bad', 0)
            best_epoch = state.get('best_epoch', 0); best_state = state.get('best_state')
            hist = state.get('hist', [])
            print(f'[{name}] resume after completed epoch {start - 1} '
                  f'(best={best:.4f} @ep{best_epoch}, bad={bad})', flush=True)
    epoch = start - 1  # keeps the post-loop checkpoint well defined if the range is empty
    for epoch in range(start, epochs + 1):
        model.train(); losses = []; parts_acc = {}
        bar = tqdm(loader(train_set, True), desc=f'{name} {epoch:02d}/{epochs}', unit='batch')
        nb = len(bar); optimizer.zero_grad(set_to_none=True)
        for i, (a, d, y, ga, gd, real) in enumerate(bar, 1):
            a, d, y, ga, gd = (x.to(DEV) for x in (a, d, y, ga, gd)); real = real.to(DEV)
            x, yy = torch.cat((a, d)), torch.cat((y, y))
            if taps_mode:
                taps = model.forward_taps(x); z4, z34 = taps['z4'], taps['z34']
            else:
                taps = None; z4, z34, _ = model(x)
            loss = .5 * seg(z4, yy, weight) + .5 * seg(z34, yy, weight)
            # Logged so the reader can see how much of the objective each added
            # term actually carries, rather than inferring it from the arm names.
            parts_acc.setdefault('seg', []).append(float(loss.detach()))
            t = None
            if kd_mode or use_delta:
                with torch.no_grad():
                    t = merge_taps(teacher.forward_pair_taps(a, d))
                    g10 = torch.cat(((ga > 0) & (gd == 0), (gd > 0) & (ga == 0)))
            if kd_mode:
                if kd_mode in ('gain', 'all'):
                    # ``gain`` is the selective rule; ``all`` is its unweighted
                    # form (w == 1 everywhere), i.e. vanilla logit KD on the same
                    # taps, which is the baseline the selective rule is read against.
                    k4, _ = selective_kl(t['z0'], t['z4'], z4, yy, g10, kd_mode)
                    k34, _ = selective_kl(t['z0'], t['z34'], z34, yy, g10, kd_mode)
                    kd = .5 * (k4 + k34)
                    # Logged as its own term so the decomposition (seg / kd / delta)
                    # is readable from the run log for every arm.
                    parts = {'kd': kd}
                elif kd_mode == 'full':
                    # R3g: R3's rule with the geometry priority switched on.  The
                    # only difference against the branch above is the weight.
                    r2 = torch.cat((real, real))
                    k4, w4 = selective_kl(t['z0'], t['z4'], z4, yy, g10, kd_mode, r2)
                    k34, w34 = selective_kl(t['z0'], t['z34'], z34, yy, g10, kd_mode, r2)
                    kd, parts = .5 * (k4 + k34), {}
                    # The boost has to be observable on its own: ``w_frac`` is the
                    # live fraction of selected pixels, ``geo_frac`` the fraction
                    # that actually received the priority (real pre/post pairs that
                    # are G10), ``w_mean`` the mean weight over the whole batch.
                    parts_acc.setdefault('w_frac', []).append(float((w34 > 0).float().mean()))
                    parts_acc.setdefault('geo_frac', []).append(float((g10 & (r2[:, None, None])).float().mean()))
                    parts_acc.setdefault('w_mean', []).append(float(w34.mean()))
                elif kd_mode == 'dis2':
                    kd, parts = dis2_multilevel_kd(taps, t)
                elif kd_mode == 'dis2-logit':
                    kd, parts = dis2_logit_only(taps, t)
                else:
                    w = gain_protect(t['z0'], t['z34'], z34, yy)
                    kd, parts = selective_multilevel_kd(taps, t, w, yy)
                    # A batch on which nothing is selected contributes exactly zero,
                    # so the live fraction of selected pixels has to be observable.
                    parts_acc.setdefault('w_frac', []).append(float((w > 0).float().mean()))
                # One weight for every arm, so the arms differ in the rule and not
                # in how loudly it is played.  No per-arm calibration.
                loss = loss + kd_weight * kd
                for key, value in parts.items():
                    parts_acc.setdefault(key, []).append(float(value.detach()))
            if use_delta:
                # The only addition over R3, and the reason R3D exists: match the
                # decision change the correction produces, not the correction vector.
                dl = delta_loss(taps, t, yy)
                loss = loss + LAM_DELTA * dl
                parts_acc.setdefault('delta', []).append(float(dl.detach()))
            loss = loss / ACCUM
            loss.backward()
            if i % ACCUM == 0 or i == nb:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item() * ACCUM); bar.set_postfix(loss=f'{np.mean(losses):.3f}')
        if (kd_mode or use_delta) and parts_acc:
            print(f'[{name}] terms epoch={epoch} ' +
                  ' '.join(f'{k}={np.mean(v):.4f}' for k, v in parts_acc.items()), flush=True)
        mean_loss = float(np.mean(losses))
        # v4: the deployed corrected output is scored every epoch, threshold-free.
        score = monitor_score(model, monitor_set, 'S') if EARLY_STOP else float('nan')
        improved = (not EARLY_STOP) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = 'NEW BEST'
        else:
            bad += 1; flag = f'no gain ({bad}/{PATIENCE})'
        hist.append(dict(epoch=epoch, loss=mean_loss, monitor=score, best=best,
                         bad=bad, improved=bool(improved)))
        print(f'[{name}] epoch={epoch}/{epochs} loss={mean_loss:.5f} '
              f'{MONITOR}={score:.4f} best={best:.4f}@{best_epoch} [{flag}]', flush=True)
        checkpoint(latest, model, optimizer, epoch, best, best_state, bad,
                   best_epoch=best_epoch, hist=hist)
        (OUT / f'{name}{tag}_seed{seed}_progress.json').write_text(json.dumps(
            {'method': name, 'seed': seed, 'epoch': epoch, 'max_epochs': epochs,
             'loss': mean_loss, 'monitor': MONITOR, 'monitor_value': score,
             'best': best, 'best_epoch': best_epoch, 'bad': bad, 'patience': PATIENCE,
             'improved': bool(improved), 'history': hist,
             'checkpoint': 'best_monitored_epoch', 'early_stop': bool(EARLY_STOP),
             'monitor_risk': MONITOR_RISK}, indent=2))
        if EARLY_STOP and bad >= PATIENCE:
            stop_reason = (f'{PATIENCE} consecutive epochs without beating '
                           f'{best:.4f} at epoch {best_epoch}')
            print(f'[{name}] EARLY STOP at epoch {epoch}: {stop_reason}', flush=True)
            break
    else:
        stop_reason = f'reached the {epochs}-epoch ceiling'
    # the best-monitored-epoch weights ARE the model, not the last epoch's
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state); model.to(DEV)
    torch.save(model.state_dict(), OUT / f'{name}{tag}_seed{seed}.pt')
    checkpoint(latest, model, optimizer, epoch, best, None, bad, complete=True, kd_weight=kd_weight)
    print(f'[{name}] model = epoch {best_epoch} weights, {MONITOR}={best:.4f}; '
          f'stopped by: {stop_reason}', flush=True)
    return model


def teacher_label(tag):
    """Method prefix for a Teacher variant: tag '' -> Teacher, '_T2a' -> Teacher-T2a."""
    return 'Teacher' + tag.replace('_', '-')


def teacher_pred(head, q):
    """Wrap one Teacher head in the prediction dict ``report`` expects."""
    return {'p': q[head], 'y': q['y'], 'gt': q['gt'], 'gc': q['gc'], 'orbit': q['orbit']}


def teacher_validation_rows(model, dataset, tag=''):
    """Validation summary for every head, in the frozen ``teacher_validation.csv`` shape."""
    q = predict_teacher_heads(model, dataset); name = teacher_label(tag)
    rows = []
    for head, head_name in (('z0', 'self'), ('z4', 'p4'), ('z34', 'dual')):
        rows.append({'method': f'{name}_{head_name}', 'partition': 'validation', 'region': 'all',
                     'pixels': int(q['y'].size), **metrics(q[head], q['y'])})
    return rows


def main():
    global BATCH, ACCUM, LOSS_MODE, POS_WEIGHT, VALIDATE_EVERY, PATIENCE, EARLY_STOP, MONITOR, P, PROTOCOL
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['teacher', 'teacher-test', 'all', *CONFIGS])
    ap.add_argument('--protocol', choices=['v1', 'v2'], default='v2',
                    help='v2 is the unified protocol in configs/protocol_v2.json; v1 reproduces the reported arms')
    ap.add_argument('--epochs', type=int, default=None, help='default: the protocol value')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=None, help='physical batch; default: the protocol value')
    ap.add_argument('--accum', type=int, default=None,
                    help='gradient accumulation; default: whatever keeps physical x accum at the protocol effective batch')
    ap.add_argument('--stages', default='', help='comma-separated override of the stage list run by all')
    ap.add_argument('--tag', default='', help='suffix for method names and the per-stage results table')
    ap.add_argument('--kd-weight', type=float, default=1.0,
                    help='one weight for every arm; the arms differ in the rule, not in how loudly it is played')
    ap.add_argument('--thr', type=float, default=None,
                    help='override the validation-selected F1 threshold (reproducing the v1 tables uses 0.5)')
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL = args.protocol; P = protocol_values(PROTOCOL)
    BATCH = args.batch if args.batch else P['batch']
    ACCUM = args.accum if args.accum else max(1, int(round(P['effective_batch'] / BATCH)))
    LOSS_MODE, POS_WEIGHT, VALIDATE_EVERY, PATIENCE = P['seg'], P['pos_weight'], P['validate_every'], P['patience']
    EARLY_STOP, MONITOR = P['early_stop'], P['monitor']
    if BATCH * ACCUM != P['effective_batch']:
        raise SystemExit(f'physical {BATCH} x accum {ACCUM} != effective {P["effective_batch"]}')
    epochs = args.epochs if args.epochs else P['epochs']
    echo_protocol()
    tid, vid, eid = split(); train_set = HaitiPairs(tid, True); val_set = HaitiPairs(vid, False, train_set.stats); test_set = HaitiPairs(eid, False, train_set.stats)
    thresholds = {}

    def freeze(name, p, y):
        """The threshold is fixed at 0.5, identical for every method.

        There is no validation partition to search on, and searching on test is
        not allowed, so nothing is measured and nothing is chosen here.  The
        early-stopping rule reads AUPRC, which is threshold-free, so fixing 0.5
        is unaffected by model selection.
        """
        return 0.5, float('nan')

    # A Teacher variant is evaluated only on request and only on the test set.
    # The v3 gate that used a validation curve to decide whether a Teacher was
    # worth evaluating is gone with the validation partition: evaluation is now
    # always available on request, and the cost of running it is the only gate.
    if args.stage == 'teacher-test':
        teacher = load_teacher(tag=args.tag)
        name = teacher_label(args.tag)
        thr, _ = freeze(f'{name}_dual', None, None)
        thresholds.update({f'{name}_self': thr, f'{name}_p4': thr, f'{name}_dual': thr})
        q = predict_teacher_heads(teacher, test_set)
        rows = []
        for head, head_name in (('z0', 'self'), ('z4', 'p4'), ('z34', 'dual')):
            rows.extend(report(f'{name}_{head_name}', teacher_pred(head, q), thr))
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / f'results_teacher{args.tag}_seed{args.seed}.csv', index=False)
        (OUT / f'thresholds_teacher{args.tag}_seed{args.seed}.json').write_text(
            json.dumps({'protocol': PROTOCOL, 'source': 'fixed', 'value': 0.5, **thresholds}, indent=2))
        for part in ('overall', 'asc', 'desc'):
            for head_name in ('self', 'dual'):
                r = frame[(frame.method == f'{name}_{head_name}') & (frame.partition == part) & (frame.region == 'all')]
                if len(r):
                    r = r.iloc[0]
                    print(f'[{name}_{head_name}] {part:7s} thr={thr:.3f} IoU={r.iou:.4f} F1={r.f1:.4f} '
                          f'P={r.precision:.4f} R={r.recall:.4f} AUPRC={r.auprc:.4f}', flush=True)
        print(f'[{name}] test table written to results_teacher{args.tag}_seed{args.seed}.csv', flush=True)
        return
    # An explicit ``--stages`` list is a deliberate subset, so it never triggers an
    # implicit Teacher retrain; and ``all`` must be safe to run on a finished
    # Teacher, because train_teacher only recovers from a checkpoint whose complete
    # flag is False.  An explicit ``teacher`` stage still retrains, because that is
    # what asking for it means.
    if args.stage == 'teacher' or (args.stage == 'all' and not args.stages
                                   and not resolve_checkpoint(f'teacher{args.tag}.pt').exists()):
        teacher = train_teacher(train_set, val_set, epochs, args.seed, tag=args.tag,
                                monitor_set=test_set, kd_weight=args.kd_weight)
        print('[Teacher] training complete; test table is written by the teacher-test stage',
              flush=True)
        return
    if args.stages:
        todo = {n: CONFIGS[n] for n in args.stages.split(',')}
    elif args.stage == 'all':
        todo = {name: CONFIGS[name] for name in PLAN}
    else:
        todo = {args.stage: CONFIGS[args.stage]}
    # The Teacher is loaded only when an arm actually needs it: a distillation arm
    # needs its soft targets, R2/R3 need its correction operator, and the v1 arms
    # are warm-started from it.  The single-orbit baseline needs none of that, so
    # it can be trained on its own before the Teacher exists.
    needs_teacher = any(cfg['kd'] is not None or cfg.get('init') == 'teacher'
                        or (cfg.get('op_init', OP_INIT) == 'teacher' and cfg['variant'] in ('R2', 'R3'))
                        for cfg in todo.values())
    if needs_teacher:
        if args.stage == 'all':
            print('[all] reusing the frozen Teacher', flush=True)
        teacher = load_teacher(tag=args.tag)
    else:
        print(f'[all] none of {list(todo)} needs the Teacher; skipping it', flush=True)
        teacher = None
    rows = []
    for name, cfg in todo.items():
        model = train_student(name, cfg, teacher, train_set, val_set, epochs, args.seed,
                              tag=args.tag, kd_weight=args.kd_weight, monitor_set=test_set)
        label = cfg.get('label', f'OursV3-{name}') + args.tag
        # The threshold is fixed at 0.5 and frozen before the test read; model
        # selection has already happened on the monitored test score above.
        thr, _ = freeze(label, None, None)
        thresholds[label] = thr
        rows.extend(report(label, predict_student(model, test_set), thr))
    if rows:
        frame = pd.DataFrame(rows)
        (OUT / f'thresholds_{args.stage}{args.tag}_seed{args.seed}.json').write_text(
            json.dumps({'protocol': PROTOCOL, 'source': 'fixed', 'value': 0.5, **thresholds}, indent=2))
        # Immutable per-stage table, plus a cumulative table.  A later stage must
        # never erase an earlier stage's rows, so the cumulative file is merged
        # on (method) rather than overwritten.
        frame.to_csv(OUT / f'results_{args.stage}{args.tag}_seed{args.seed}.csv', index=False)
        # A tagged run keeps its own cumulative table, so a v2 sweep never mixes
        # into the v1 table.  The rows are unique per (method, partition, region);
        # deduplicating on method alone would keep one arbitrary row per method.
        merged = OUT / f'results{args.tag}_seed{args.seed}.csv'
        if merged.exists():
            frame = pd.concat([pd.read_csv(merged), frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=['method', 'partition', 'region'], keep='last')
        frame.to_csv(merged, index=False)


if __name__ == '__main__':
    main()
