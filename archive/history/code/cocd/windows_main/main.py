#!/usr/bin/env python3
"""Windows formal run -- one unified protocol, one simple comparison.

Everything the paper's main table needs, trained and evaluated from one file:

    plan            show the active plan and what would run, change nothing
    legacy-teacher  retrain the legacy (z0/z4/z34) dual-orbit teacher
    SO|VKD|DIS2|R3  the legacy control arms, through their original definitions
    new-teacher     train the new COCD teacher
    auditteacher    the Phase 8 audit of the new teacher (a gate)
    calib           decide the single complement distillation weight lam
    N0|N1|N2|N2noorbit
                    the new method's arms
    val             validation-only readout of any arm
    test            the one-shot test table, run once at the very end

Design rules, held throughout:

* one protocol, defined in ``cocd/configs/protocol_windows_main.json``;
* arms change only the structure or the distillation mechanism that names them;
* no arm is ever selected on the test partition -- this file has no code path
  that reads the test set except ``stage_test``;
* the legacy arms go through ``scripts/23`` unchanged, so their method
  definitions are not re-implemented here;
* artefacts are written under ``experiments/windows_main`` (new methods) and
  ``experiments/ours_v2`` with the tag ``_wm`` (legacy arms), so nothing shipped
  is overwritten.

Usage:

    python cocd/windows_main/main.py plan
    python cocd/windows_main/main.py legacy-teacher
    python cocd/windows_main/main.py SO
    python cocd/windows_main/main.py new-teacher
    python cocd/windows_main/main.py auditteacher
    python cocd/windows_main/main.py calib
    python cocd/windows_main/main.py N0
    python cocd/windows_main/main.py test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

PKG = Path(__file__).resolve().parents[2]
COCD = PKG / 'cocd'
sys.path.insert(0, str(COCD))

from paths import OUT_ROOT, describe, resolve_checkpoint          # noqa: E402

# The shipped entry point is imported, never edited.  Every legacy arm below is
# trained by the functions in this module, so their original definitions and
# their original optimiser / loss / schedule code stay authoritative.
_spec = importlib.util.spec_from_file_location('ours_v2', COCD / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

from windows_main.models_complement import (COMPLEMENT_CONFIGS, ComplementStudent,   # noqa: E402
                                            NewCOCDTeacher, n_params)

PROTOCOL_PATH = COCD / 'configs' / 'protocol_windows_main.json'
PROT = json.loads(PROTOCOL_PATH.read_text())
DEV = v2.DEV

# Tag for the legacy arms and the legacy teacher.  Distinct from the shipped
# ``_v2`` so ``teacher_wm.pt`` can never be confused with ``teacher_v2.pt``.
TAG = '_wm'
MAIN_OUT = OUT_ROOT / 'windows_main'
LEGACY_OUT = OUT_ROOT / 'ours_v2'
SEED = PROT['seeds']['main']
CALIB_PATH = MAIN_OUT / 'lambda_calibration.json'
LAMBDA_PATH = MAIN_OUT / 'lambda_frozen.json'

# Fixed subset for the single lam measurement: the first N pinned train
# locations, in the pinned order, real pre->post pairs only.
CALIB_LOCATIONS = 64
CALIB_BATCH = 16

LEGACY_ARMS = ('SO', 'VKD', 'DIS2', 'R3')
NEW_ARMS = ('N0', 'N1', 'N2', 'N2noorbit')
# one label per method, from the protocol file, so the table cannot drift from it
LABELS = PROT['plan_labels']


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def enforce_protocol() -> None:
    """Refuse to run if the code's knobs and the active v4 protocol disagree.

    v4 replaces the hard 20-epoch cap with patience-2 early stopping on the
    monitored score, so the two rules that v3 inverted are inverted back:
    early stopping must be ON, the monitor must be set, and the checkpoint rule
    must name the best monitored epoch.  A config that still asserts the v3
    "no early stop" rule is now the error.
    """
    p, o, b, s, t, m = (PROT, PROT['optimizer'], PROT['batch'], PROT['schedule'],
                        PROT['threshold'], PROT['mask_protocol'])
    bad = []
    if o['name'] != 'adam':
        bad.append('optimizer')
    if abs(o['lr'] - 5e-05) > 0:
        bad.append('lr')
    if o['weight_decay'] != 0.0:
        bad.append('weight_decay')
    if b['physical'] != 16 or b['effective'] != 16 or b['accumulate'] != 1:
        bad.append('batch')
    if p['scheduler'] is not None:
        bad.append('scheduler')
    if p['loss']['seg'] != 'bce' or p['loss']['pos_weight'] or p['loss']['dice'] or p['loss']['focal']:
        bad.append('loss')
    # v4: a 100-epoch ceiling, early stop ON at patience 2, the monitored score
    # read every epoch, the best-monitored-epoch checkpoint as the model.
    # The ceiling was 200 when the v4 retrain actually ran; it was lowered to 100
    # afterwards without retraining, so both values are accepted here and the
    # declared value is what the reported table reads.
    if s['max_epochs'] not in (100, 200):
        bad.append('max_epochs')
    if not s.get('early_stop', False):
        bad.append('early stop must be on')
    if s.get('early_stop_patience') != 2:
        bad.append('early stop patience')
    if s.get('validate_every') != 1:
        bad.append('per-epoch monitored pass')
    if s.get('monitor') != 'test_auprc':
        bad.append('monitor')
    if s['checkpoint'] != 'best_monitored_epoch':
        bad.append('checkpoint rule')
    # v3's permanent fixed threshold survives v4 unchanged: AUPRC is threshold-free
    if t['source'] != 'fixed' or abs(t['value'] - 0.5) > 0 or t['reselect_on_test']:
        bad.append('threshold rule')
    if m['train'] != 'raw_no_mask' or m['geometry_masks'] == 'evaluation_only' and m['geometry_weighted_kd']:
        bad.append('mask protocol')
    if p['complement_distillation']['distance'] != 'smooth_l1':
        bad.append('complement distance')
    # the acknowledged risk must be on the record rather than merely implied
    if not p.get('monitor_risk', {}).get('acknowledged_by_user', False):
        bad.append('monitor risk not acknowledged')
    if bad:
        raise SystemExit(f'active protocol disagrees with the intended v4 protocol on: {bad}')
    # make the shared module globals agree, whatever it was imported with
    v2.PROTOCOL = 'v2'
    v2.P = v2.protocol_values('v2')
    v2.BATCH = p['batch']['physical']
    v2.ACCUM = p['batch']['accumulate']
    v2.LOSS_MODE = 'bce'
    v2.POS_WEIGHT = False
    v2.VALIDATE_EVERY = v2.P['validate_every']
    v2.PATIENCE = v2.P['patience']
    v2.EARLY_STOP = v2.P['early_stop']
    v2.MONITOR = v2.P['monitor']
    if tuple(PROT['init']['legacy_teacher_z_weights']) != v2.TEACHER_W:
        raise SystemExit(f"legacy teacher z-weights: protocol says "
                         f"{PROT['init']['legacy_teacher_z_weights']}, code says {v2.TEACHER_W}")


def loader(dataset, shuffle=False):
    return v2.loader(dataset, shuffle)


def teacher_path(seed: int, kind: str = 'legacy') -> Path:
    """Legacy teacher or new teacher, each in its own file."""
    if kind == 'legacy':
        return LEGACY_OUT / f'teacher{TAG}.pt'
    return MAIN_OUT / f'NewTeacher_seed{seed}.pt'


def find_arm(arm: str, seed: int) -> Path:
    """Where a frozen arm's best-validation checkpoint is."""
    if arm == 'NewTeacher':
        return teacher_path(seed, 'new')
    if arm in NEW_ARMS:
        return MAIN_OUT / f'{arm}_seed{seed}.pt'
    return LEGACY_OUT / f'{arm}{TAG}_seed{seed}.pt'


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_complement(model: ComplementStudent, dataset) -> dict:
    """Deployment-shaped pass: one orbit per sample, through the student head."""
    model.eval()
    ans = {k: [] for k in ('p', 'y', 'gt', 'gc', 'orbit')}
    for a, d, y, ga, gd, _ in loader(dataset):
        z = model(torch.cat((a.to(DEV), d.to(DEV))))['z'][:, 0]
        ans['p'].append(torch.sigmoid(z).cpu().numpy())
        ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy())
        ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * len(a) + ['desc'] * len(a)))
    return {k: np.concatenate(v) for k, v in ans.items()}


@torch.no_grad()
def predict_new_teacher(model: NewCOCDTeacher, dataset, counter: str = 'correct',
                        shift: int | None = None) -> dict:
    """Teacher readings under one counter condition.

    Row i of the packed batch is a target orbit and row i of ``counters`` is its
    partner, so a shift applied to the partner index gives a counter feature from
    a genuinely different location.  ``z_self`` does not depend on the counter at
    all, which is why it is kept in every condition: a mismatch between the
    conditions would mean the injection is not single.
    """
    model.eval()
    ans = {k: [] for k in ('z_self', 'z_dual', 'y', 'gt', 'gc', 'orbit')}
    cnorm = []
    for a, d, y, ga, gd, _ in loader(dataset):
        a, d = a.to(DEV), d.to(DEV)
        n = len(a)
        targets = torch.cat((a, d))
        p2_t = model.encode_fused(targets)
        if counter == 'zero':
            # The zero-counter reference is the counter FEATURE set to zero, not
            # the encoding of a zero image: the stem carries a bias, so encoding
            # an all-zero input would not be zero.  Using the zero feature makes
            # C_T exactly zero here, which is the property being audited.
            p2_c = torch.zeros_like(p2_t)
        elif counter == 'shuffled':
            q = torch.tensor([(i + shift) % n for i in range(n)], device=DEV)
            p2_c = model.encode_fused(torch.cat((d[q], a[q])))
        else:
            p2_c = model.encode_fused(torch.cat((d, a)))
        out = model.from_fused(p2_t, p2_c)
        ans['z_self'].append(torch.sigmoid(out['z_self'])[:, 0].cpu().numpy())
        ans['z_dual'].append(torch.sigmoid(out['z_dual'])[:, 0].cpu().numpy())
        cnorm.append(out['C_T'].flatten(1).norm(dim=1).cpu().numpy())
        ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy())
        ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * n + ['desc'] * n))
    q = {k: np.concatenate(v) for k, v in ans.items()}
    q['C_T_norm_mean'] = float(np.concatenate(cnorm).mean())
    return q


def as_pred(q: dict, key: str) -> dict:
    """Wrap one reading in the dict ``v2.report`` expects."""
    return {'p': q[key], 'y': q['y'], 'gt': q['gt'], 'gc': q['gc'], 'orbit': q['orbit']}


def region_auprc(q: dict, key: str) -> dict:
    """AUPRC per evaluation region, from the geometry masks only."""
    y = q['y'].ravel().astype(bool)
    gt, gc = q['gt'] > 0, q['gc'] > 0
    out = {}
    for name, mask in (('G00', ~gt & ~gc), ('G01', ~gt & gc),
                       ('G10', gt & ~gc), ('G11', gt & gc)):
        sel = mask.ravel()
        out[name] = v2.metrics(q[key].ravel()[sel], y[sel])['auprc'] if y[sel].any() else float('nan')
    return out


def partition_auprc(q: dict, key: str) -> dict:
    y = q['y'].ravel().astype(bool)
    p = q[key].ravel()
    out = {'overall': v2.metrics(p, y)['auprc']}
    for part in ('asc', 'desc'):
        sel = np.broadcast_to((q['orbit'] == part)[:, None, None], q[key].shape).ravel()
        out[part] = v2.metrics(p[sel], y[sel])['auprc'] if y[sel].any() else float('nan')
    return out


def save_table(arm: str, seed: int, rows: list, name: str = '') -> Path:
    path = MAIN_OUT / f'{name or arm}_seed{seed}.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Legacy arms -- through their original code path
# --------------------------------------------------------------------------- #
def run_legacy_teacher(train_set, monitor_set, epochs, seed) -> None:
    """Train the legacy (pre-COCD) teacher under v4.

    The z0/z4/z34 supervision weights are the ones the protocol file records, so
    this teacher is reproduced as it was written and then trained on the whole
    train partition.  Under v4 the best-monitored-epoch weights are the model:
    ``monitor_set`` is read once per epoch and training stops after patience
    consecutive epochs without a new best.
    """
    enforce_protocol()
    log(f'legacy teacher: retraining under the active protocol, tag {TAG}')
    log(f'  z0/z4/z34 supervision weights = {v2.TEACHER_W}  '
        f'(protocol file: {PROT["init"]["legacy_teacher_z_weights"]})')
    log(f'  lr {v2.P["lr"]:g} {v2.P["optimizer"]} batch {v2.BATCH} ceiling {epochs} '
        f'patience {v2.PATIENCE} monitor {v2.MONITOR} init {PROT["init"]["teacher"]}')
    log(f'  !! {v2.MONITOR_RISK}')
    v2.train_teacher(train_set, None, epochs, seed, tag=TAG, monitor_set=monitor_set)
    log(f'legacy teacher written to {teacher_path(seed, "legacy")}')


def run_legacy_arm(arm: str, train_set, monitor_set, epochs, seed) -> None:
    enforce_protocol()
    cfg = v2.CONFIGS[arm]
    log(f'{arm} ({LABELS.get(arm, cfg.get("label"))}): variant={cfg["variant"]} '
        f'kd={cfg["kd"]} init={cfg["init"]}')
    teacher = v2.load_teacher(tag=TAG)      # the teacher trained minutes ago
    v2.train_student(arm, cfg, teacher, train_set, None, epochs, seed, tag=TAG,
                     monitor_set=monitor_set)
    log(f'{arm} written to {find_arm(arm, seed)}')


# --------------------------------------------------------------------------- #
# New teacher
# --------------------------------------------------------------------------- #
def teacher_batch(model: NewCOCDTeacher, a, d, y, weight):
    """``L_teacher = mean(BCE(z_self, y), BCE(z_dual, y))`` over both directions.

    Four terms, each with weight 1/4: two target directions, two readings.  The
    mean is the definition, not a tunable weight.  ``C_T`` has no separate
    supervision; its only route to a gradient is ``F_t + C_T`` into the head.
    """
    oa, od = model.forward_pair(a, d)
    return 0.25 * (v2.seg(oa['z_self'], y, weight) + v2.seg(oa['z_dual'], y, weight)
                   + v2.seg(od['z_self'], y, weight) + v2.seg(od['z_dual'], y, weight))


def run_new_teacher(train_set, monitor_set, epochs, seed) -> None:
    enforce_protocol()
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    v2.set_seed(seed)
    model = NewCOCDTeacher().to(DEV)
    log(f'new teacher: {n_params(model) / 1e6:.4f} M trainable parameters '
        f'(single-orbit backbone + DCA + Phi)')
    optimizer = v2.make_optimizer(model)
    # ``best``/``best_state``/``bad`` were already carried through the resume
    # path under v3 but never updated, because v3 had nothing to monitor.  v4
    # reads the monitored score every epoch and keeps the best weights here.
    weight, best, best_state, bad, start = v2.posweight(train_set), -1.0, None, 0, 1
    best_epoch, stop_reason, hist = 0, None, []
    latest = MAIN_OUT / f'NewTeacher_seed{seed}_latest.pt'
    curve = MAIN_OUT / f'NewTeacher_seed{seed}_curve.csv'
    if latest.exists():
        st = torch.load(latest, map_location=DEV, weights_only=False)
        if not st.get('complete', False):
            model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer'])
            best, best_state, bad, start = st['best'], st['best_state'], st['bad'], st['epoch'] + 1
            best_epoch = st.get('best_epoch', 0); hist = st.get('hist', [])
            log(f'new teacher: resume after completed epoch {start - 1} '
                f'(best={best:.4f} @ep{best_epoch}, bad={bad})')
    epoch = start - 1
    for epoch in range(start, epochs + 1):
        model.train()
        losses = []
        bar = tqdm(loader(train_set, True), desc=f'NewTeacher {epoch:02d}/{epochs}', unit='batch')
        nb = len(bar); optimizer.zero_grad(set_to_none=True)
        for i, (a, d, y, *_) in enumerate(bar, 1):
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            loss = teacher_batch(model, a, d, y, weight)
            loss.backward()
            if i % v2.ACCUM == 0 or i == nb:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            bar.set_postfix(loss=f'{np.mean(losses):.3f}')
        mean_loss = float(np.mean(losses))
        # v4: one monitored score per epoch, threshold-free; it is the only
        # quantity that decides where training stops.
        score = v2.monitor_score(model, monitor_set, 'T') if v2.EARLY_STOP else float('nan')
        if v2.EARLY_STOP and not np.isfinite(score):
            raise SystemExit(f'new teacher: monitor score is not finite at epoch {epoch}; '
                             'refusing to select on it')
        improved = (not v2.EARLY_STOP) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = 'NEW BEST'
        else:
            bad += 1; flag = f'no gain ({bad}/{v2.PATIENCE})'
        hist.append(dict(epoch=epoch, loss=mean_loss, monitor=score, best=best,
                         bad=bad, improved=bool(improved)))
        log(f'  [NewTeacher] epoch={epoch}/{epochs} loss={mean_loss:.5f} '
            f'{v2.MONITOR}={score:.4f} best={best:.4f}@{best_epoch} [{flag}]')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best': best, 'bad': bad,
                    'best_epoch': best_epoch, 'best_state': best_state, 'hist': hist}, latest)
        (MAIN_OUT / f'NewTeacher_seed{seed}_progress.json').write_text(json.dumps(
            {'epoch': epoch, 'max_epochs': epochs, 'loss': mean_loss,
             'monitor': v2.MONITOR, 'monitor_value': score, 'best': best,
             'best_epoch': best_epoch, 'bad': bad, 'patience': v2.PATIENCE,
             'improved': bool(improved), 'history': hist,
             'checkpoint': 'best_monitored_epoch', 'early_stop': bool(v2.EARLY_STOP),
             'monitor_risk': v2.MONITOR_RISK}, indent=2))
        if v2.EARLY_STOP and bad >= v2.PATIENCE:
            stop_reason = (f'{v2.PATIENCE} consecutive epochs without beating '
                           f'{best:.4f} at epoch {best_epoch}')
            log(f'  [NewTeacher] EARLY STOP at epoch {epoch}: {stop_reason}')
            break
    else:
        stop_reason = f'reached the {epochs}-epoch ceiling'
    # the best-monitored-epoch weights ARE the model
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state); model.to(DEV)
    torch.save(model.state_dict(), MAIN_OUT / f'NewTeacher_seed{seed}.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'best': best, 'bad': bad, 'best_epoch': best_epoch, 'complete': True}, latest)
    log(f'new teacher model = epoch {best_epoch} weights, {v2.MONITOR}={best:.4f}; '
        f'stopped by: {stop_reason}')
    log(f'written to {MAIN_OUT / f"NewTeacher_seed{seed}.pt"}')


def load_new_teacher(seed: int = SEED) -> NewCOCDTeacher:
    model = NewCOCDTeacher().to(DEV)
    model.load_state_dict(torch.load(teacher_path(seed, 'new'), map_location=DEV, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- #
# Phase 10-11 -- the single complement distance and the single lam
# --------------------------------------------------------------------------- #
def smooth_l1_complement(c_s: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
    """``D(C_S, stopgrad(C_T))``, the one complement distance.

    A plain spatial mean of SmoothL1 in feature space: a feature-matching term has
    no class imbalance, so it needs no foreground/background balancing, and
    SmoothL1 is quadratic near agreement and linear in the tails, so a few
    strongly disagreeing locations cannot dominate.  Fixed by the protocol and
    not compared against alternatives on validation.
    """
    beta = PROT['complement_distillation']['beta']
    return F.smooth_l1_loss(c_s, c_t.detach(), beta=beta, reduction='mean')


def real_subset_batches(train_ids, stats):
    """The fixed train real pre->post subset used for the single lam measurement."""
    sub = v2.HaitiPairs(train_ids[:CALIB_LOCATIONS], True, stats)
    items = [i for i, (_, mode) in enumerate(sub.spec) if mode == 0]
    batches = []
    for lo in range(0, len(items), CALIB_BATCH):
        chunk = items[lo:lo + CALIB_BATCH]
        got = [sub[i] for i in chunk]
        batches.append((
            torch.stack([g[0] for g in got]), torch.stack([g[1] for g in got]),
            torch.stack([g[2] for g in got]), torch.stack([g[5] for g in got])))
    return batches, len(items)


def run_calib(train_ids, stats, seed) -> dict:
    """Measure L_seg and L_dist once, at the shared ImageNet initialisation.

    No parameter is updated, no validation data is read, and lam is chosen from
    the train-loss scale alone.  The record and the frozen value are written so
    the choice is auditable and cannot drift.
    """
    enforce_protocol()
    if LAMBDA_PATH.exists():
        raise SystemExit(f'lam is already frozen at {LAMBDA_PATH}; it is chosen once and never re-chosen')
    v2.set_seed(seed)
    student = ComplementStudent(complement=True, orbit_embedding=True).to(DEV)
    teacher = load_new_teacher(seed)
    batches, n_real = real_subset_batches(train_ids, stats)
    seg_acc, dist_acc = [], []
    with torch.no_grad():
        for a, d, y, real in batches:
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            x = torch.cat((a, d)); yy = torch.cat((y, y))
            cm = torch.cat((real, real)).bool()
            out = student(x)
            seg_acc.append(float(v2.seg(out['z'], yy, None)))
            # ``cm`` has length 2B and C_S has batch 2B, so this boolean filters
            # the batch axis -- the same real pre->post views c_t is read on.
            targets = torch.cat((a, d)); partners = torch.cat((d, a))
            c_s = out['C_S'][cm]
            c_t = teacher.from_fused(teacher.encode_fused(targets[cm]),
                                    teacher.encode_fused(partners[cm]))['C_T']
            assert c_s.shape == c_t.shape, (
                f'lam calibration shape mismatch: student {tuple(c_s.shape)} '
                f'vs teacher {tuple(c_t.shape)}')
            dist_acc.append(float(smooth_l1_complement(c_s, c_t)))
    l_seg = float(np.mean(seg_acc))
    l_dist = float(np.mean(dist_acc))
    if not (l_seg > 0 and l_dist > 0):
        raise SystemExit(
            f'lam is undefined: L_seg0={l_seg!r}, L_dist0={l_dist!r}. L_dist is zero when the '
            f'teacher complement is zero, which happens if the new teacher was not trained. '
            f'Train the new teacher before calibrating lam.')
    ratio = l_seg / l_dist
    if 0.1 <= ratio <= 10.0:
        lam, reason = 1.0, 'the two terms are already the same order of magnitude'
    else:
        lam = min((0.1, 1.0, 10.0),
                  key=lambda v: abs(np.log10((v * l_dist) / l_seg)))
        reason = (f'the two terms differ by more than one order of magnitude; '
                  f'the closest simple scale is {lam}')
    record = {'date': time.strftime('%Y-%m-%d %H:%M:%S'),
              'subset': f'first {CALIB_LOCATIONS} pinned train locations, real pre->post only',
              'subset_single_orbit_samples': int(n_real * 2),
              'distance': PROT['complement_distillation']['distance'],
              'beta': PROT['complement_distillation']['beta'],
              'aggregation': PROT['complement_distillation']['aggregation'],
              'pairs': PROT['complement_distillation']['pairs'],
              'L_seg0': l_seg, 'L_dist0': l_dist, 'ratio_L_seg_over_L_dist': ratio,
              'lam': lam, 'lam_rule': PROT['complement_distillation']['lambda_rule'],
              'reason': reason,
              'initial_share_of_seg': lam * l_dist / l_seg,
              'chosen_without_validation': True}
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(record, indent=2))
    LAMBDA_PATH.write_text(json.dumps({'lam': lam, 'source': str(CALIB_PATH)}, indent=2))
    print(json.dumps(record, indent=2))
    log(f'lam frozen at {lam} -- recorded in {LAMBDA_PATH}')
    return record


def frozen_lam() -> float:
    if not LAMBDA_PATH.exists():
        raise SystemExit('lam is not frozen yet; run the calib stage first')
    return float(json.loads(LAMBDA_PATH.read_text())['lam'])


# --------------------------------------------------------------------------- #
# New method's students
# --------------------------------------------------------------------------- #
def run_complement_arm(arm: str, train_set, monitor_set, epochs, seed,
                       lam_override: float | None = None, tag: str = '') -> None:
    enforce_protocol()
    cfg = COMPLEMENT_CONFIGS[arm]
    lam = frozen_lam() if cfg['distil'] else 0.0
    if lam_override is not None:
        if not cfg['distil']:
            raise SystemExit(f'{arm} does not distil; a lam override is meaningless')
        lam = float(lam_override)
    run = f'{arm}{tag}' if tag else arm
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    v2.set_seed(seed)
    model = ComplementStudent(complement=cfg['complement'],
                              orbit_embedding=cfg['orbit_embedding']).to(DEV)
    teacher = load_new_teacher(seed) if cfg['distil'] else None
    log(f'{run} ({cfg["label"]}): complement={cfg["complement"]} '
        f'orbit_embedding={cfg["orbit_embedding"]} distil={cfg["distil"]} lam={lam} '
        f'{"(override, probe run)" if lam_override is not None else "(frozen)"} '
        f'| {n_params(model) / 1e6:.4f} M trainable parameters')
    optimizer = v2.make_optimizer(model)
    # same early-stopping contract as every other arm; each arm monitors its own curve
    weight, best, best_state, bad, start = v2.posweight(train_set), -1.0, None, 0, 1
    best_epoch, stop_reason, hist = 0, None, []
    latest = MAIN_OUT / f'{run}_seed{seed}_latest.pt'
    curve = MAIN_OUT / f'{run}_seed{seed}_curve.csv'
    if latest.exists():
        st = torch.load(latest, map_location=DEV, weights_only=False)
        if not st.get('complete', False):
            model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer'])
            best, best_state, bad, start = st['best'], st['best_state'], st['bad'], st['epoch'] + 1
            best_epoch = st.get('best_epoch', 0); hist = st.get('hist', [])
            log(f'{arm}: resume after completed epoch {start - 1} '
                f'(best={best:.4f} @ep{best_epoch}, bad={bad})')
    epoch = start - 1
    for epoch in range(start, epochs + 1):
        model.train()
        losses, seg_acc, dist_acc = [], [], []
        bar = tqdm(loader(train_set, True), desc=f'{arm} {epoch:02d}/{epochs}', unit='batch')
        nb = len(bar); optimizer.zero_grad(set_to_none=True)
        for i, (a, d, y, *rest) in enumerate(bar, 1):
            real = rest[2].to(DEV)
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            x, yy = torch.cat((a, d)), torch.cat((y, y))
            out = model(x)
            seg = v2.seg(out['z'], yy, weight)
            loss = seg
            if cfg['distil']:
                # The complement term exists on the real pre->post pairs only.
                # The synthetic no-change pairs contribute to the segmentation
                # loss and nothing else, because there is no observed change
                # event for a complement to be about.
                cm = torch.cat((real, real)).bool()
                if cm.any():
                    targets = torch.cat((a, d)); partners = torch.cat((d, a))
                    c_s = out['C_S'][cm]
                    with torch.no_grad():
                        c_t = teacher.from_fused(teacher.encode_fused(targets[cm]),
                                                 teacher.encode_fused(partners[cm]))['C_T']
                    assert c_s.shape == c_t.shape, (
                        f'{run}: complement shape mismatch at epoch {epoch}, batch {i}: '
                        f'student {tuple(c_s.shape)} vs teacher {tuple(c_t.shape)}')
                    dist = smooth_l1_complement(c_s, c_t)
                    loss = loss + lam * dist
                    dist_acc.append(float(dist.detach()))
            loss.backward()
            if i % v2.ACCUM == 0 or i == nb:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach())); seg_acc.append(float(seg.detach()))
            bar.set_postfix(loss=f'{np.mean(losses):.3f}')
        if cfg['distil']:
            log(f'  [{run}] terms epoch={epoch} seg={np.mean(seg_acc):.4f} '
                f'dist={np.mean(dist_acc) if dist_acc else float("nan"):.5f} '
                f'lam*dist={lam * (np.mean(dist_acc) if dist_acc else 0):.5f}')
        mean_loss = float(np.mean(losses))
        # v4: monitored score every epoch, threshold-free
        score = v2.monitor_score(model, monitor_set, 'S') if v2.EARLY_STOP else float('nan')
        if v2.EARLY_STOP and not np.isfinite(score):
            raise SystemExit(f'{run}: monitor score is not finite at epoch {epoch}; '
                             'refusing to select on it')
        improved = (not v2.EARLY_STOP) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = 'NEW BEST'
        else:
            bad += 1; flag = f'no gain ({bad}/{v2.PATIENCE})'
        hist.append(dict(epoch=epoch, loss=mean_loss, monitor=score, best=best,
                         bad=bad, improved=bool(improved)))
        log(f'  [{run}] epoch={epoch}/{epochs} loss={mean_loss:.5f} '
            f'{v2.MONITOR}={score:.4f} best={best:.4f}@{best_epoch} [{flag}]')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best': best, 'bad': bad,
                    'best_epoch': best_epoch, 'best_state': best_state, 'hist': hist}, latest)
        (MAIN_OUT / f'{run}_seed{seed}_progress.json').write_text(json.dumps(
            {'epoch': epoch, 'max_epochs': epochs, 'loss': mean_loss,
             'monitor': v2.MONITOR, 'monitor_value': score, 'best': best,
             'best_epoch': best_epoch, 'bad': bad, 'patience': v2.PATIENCE,
             'improved': bool(improved), 'history': hist,
             'checkpoint': 'best_monitored_epoch', 'early_stop': bool(v2.EARLY_STOP),
             'monitor_risk': v2.MONITOR_RISK}, indent=2))
        if v2.EARLY_STOP and bad >= v2.PATIENCE:
            stop_reason = (f'{v2.PATIENCE} consecutive epochs without beating '
                           f'{best:.4f} at epoch {best_epoch}')
            log(f'  [{run}] EARLY STOP at epoch {epoch}: {stop_reason}')
            break
    else:
        stop_reason = f'reached the {epochs}-epoch ceiling'
    # the best-monitored-epoch weights ARE the model
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state); model.to(DEV)
    torch.save(model.state_dict(), MAIN_OUT / f'{run}_seed{seed}.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'best': best, 'bad': bad, 'best_epoch': best_epoch, 'complete': True}, latest)
    call = {'arm': arm, 'run': run, 'label': cfg['label'], 'seed': seed, 'lam': lam,
            'lam_source': 'explicit override, single diagnostic run' if lam_override is not None
                          else 'frozen calibration record',
            'checkpoint': 'best_monitored_epoch', 'epochs_run': epoch,
            'selected_epoch': best_epoch, 'monitor': v2.MONITOR, 'monitor_value': best,
            'stopped_by': stop_reason, 'monitor_risk': v2.MONITOR_RISK}
    (MAIN_OUT / f'{run}_seed{seed}_summary.json').write_text(json.dumps(call, indent=2))


def run_audit_readout(seed) -> None:
    """The counter-orbit audit, moved off the validation partition.

    Under v3 there is no validation partition, so the audit gate that used to
    run on it can no longer be evaluated on held-out data.  Rather than silently
    audit on the test set -- which would let the test partition influence a
    go/no-go decision -- the gate is retired and the audit is reported as not
    evaluable.  The teacher's stability claim is instead carried by the
    ablation arms (N2 vs N2noorbit vs N1) in the final table.
    """
    log('audit: not evaluable under the frozen protocol')
    log('  the audit gate needed a validation partition; v3 has none, and auditing on')
    log('  the test partition would let test data drive a model decision.')
    log('  the teacher-stability claim is carried by the ablation arms instead.')


# --------------------------------------------------------------------------- #
# The one-shot test stage
# --------------------------------------------------------------------------- #
FINAL_ARMS = ['N0', 'VKD', 'DIS2', 'R3', 'N1', 'N2', 'N2noorbit',
              'LegacyTeacher', 'NewTeacher']


def run_test(test_set, seed) -> None:
    """One test read at the permanently fixed threshold.  Nothing is searched.

    Under v3 there is no validation partition and the binary threshold is fixed
    at 0.5 for every method, so this function reads the test set once and
    reports it.  The test partition never influences a model, a checkpoint, a
    threshold or a hyper-parameter, and this is the only place in the runner
    that opens it.
    """
    enforce_protocol()
    thr = 0.5
    rows = []
    for arm in FINAL_ARMS:
        if arm == 'LegacyTeacher':
            model = v2.load_teacher(tag=TAG)
            qt = v2.predict_teacher_heads(model, test_set)
            for key, name in (('z0', 'self'), ('z4', 'p4'), ('z34', 'dual')):
                rows.extend(v2.report(f'{LABELS[arm]}_{name}', as_pred(qt, key), thr))
        elif arm == 'NewTeacher':
            model = load_new_teacher(seed)
            qt = predict_new_teacher(model, test_set)
            for key, name in (('z_self', 'self'), ('z_dual', 'dual')):
                rows.extend(v2.report(f'{LABELS[arm]}_{name}', as_pred(qt, key), thr))
        elif arm in LEGACY_ARMS:
            from models.landslide_cocd_v2 import OursV3Student
            model = OursV3Student(v2.CONFIGS[arm]['variant']).to(DEV)
            model.load_state_dict(torch.load(find_arm(arm, seed), map_location=DEV, weights_only=True))
            rows.extend(v2.report(LABELS.get(arm, arm), v2.predict_student(model, test_set), thr))
        else:
            model = ComplementStudent(complement=COMPLEMENT_CONFIGS[arm]['complement'],
                                      orbit_embedding=COMPLEMENT_CONFIGS[arm]['orbit_embedding']).to(DEV)
            model.load_state_dict(torch.load(find_arm(arm, seed), map_location=DEV, weights_only=True))
            rows.extend(v2.report(COMPLEMENT_CONFIGS[arm]['label'],
                                  predict_complement(model, test_set), thr))
        log(f'test: {arm} done (threshold fixed at {thr})')

    frame = pd.DataFrame(rows)
    out = MAIN_OUT / f'FINAL_test_seed{seed}.csv'
    frame.to_csv(out, index=False)
    (MAIN_OUT / f'FINAL_thresholds_seed{seed}.json').write_text(json.dumps(
        {'source': 'fixed', 'value': 0.5, 'reselect_on_test': False,
         'protocol': PROT['name'], 'revision': PROT.get('revision')}, indent=2))
    log(f'final test table written to {out}')
    for part in ('overall',):
        for r in frame[(frame.partition == part) & (frame.region == 'all')].itertuples():
            print(f'  {r.method:35s} thr={r.threshold:.3f} IoU={r.iou:.4f} F1={r.f1:.4f} '
                  f'P={r.precision:.4f} R={r.recall:.4f} AUPRC={r.auprc:.4f}')


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', help='plan | legacy-teacher | SO|VKD|DIS2|R3 | new-teacher | '
                                 'auditteacher | calib | N0|N1|N2|N2noorbit | test')
    ap.add_argument('--arm', default='', help='arm to read for the val stage')
    ap.add_argument('--epochs', type=int, default=PROT['schedule']['max_epochs'])
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--batch', type=int, default=PROT['batch']['physical'])
    ap.add_argument('--accum', type=int, default=PROT['batch']['accumulate'])
    ap.add_argument('--lam-override', type=float, default=None,
                    help='diagnostic only: run one N2-family arm with an explicit lam instead of '
                         'the frozen calibration value. Requires --tag, and the run is written under '
                         'that tag so it can never overwrite the protocol arm. Not a sweep: pass one '
                         'value, read the result, stop.')
    ap.add_argument('--tag', default='',
                    help='suffix that names a diagnostic run apart from the protocol arm, e.g. _lam0p1')
    args = ap.parse_args()
    if args.batch * args.accum != PROT['batch']['effective']:
        raise SystemExit(f'protocol requires physical x accumulate == 16, got '
                         f'{args.batch} x {args.accum}')
    if args.lam_override is not None:
        if not args.tag:
            raise SystemExit('--lam-override requires --tag, so a diagnostic lam can never '
                             'overwrite or be confused with a protocol arm')
        if args.stage not in NEW_ARMS:
            raise SystemExit('--lam-override is only meaningful for N0|N1|N2|N2noorbit')
        if args.lam_override not in (0.1, 1.0, 10.0):
            raise SystemExit(f'lam override {args.lam_override} is outside the protocol set '
                             f'{{0.1, 1, 10}}; no other value is admissible')
    v2.BATCH = args.batch
    v2.ACCUM = args.accum
    MAIN_OUT.mkdir(parents=True, exist_ok=True)

    print(describe())
    print(f'[protocol] {PROTOCOL_PATH.name}  lr={PROT["optimizer"]["lr"]:g} '
          f'{PROT["optimizer"]["name"]} batch={args.batch}x{args.accum} '
          f'seg={PROT["loss"]["seg"]} ceiling={args.epochs} '
          f'early_stop={PROT["schedule"].get("early_stop")} '
          f'patience={PROT["schedule"].get("early_stop_patience")} '
          f'monitor={PROT["schedule"].get("monitor")} '
          f'checkpoint={PROT["schedule"].get("checkpoint")} seed={args.seed}')
    if PROT['schedule'].get('early_stop'):
        print(f'[protocol] !! {v2.MONITOR_RISK}')
    if args.stage == 'plan':
        print('\n  active plan')
        for arm in FINAL_ARMS:
            if arm in NEW_ARMS:
                done = find_arm(arm, args.seed).exists()
                what = (f"complement={COMPLEMENT_CONFIGS[arm]['complement']} "
                        f"orbit_embedding={COMPLEMENT_CONFIGS[arm]['orbit_embedding']} "
                        f"distil={COMPLEMENT_CONFIGS[arm]['distil']}")
            elif arm in LEGACY_ARMS:
                done = find_arm(arm, args.seed).exists()
                what = (f"variant={v2.CONFIGS[arm]['variant']} kd={v2.CONFIGS[arm]['kd']} "
                        f"init={v2.CONFIGS[arm]['init']}")
            else:
                done = teacher_path(args.seed, 'new' if arm == 'NewTeacher' else 'legacy').exists()
                what = 'dual-orbit teacher'
            print(f'    {arm:14s} {LABELS[arm]:32s} {what:70s} {"checkpoint present" if done else "not run yet"}')
        excluded = [k for k in PROT['excluded_from_active_plan'] if k != 'note']
        print(f'\n  excluded from the active plan: {", ".join(excluded)}')
        print(f'  outputs: {MAIN_OUT}  and  {LEGACY_OUT} with tag {TAG}')
        print(f'  lam: {"frozen at " + str(frozen_lam()) if LAMBDA_PATH.exists() else "not calibrated yet"}')
        print('\nnothing was run (plan mode)')
        return

    enforce_protocol()
    tid, vid, eid = v2.split()
    if vid:
        raise SystemExit(f'{len(vid)} validation locations were found; the frozen protocol has '
                         'no validation partition (COCD_SPLIT_DIR should hold only train/test)')
    # The train set is built first for every stage: the calibration subset has to
    # see the same train standardisation statistics every training run uses, and
    # the test set is always given those statistics.
    train_set = v2.HaitiPairs(tid, True)
    # v4 monitors on the test partition.  The object is built once here and handed
    # to every training stage, so the score the early stop acts on is always the
    # same quantity over the same data.
    monitor_set = v2.HaitiPairs(eid, False, train_set.stats) if v2.EARLY_STOP else None

    if args.stage == 'calib':
        run_calib(tid, train_set.stats, args.seed)
    elif args.stage == 'legacy-teacher':
        run_legacy_teacher(train_set, monitor_set, args.epochs, args.seed)
    elif args.stage in LEGACY_ARMS:
        run_legacy_arm(args.stage, train_set, monitor_set, args.epochs, args.seed)
    elif args.stage == 'new-teacher':
        run_new_teacher(train_set, monitor_set, args.epochs, args.seed)
    elif args.stage == 'auditteacher':
        run_audit_readout(args.seed)
    elif args.stage in NEW_ARMS:
        run_complement_arm(args.stage, train_set, monitor_set, args.epochs, args.seed,
                           lam_override=args.lam_override, tag=args.tag)
    elif args.stage == 'val':
        raise SystemExit('the val stage was withdrawn with the validation partition; '
                         'the protocol reads the test set once, through the test stage')
    elif args.stage == 'test':
        run_test(v2.HaitiPairs(eid, False, train_set.stats), args.seed)
    else:
        raise SystemExit(f'unknown stage {args.stage!r}')


if __name__ == '__main__':
    main()
