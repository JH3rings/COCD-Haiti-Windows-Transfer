#!/usr/bin/env python3
"""Train and validate the EAR Teacher/Student pair.

This is a deliberately separate entry point from the historical v4 runner.
The current run merges the pinned train and validation IDs into one training
set at the user's request.  It therefore has no validation selection: the last
requested training epoch is the model and every readout is labelled train-only.
The old N-series classes and checkpoints are left untouched.

Stages::

    teacher       train/select the new EAR Teacher
    calibrate     freeze lambda_EAR once on up to eight real train batches
    EAR-GT        inherited Student with segmentation loss only
    EAR-Full      inherited Student with G10-prioritised action loss
    EAR-noGeo     same as EAR-Full with w_geo=1
    validate      write train-only tables and mechanism readouts
    all           teacher -> calibrate -> EAR-GT -> EAR-Full -> EAR-noGeo -> validate

The test partition is intentionally not opened by this script.  A separate
one-shot test read should only be added after the structure, lambda, checkpoint
rule, and validation comparison are frozen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PACKAGE = Path(__file__).resolve().parents[2]
# The source files are the original pinned train/val/test split.  build_sets()
# merges train and validation IDs below, without redrawing or touching test.
os.environ.setdefault('COCD_SPLIT_DIR', str(PACKAGE / 'data' / 'splits'))
os.environ.setdefault('COCD_OUT_ROOT', str(PACKAGE / 'experiments'))
COCD = PACKAGE / 'cocd'
sys.path.insert(0, str(COCD))

_spec = importlib.util.spec_from_file_location('ear_v2', COCD / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# The Windows shell may default Python's text codec to GBK, while the shipped
# JSON protocol contains UTF-8 punctuation.  The legacy module calls
# Path.read_text() without an explicit encoding, so make this import robust and
# restore the standard method immediately afterwards.
_path_read_text = Path.read_text
Path.read_text = lambda self, encoding=None, errors=None: _path_read_text(
    self, encoding='utf-8' if encoding is None else encoding, errors=errors)
try:
    _spec.loader.exec_module(v2)
finally:
    Path.read_text = _path_read_text

from paths import OUT_ROOT, describe  # noqa: E402
from windows_main.models_complement import (  # noqa: E402
    EARStudent,
    EARTeacher,
    init_ear_student_from_teacher,
    n_params,
)

DEV = v2.DEV
OUT = OUT_ROOT / 'windows_main' / 'ear_reallocation'
SEED = 42
EPS = 1e-8
BATCH = 16
MAX_EPOCHS = 20
LAMBDA_PATH = OUT / 'lambda_EAR_frozen.json'


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def seg_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The active protocol's ordinary segmentation loss is plain BCE.
    return F.binary_cross_entropy_with_logits(logits[:, 0], y)


def metric(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    p = np.asarray(p).ravel()
    y = np.asarray(y).astype(bool).ravel()
    h = p >= threshold
    tp = int((h & y).sum())
    fp = int((h & ~y).sum())
    fn = int((~h & y).sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return {
        'iou': float(tp / (tp + fp + fn + EPS)),
        'f1': float(2 * precision * recall / (precision + recall + EPS)),
        'precision': float(precision),
        'recall': float(recall),
        'auprc': float(average_precision_score(y, p)) if y.any() else float('nan'),
    }


def _empty_pred() -> dict:
    return {k: [] for k in ('p', 'self', 'dual', 'y', 'gt', 'gc', 'orbit')}


@torch.no_grad()
def predict_teacher(model: EARTeacher, dataset) -> dict:
    model.eval()
    ans = _empty_pred()
    for a, d, y, ga, gd, _real in loader(dataset):
        a, d = a.to(DEV), d.to(DEV)
        oa, od = model.forward_pair(a, d)
        dual = torch.sigmoid(torch.cat((oa['z_dual'], od['z_dual'])))[:, 0]
        self_p = torch.sigmoid(torch.cat((oa['z_self'], od['z_self'])))[:, 0]
        ans['p'].append(dual.cpu().numpy())
        ans['dual'].append(dual.cpu().numpy())
        ans['self'].append(self_p.cpu().numpy())
        ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy())
        ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * len(a) + ['desc'] * len(a)))
    return {k: np.concatenate(v) for k, v in ans.items()}


@torch.no_grad()
def predict_student(model: EARStudent, dataset) -> dict:
    model.eval()
    ans = _empty_pred()
    for a, d, y, ga, gd, _real in loader(dataset):
        out = model(torch.cat((a.to(DEV), d.to(DEV))))
        p = torch.sigmoid(out['z'])[:, 0]
        self_p = torch.sigmoid(out['z_self'])[:, 0]
        ans['p'].append(p.cpu().numpy())
        ans['self'].append(self_p.cpu().numpy())
        ans['dual'].append(p.cpu().numpy())
        ans['y'].append(torch.cat((y, y)).numpy())
        ans['gt'].append(torch.cat((ga, gd)).numpy())
        ans['gc'].append(torch.cat((gd, ga)).numpy())
        ans['orbit'].append(np.array(['asc'] * len(a) + ['desc'] * len(a)))
    return {k: np.concatenate(v) for k, v in ans.items()}


def rows(name: str, pred: dict) -> list[dict]:
    """Overall/orbit/G-region validation metrics at the fixed threshold."""
    p, y = pred['p'], pred['y']
    gt, gc = pred['gt'] > 0, pred['gc'] > 0
    out = []
    for part, part_mask in (
            ('overall', np.ones(len(pred['orbit']), dtype=bool)),
            ('asc', pred['orbit'] == 'asc'),
            ('desc', pred['orbit'] == 'desc')):
        where = np.broadcast_to(part_mask[:, None, None], p.shape)
        for region, region_mask in (
                ('all', np.ones_like(gt, dtype=bool)),
                ('G00', ~gt & ~gc), ('G01', ~gt & gc),
                ('G10', gt & ~gc), ('G11', gt & gc)):
            mask = where & region_mask
            if not mask.any():
                continue
            mm = metric(p[mask], y[mask])
            out.append({
                'method': name,
                'partition': part,
                'region': region,
                'threshold': 0.5,
                'pixels': int(mask.sum()),
                'positive_pixels': int(y[mask].sum()),
                **mm,
            })
    return out


def overall_auprc(pred: dict) -> float:
    return metric(pred['p'], pred['y'])['auprc']


def clone_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_teacher() -> EARTeacher:
    path = OUT / f'EARTeacher_seed{SEED}.pt'
    if not path.exists():
        raise SystemExit(f'missing {path}; run the teacher stage first')
    model = EARTeacher().to(DEV)
    model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def teacher_batch(model: EARTeacher, a: torch.Tensor, d: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    oa, od = model.forward_pair(a, d)
    return 0.25 * (seg_loss(oa['z_self'], y) + seg_loss(oa['z_dual'], y)
                   + seg_loss(od['z_self'], y) + seg_loss(od['z_dual'], y))


def train_teacher(train_set, val_set=None, epochs: int = MAX_EPOCHS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / f'EARTeacher_seed{SEED}.pt'
    if final.exists():
        log(f'Teacher checkpoint already exists: {final}')
        return
    set_seed(SEED)
    model = EARTeacher().to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    latest = OUT / f'EARTeacher_seed{SEED}_latest.pt'
    progress = OUT / f'EARTeacher_seed{SEED}_progress.json'
    best, best_epoch, bad, best_state, start, hist = float('nan'), 0, 0, None, 1, []
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get('complete', False):
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start, hist = state['epoch'] + 1, state.get('history', [])
            log(f'resuming EAR Teacher after epoch {start - 1} on merged train set')
    for epoch in range(start, epochs + 1):
        model.train()
        losses = []
        bar = tqdm(loader(train_set, True), desc=f'EARTeacher {epoch:03d}/{epochs}', unit='batch')
        optimizer.zero_grad(set_to_none=True)
        for a, d, y, *_ in bar:
            loss = teacher_batch(model, a.to(DEV), d.to(DEV), y.to(DEV))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            bar.set_postfix(loss=f'{np.mean(losses):.4f}')
        # No validation partition is used in this continuation.  The current
        # epoch is the candidate model and no early-stop decision is made.
        score = float('nan')
        improved = False
        best_epoch, best_state = epoch, clone_state(model)
        rec = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'val_auprc_dual': None,
               'val_auprc_self': None,
               'best': best, 'best_epoch': best_epoch, 'improved': bool(improved),
               'bad': bad}
        hist.append(rec)
        log(f'EARTeacher epoch={epoch} loss={rec["train_loss"]:.5f} '
            f'(train-only continuation; last epoch is the model)')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'bad': bad,
                    'best_state': best_state, 'history': hist, 'complete': False}, latest)
        progress.write_text(json.dumps({
            'method': 'EARTeacher', 'seed': SEED, 'epoch': epoch,
            'max_epochs': epochs, 'selection': 'last epoch on merged train set',
            'best': None, 'best_epoch': best_epoch, 'history': hist,
            'validation_used': False, 'test_opened': False,
        }, indent=2))
    if best_state is None:
        best_state = clone_state(model)
        best_epoch = epoch
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'best': best,
                'best_epoch': best_epoch, 'complete': True}, latest)
    progress.write_text(json.dumps({
        'method': 'EARTeacher', 'seed': SEED, 'epoch': epoch, 'max_epochs': epochs,
        'selection': 'last epoch on merged train set', 'best': None,
        'best_epoch': best_epoch, 'history': hist, 'checkpoint': str(final),
        'validation_used': False, 'test_opened': False,
    }, indent=2))
    log(f'wrote {final}; last train epoch={best_epoch}; no validation selection was performed')


def _real_batches(train_set, limit: int = 8):
    """Yield at most eight real pre->post batches in pinned order."""
    real_items = [i for i, (_location, mode) in enumerate(train_set.spec) if mode == 0]
    got = []
    for lo in range(0, min(len(real_items), limit * BATCH), BATCH):
        items = [train_set[i] for i in real_items[lo:lo + BATCH]]
        got.append(tuple(torch.stack([x[j] for x in items]) for j in (0, 1, 2, 3, 4)))
    return got


def action_loss(rho_s: torch.Tensor, rho_t: torch.Tensor,
                e0: torch.Tensor, e1: torch.Tensor,
                g10: torch.Tensor, y: torch.Tensor,
                use_geo: bool) -> tuple[torch.Tensor, dict]:
    """The sole EAR auxiliary loss, with teacher-side quantities detached."""
    with torch.no_grad():
        gain = ((e0 - e1) / (e0 + EPS)).clamp(0, 1)
        weight = 1.0 + (g10 if use_geo else torch.zeros_like(g10))
        valid = torch.ones_like(y, dtype=torch.bool)
    d_action = (rho_s - rho_t.detach()).abs().sum(dim=1) / 4.0
    d_up = F.interpolate(d_action[:, None], size=y.shape[-2:], mode='nearest')[:, 0]
    numerator = (valid * weight * gain * d_up).sum()
    denominator = (valid * weight).sum()
    value = numerator / (denominator + EPS)
    return value, {'gain_mean': float(gain.mean().detach()),
                   'd_action_mean': float(d_up.mean().detach()),
                   'g10_fraction': float(g10.mean().detach())}


def calibrate_lambda(train_set, teacher: EARTeacher) -> float:
    """Static train-only lambda calibration, never reading validation labels."""
    OUT.mkdir(parents=True, exist_ok=True)
    if LAMBDA_PATH.exists():
        return float(json.loads(LAMBDA_PATH.read_text())['lambda_EAR'])
    set_seed(SEED)
    student = EARStudent().to(DEV)
    init_ear_student_from_teacher(student, teacher)
    student.eval()
    teacher.eval()
    seg_values, lear_values = [], []
    with torch.no_grad():
        for a, d, y, ga, gd in _real_batches(train_set, limit=8):
            a, d, y, ga, gd = (z.to(DEV) for z in (a, d, y, ga, gd))
            out_s = student(torch.cat((a, d)))
            oa, od = teacher.forward_pair(a, d)
            y2 = torch.cat((y, y))
            seg_values.append(float(seg_loss(out_s['z'], y2)))
            e0 = F.binary_cross_entropy_with_logits(
                torch.cat((oa['z_self'], od['z_self']))[:, 0], y2, reduction='none')
            e1 = F.binary_cross_entropy_with_logits(
                torch.cat((oa['z_dual'], od['z_dual']))[:, 0], y2, reduction='none')
            rho_t = torch.cat((oa['rho'], od['rho']))
            g10 = torch.cat(((ga > 0) & ~(gd > 0), (gd > 0) & ~(ga > 0)))
            zero_geo = torch.zeros_like(g10, dtype=torch.float32)
            lear, _ = action_loss(out_s['rho'], rho_t, e0, e1, zero_geo, y2, False)
            lear_values.append(float(lear))
    lseg = float(np.mean(seg_values))
    lear = float(np.mean(lear_values))
    lam = min(0.05, 0.05 * lseg / (lear + EPS)) if lear > EPS else 0.05
    record = {
        'method': 'EAR', 'seed': SEED, 'batches': min(8, len(seg_values)),
        'Lseg0': lseg, 'LEAR0': lear, 'lambda_EAR': lam,
        'rule': 'min(0.05, 0.05 * mean(Lseg0)/(mean(LEAR0)+epsilon))',
        'chosen_without_validation': True, 'test_opened': False,
    }
    (OUT / 'lambda_EAR_calibration.json').write_text(json.dumps(record, indent=2))
    LAMBDA_PATH.write_text(json.dumps(record, indent=2))
    log(f'lambda_EAR frozen at {lam:.8f} (Lseg0={lseg:.6f}, LEAR0={lear:.6f})')
    return lam


def student_batch_loss(model: EARStudent, teacher: EARTeacher,
                       a: torch.Tensor, d: torch.Tensor, y: torch.Tensor,
                       ga: torch.Tensor, gd: torch.Tensor,
                       real: torch.Tensor, lam: float,
                       use_geo: bool) -> tuple[torch.Tensor, dict]:
    x, y2 = torch.cat((a, d)), torch.cat((y, y))
    out = model(x)
    seg = seg_loss(out['z'], y2)
    stats = {'seg': float(seg.detach()), 'lear': 0.0, 'weighted_lear': 0.0,
             'gain_mean': 0.0, 'd_action_mean': 0.0, 'g10_fraction': 0.0}
    cm = torch.cat((real, real)).bool()
    if lam <= 0 or not cm.any():
        return seg, stats
    with torch.no_grad():
        oa, od = teacher.forward_pair(a[real.bool()], d[real.bool()])
        yt = torch.cat((y[real.bool()], y[real.bool()]))
        e0 = F.binary_cross_entropy_with_logits(
            torch.cat((oa['z_self'], od['z_self']))[:, 0], yt, reduction='none')
        e1 = F.binary_cross_entropy_with_logits(
            torch.cat((oa['z_dual'], od['z_dual']))[:, 0], yt, reduction='none')
        rho_t = torch.cat((oa['rho'], od['rho']))
        g10 = torch.cat(((ga[real.bool()] > 0) & ~(gd[real.bool()] > 0),
                         (gd[real.bool()] > 0) & ~(ga[real.bool()] > 0))).float()
    lear, extra = action_loss(out['rho'][cm], rho_t, e0, e1, g10, yt, use_geo)
    loss = seg + lam * lear
    stats.update({'lear': float(lear.detach()), 'weighted_lear': float((lam * lear).detach()),
                  **extra})
    return loss, stats


def train_student(variant: str, train_set, val_set=None, teacher: EARTeacher = None,
                  epochs: int = MAX_EPOCHS) -> None:
    final = OUT / f'{variant}_seed{SEED}.pt'
    if final.exists():
        log(f'{variant} checkpoint already exists: {final}')
        return
    set_seed(SEED)
    model = EARStudent().to(DEV)
    init_ear_student_from_teacher(model, teacher)
    lam = 0.0 if variant == 'EAR-GT' else float(json.loads(LAMBDA_PATH.read_text())['lambda_EAR'])
    use_geo = variant != 'EAR-noGeo'
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    latest = OUT / f'{variant}_seed{SEED}_latest.pt'
    best, best_epoch, bad, best_state, start, hist = float('nan'), 0, 0, None, 1, []
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get('complete', False):
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start, hist = state['epoch'] + 1, state.get('history', [])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for epoch in range(start, epochs + 1):
        model.train()
        terms = []
        bar = tqdm(loader(train_set, True), desc=f'{variant} {epoch:03d}/{epochs}', unit='batch')
        for a, d, y, ga, gd, real in bar:
            loss, stats = student_batch_loss(model, teacher, a.to(DEV), d.to(DEV), y.to(DEV),
                                             ga.to(DEV), gd.to(DEV), real.to(DEV), lam, use_geo)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            terms.append(stats)
            bar.set_postfix(loss=f'{float(loss.detach()):.4f}',
                            seg=f'{stats["seg"]:.4f}', lear=f'{stats["lear"]:.4f}')
        score = float('nan')
        improved = False
        best_epoch, best_state = epoch, clone_state(model)
        mean = {k: float(np.mean([t[k] for t in terms])) for k in terms[0]}
        rec = {'epoch': epoch, 'train_loss': mean['seg'] + lam * mean['lear'],
               'Lseg': mean['seg'], 'LEAR': mean['lear'],
               'lambda_EAR_times_LEAR': lam * mean['lear'],
               'aux_to_seg': (lam * mean['lear'] / (mean['seg'] + EPS)),
               'val_auprc': None, 'best': None, 'best_epoch': best_epoch,
               'improved': bool(improved), 'bad': bad, **{k: mean[k] for k in
               ('gain_mean', 'd_action_mean', 'g10_fraction')}}
        hist.append(rec)
        log(f'{variant} epoch={epoch} seg={mean["seg"]:.5f} '
            f'LEAR={mean["lear"]:.5f} lam*LEAR={lam * mean["lear"]:.5f} '
            '(train-only continuation; last epoch is the model)')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'bad': bad,
                    'best_state': best_state, 'history': hist, 'complete': False}, latest)
        (OUT / f'{variant}_seed{SEED}_progress.json').write_text(json.dumps({
            'method': variant, 'seed': SEED, 'epoch': epoch, 'max_epochs': epochs,
            'lambda_EAR': lam, 'geometry_weighted': use_geo,
            'selection': 'last epoch on merged train set', 'best': None,
            'best_epoch': best_epoch, 'history': hist, 'validation_used': False,
            'test_opened': False,
        }, indent=2))
    if best_state is None:
        best_state, best_epoch = clone_state(model), epoch
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'best': best,
                'best_epoch': best_epoch, 'complete': True}, latest)
    log(f'wrote {final}; last train epoch={best_epoch} (no validation selection)')


def mechanism_readout(name: str, pred: dict) -> list[dict]:
    """Two lightweight checks: Teacher gain and disabling Student rho."""
    gt, gc = pred['gt'] > 0, pred['gc'] > 0
    result = []
    for region, mask in (('overall', np.ones_like(gt, dtype=bool)),
                         ('G10', gt & ~gc)):
        m = mask
        a = metric(pred['self'][m], pred['y'][m])
        b = metric(pred['p'][m], pred['y'][m])
        result.append({'method': name, 'region': region,
                       'normal_auprc': b['auprc'], 'rho0_auprc': a['auprc'],
                       'normal_f1': b['f1'], 'rho0_f1': a['f1'],
                       'delta_auprc_normal_minus_rho0': b['auprc'] - a['auprc'],
                       'delta_f1_normal_minus_rho0': b['f1'] - a['f1']})
    return result


def load_student(variant: str) -> EARStudent:
    model = EARStudent().to(DEV)
    model.load_state_dict(torch.load(OUT / f'{variant}_seed{SEED}.pt',
                                     map_location=DEV, weights_only=True))
    return model.eval()


def validate(train_set, val_set, teacher: EARTeacher) -> None:
    all_rows = []
    eval_set = train_set if val_set is None else val_set
    teacher_pred = predict_teacher(teacher, eval_set)
    all_rows.extend(rows('EAR-Teacher-self', {**teacher_pred, 'p': teacher_pred['self']}))
    all_rows.extend(rows('EAR-Teacher-dual', teacher_pred))
    mechanisms = mechanism_readout('EAR-Teacher', teacher_pred)
    for variant in ('EAR-GT', 'EAR-Full', 'EAR-noGeo'):
        path = OUT / f'{variant}_seed{SEED}.pt'
        if not path.exists():
            continue
        pred = predict_student(load_student(variant), eval_set)
        all_rows.extend(rows(variant, pred))
        mechanisms.extend(mechanism_readout(variant, pred))
    # A current DIS2 checkpoint is included only as a labelled re-readout.  It
    # was selected under the v4 test-monitoring protocol, so it is not treated
    # as a clean validation baseline or used for any EAR decision.
    dis2_path = OUT_ROOT / 'ours_v2' / 'DIS2_wm_seed42.pt'
    if dis2_path.exists():
        from models.landslide_cocd_v2 import OursV3Student
        dis2 = OursV3Student('R1').to(DEV)
        dis2.load_state_dict(torch.load(dis2_path, map_location=DEV, weights_only=True))
        q = v2.predict_student(dis2, eval_set)
        for row in v2.report('DIS2-current-v4-checkpoint', q, 0.5):
            row['evidence_note'] = 'checkpoint selected with v4 test AUPRC; contaminated re-readout'
            all_rows.append(row)
    result_name = 'train_only_results_seed42.csv' if val_set is None else 'validation_results_seed42.csv'
    pd.DataFrame(all_rows).to_csv(OUT / result_name, index=False)
    mechanism_name = 'mechanism_train_only_seed42.json' if val_set is None else 'mechanism_validation_seed42.json'
    (OUT / mechanism_name).write_text(json.dumps({
        'rows': mechanisms,
        'selection': 'last epoch on merged train set; fixed threshold 0.5',
        'evaluation_partition': 'train_only' if val_set is None else 'validation',
        'test_opened': False,
        'note': 'DIS2 current checkpoint is a contaminated comparison only and did not drive selection.',
    }, indent=2))
    log(f'wrote {result_name} and {mechanism_name}')


def build_sets():
    tid, vid, eid = v2.split()
    merged = list(tid) + list(vid)
    train_set = v2.HaitiPairs(merged, True)
    return train_set, None, eid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('teacher', 'calibrate', 'EAR-GT', 'EAR-Full',
                                      'EAR-noGeo', 'validate', 'all'))
    ap.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(describe())
    print(f'[EAR protocol] split={os.environ.get("COCD_SPLIT_DIR")} batch={BATCH} '
          f'epochs={args.epochs} lr=5e-5 seed={SEED} '
          'train=merged(train+validation) selection=last_epoch test_opened=False')
    train_set, val_set, _test_ids = build_sets()
    if args.stage in ('teacher', 'all'):
        train_teacher(train_set, val_set, args.epochs)
    teacher = load_teacher() if args.stage != 'teacher' or (OUT / f'EARTeacher_seed{SEED}.pt').exists() else None
    if args.stage in ('calibrate', 'all'):
        assert teacher is not None
        calibrate_lambda(train_set, teacher)
    if args.stage in ('EAR-GT', 'EAR-Full', 'EAR-noGeo', 'all'):
        assert teacher is not None
        if args.stage in ('EAR-Full', 'EAR-noGeo', 'all') and not LAMBDA_PATH.exists():
            calibrate_lambda(train_set, teacher)
        variants = ('EAR-GT', 'EAR-Full', 'EAR-noGeo') if args.stage == 'all' else (args.stage,)
        for variant in variants:
            train_student(variant, train_set, val_set, teacher, args.epochs)
    if args.stage in ('validate', 'all'):
        assert teacher is not None
        validate(train_set, val_set, teacher)


if __name__ == '__main__':
    main()
