#!/usr/bin/env python3
"""Stage 1 fact-check: batch census, zero-gradient scope, tap paths, inheritance.

Read-only.  No training, no checkpoint is written, no test partition is touched.
Every number here is measured or read off the code, never inferred from the batch
size.  Sections:

  [A] loader/spec census      -- samples, batches, optimiser updates per epoch
  [B] gradient scope          -- per-parameter grad census on synthetic / mixed /
                                 real-only batches, for the Teacher objective and
                                 for the Student objective (seg and KD apart)
  [C] tap paths               -- are z0/z4/z34 one decoder, does zeroing the
                                 counter return to self, does r4 reach P3
  [D] inheritance             -- which student copies the Teacher's correction
                                 operator, and how far the trained copy drifted
  [E] counter-orbit gain      -- where the dual track beats the self track on
                                 validation, and what each student does there

Results: experiments/ours_v2/stage1_factcheck.json (+ .csv for [A] and [E]).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1].parent
A = Path(__file__).resolve().parents[1]
OUT = A / 'experiments' / 'ours_v2'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = load_module('v2', A / 'scripts' / '23_train_ours_v2.py')
DEV = v2.DEV
res = {}
rows_a, rows_e = [], []


def say(*a):
    print(*a, flush=True)


def batch_of(ds, idxs):
    """Collate exactly as the default DataLoader collate does."""
    items = [ds[i] for i in idxs]
    return tuple(torch.stack([it[k] for it in items]) for k in range(6))


def grad_census(model):
    none_, zero_, nonzero = [], [], []
    for n, p in model.named_parameters():
        if p.grad is None:
            none_.append(n)
        elif float(p.grad.abs().max()) == 0.0:
            zero_.append(n)
        else:
            nonzero.append(n)
    return none_, zero_, nonzero


def census(model, tag):
    none_, zero_, nonzero = grad_census(model)
    say(f'  {tag}: params={len(none_) + len(zero_) + len(nonzero)} '
        f'grad=None:{len(none_)} grad==0:{len(zero_)} grad!=0:{len(nonzero)}')
    return {'params': len(none_) + len(zero_) + len(nonzero), 'grad_none': len(none_),
            'grad_zero_tensor': len(zero_), 'grad_nonzero': len(nonzero),
            'names_grad_zero': zero_, 'names_grad_nonzero': nonzero}


# --------------------------------------------------------------------------- #
# [A] loader / spec census
# --------------------------------------------------------------------------- #
say('=== [A] loader / spec census ===')
tid, vid, eid = v2.split()
train_set = v2.HaitiPairs(tid, True)
val_set = v2.HaitiPairs(vid, False, train_set.stats)
spec = train_set.spec
modes = np.array([m for _, m in spec])
res['A_dataset'] = {
    'train_rows': len(train_set.rows), 'val_rows': len(val_set.rows),
    'test_rows': len(eid), 'train_spec_len': len(spec),
    'spec_mode_counts': {int(m): int((modes == m).sum()) for m in (0, 1, 2)},
    'val_spec_len': len(val_set.spec),
    'posweight_teacher_and_student': float(v2.posweight(train_set)),
    'train_modes_flag_teacher_and_student': True,
}
say(f"  train rows={len(train_set.rows)} val rows={len(val_set.rows)} test rows={len(eid)}")
say(f"  train spec={len(spec)} (mode0={int((modes == 0).sum())} mode1={int((modes == 1).sum())} "
    f"mode2={int((modes == 2).sum())})  val spec={len(val_set.spec)}")

for B in (2, 8):
    for shuffle in (False, True):
        rounds = 10 if shuffle else 1
        hist, n_batch, all_syn, sum_r_over_B, last_size, n_updates = {}, 0, 0, 0.0, None, 0
        for _ in range(rounds):
            for a, d, y, ga, gd, real in DataLoader(train_set, batch_size=B, shuffle=shuffle, num_workers=0):
                r = int(real.sum())
                hist[r] = hist.get(r, 0) + 1
                n_batch += 1
                all_syn += int(r == 0)
                sum_r_over_B += r / B
                last_size = len(a)
                n_updates += 1
        per_epoch = n_batch / rounds
        row = {'batch': B, 'shuffle': shuffle, 'epochs_measured': rounds,
               'batches_per_epoch': per_epoch, 'last_batch_size': last_size,
               'optimizer_updates_per_epoch': per_epoch,
               'all_synthetic_batches_frac': all_syn / n_batch,
               'real_per_batch_histogram': hist,
               'sum_r_over_B_per_epoch': sum_r_over_B / rounds,
               'expected_sum_r_over_B': len(spec) / 3 / B,
               'updates_over_50_epochs': per_epoch * 50}
        rows_a.append(row)
        say(f"  B={B} shuffle={shuffle}: batches/epoch={per_epoch:.1f} all-syn={all_syn / n_batch:.4f} "
            f"hist={hist} sum(r/B)/epoch={sum_r_over_B / rounds:.3f} last_batch={last_size}")
res['A_loader'] = rows_a

# --------------------------------------------------------------------------- #
# [B] gradient scope on synthetic / mixed / real-only batches
# --------------------------------------------------------------------------- #
say('=== [B] gradient scope ===')
S0 = torch.load(A / 'experiments' / 'rapid_landslide_cocd' / 'S0.pt', map_location=DEV, weights_only=True)
weight = v2.posweight(train_set)
real_idx = [i for i, (_, m) in enumerate(spec) if m == 0]
syn_idx = [i for i, (_, m) in enumerate(spec) if m != 0]
b_real1 = batch_of(train_set, real_idx[:1])
b_syn1 = batch_of(train_set, syn_idx[:1])
b_syn2 = batch_of(train_set, syn_idx[:2])
b_mixed = batch_of(train_set, [real_idx[0], syn_idx[0]])
b_2real = batch_of(train_set, real_idx[:2])


def teacher_loss(bt):
    a, d, y = (x.to(DEV) for x in bt[:3])
    return v2.teacher_batch(t_model, a, d, y, weight)


def teacher_probs(bt):
    a, d = bt[0].to(DEV), bt[1].to(DEV)
    outs = [pair[i].detach() for pair in t_model.forward_pair(a, d) for i in range(3)]
    return [float(torch.sigmoid(o[:, 0]).max()) for o in outs], [
        float((torch.sigmoid(o[:, 0]) == 0).float().mean()) for o in outs]


t_model = v2.OursV2Teacher().to(DEV)
t_model.load_s0(S0)
t_model.train()
losses = {}
for tag, bt in (('syn1', b_syn1), ('syn2', b_syn2), ('mixed', b_mixed),
                ('real1', b_real1), ('real2', b_2real)):
    t_model.zero_grad(set_to_none=True)
    loss = teacher_loss(bt)
    loss.backward()
    losses[tag] = float(loss.detach())
    res.setdefault('B_teacher', {})[tag] = {'loss': float(loss.detach()), **census(t_model, f'Teacher/{tag}')}
t_model.eval()
for tag, bt in (('syn1', b_syn1), ('real1', b_real1), ('mixed', b_mixed)):
    mx, frac0 = teacher_probs(bt)
    res['B_teacher'][tag]['sigmoid_max_per_head'] = mx
    res['B_teacher'][tag]['pixel_frac_p_exactly_0_per_head'] = frac0
    say(f"  Teacher/{tag}: sigmoid max per head={['%.3e' % v for v in mx]} "
        f"frac(p==0)={['%.4f' % v for v in frac0]}")
res['B_teacher']['dilution_train_stochastic'] = {
    'loss_real1': losses['real1'], 'loss_mixed': losses['mixed'],
    'ratio_mixed_over_real1': losses['mixed'] / losses['real1'],
    'note': 'train mode: DropPath makes two forwards of the same sample non-comparable',
}
# Deterministic version: the same two batches with DropPath disabled, so the only
# difference between them is the co-resident synthetic sample.
with torch.no_grad():
    det = {tag: float(teacher_loss(bt)) for tag, bt in
           (('real1', b_real1), ('mixed', b_mixed), ('real2', b_2real), ('syn1', b_syn1), ('syn2', b_syn2))}
res['B_teacher']['dilution'] = {**det,
                                'ratio_mixed_over_real1': det['mixed'] / det['real1'],
                                'ratio_real2_over_real1': det['real2'] / det['real1']}
say(f"  dilution (eval, deterministic): loss[real]={det['real1']:.8f} loss[real,syn]={det['mixed']:.8f} "
    f"ratio={det['mixed'] / det['real1']:.8f}  loss[2real]={det['real2']:.8f} "
    f"ratio={det['real2'] / det['real1']:.8f}  loss[syn]={det['syn1']:.3e}")

# Student objective: seg and KD apart, on the same batches.
teacher_frozen = v2.load_teacher()
trained_teacher = v2.load_teacher()


def student_parts(model, bt, kd_mode='gain'):
    a, d, y, ga, gd, real = (x.to(DEV) for x in bt)
    x, yy = torch.cat((a, d)), torch.cat((y, y))
    taps = model.forward_taps(x)
    seg = .5 * v2.seg(taps['z4'], yy, weight) + .5 * v2.seg(taps['z34'], yy, weight)
    with torch.no_grad():
        t = v2.merge_taps(teacher_frozen.forward_pair_taps(a, d))
        g10 = torch.cat(((ga > 0) & (gd == 0), (gd > 0) & (ga == 0)))
    k4, w4 = v2.selective_kl(t['z0'], t['z4'], taps['z4'], yy, g10, kd_mode)
    k34, w34 = v2.selective_kl(t['z0'], t['z34'], taps['z34'], yy, g10, kd_mode)
    kd = .5 * (k4 + k34)
    from losses.distill import dis2_multilevel_kd
    kd_dis2, _ = dis2_multilevel_kd(taps, t)
    return {'seg': seg, 'kd_gain': kd, 'kd_dis2': kd_dis2,
            'w34_frac_positive': float((w34 > 0).float().mean()),
            'w34_mean': float(w34.mean())}


for variant, tag in (('R1', 'R1'), ('R3', 'R3')):
    s_model = v2.OursV3Student(variant).to(DEV)
    s_model.load_teacher_self(teacher_frozen)
    s_model.train()
    res.setdefault('B_student', {})
    for btag, bt in (('syn1', b_syn1), ('syn2', b_syn2), ('mixed', b_mixed), ('real2', b_2real)):
        parts = student_parts(s_model, bt)
        s_model.zero_grad(set_to_none=True)
        total = parts['seg'] + parts['kd_gain']
        total.backward()
        res['B_student'].setdefault(tag, {})[btag] = {
            'seg': float(parts['seg'].detach()), 'kd_gain': float(parts['kd_gain'].detach()),
            'kd_dis2': float(parts['kd_dis2'].detach()),
            'w34_frac_positive': parts['w34_frac_positive'], 'w34_mean': parts['w34_mean'],
            **census(s_model, f'Student-{tag}/{btag}')}
        say(f"  Student-{tag}/{btag}: seg={float(parts['seg'].detach()):.3e} "
            f"kd_gain={float(parts['kd_gain'].detach()):.3e} kd_dis2={float(parts['kd_dis2'].detach()):.4f} "
            f"w>0 frac={parts['w34_frac_positive']:.4f}")
    del s_model

# Does the DIS2 rule also go inert on a synthetic batch?  It has no gain/protect
# weighting, so the question has to be measured rather than assumed.
s_dis2 = v2.OursV3Student('R1').to(DEV)
s_dis2.load_teacher_self(teacher_frozen)
s_dis2.train()
for btag, bt in (('syn1', b_syn1), ('syn2', b_syn2), ('real2', b_2real)):
    parts = student_parts(s_dis2, bt)
    s_dis2.zero_grad(set_to_none=True)
    parts['kd_dis2'].backward()
    res['B_student'].setdefault('DIS2', {})[btag] = {
        'kd_dis2': float(parts['kd_dis2'].detach()),
        'note': 'backward through the DIS2 KD term alone (no gain/protect weighting)',
        **census(s_dis2, f'Student-DIS2/{btag}')}
    say(f"  Student-DIS2/{btag}: kd_dis2={float(parts['kd_dis2'].detach()):.6f}")
del s_dis2

# --------------------------------------------------------------------------- #
# [C] tap paths
# --------------------------------------------------------------------------- #
say('=== [C] tap paths ===')
t_eval = v2.load_teacher()
a, d = b_real1[0].to(DEV), b_real1[1].to(DEV)
ft, fc = t_eval.backbone.encode(a), t_eval.backbone.encode(d)
r3, r4 = t_eval.from_features(ft, fc)[3]
c2, p3, p4 = t_eval.decoder.pyramid(ft)
z0, z4, z34 = t_eval.decoder.states(ft, r3, r4)
c = {}
c['correction_norm_p3'] = float(r3.norm())
c['correction_norm_p4'] = float(r4.norm())
c['z0_vs_z4_maxdiff'] = float((z0 - z4).abs().max())
c['z0_vs_z34_maxdiff'] = float((z0 - z34).abs().max())
_, _, _, r_zero = t_eval.from_features(ft, tuple(torch.zeros_like(x) for x in fc))
c['r_with_zero_counter_max'] = max(float(r_zero[0].abs().max()), float(r_zero[1].abs().max()))
za, zd = t_eval.forward_pair(a, torch.zeros_like(d))
c['zero_counter_all_heads_identical'] = bool(torch.equal(za[0], za[1]) and torch.equal(za[0], za[2])
                                             and torch.equal(zd[0], zd[1]) and torch.equal(zd[0], zd[2]))
s_none = t_eval.decoder.states(ft, None, None)
s_r4 = t_eval.decoder.states(ft, None, r4)
s_r3 = t_eval.decoder.states(ft, r3, None)
c['states_none_all_identical'] = bool(torch.equal(s_none[0], s_none[1]) and torch.equal(s_none[0], s_none[2]))
c['with_r4_only: z4-vs-z0'] = float((s_r4[1] - s_r4[0]).abs().max())
c['with_r4_only: z34-vs-z4'] = float((s_r4[2] - s_r4[1]).abs().max())
c['with_r3_only: z34-vs-z4'] = float((s_r3[2] - s_r3[1]).abs().max())
c['p4_correction_propagates_into_p3_l2'] = float(
    ((t_eval.backbone.l3(ft[1]) + torch.nn.functional.interpolate(p4 + r4, size=p3.shape[-2:], mode='nearest')) - p3
     ).norm())
c['r3_computed_on_uncorrected_p3'] = True  # code: a3(cat(p3t, p3c)) - a3(cat(p3t, 0)), p3t from pyramid(ft)
res['C_paths'] = c
say('  ' + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()}, indent=2))

# --------------------------------------------------------------------------- #
# [D] inheritance of the correction operator
# --------------------------------------------------------------------------- #
say('=== [D] inheritance ===')
fresh_t = v2.OursV2Teacher().to(DEV)
fresh_t.load_s0(S0)
inh = {}
for variant in ('R0', 'R1', 'R2', 'R3'):
    s = v2.OursV3Student(variant).to(DEV)
    s.load_teacher_self(fresh_t)
    entry = {'has_omega_op': variant in ('R2', 'R3')}
    if variant in ('R2', 'R3'):
        entry['op3_bitwise_equal_a3'] = all(
            torch.equal(s.op3.state_dict()[k], fresh_t.a3.state_dict()[k]) for k in s.op3.state_dict())
        entry['op4_bitwise_equal_a4'] = all(
            torch.equal(s.op4.state_dict()[k], fresh_t.a4.state_dict()[k]) for k in s.op4.state_dict())
        entry['op3_requires_grad'] = bool(next(s.op3.parameters()).requires_grad)
        entry['driver_d3_last_layer_is_zero'] = bool(float(s.d3.net[-1].weight.abs().max()) == 0.0)
    inh[variant] = entry
# drift of the trained copies
for ckpt, variant, keys in (('R3_seed42.pt', 'R3', ('op3', 'op4')), ('R2_seed42.pt', 'R2', ('op3', 'op4')),
                            ('R1_seed42.pt', 'R1', ())):
    state = torch.load(OUT / ckpt, map_location='cpu', weights_only=True)
    e = {}
    for k, tgt in (('op3', 'a3'), ('op4', 'a4')):
        if k in keys:
            src = torch.load(OUT / 'teacher.pt', map_location='cpu', weights_only=True)
            num = sum(float((state[f'{k}.{p}'] - src[f'{tgt}.{p}']).pow(2).sum()) for p in
                      [n.split('.', 1)[1] for n in src if n.startswith(f'{tgt}.')]) ** .5
            den = sum(float(src[f'{tgt}.{p}'].pow(2).sum()) for p in
                      [n.split('.', 1)[1] for n in src if n.startswith(f'{tgt}.')]) ** .5
            e[f'{k}_relative_l2_drift_vs_teacher'] = num / (den + 1e-12)
        else:
            e[f'{k}_absent'] = True
    inh[f'trained_{ckpt}'] = e
    say(f'  {ckpt}: {e}')
res['D_inheritance'] = inh

# --------------------------------------------------------------------------- #
# [E] where the counter track helps, and what each student does there
# --------------------------------------------------------------------------- #
say('=== [E] counter-orbit gain analysis (validation) ===')
v2.BATCH = 8
tq = v2.predict_teacher_heads(v2.load_teacher(), val_set)
truth = tq['y'] > 0
self_h = tq['z0'] >= .5
dual_h = tq['z34'] >= .5
gt, gc = tq['gt'] > 0, tq['gc'] > 0
orbit = tq['orbit']

U_TPgain = truth & ~self_h & dual_h       # counter recovers a positive the self track missed
U_TPloss = truth & self_h & ~dual_h       # counter loses a positive the self track had
U_FPcut = ~truth & self_h & ~dual_h       # counter removes a self false positive
U_FPadd = ~truth & ~self_h & dual_h       # counter introduces a false positive
U = U_TPgain | U_FPcut                    # pixels where the counter track is net better
res['E_teacher_cells'] = {
    'val_pixels_total': int(truth.size),
    'val_positive_pixels': int(truth.sum()),
    'TPgain': int(U_TPgain.sum()), 'TPloss': int(U_TPloss.sum()),
    'FPcut': int(U_FPcut.sum()), 'FPadd': int(U_FPadd.sum()),
    'U_net_better': int(U.sum()),
    'by_orbit': {o: int((U & (orbit == o)[:, None, None]).sum()) for o in ('asc', 'desc')},
    'by_region': {r: int((U & m).sum()) for r, m in
                  (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc))},
    'TPgain_by_region': {r: int((U_TPgain & m).sum()) for r, m in
                         (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc))},
    'FPcut_by_region': {r: int((U_FPcut & m).sum()) for r, m in
                        (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc))},
}
res['E_region_sizes'] = {r: int(m.sum()) for r, m in
                         (('G00', ~gt & ~gc), ('G01', ~gt & gc), ('G10', gt & ~gc), ('G11', gt & gc))}
say('  ' + json.dumps(res['E_teacher_cells'], indent=2))


def ev(name, p):
    h = p >= .5
    tp, fp, fn = (h & truth).sum(), (h & ~truth).sum(), (~h & truth).sum()
    row = {'model': name, 'val_iou': tp / (tp + fp + fn + 1e-8),
           'TPgain_recovered': float((h & U_TPgain).sum()) / max(int(U_TPgain.sum()), 1),
           'FPadd_carried': float((h & U_FPadd).sum()) / max(int(U_FPadd.sum()), 1),
           'FPcut_still_fired': float((h & U_FPcut).sum()) / max(int(U_FPcut.sum()), 1),
           'FP_outside_U': int((h & ~truth & ~U).sum()),
           'FN_outside_U': int((~h & truth & ~U).sum()),
           'FP_total': int(fp), 'FN_total': int(fn), 'TP_total': int(tp),
           'FP_rate_all_negatives': float(fp) / max(int((~truth).sum()), 1),
           'FP_rate_on_self_fired_negatives':
               float((h & ~truth & self_h).sum()) / max(int((~truth & self_h).sum()), 1),
           'recall_on_U_positives': float((h & truth & U).sum()) / max(int((truth & U).sum()), 1),
           'recall_off_U_positives': float((h & truth & ~U).sum()) / max(int((truth & ~U).sum()), 1),
           'iou_on_U': float((h & truth & U).sum()) / max(int(((h | truth) & U).sum()), 1),
           'iou_off_U': float((h & truth & ~U).sum()) / max(int(((h | truth) & ~U).sum()), 1)}
    rows_e.append(row)
    say(f"  {name:14s} IoU={row['val_iou']:.4f} TPgain_rec={row['TPgain_recovered']:.3f} "
        f"FPadd_carried={row['FPadd_carried']:.3f} FP_out_U={row['FP_outside_U']} FN_out_U={row['FN_outside_U']}")


preds = {'Teacher_self': tq['z0'], 'Teacher_dual': tq['z34']}
ev('Teacher_self', tq['z0'])
ev('Teacher_dual', tq['z34'])
for ckpt, variant, name in (('R1_seed42.pt', 'R1', 'Ours-R1'), ('R2_seed42.pt', 'R2', 'Ours-R2'),
                            ('R3_seed42.pt', 'R3', 'Ours-R3'), ('R3g_seed42.pt', 'R3', 'Ours-R3g'),
                            ('DIS2_seed42.pt', 'R1', 'DIS2-port')):
    m = v2.OursV3Student(variant).to(DEV)
    m.load_state_dict(torch.load(OUT / ckpt, map_location=DEV, weights_only=True))
    q = v2.predict_student(m, val_set)
    preds[name] = q['p']
    ev(name, q['p'])
    del m
# Keep the raw maps so the region analysis can be re-derived without re-running
# any model.
np.savez_compressed(OUT / 'stage1_counter_gain_val_preds.npz', y=tq['y'], gt=tq['gt'], gc=tq['gc'],
                    orbit=tq['orbit'].astype('U8'), **preds)

pd.DataFrame(rows_a).to_csv(OUT / 'stage1_factcheck_loader.csv', index=False)
pd.DataFrame(rows_e).to_csv(OUT / 'stage1_counter_gain_val.csv', index=False)
(OUT / 'stage1_factcheck.json').write_text(json.dumps(res, indent=2, default=str))
say(f'\nwrote {OUT / "stage1_factcheck.json"}')
