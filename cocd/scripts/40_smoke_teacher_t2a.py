#!/usr/bin/env python3
"""Checks for the Teacher-T2a change: the reweighting, the curve, and the plumbing.

Nothing here trains.  What it has to establish before a 4-hour run is started:

  [A] the reweighted objective equals what it claims -- 0.25/0.25/0.50 per target
      direction, total 1.0, so the gradient scale is untouched;
  [B] the refactored expression reproduces the *frozen* objective bit for bit
      when the frozen weights are put back, so the only difference between the
      old Teacher and T2a is where the supervision sits;
  [C] the three-head curve pass reproduces the frozen Teacher's known validation
      numbers, and its taps agree with the deployed forward path;
  [D] the curve row carries every quantity the stage is required to record;
  [E] a tagged Teacher run reads and writes tagged files, and cannot overwrite
      the frozen ``teacher.pt`` that every reported arm distils from;
  [F] the test half writes the same grouped table shape as the student arms.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

A = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ours_v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

PASS = []


def check(name, ok, detail=''):
    PASS.append((name, bool(ok)))
    print(f'[{"ok " if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(f'failed: {name}')


def original_teacher_batch(model, a, d, y, weight):
    """The frozen objective, transcribed from the pre-T2a script."""
    outputs_a, outputs_d = model.forward_pair(a, d)
    loss = sum(.5 * v2.seg(x, y, weight) for x in (outputs_a[0], outputs_d[0]))
    loss += sum(.25 * v2.seg(x, y, weight) for x in (outputs_a[1], outputs_a[2], outputs_d[1], outputs_d[2]))
    return .5 * loss


def main():
    print('T2a weights:', v2.TEACHER_W, flush=True)
    check('[A] T2a weights are (0.25, 0.25, 0.50)', tuple(v2.TEACHER_W) == (0.25, 0.25, 0.50))
    check('[A] total weight is 1.0', abs(sum(v2.TEACHER_W) - 1.0) < 1e-12)
    check('[A] z34 carries twice z0', abs(v2.TEACHER_W[2] - 2 * v2.TEACHER_W[0]) < 1e-12)

    tid, vid, eid = v2.split()
    train_set = v2.HaitiPairs(tid, True)
    weight = v2.posweight(train_set)
    a, d, y, ga, gd, real = next(iter(v2.loader(train_set, False)))
    a, d, y = a.to(v2.DEV), d.to(v2.DEV), y.to(v2.DEV)

    model = v2.OursV2Teacher().to(v2.DEV)
    model.load_s0(torch.load(A / 'experiments' / 'rapid_landslide_cocd' / 'S0.pt',
                             map_location=v2.DEV, weights_only=True))
    # The backbone carries drop-path, so two forwards only agree in eval mode.
    # The loss checks below compare two evaluations of the objective and would
    # otherwise differ by the randomisation rather than by the weights.
    model.eval()

    # --- [A] the loss is the stated weighted sum --------------------------------
    outs = [o.detach() for pair in model.forward_pair(a, d) for o in pair[:3]]
    expected = .5 * sum(w * v2.seg(outs[3 * i + j], y, weight)
                        for i in range(2) for j, w in enumerate(v2.TEACHER_W))
    got = v2.teacher_batch(model, a, d, y, weight)
    check('[A] loss equals the stated weighted sum', torch.allclose(got.detach(), expected, atol=0, rtol=0),
          f'{got.detach().item():.8f} vs {expected.detach().item():.8f}')
    check('[A] loss carries grad', got.requires_grad)

    # --- [B] frozen weights reproduce the frozen objective exactly --------------
    v2.TEACHER_W = (0.5, 0.25, 0.25)
    t2a_expr = v2.teacher_batch(model, a, d, y, weight)
    frozen_expr = original_teacher_batch(model, a, d, y, weight)
    check('[B] refactor is bit-identical to the frozen objective',
          torch.equal(t2a_expr.detach(), frozen_expr.detach()),
          f'{t2a_expr.detach().item():.10f} vs {frozen_expr.detach().item():.10f}')
    v2.TEACHER_W = (0.25, 0.25, 0.50)

    # --- [C] the curve pass reproduces the frozen Teacher -----------------------
    teacher = v2.load_teacher()
    val_set = v2.HaitiPairs(vid, False, train_set.stats)
    q = v2.predict_teacher_heads(teacher, val_set)
    check('[C] every head and mask is present',
          all(k in q for k in ('z0', 'z4', 'z34', 'y', 'gt', 'gc', 'orbit')))
    check('[C] taps agree with the deployed forward path', q['tap_path_max_abs_diff'] == 0.0,
          f"max|d|={q['tap_path_max_abs_diff']:.1e}")
    ms, md = v2.metrics(q['z0'], q['y']), v2.metrics(q['z34'], q['y'])
    check('[C] frozen Teacher self IoU 0.6057', abs(ms['iou'] - 0.6057) < 5e-4, f"{ms['iou']:.4f}")
    check('[C] frozen Teacher dual IoU 0.6300', abs(md['iou'] - 0.6300) < 5e-4, f"{md['iou']:.4f}")
    check('[C] frozen Teacher self AUPRC 0.8809', abs(ms['auprc'] - 0.8809) < 5e-4, f"{ms['auprc']:.4f}")
    check('[C] frozen Teacher dual AUPRC 0.89626', abs(md['auprc'] - 0.89626) < 5e-4, f"{md['auprc']:.5f}")

    # --- [D] the curve row carries every required quantity ----------------------
    row = v2.teacher_curve_row(teacher, val_set, 50)
    for head in ('z0', 'z4', 'z34'):
        for metric in ('iou', 'f1', 'auprc'):
            check(f'[D] {head}_{metric} recorded', f'{head}_{metric}' in row and np.isfinite(row[f'{head}_{metric}']))
    for key in ('gain_auprc_z34_minus_z0', 'p3_corr_norm_mean', 'p4_corr_norm_mean',
                'p3_corr_norm_max', 'p4_corr_norm_max', 'tap_path_max_abs_diff'):
        check(f'[D] {key} recorded', key in row and np.isfinite(row[key]), f'{row[key]}')
    check('[D] gain is z34 minus z0', abs(row['gain_auprc_z34_minus_z0'] -
                                          (row['z34_auprc'] - row['z0_auprc'])) < 1e-12)
    check('[D] gains reported per region',
          all(f'gain_auprc_{r}' in row for r in ('G00', 'G01', 'G10', 'G11')))
    check('[D] correction norms are non-zero', row['p3_corr_norm_mean'] > 0 and row['p4_corr_norm_mean'] > 0,
          f"P3={row['p3_corr_norm_mean']:.4f} P4={row['p4_corr_norm_mean']:.4f}")

    # --- [E] a tagged run keeps its own files ----------------------------------
    import inspect
    src = (A / 'scripts' / '23_train_ours_v2.py').read_text()
    check('[E] train_teacher is parameterised by tag', 'tag' in inspect.signature(v2.train_teacher).parameters)
    check('[E] train_teacher is parameterised by early_stop',
          'early_stop' in inspect.signature(v2.train_teacher).parameters
          and inspect.signature(v2.train_teacher).parameters['early_stop'].default is True)
    check('[E] load_teacher is parameterised by tag', 'tag' in inspect.signature(v2.load_teacher).parameters)
    check('[E] the Teacher writes only tagged checkpoint paths',
          "f'teacher{tag}_latest.pt'" in src and "OUT / 'teacher_latest.pt'" not in src
          and "OUT / 'teacher_progress.json'" not in src)
    check('[E] the Teacher writes only tagged result paths',
          "f'teacher{tag}.pt'" in src and "OUT / 'teacher.pt'))" not in src.split('def load_teacher')[0])
    frozen = A / 'experiments' / 'ours_v2' / 'teacher.pt'
    check('[E] the frozen teacher.pt is present', frozen.exists())
    check('[E] label derives from the tag', v2.teacher_label('_T2a') == 'Teacher-T2a'
          and v2.teacher_label('') == 'Teacher')

    # --- [F] the test half emits the student table's shape ---------------------
    head = v2.predict_teacher_heads(teacher, torch.utils.data.Subset(val_set, range(0, 20)))
    rows = v2.report('Teacher-T2a_dual', v2.teacher_pred('z34', head))
    cols = {'method', 'partition', 'region', 'pixels', 'iou', 'f1', 'precision', 'recall', 'auprc'}
    check('[F] report rows have the student table schema', cols <= set(rows[0]))
    check('[F] all four regions are emitted', {'G00', 'G01', 'G10', 'G11'} <= {r['region'] for r in rows})
    check('[F] partitions are emitted', {'overall', 'asc', 'desc'} == {r['partition'] for r in rows})

    print(f'\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed', flush=True)


if __name__ == '__main__':
    main()
