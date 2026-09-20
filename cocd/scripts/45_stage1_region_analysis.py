#!/usr/bin/env python3
"""Region-wise reading of the counter-orbit effect, from the saved raw maps.

Reads experiments/ours_v2/stage1_counter_gain_val_preds.npz (written by
scripts/44) so no model is run again.  Answers, on validation:

  * region sizes and each model's IoU / AUPRC inside G00..G11;
  * where the Teacher's dual track beats its own self track, region by region;
  * how much of that the students carry, and whether they pay for it with extra
    false positives elsewhere.

Writes experiments/ours_v2/stage1_region_val.csv and prints the table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

A = Path(__file__).resolve().parents[1]
OUT = A / 'experiments' / 'ours_v2'
d = np.load(OUT / 'stage1_counter_gain_val_preds.npz', allow_pickle=False)
y = d['y'] > 0
gt, gc = d['gt'] > 0, d['gc'] > 0
models = [k for k in d.files if k not in ('y', 'gt', 'gc', 'orbit')]
REGIONS = {'G00': ~gt & ~gc, 'G01': ~gt & gc, 'G10': gt & ~gc, 'G11': gt & gc}


def metrics(p, yy):
    h = p >= .5
    tp, fp, fn = (h & yy).sum(), (h & ~yy).sum(), (~h & yy).sum()
    pr = tp / (tp + fp + 1e-8)
    re = tp / (tp + fn + 1e-8)
    return (tp / (tp + fp + fn + 1e-8), 2 * pr * re / (pr + re + 1e-8),
            float(average_precision_score(yy, p)) if yy.any() and (~yy).any() else float('nan'))


rows = []
for region, mask in REGIONS.items():
    yv = y.ravel()[mask.ravel()]
    for name in models:
        iou, f1, ap = metrics(d[name].ravel()[mask.ravel()], yv)
        rows.append({'region': region, 'pixels': int(mask.sum()), 'positive_pixels': int(yv.sum()),
                     'model': name, 'iou': iou, 'f1': f1, 'auprc': ap})
region_all = np.ones_like(y)
for name in models:
    iou, f1, ap = metrics(d[name].ravel(), y.ravel())
    rows.append({'region': 'all', 'pixels': int(y.size), 'positive_pixels': int(y.sum()),
                 'model': name, 'iou': iou, 'f1': f1, 'auprc': ap})
frame = pd.DataFrame(rows)
frame.to_csv(OUT / 'stage1_region_val.csv', index=False)

wide = frame.pivot_table(index=['region', 'pixels', 'positive_pixels'], columns='model',
                         values=['iou', 'auprc'])
pd.set_option('display.width', 220, 'display.max_columns', 40)
print('=== IoU / AUPRC per region (validation) ===')
print(wide.round(4).to_string())

print('\n=== Teacher dual-vs-self delta per region ===')
for region, mask in list(REGIONS.items()) + [('all', region_all)]:
    sel = mask.ravel()
    a, b = d['Teacher_self'].ravel()[sel], d['Teacher_dual'].ravel()[sel]
    yv = y.ravel()[sel]
    ia, _, apa = metrics(a, yv)
    ib, _, apb = metrics(b, yv)
    ha, hb = a >= .5, b >= .5
    fp_cut = int((ha & ~hb & ~yv).sum())
    fp_add = int((~ha & hb & ~yv).sum())
    tp_gain = int((~ha & hb & yv).sum())
    tp_loss = int((ha & ~hb & yv).sum())
    print(f'  {region:4s} px={int(mask.sum()):>9,} pos={int(yv.sum()):>7,} '
          f'dIoU={ib - ia:+.4f} dAUPRC={apb - apa:+.4f}  '
          f'TPgain={tp_gain:>6,} TPloss={tp_loss:>5,} FPcut={fp_cut:>6,} FPadd={fp_add:>6,}')
