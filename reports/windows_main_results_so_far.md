# Windows formal run set — results so far

Status: **validation only, seed 42, no test evaluation has been performed on any
Windows artefact.** Every number below is best-validation, threshold-free AUPRC
unless a column says otherwise, under the unified protocol
(`cocd/configs/protocol_windows_main.json`): Adam, lr 5e-5 constant, batch 16,
plain BCE, 20 epochs, patience 3, best checkpoint = val AUPRC, val-selected max-F1
threshold then frozen, ImageNet init, raw / no mask, geometry only for G-regions.

## The formal methods, overall partition

| # | run | what it is | lam | best val AUPRC | vs N0 | vs N1 | IoU@thr | F1@thr | P | R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | N0 | single-orbit baseline, no complement branch | 0 | 0.64428 | — | −0.0062 | 0.32780 | 0.49375 | 0.73615 | 0.37144 |
| 2 | VKD | Vanilla KD student | — | 0.60333 | −0.0410 | −0.0471 | 0.36870 | 0.53876 | 0.61977 | 0.47648 |
| 3 | DIS2 | DIS2-port student | — | 0.59898 | −0.0453 | −0.0515 | 0.27743 | 0.43436 | 0.71497 | 0.31194 |
| 4 | R3 | legacy R3 student | — | 0.61073 | −0.0336 | −0.0397 | 0.36279 | 0.53242 | 0.65087 | 0.45044 |
| 5 | **N1** | complement predictor, BCE only | 0 | **0.65045** | **+0.0062** | — | 0.32310 | 0.48840 | 0.74490 | 0.36330 |
| 6 | N2 | full COCD, complement distillation | 1.0 | 0.55830 | −0.0860 | −0.0922 | 0.25048 | 0.40062 | 0.69685 | 0.28111 |
| 7 | N2-no-orbit | N2 without orbit embedding | 1.0 | 0.55575 | −0.0885 | −0.0947 | 0.27773 | 0.43473 | 0.65666 | 0.32491 |

## Teachers (not ranked against the students; they are the sources)

| run | reading | best val AUPRC |
|---|---|---:|
| Legacy teacher (re-trained, z-weights 0.25/0.25/0.50) | z0 self | 0.62766 |
| Legacy teacher | z4 | 0.64957 |
| Legacy teacher | dual (z34) | 0.66635 |
| New COCD teacher | z_self | 0.64594 |
| **New COCD teacher** | **z_dual** | **0.68967** |

Teacher audit passed: `z_self` identical under all counter conditions (0.000e+00
difference, so the single-point injection holds), dual correct 0.68967 vs zero
counter exactly 0.64594 (= self), and correct beat all 5 fixed shuffles
(0.309-0.351), margin +0.33851 over the best shuffle.

## Diagnostic run, outside the protocol

| run | lam | best val AUPRC | vs N0 | vs N1 | share of seg @20 |
|---|---:|---:|---:|---:|---:|
| N2 diagnostic | 0.1 | 0.63136 | −0.0129 | −0.0191 | 0.188 |
| N2 as specified | 1.0 | 0.55830 | −0.0860 | −0.0922 | 0.677 |

Registered under `diagnostic_runs` in the protocol file. Does not enter the table
above; `lam = 1.0` remains the frozen protocol value.

## By region and orbit partition (AUPRC)

| arm | part | all | G00 | G01 | G10 | G11 |
|---|---|---:|---:|---:|---:|---:|
| N0 | overall | 0.6443 | 0.6373 | 0.6680 | 0.6439 | 0.6743 |
| N0 | asc | 0.6391 | 0.6353 | 0.6245 | 0.6546 | 0.6706 |
| N0 | desc | 0.6495 | 0.6394 | 0.6882 | 0.6217 | 0.6790 |
| N1 | overall | 0.6504 | 0.6425 | 0.6761 | 0.6541 | 0.6682 |
| N1 | asc | 0.6465 | 0.6405 | 0.6437 | 0.6622 | 0.6844 |
| N1 | desc | 0.6543 | 0.6443 | 0.6914 | 0.6370 | 0.6590 |
| N2 | overall | 0.5583 | 0.5472 | 0.6034 | 0.5542 | 0.5778 |
| N2 | asc | 0.5577 | 0.5507 | 0.5628 | 0.5756 | 0.5695 |
| N2 | desc | 0.5590 | 0.5438 | 0.6210 | 0.5077 | 0.5896 |
| N2 lam 0.1 | overall | 0.6314 | 0.6244 | 0.6598 | 0.6314 | 0.6438 |
| N2-no-orbit | overall | 0.5558 | 0.5422 | 0.6003 | 0.5591 | 0.5867 |
| VKD | overall | 0.6033 | 0.5968 | 0.6277 | 0.5961 | 0.6446 |
| DIS2 | overall | 0.5990 | 0.5901 | 0.6305 | 0.6004 | 0.6385 |
| R3 | overall | 0.6107 | 0.6023 | 0.6433 | 0.6041 | 0.6535 |

## What the numbers say

**One arm above the single-orbit baseline: N1.** It is +0.0062 over N0, which is
smaller than the ±0.01 epoch-to-epoch platform noise measured in the Phase 8
audit. Treat it as "at or slightly above N0", not as a demonstrated gain.

**All three distillation controls land below their own student baseline**
(VKD 0.6033, DIS2 0.5990, R3 0.6107 against N0 0.6443). Their training logs show
the same pattern: segmentation settles at 0.036-0.041 while the KD term is still
0.021 / 0.007 / 0.003, so the auxiliary term dominates the late objective. This is
the legacy KD being a weak control, matching what
`reports/protocol_v2_unification.md` section 7 predicted.

**The internal chain N0 -> N1 -> N2 is broken at N1 -> N2.** The complement branch
helps a little; the complement distillation term costs 0.0922. The one diagnostic
run at lam = 0.1 recovers 0.0731 of that, which attributes most of the damage to an
over-large weight, but it still ends 0.0191 below N1. So changing the weight is
necessary and not sufficient: the residual points at raw `C_T` being a poor
matching target (it lives in the teacher's separately trained feature space and
carries counter-orbit information the student never sees).

**Notable divergence between AUPRC and the thresholded scores.** VKD has the
lowest AUPRC among the controls (0.6033) but the best IoU (0.3687) and F1 (0.5388)
in the whole set. Ranking arms by AUPRC and ranking them by F1@thr give different
orders, so the reported table should carry both and say which is threshold-free.

**Uniform pattern across every arm:** all four region groups score above the
overall average, because the G-regions are geometry-defined and the `all` partition
also carries the geometry-free background. G01 and G11 are consistently the easy
groups, G00 the hard one. The core group G10 only moves meaningfully in the
teacher: new teacher +0.0522, legacy teacher +0.0497 on the dual reading.

## Shipped Mac reference (not part of the Windows run set)

`results/ours_v2/val_report_v2.csv`, legacy teacher validation: self 0.63043,
p4 0.65121, dual 0.66708. The re-trained Windows teacher gives 0.62766 / 0.64957 /
0.66635, agreeing within 0.0007, which is what closed conflict 1: the 0.25 / 0.25 /
0.50 z-weights are what the shipped code actually used, and the docstring was stale.

## Still open

- Seeds 123 and 2026 not run. A per-seed legacy tag is needed first, otherwise
  re-training the teacher overwrites the seed-42 teacher.
- Test partition not evaluated anywhere.
- Whether N2 is reported as a negative result, or the knowledge object is changed,
  is a method decision and is not made by this document.
