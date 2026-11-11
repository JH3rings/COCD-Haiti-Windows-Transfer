# Optical-guided single-orbit SAR landslide mapping

Optical imagery is used **only during training** to help a single-orbit SAR
U-Net organise its own features. The deployed model takes only SAR and terrain
as input.

The earlier dual-orbit method ("counter-guided target evidence search",
Haiti) is no longer the active method on this branch. Its code and paper
remain in git history and on the `archive/*` and `paper/grsl-final-freeze`
branches.

## Problem

Landslide inventories are grouped by disaster **event**, and a SAR network
learns to recognise events instead of landslides. On the DLR benchmark the
plain U-Net reaches 0.909 IoU on validation patches from the training events,
but only 54.6 on new events.

Standard distillation from an optical teacher does not fix this:

- **Output distillation** carries nothing beyond the labels, because the
  teacher memorises them.
- **Feature and frequency distillation** ask SAR to imitate optical content,
  and most of that content is invisible to SAR.

What does transfer is the teacher's **relations between regions of different
events**.

## Method

| Part | Where | What it does |
|---|---|---|
| Event-shared concept bank (EPRM, structure) | After U-Net decoder stage 2 (32×32, 256 ch); kept at deployment | 16 learnable prototypes shared by all events. Each pixel compares itself with the prototypes and is refined, `f ← f + O(softmax(cos(q(f), P)/τ) · V(P))`. The output projection is zero-initialised, so training starts exactly at the plain U-Net. |
| Cross-event correspondence distillation (loss) | Same stage, training only | Region tokens (4×4 per patch, pooled over the batch). For each region, the frozen optical+SAR teacher gives a soft distribution over **all regions of other events** ("which regions in other disasters are my kind"). The SAR student's concept-refined regions give the same distribution, and the two are aligned with a KL divergence. Regions of the same event are never compared. |

Implementation: [`pr.py`](experiments/dlr_official6_faithful_20261110/pr.py)

- arm `EPRM_CC`: the correspondence loss described above;
- arm `EPRM`: structure-only ablation;
- arm `EPRM_CEraw`: the earlier variant that uses an optimal-transport relation term.

Status (2026-11-11): the correspondence form (`EPRM_CC`) did **not** work,
scoring 52.48. A soft attraction to the teacher's most similar cross-event
regions mostly re-states the labels. The working ingredient is cross-event
landslide-vs-background **separation**, so the loss is being redesigned
around it. See section 16 of the log.

The experiment log and every diagnostic (in Chinese) are in
[`ANALYSIS_ZH.md`](experiments/dlr_official6_faithful_20261110/ANALYSIS_ZH.md).

## Setting

- **Data:** the official DLR split, 355 / 88 / 120 patches.
- **Inputs:**
  - Student: PRE1/POST1 VV/VH, DEM and slope (6 channels).
  - Teacher: the same 6 channels plus 12 Sentinel-2 bands.
- **Model:** the DLR paper U-Net.
- **Training:**
  - BCE + soft Dice (1:1), AdamW at 1e-4, 200 epochs, dihedral augmentation;
  - validation every 20 epochs, and validation IoU selects the checkpoint;
  - each arm is tested once, after it is frozen.

## Results (test IoU %, seed 42)

Teacher: 70.38. Three-seed means: B0 54.60 ± 2.10, PKDOT 56.19 ± 0.47.

| Arm | all120 | ind44 | spt76 | AUPRC |
|---|---:|---:|---:|---:|
| B0 (SAR U-Net) | 52.20 | 63.24 | 46.03 | 72.91 |
| KD / AT / DIS2 | 53.45 / 53.12 / 55.97 | | | |
| PKDOT | 55.74 | 70.52 | 47.80 | 74.75 |
| Label-only relations (no optical) | 54.09 | 65.78 | 47.62 | 74.96 |
| Cross-event relations only | 56.27 | 71.29 | 48.19 | 77.24 |
| Concept bank only (no optical) | 54.78 | 64.61 | 49.21 | 74.94 |
| Concept bank + cross-event OT relations (`EPRM_CEraw`) | 56.68 | 69.01 | 50.14 | 76.83 |
| Concept bank + cross-event correspondence KL (`EPRM_CC`) | 52.48 | 65.66 | 45.22 | 72.43 |

The results are from one seed. Because the B0 seed spread is 2.1 points,
differences smaller than that are not yet conclusive.

## Run

The data are the DLR S1/S2 reference HDF5 files listed in
`experiments/dlr_unet_study_20261104/split_manifest.csv` (`h5_path`). They are
not included in this repository.

```powershell
pip install -r requirements.txt
cd experiments/dlr_official6_faithful_20261110
python run.py --arms B0 T_fusion        # SAR baseline and the optical teacher
python pr.py --check                    # identity-at-init and gradient checks
python pr.py --arms EPRM_CC             # our method (needs the frozen T_fusion)
python pr.py --test EPRM EPRM_CC        # one test after freezing
```
