# Haiti COCD paper submission package — 2026-09-29

## Start here

This is the new paper-facing package. Move the entire project `experiments/` folder to preserve run checkpoints and source outputs; this folder is the compact paper bundle with tables, plots, protocol cards, and links back to retained runs. Nothing here requires re-running training.

## Contents

- `01_results/`: current main comparison, all-region metrics, per-method metric cross-checks, exact source-table snapshots, and saved convergence curves.
- `02_figures/`: generated manuscript comparisons and previously generated project charts (PNG and vector PDF), including Raw/masked diagnostics, Teacher counter shuffle, matched KD budget, training curves, and geometry audit panels.
- `03_qualitative/`: the existing candidate contact sheet and case panels. These are automatically selected candidates; final human case selection is still blank.
- `04_geometry_mask/`: full mask/no-mask diagnostic rows, each retaining its original training regime and split description. The old validation-split experiment and the full-train v4 experiment are not pooled.
- `05_diagnostics/`: compact teacher/search, TGD, and equal-KD reports/curves; equal-KD controllers remain separate.
- `06_geometry_audit/`: authoritative geometry-code, joint-region, patch statistics, and audit plots. Codes are 0=valid, 1=layover, 2=shadow, 3=layover+shadow.
- `07_provenance/`: current GRSL tables/specs, compact run cards, external-method notes, and source outputs.
- `00_manifest/`: paths to retained checkpoints/runs and snapshot of the full run catalog.

The qualitative source prediction cache used by the current candidate panels is retained in `../GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz`.

## Recommended paper table

Use `01_results/paper_main_comparison_all_regions.csv`. It contains the 7 current GRSL core rows plus 5 single-orbit external baselines, with Overall, BothValid, TargetValid, G10 and G01 AUPRC/IoU/F1. External single-orbit baseline region metrics were re-evaluated from saved test predictions using the project metric definition; see `01_results/external_baseline_metric_crosscheck.csv`. The matching complete source predictions remain under `../single_orbit_baselines_v3/`.

`S2-Raw-native` is explicitly marked historical because its native KD query weighting uses geometry-derived G10/G01 information. Do not fold it into the geometry-free S2-2Stage claim. DIS2-Port remains in the table despite its low score. Multi2Uni-Port and MMOKD-Port are mechanism-faithful ports, not official numerical reproductions.

## Protocol limits

The locked project protocol has no validation split and uses fixed-test Overall AUPRC as the per-epoch checkpoint monitor. These are test-monitored development results, not independent test estimates. Raw/no-mask and masked-training rows have distinct protocols; geometry is post-hoc for the raw main benchmark. `G10` and `G01` are diagnostic partitions.

## Package integrity

`00_manifest/package_sha256.csv` lists every asset in this folder except the checksum file itself. The build script is under `scripts/`; it only copies existing artifacts and re-evaluates saved baseline predictions, and does not train models.
