# Experiment summary and evidence map

The consolidated experiment data lives under `experiments/`. Use these files as the current entry points:

- `experiments/summary/MASTER_METRICS.csv` — one row per run, method, and evaluation region.
- `experiments/summary/MASTER_RUN_CATALOG.csv` — protocol, checkpoints, metrics, and artifact counts for retained runs.
- `experiments/summary/ALL_FIGURE_INVENTORY.csv` — paths to current, historical, and quarantined visual artifacts.
- `experiments/summary/figures/` — consolidated method-comparison plots.
- `experiments/GRSL_complete_paper_package/04_tables/` — current paper tables and full metric exports.

## Current raw benchmark

The current paper package reports S2-2Stage against S0-Raw on the fixed test set. In the source table, S2-2Stage improves AUPRC by about 0.0117 overall, 0.0121 on TargetValid, and 0.0125 on BothValid. This is test-monitored development evidence under the project protocol; it should not be presented as an independent test estimate.

## Geometry-masked evidence

The legacy validation-split experiment (`windows_main/geometry_transfer_restored_v1`, train/validation/test = 1233/137/343) shows a clear masked Teacher gain on valid pixels: BothValid CG−SG is +0.01956 AUPRC, and CG−counter-shuffle is +0.08417. Student Search-KD is mixed: BothValid AUPRC is 0.00092 below S0, while IoU and F1 are slightly higher.

The full-train masked runs (`mask_aware_search_kd_v5_seed42` and `sgri_powc_masked_v4_seed42_20260925`, 1370/0/343) are kept separately. In both, S2 is below its matched S0 on BothValid AUPRC. Their test set was used by the recorded monitoring/selection rule. The teacher rows in the v4 package reuse the earlier validation-selected geometry-masked checkpoints. These protocols must not be pooled with the legacy validation-split experiment.

## Repeated and matched-budget runs

All completed 10% KD controller runs are retained independently under `experiments/windows_main/multitoone_kd_*`; the initial EMA, batch-ratio EMA, and pooled-loss EMA variants are distinct controller protocols. Their metrics and per-epoch KD budget curves remain in each run folder and are indexed in the master catalog. No averages across these runs are implied.

## Geometry-region definitions

- `BothValid`: target geometry code = 0 and counter geometry code = 0.
- `TargetValid`: target geometry code = 0.
- `OverallValid`: at least one orbit is geometry-valid; this includes the G10 reference region.
- `G10`: target invalid, counter valid. Diagnostic only.
- `G01`: target valid, counter invalid. Diagnostic only.

Geometry codes are stored separately for layover, shadow, and their overlap. The authoritative data audit and human-review forms are in `experiments/dataset_audit_final/`.

## Interpretation and provenance

Raw/no-mask, masked, and validation-split runs are explicitly tagged in `MASTER_METRICS.csv`. Results from different selection protocols are evidence records, not directly matched comparisons. Completed, non-duplicate runs are preserved; superseded and temporary files are isolated under `experiments/00_curation/superseded/` and recorded in the curation log.
