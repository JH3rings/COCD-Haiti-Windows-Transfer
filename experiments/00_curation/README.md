# Experiment organization and cleanup record — 2026-09-29

The sole portable experiment-data directory is `experiments/`. The paper-facing entry point is `../paper_submission_package_20260929/README.md`; model checkpoints and full run cards remain in their named run folders inside this directory.

## Current retained results

- Current own-method runs: `windows_main/raw_cver_v1_seed42/`, `windows_main/teacher_target_search_v2/`, `windows_main/student_target_search_v2/`, and `distillation_final/ours_s2_2stage/`.
- External comparisons: `distillation_final/`, `single_orbit_baselines_v3/`, and the MMOKD run under `GRSL_complete_paper_package/baselines/mmokd_port/`.
- Geometry-mask and equal-KD diagnostics retain their metric/report/progress files and figures. They are separated by original protocol and are not pooled into the main table.
- Dataset integrity and geometry audit: `dataset_audit_final/`.

## Cleanup audit

- `purge_manifest_20260929.csv` records each proposed project-local deletion, its pre-deletion size, SHA256, category, and reason.
- `purge_skipped_external_links_20260929.csv` records paths that resolve outside `experiments/`; external runtime files were excluded.
- The bulk deletion command was rejected by the automatic approval policy before execution. No candidate in the new purge manifest was removed; no execution log exists. The old outputs and interim packages still remain on disk.
- `removed_files_sha256.csv` and `cleanup_execution_log.csv` describe an earlier quarantine move. Their files still exist under `superseded/`; the new deletion attempt did not alter them.
- Source code under `cocd/`, raw `data/`, the authoritative `dataset_audit_final/`, final own-method checkpoints, external baseline checkpoints, and the current geometry diagnostic metrics were not cleanup targets.

## Portable package

Copy the whole `experiments/` directory to carry metrics, figures, run cards, checkpoints, and dataset-audit evidence together. For paper tables and figures alone, use `paper_submission_package_20260929/`. The older `GRSL_complete_paper_package/` and intermediate `paper_submission_20260929*` folders are superseded; only the dated `paper_submission_package_20260929/` directory is the current paper entry point.
