> Historical experiment catalogue/package. The final paper authority is `paper_release/README.md` at the repository root. Retained numbers are not automatically same-protocol eligible.

# Experiment data hub

This folder is the single portable home for Haiti COCD experiment data. Copy `experiments/` to move metrics, checkpoints, logs, figures, packages, and dataset-audit evidence together. Source code and the raw `data/` tree remain at the project root.

## Start here

- Final paper-ready package: [portable paper package](paper_submission_package_20260929/README.md)
- Master run/protocol catalog: [MASTER_RUN_CATALOG.csv](summary/MASTER_RUN_CATALOG.csv)
- Unified long-form metrics (one row per run × method × region): [MASTER_METRICS.csv](summary/MASTER_METRICS.csv)
- Complete image/PDF/PPTX inventory: [ALL_FIGURE_INVENTORY.csv](summary/ALL_FIGURE_INVENTORY.csv)
- Comparison charts: [summary/figures](summary/figures/)
- Dataset audit package: [dataset_audit_final/README.md](dataset_audit_final/README.md)
- External KD implementations, runs and source notes: [distillation_final](distillation_final/)
- Current and historical checkpoints: [checkpoints](checkpoints/) and [legacy_checkpoints](legacy_checkpoints/)
- Cleanup/integrity records: [curation README](00_curation/README.md)

## How to read comparisons

`MASTER_METRICS.csv` keeps Raw/no-mask and geometry-masked runs in separate `TrainingRegime` and `RunID` fields. `Overall`, `OverallValid`, `TargetValid`, `BothValid`, `G10`, and `G01` remain distinct; do not infer one from another. Protocol variants and repeated completed runs are separate rows and were not averaged. The historical 1233/137/343 geometry-transfer split, the active full-train 1370/0/343 masked runs, and raw final benchmarks use different selection/training conditions; compare within the matching protocol family.

## Key comparison charts

- [Raw final method comparison](summary/figures/raw_final_method_comparison.png)
- [Raw vs masked S0/S2](summary/figures/raw_vs_masked_s0_s2.png)
- [Masked Teacher counter-view test](summary/figures/masked_teacher_counter_view.png)
- [All masked full-train v4 methods](summary/figures/masked_v4_all_methods.png)
- [Masked legacy Student mechanisms](summary/figures/masked_legacy_student_mechanisms.png)
- [Completed equal-KD controller runs](summary/figures/equal_kd_completed_runs.png)
- [Equal-KD epoch contribution curves](summary/figures/equal_kd_epoch_curves.png)

Each chart is saved as PNG and vector PDF under `summary/figures/`. Existing method, qualitative, search-behavior, training-curve, and audit figures remain in their owning run/package folders and are indexed in `ALL_FIGURE_INVENTORY.csv`.

## Retention and cleanup

Completed runs with distinct protocols and repeated valid runs are retained. Fixed-30 diagnostics, superseded loss-controller pilots, aborted calibration artefacts, duplicate files, and rendering caches are isolated under `00_curation/superseded/`; they are not part of the current comparison package. Their hashes and paths are recorded in the curation manifests. Files isolated there were moved rather than deleted because the execution policy rejected file deletion.
