# Haiti SAR GRSL assets v1

This is a read-only paper-material package built from the locked local Haiti
SAR experiment. It does not retrain, fine-tune, redraw splits, change losses,
change thresholds, or overwrite checkpoints.

The active current chain is `T0 -> T1 -> S0`. `S0` is the only active
target-only Student. `SO-reference` is a separate ordinary target-only
reference, not another Student. G10 is retained internally and mapped to the
paper term **TGD** (target-only geometric distortion regions); Remaining is its
valid-domain complement.

Start here:

- [`paper_summary.md`](paper_summary.md) — concise evidence and boundaries.
- [`missing_items.md`](missing_items.md) — items not available without new work.
- [`data/model_registry.csv`](data/model_registry.csv) — checkpoint/result provenance.
- [`tables/table1_deployment.md`](tables/table1_deployment.md) — single-orbit deployment table.
- [`tables/table2_existing_components.md`](tables/table2_existing_components.md) — stage/transfer comparison.
- [`tables/table3_tgd_mechanism.md`](tables/table3_tgd_mechanism.md) — counter intervention and fixed diagnostic subset.
- [`preview/index.html`](preview/index.html) — figures and table index.

Main figures are in `figures/main/` as PNG, PDF and SVG. Candidate PR,
training and DA behavior figures are in `figures/candidates/`. Re-drawable
prediction caches and PR arrays are in `figure_data/`; they are intentionally
not included in the review zip. The review zip excludes checkpoints, full
SAR data, large prediction caches and fonts.

The reproducible entry point is
`scripts/run_assets.py`. It calls only evaluation/statistics/plotting code and
uses the existing local checkpoints and prediction caches. The run records
the environment and source hashes in `audit_report.json` and
`source_manifest.json`.

The locked training protocol uses the merged 1,370-location train split,
343-location test split, batch 16, Adam at 5e-5, max 100 epochs, patience 2,
threshold 0.5, and test AUPRC for development monitoring/checkpoint selection.
Consequently the reported numbers are development-monitor evidence, not an
untouched held-out estimate.
