# Retained GRSL result sources and MMOKD run

The new manuscript entry point is [`../paper_submission_package_20260929/README.md`](../paper_submission_package_20260929/README.md). This folder remains because it holds the MMOKD-Port checkpoint, its run card, and the source artifacts used to assemble the paper materials. Superseded dated package copies and obsolete run outputs are still on disk: the requested bulk cleanup was blocked by the app's automatic policy before execution, so no candidate files were removed. The current cleanup inventory is `../00_curation/purge_manifest_20260929.csv`. The project-wide metrics catalog and raw/masked comparison charts are linked from [`../README.md`](../README.md).

- `01_method/fig1_method.svg` and `fig1_method.png`: editable method figure source and rendered preview; counter is restricted to the teacher policy and the decoder reads target-side retrieved evidence.
- `02_qualitative/`: automatic test-set candidates and the contact sheet; human final case choices are recorded separately and are not inferred from ranking.
- `03_search_behavior/`: compact search-policy and retrieval diagnostics.
- `04_tables/`: paper tables and the complete metric export.
- `05_analysis/`: region gains, rescue counts, mechanism analyses, and provenance pointers.
- `06_baselines/`: baseline summaries; the full MMOKD implementation and checkpoints are in `baselines/mmokd_port/`.
- `07_reproduction/`: method reproduction notes and related-work source notes.

For project-wide comparison across raw/no-mask, masked, repeated, and historical
protocols, use `../summary/MASTER_METRICS.csv` and
`../summary/MASTER_RUN_CATALOG.csv`. The plot inventory is
`../summary/ALL_FIGURE_INVENTORY.csv`; generated comparison figures are in
`../summary/figures/`.

## Provenance

S0-Raw, Hinton-KD, DIS2-Port, Multi2Uni-Port, S2-Raw-native, and S2-2Stage are frozen project runs. OGP-Net-Port failed its sanity gate and is excluded from quantitative tables. MMOKD-Port is the only new training arm in this closeout; it is a mechanism-faithful port on the project ConvNeXt-Tiny/FPN, not an official numerical reproduction. Exact published details that were unavailable are listed in `07_reproduction/mmokd_spec.md`.

All model metrics are reused from their recorded run summaries or final CSVs,
except MMOKD-Port which is evaluated from its newly selected checkpoint. The
locked project protocol has no validation split and selects checkpoints using
fixed-test Overall AUPRC; treat comparisons as test-monitored development
evidence. Geometry-masked and legacy validation-split results remain separate
in the project-wide catalog and are not silently mixed into these raw benchmark
tables.
