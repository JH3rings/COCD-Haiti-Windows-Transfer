# Baseline notes

- **S0-Raw:** existing target-only baseline, frozen; source metrics in `experiments/distillation_final/comparison/final_table.csv`.
- **Hinton-KD:** frozen target-only student; adapted to the project SAR pair protocol.
- **DIS2-Port:** frozen adapted baseline. Its low recorded score is retained as observed; no rerun or correction was made in this closeout.
- **Multi2Uni-Port:** mechanism-faithful MEF/FAD/CAD adaptation; exact details unavailable in the article are listed in its summary and reproduction note.
- **MMOKD-Port:** mechanism-faithful online fusion-teacher / two single-orbit student adaptation. The paper source does not expose its full implementation coefficients/operators; `mmokd_spec.md` records the adaptation boundary.
- **S2-Raw-native:** historical reference; used geometry-derived region weighting, so report it in mechanism analysis rather than the clean main table.
- **S2-2Stage:** final Ours; frozen canonical checkpoint, target-only deployment.
- **OGP-Net-Port:** sanity failed; no quantitative result.

All test metrics use the fixed project test set; the project has no validation split and uses test Overall AUPRC for epoch selection. This is a protocol limitation and must be stated in the paper.
