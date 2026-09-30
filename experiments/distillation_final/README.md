# Haiti knowledge-distillation benchmark

This directory contains the final single-seed, method-faithful benchmark outputs and runners for target-only deployment.

## Locked protocol

- Existing split only: 1,370 train locations, 0 validation locations, 343 test locations. Do not create a new split.
- Seed 42; batch 16; Adam `lr=5e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0; maximum 100 epochs; patience 2.
- The existing project protocol monitors test Overall AUPRC after every epoch and selects the best checkpoint. Treat reported results as test-monitored development evidence, not an independent holdout estimate.
- Geometry is post-hoc only for the new Hinton/DIS2/S2-2Stage training runs. The evaluators report Overall, BothValid (`target=0,counter=0`), TargetValid (`target=0`), G10, and G01.
- Single-orbit deployment input is `[VV_pre,VH_pre,VV_post,VH_post,orbit_id]`. DIS2 uses its source-native paired tensor API with the non-target modality slot zeroed; no counter SAR values are supplied at inference.

## Methods and source fidelity

| Method | Runner / output | Source logic kept | Haiti adaptation |
|---|---|---|---|
| Hinton-KD-Port | `hinton_kd/train_hinton.py` | T=2 soft response KD with `L_soft + 0.5 L_seg` | Binary logit `z` maps to `[-z,z]`; frozen clean CG0 Teacher; mean per-pixel KL; target-only S0 Student. |
| DIS2-Port | `dis2_port/train_dis2.py` | Official DLKD_ver4 distinct/supplementary branches, fusion/classwise decoder, normalized multi-level + penultimate feature KD, T=2 logit KD, orthogonality, multi-scale/full/missing/unimodal supervision; epoch 1–5 KL off then weight 5. | ASC/DESC symmetric target directions and two semantic classes. Official architecture retained because it is the method's core; random seed-42 initialization is not ConvNeXt-compatible. A narrow upstream class-map append duplication is repaired in the local port only; see `unresolved_hyperparameters.md`. |
| OGP-Net-Port | `ogp_net_port/`; sanity gate failed, no formal checkpoint | DMC feature exchange, last-two encoder/decoder MSE, pixel semi-hard contrast, DIST/DUR, GSU and auxiliary head implemented. | Reused frozen S0-Raw as Stage-1 Teacher; tau=0.2 and near-zero exchange cutoff=1e-3 fixed once. Unclipped Eq.12 updates overflowed; fixed global grad norm clip 1.0 kept the 32-sample run finite, but deployed-branch IoU/F1 fell, so the user-required sanity gate blocked formal training. |
| Multi2Uni-Port | `multi2uni_port/`; best checkpoint and progress recorded | MEF Teacher + DWT detail FAD + global c5 affinity CAD, with a target-only ConvNeXtTinyFPN Student. | Mechanism-faithful port: one bidirectional gated residual MEF block per encoder level, one-level Haar FAD at c2/c3, global cross-view c5 affinity CAD. One-time initial magnitude calibration; source details remain unresolved. |
| S2-Raw-native | existing `experiments/windows_main/raw_cver_v1_seed42/` | Preserved as historical formal result. | Old code uses CG10 and geometry-derived G10/G01 KD query weighting; shown only as historical context, not compliant with the current geometry post-hoc-only training rule. |
| S2-2Stage | `ours_s2_2stage/train_s2_2stage.py` | Existing policy JS and same-target-memory evidence cosine KD targets; no new module/loss. | Uses geometry-free CG0. Epochs 1–5 segmentation only, epoch-level calibration after epoch 5, equal initial normalized policy/evidence weights, then one cosine multiplier from 1 to 0.25. KD queries are uniform, with no geometry input. |

## Files

- `comparison/final_table.csv`: per-method overall and geometry-region metrics, best epoch, inference parameters, and supported-operator FLOP trace.
- `comparison/loss_curves.csv`: per-epoch segmentation/KD losses, contributions, monitoring metrics, and S2 gradient norms.
- `comparison/kd_contribution_curves.csv`: per-epoch KD scalar share.
- `comparison/compute_cost.csv`: inference cost and observed-epoch supported forward FLOPs; backward cost is excluded.
- `comparison/flop_profile_external.json` and `profile_external_ports.py`: supported-op batch-one target-only inference traces for both ports; OGP is architecture-only because no formal checkpoint passed sanity.
- `comparison/final_report.md`: results, interpretation, method limits, and cleanup record.
- `comparison/profile_cost.py` and `comparison/flop_profile.json`: reproducible fvcore trace and unsupported-op accounting.
- `method_reproduction_spec.md`: source audit and reproduction checklist.
- `distillation_deletion_manifest.txt`: exact legacy outputs and runners selected for deletion after the new results were verified.

All results are single seed. New methods and ports are not described as exact numerical reproductions on their original datasets. OGP-Net-Port did not pass its preflight and has no scored formal run; Multi2Uni-Port completed with an explicit mechanism-faithful adaptation. Source citations and unresolved parameters are recorded in the method specification and final report.
