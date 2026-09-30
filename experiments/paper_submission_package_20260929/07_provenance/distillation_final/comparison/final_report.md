# Final knowledge-distillation benchmark

## Locked protocol and evidence boundary

- Fixed project split: **1,370 train / 0 validation / 343 test locations**, 4,110 train rows after the project's three training modes. No split was redrawn.
- Seed 42, batch 16, Adam lr `5e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0`, max 100 epochs, patience 2. Every completed new arm selects best checkpoint by the inherited **test Overall AUPRC** monitor; this is test-monitored development evidence, not an independent unbiased holdout estimate.
- Geometry codes are post-hoc only in Hinton-KD-Port, DIS2-Port, and S2-2Stage. BothValid means target=0 and counter=0; TargetValid means target=0; G10/G01 are diagnostic regions. No geometry enters these three methods' training loss, model input, search, or KD.
- All primary results are single-seed first-pass. Multi2Uni-Port completed as an explicit mechanism-faithful port with unresolved paper details listed below. OGP-Net-Port failed the required deployed-branch sanity gate; it has no formal training checkpoint or test score.

## Results

| Method | Status | Overall AP | IoU | F1 | BothValid AP | BothValid IoU | BothValid F1 | TargetValid AP | TargetValid IoU | TargetValid F1 | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0-Raw | COMPLETE | 0.87983 | 0.62914 | 0.77236 | 0.86909 | 0.61344 | 0.76041 | 0.87552 | 0.62268 | 0.76747 | 19 |
| Hinton-KD-Port | COMPLETE | 0.88074 | 0.64230 | 0.78219 | 0.86979 | 0.62765 | 0.77123 | 0.87633 | 0.63659 | 0.77795 | 26 |
| DIS2-Port | COMPLETE | 0.32681 | 0.21615 | 0.35547 | 0.29762 | 0.19797 | 0.33051 | 0.31086 | 0.20795 | 0.34430 | 16 |
| OGP-Net-Port | SANITY_FAILED_NOT_TRAINED | — | — | — | — | — | — | — | — | — | — |
| Multi2Uni-Port | COMPLETE_MECHANISM_FAITHFUL_PORT | 0.88626 | 0.63997 | 0.78046 | 0.87619 | 0.62593 | 0.76994 | 0.88244 | 0.63480 | 0.77661 | 20 |
| S2-Raw-native | HISTORICAL_REFERENCE_GEOMETRY_WEIGHTED_KD | 0.88883 | 0.64234 | 0.78223 | 0.87844 | 0.62667 | 0.77050 | 0.88477 | 0.63596 | 0.77748 | 44 |
| S2-2Stage | COMPLETE | 0.89156 | 0.65098 | 0.78860 | 0.88154 | 0.63717 | 0.77838 | 0.88767 | 0.64651 | 0.78531 | 44 |

Full precision, G10/G01 metrics, best epochs, parameter counts, and supported-operator FLOPs are in `final_table.csv` and `compute_cost.csv`.

## Main findings

1. Among scored methods, S2-2Stage has the highest test-monitored Overall AP (**0.891562**), followed by historical S2-Raw-native (**0.888835**), Multi2Uni-Port (**0.886259**), Hinton-KD-Port (**0.880736**), S0-Raw (**0.879834**), and DIS2-Port (**0.326807**). The ranking is conditional on the one seed and test-based checkpoint selection. OGP-Net-Port is unscored because its sanity gate failed.
2. In the completed run, S2-2Stage exceeds historical S2-Raw-native by **+0.002727 AP** and S0-Raw by **+0.011728 AP**. This is not evidence of a statistically stable difference.
3. S2-2Stage best epoch 44 has weighted KD scalar share **6.39%** (policy 0.000558789; evidence 0.000459662); the configured initial target was 20%, while epoch 6 actually measured 13.93% after epoch-level losses changed. At best epoch 44, the historical S2-Raw-native KD scalar share was **67.17%**. The two-stage schedule therefore avoided the native run's high KD scalar share while retaining its selected AP in this run.
4. S2-2Stage remains a single-orbit Student at inference. Its model has 28.27M inference parameters; DIS2's two target-orbit deploy paths together have 8.90M. Thus Search-KD is not smaller by parameter count than the DIS2 deploy paths, although its supported-operator fvcore estimate is similar (1.708 vs 1.729 GFLOPs per orbit). The FLOP counts omit listed unsupported ops and are partial estimates.
5. Multi2Uni-Port's best target-only Student reached Overall/BothValid/TargetValid AUPRC **0.886259/0.876185/0.882435**, with Overall IoU/F1 **0.639968/0.780464**. This is +0.006425 AP over S0-Raw in this single test-monitored run; it does not exceed S2-2Stage. Because detailed paper equations/configuration were unavailable, interpret it as a mechanism-faithful port, not an official numerical reproduction.

## Loss and method-faithfulness notes

- **Hinton-KD-Port:** `L_soft + 0.5 L_seg`, two-class logits `[-z,z]`, T=2, per-pixel mean `T² KL`. Best epoch 26, KD scalar share 44.38%.
- **DIS2-Port:** author DLKD_ver4 distinct/supplementary branches, hierarchical fusion, classwise decoder, normalized multi-level/penultimate feature KD, T=2 logit KD, orthogonality, full/missing/unimodal and multi-scale segmentation; symmetric ASC/DESC target directions; KL weight 0 for epochs 1–5, 5 from epoch 6. Best epoch 16, auxiliary share 5.52%. It is a Haiti mechanism port, not numerical reproduction of the original Potsdam result. The port fixes only an upstream classwise map append duplication in three lines; see `unresolved_hyperparameters.md`.
- **S2-Raw-native:** retained historical canonical checkpoint and scores, but code inspection shows its old KD query weight uses geometry-derived G10/G01 and its Teacher is CG10 (geometry-TGD weighted training). It is shown as historical context, not compliant with the current geometry-post-hoc-only rule.
- **S2-2Stage:** uses geometry-free CG0 as frozen Teacher; epochs 1–5 segmentation only; epoch-level loss normalization after epoch 5; equal policy/evidence weight and one cosine decay from 1 to 0.25. No additional module or loss was introduced. The KD inputs stop-gradient before the shared backbone by the canonical Search-KD definition, so best-epoch policy/evidence-to-backbone gradient norms are 0; segmentation-to-backbone norm was 0.0904836 on the recorded first batch.
- **OGP-Net-Port sanity failed:** the port implements the paper's Eq.12 components and uses tau=0.2, gamma cutoff=1e-3, and a fixed global gradient norm clip=1 after the unbounded SGD update overflowed. On 32 rows, loss fell and DIST/feature-MSE/contrastive had nonzero gradients, but deployed single-orbit IoU/F1 fell from 0.8143/0.8977 to 0.5388/0.7003. No formal Stage-2 training was launched. The Stage-1 Teacher is the reused, frozen canonical S0-Raw checkpoint. Architecture-only target branch cost is 28.080M parameters and 1.56214 supported GFLOPs/orbit; there is no inference checkpoint.
- **Multi2Uni-Port completed:** two-encoder MEF Teacher, target-only ConvNeXtTinyFPN Student, one-level Haar detail FAD at c2/c3, and global c5 cross-view affinity CAD. One-time initial magnitude calibration yielded fixed lambdas 0.2943639/0.0325334. At best epoch 20, mean weighted FAD/CAD were 10.47%/11.32% of segmentation loss. The run stopped at epoch 22 under patience 2. Total training model parameters: 104.358M; deployed Student: 28.080M and 1.56201 supported GFLOPs/orbit. Exact MEF equation, FAD wavelet/levels/loss, CAD equation/layer, and paper coefficients remain unresolved; see `unresolved_hyperparameters.md`.

## Compute accounting

`flop_profile.json` and `profile_cost.py` record the original methods' batch-one JIT trace. `flop_profile_external.json` and `profile_external_ports.py` record the two new ports at 128×128. All report fvcore-supported forward operations only; unsupported elementwise/GELU, sampling, softmax/attention, padding, and backward work are omitted. OGP's cost is architecture-only because it has no formal checkpoint. Multi2Uni's inference path is only its single target-orbit Student; teacher/MEF/FAD/CAD are training-only.

## References

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- Kieu et al., [DIS2 paper](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html), [author source](https://github.com/nhikieu/DIS2).
- Xu et al., [OGP-Net, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32743).
- Sun et al., [Multi-to-Uni Modality Knowledge Distillation, Information Fusion 2026](https://doi.org/10.1016/j.inffus.2026.104696).

## Cleanup

`distillation_deletion_manifest.txt` records the obsolete KD experiment trees and dedicated runners requested for deletion. Deletion has **not** been completed: the execution environment rejected the recursive filesystem deletion command by policy. Canonical S0-Raw/S2-Raw evidence, current Teacher checkpoints, fixed split/data audit, shared source modules, third-party DIS2 source, and the paper package were retained. `retest_dis2_tgd.py` was retained because existing baseline and audit documents still depend on it. No deletion target should be described as removed until this operation can run successfully.
