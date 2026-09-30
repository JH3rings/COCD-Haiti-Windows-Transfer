# Method reproduction specification

Status: executable runs complete for Hinton-KD-Port, DIS2-Port, S2-2Stage, and Multi2Uni-Port. OGP-Net-Port was implemented and source-audited but failed the required 32-row deployed-branch sanity gate, so no formal OGP training/checkpoint/metrics are claimed. This is a method-port record, not a claim of exact reproduction on the papers' original datasets.

## Locked Haiti protocol

- Dataset IDs are read from the pinned project split: 1,370 train locations, 0 validation locations, 343 test locations. No split is created or changed.
- Raw inputs come from the existing HaitiPairs pipeline. Each deployed Student sees one orbit in the five-channel format VV-pre, VH-pre, VV-post, VH-post, orbit ID. ASC and DESC are evaluated independently and concatenated for metrics.
- Training uses the project's three existing train modes, seed 42, batch size 16, Adam at 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0, max 100 epochs, patience 2. Best epoch is selected by test Overall AUPRC because this is the already locked project monitor; therefore all selected test metrics are test-monitored and not an independent unbiased test estimate.
- No geometry raster is passed to the model, training objective, search, or KD. Geometry codes are read only by the post-hoc evaluator: 0 valid, 1 layover, 2 shadow, 3 layover+shadow. Report Overall, BothValid (target=0 and counter=0), TargetValid (target=0), G10 (target>0 and counter=0), and G01 (target=0 and counter>0).
- The canonical ConvNeXt-Tiny/FPN Student constructor uses ImageNet-1K pretrained weights. Hinton-KD-Port, S2-2Stage, and Multi2Uni-Port initialize the target-only Student from the locked SO_backbone_seed42 checkpoint. OGP-Net-Port follows its Stage-1 pre-trained segmentation Teacher by reusing the locked S0-Raw checkpoint to initialize both Stage-2 orbit branches. DIS2 uses its paper's own U-Net-style encoder because the Distinct/Supplement branches, hierarchical hybrid fusion and classwise decoder are the method's core; this is an explicit architecture exception.
- Existing evidence is retained unchanged: S0-Raw best epoch 19, Overall AP/IoU/F1 0.8798338/0.6291364/0.7723557; S2-Raw-native best epoch 44, 0.8888350/0.6423411/0.7822262. S2-Raw-native has geometry-conditioned query weighting in its historical implementation, so it is a historical reference and not evidence for the no-geometry S2-2Stage run.

## Teacher provenance decision

The formal Teacher trainer defines CG10 with gamma=0.10 and adds target-geometry-specific TGD loss; that checkpoint does not satisfy the current no-geometry training rule. The formal CG0 arm uses gamma=0 and full-pixel BCE without geometry-specific weighting, so new teacher-guided arms use the existing CG0 checkpoint. It is the clean paired Teacher under the locked protocol, although it is not the highest-scoring CG10 variant.

## Hinton-KD-Port

- Source: Hinton, Vinyals and Dean, Distilling the Knowledge in a Neural Network.
- Knowledge target: frozen CG0 paired Teacher output for each current target orbit.
- Loss: student binary logits z become two-class logits [-z, z]; T=2; mean per-pixel T-squared KL(Teacher || Student). Haiti hard supervision is the existing binary BCE, and total loss is L_soft + 0.5 L_seg as requested.
- Both student orbit branches receive equal averaged supervision. No geometry-based query or pixel weighting is used.
- Adaptation: two-class output and per-pixel normalization are necessary for binary dense prediction. The deployed model remains the existing target-only S0 architecture.

## DIS2-Port

- Sources: DIS2 WACV 2026 CV4EO paper and author repository. The local official snapshot is commit 497afc2d86ea6a85271ccace0a052603d918a9b3. Its training script has unresolved import paths in this snapshot, so the Haiti runner imports the checked-in official model/loss modules directly and implements the paper's published training equation.
- Knowledge target and structure: full ASC/DESC distinct-branch fused features/logits guide each missing-counter target branch; each modality also has a distinct and a supplementary encoder; orthogonality separates distinct/supplementary pooled features; fusion is hierarchical convolution plus transformer; CFLM supplies classwise attention and classwise multi-scale decoder; feature KD includes pooled multi-level normalized feature loss and penultimate feature loss; logit KD uses T=2; multi-scale segmentation and unimodal/full/missing segmentation are retained.
- Weights/schedule: official training equation sums full, missing, and unimodal segmentation losses, DLKD (orthogonality + feature/logit distillation), and multiscale auxiliary losses. The author code sets KL weight to 0 for epochs 1-5 and 5 from epoch 6 onward. The project optimizer/split/monitoring rule overrides the original optimizer. All losses supervise raw pixels; geometry is not used.
- Haiti adaptation: two modalities are ASC and DESC five-channel observations; the model is trained on both target/counter missing directions; number of semantic classes is 2 and input patch size is 128. Deployment selects only the relevant target encoder+supplement encoder path for a single orbit. Since DIS2's encoder/fusion/classwise decoder define its method, its official U-Net-style backbone is retained rather than forcing ConvNeXt into an incompatible source architecture.

## OGP-Net-Port

- Source: [AAAI 2025 OGP-Net paper](https://ojs.aaai.org/index.php/AAAI/article/view/32743). The code implements Stage-1 reuse/freeze of the existing target-only S0-Raw Teacher, shared ConvNeXt convolutions with separate LayerNorms, gamma-guided channel exchange, odd-column exchange in the last two stages, last-two-stage plus decoder MSE, pixel semi-hard contrast (closest negative top 10%, most distant positive), two-class DIST inter/intra relations, GSU, and the c5 auxiliary head. Eq.12 is followed as fused segmentation + auxiliary segmentation + two-direction averaged DIST + target-branch feature MSE + pixel contrastive; no extra modality-head segmentation terms.
- Unresolved numeric choices: the paper gives no numeric contrastive temperature or near-zero gamma threshold and no author code was located. Fixed once at tau=0.2 (the cited PCD implementation value, explicitly an adaptation) and gamma cutoff 1e-3. SGD lr 5e-3, momentum 0.9, weight decay 0; a fixed global gradient norm clip of 1.0 was required for finite updates on the ConvNeXt/FPN adaptation. The OGP paper uses EfficientNet-B3/DeepLabV3+, so ConvNeXt/FPN and two-class [-z,z] output are Haiti adaptations.
- Sanity outcome: 100 finite steps; total loss 32.66 to last-10 mean 14.55; DIST/feature-MSE/contrastive were nonzero and each backpropagated. However, the target-only deployed branch's 32-row IoU/F1 fell from 0.8143/0.8977 to 0.5388/0.7003. It therefore failed the required sanity criterion. No full Stage-2 run or formal OGP checkpoint/metrics were produced. Stage-1 reuses the existing S0-Raw checkpoint and was not retrained.

## Multi2Uni-Port

- Source: Sun et al., [Multi-to-Uni Modality Knowledge Distillation for Remote Sensing Image Semantic Segmentation](https://doi.org/10.1016/j.inffus.2026.104696). The available publisher preview confirms MEF, low-level DWT-FAD for high-frequency/detail knowledge, and high-level CAD structural affinity transfer. Full paper, supplement, and author code were not accessible, so the selected details below are explicit port adaptations, not official numerical reproduction.
- Fixed one-time mechanism choices: independent ASC/DESC ConvNeXt encoders with bidirectional gated residual MEF at c2-c5 and shared FPN teacher decoder; target-only ConvNeXtTinyFPN student; one-level orthonormal Haar DWT detail bands LH/HL/HH at c2/c3 with MSE FAD; global c5 teacher cross-view affinity supervising each target-only student's self-affinity with L1 CAD. No local affinity window and no retired CO-FAD/CO-CAD results are reused.
- Loss magnitude calibration was performed once on the first training batch: `L_seg=0.0437179`, raw FAD `0.0148516`, raw CAD `0.1343782`; fixed `lambda_FAD=0.2943639`, `lambda_CAD=0.0325334`, giving 10% initial contribution per auxiliary term. No sweep. The formal run used the locked project optimizer and stopping policy: seed 42, batch 16, Adam `5e-5`, 100 max epochs, patience 2, test Overall AUPRC monitor. It stopped at epoch 22 and selected best epoch 20. Best epoch mean contribution ratios were FAD 10.47% and CAD 11.32% of segmentation loss.
- 32-sample sanity passed: 500 steps finite, loss 0.00703 to 0.00427 last-10 mean, FAD/CAD had nonzero Student gradients (0.1313/1.0613), and target-only Student IoU/F1 rose 0.8279/0.9058 to 0.8928/0.9433. The trained target-only Student is the only inference model.

## S2-2Stage

- Knowledge targets remain the canonical Search-KD targets: counter-guided Teacher policy distribution over target-memory queries and same-target-memory retrieved-evidence direction. The policy objective is JS; evidence objective is cosine distance.
- The old successful S2 runner weights KD queries with geometry-derived G10/G01 masks. That historical spatial weighting conflicts with the current locked rule that geometry is post-hoc only. S2-2Stage therefore preserves the policy/evidence loss definitions but computes them uniformly over all queries, with no geometry input.
- Epochs 1-5 use segmentation BCE only. After epoch 5, epoch-level means of L_seg, L_policy and L_evidence calibrate equal normalized policy/evidence weights so the initial combined KD scalar share at epoch 6 is 20%. A single cosine multiplier runs from 1 to 0.25 over epochs 6-100. No gradient norm is used to change lambda. The project test-AUPRC monitor and patience-2 rule select the best checkpoint.

## Reproduction status and references

- Completed best checkpoints and per-epoch progress are saved under `hinton_kd/`, `dis2_port/`, and `ours_s2_2stage/`; final per-region metrics and stop epochs are in each `summary.json` / `progress.json`.
- Hinton-KD-Port: best epoch 26, Overall AP 0.8807358; stop epoch 28.
- DIS2-Port: best epoch 16, Overall AP 0.3268071; stop epoch 18.
- S2-2Stage: best epoch 44, Overall AP 0.8915616; stop epoch 46; best-epoch KD scalar share 0.0639134.
- OGP-Net-Port: code and 32-row preflight exist; the deployed branch failed the sanity metric criterion, so no formal training or checkpoint exists. See `unresolved_hyperparameters.md` and `ogp_net_port/sanity_32_report.json`.
- Multi2Uni-Port: completed mechanism-faithful adaptation; best epoch 20, stopped at 22. Source-specific MEF/FAD/CAD choices remain unresolved and are enumerated in `unresolved_hyperparameters.md`.

- Hinton: [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- DIS2 paper: [WACV Workshop paper](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html); official author source [nhikieu/DIS2](https://github.com/nhikieu/DIS2)
- OGP-Net: [AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/32743); cited pixel-level contrastive source [ICCV 2023 PCD](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Pixel-Wise_Contrastive_Distillation_ICCV_2023_paper.html)
- Multi2Uni: [Information Fusion DOI](https://doi.org/10.1016/j.inffus.2026.104696)
