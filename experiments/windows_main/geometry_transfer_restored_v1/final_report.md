# Geometry masked cross-view transfer experiment

## Protocol and mask audit

- Original model definitions and initialization path were reused. `SO_backbone_seed42.pt` is the common initialization loaded by the official Teacher/Student runners; the backbone class is torchvision ConvNeXt-Tiny with ImageNet-1K V1 initialization.
- Seed 42, batch 16, Adam (`lr=5e-5`, `weight_decay=0`, betas 0.9/0.999, eps 1e-8), max 100 epochs, patience 2. The archived 137-location validation split was used for per-epoch overall AUPRC selection because the active v3 split merges those IDs back into training; validation+train IDs exactly equal v3's 1370 train IDs and test IDs match exactly (343). Test did not select any checkpoint.
- Best-validation epochs: SG 15, CG 27, S0 11, S2 12, ReliabilityKD 18, StableEvidenceKD 11.
- Geometry metadata confirms code 0=valid, 1=layover, 2=shadow, 3=layover+shadow. Both masks contain all four codes, so shadow and layover are separately measurable. Every segmentation BCE pixel with target code >0 is excluded. All KD query cells require every pixel in the corresponding 4x4 target block to be valid; reliability/stable objectives require BothValid.
- Test region definitions: BothValid=`target==0 && counter==0`; TargetValid=`target==0`; OverallValid=`target==0 || counter==0` (therefore includes the TGD/G10 reference area); TGD/G10=`target>0 && counter==0`, reported only as a diagnostic. IoU/F1 threshold is 0.5.
- In the test set, BothValid is 80.60%, TargetValid 89.84%, OverallValid 99.09%, and TGD/G10 9.25% of the two target-orbit views. Across those two views, geometry codes 1/2/3 account for 1,094,965 / 44,741 / 1,676 pixels.

## Test metrics

Each cell is **AUPRC / IoU / F1**.

| Method | BothValid | TargetValid | OverallValid |
|---|---:|---:|---:|
| SG Teacher | 0.867594 / 0.614588 / 0.761294 | 0.874160 / 0.623772 / 0.768300 | 0.869498 / 0.614947 / 0.761569 |
| CG Teacher | 0.887153 / 0.636824 / 0.778122 | 0.891836 / 0.644211 / 0.783611 | 0.889177 / 0.638550 / 0.779408 |
| CG shuffle | 0.802984 / 0.500009 / 0.666675 | 0.816565 / 0.513378 / 0.678453 | 0.815598 / 0.509507 / 0.675064 |
| S0 | 0.862814 / 0.610832 / 0.758406 | 0.869667 / 0.620634 / 0.765915 | 0.866846 / 0.614325 / 0.761092 |
| S2 Search-KD | 0.861890 / 0.614975 / 0.761591 | 0.868618 / 0.623786 / 0.768310 | 0.865082 / 0.616305 / 0.762610 |
| Reliability-KD | 0.857993 / 0.592927 / 0.744450 | 0.865221 / 0.602632 / 0.752052 | 0.858974 / 0.590325 / 0.742395 |
| Stable-Evidence-KD | 0.860773 / 0.601731 / 0.751351 | 0.867885 / 0.612038 / 0.759334 | 0.864792 / 0.605454 / 0.754247 |

## Paired differences and mechanism measurements

| Region | CG−SG (AUPRC / IoU / F1) | CG−shuffle (AUPRC / IoU / F1) | S2−S0 (AUPRC / IoU / F1) |
|---|---:|---:|---:|
| BothValid | +0.019559 / +0.022236 / +0.016828 | +0.084169 / +0.136815 / +0.111447 | −0.000924 / +0.004143 / +0.003185 |
| TargetValid | +0.017676 / +0.020438 / +0.015311 | +0.075271 / +0.130833 / +0.105158 | −0.001050 / +0.003152 / +0.002395 |
| OverallValid | +0.019679 / +0.023603 / +0.017839 | +0.073579 / +0.129043 / +0.104344 | −0.001764 / +0.001980 / +0.001518 |

Mechanism values are means on test query cells; smaller JS/evidence distance is closer to CG teacher policy/evidence. Reliability alignment is mean absolute reliability-map error; stable-evidence alignment is cosine similarity under the stable weighting.

| Student | Policy JS BothValid / TargetValid | Evidence distance BothValid / TargetValid | Reliability alignment | Stable-evidence alignment |
|---|---:|---:|---:|---:|
| S0 | 0.049831 / 0.049119 | 0.015738 / 0.016588 | — | 0.984644 |
| S2 Search-KD | 0.021960 / 0.021721 | 0.006838 / 0.007041 | — | 0.993430 |
| Reliability-KD | 0.047107 / 0.046365 | 0.011918 / 0.012722 | 0.017297 MAE | 0.988356 |
| Stable-Evidence-KD | 0.027331 / 0.026735 | 0.006517 / 0.006625 | — | 0.993772 |

Reliability and stable-evidence alignment were learned/measured, but neither improved segmentation over S0. S2 improved thresholded IoU/F1 slightly while reducing AUPRC; with one seed this is mixed evidence, not a stable transfer result.

## Requested decisions

| Question | YES / NO / UNCLEAR | Evidence |
|---|---|---|
| 1. After masking all geometry-invalid pixels, is the dual-view Teacher still clearly better than the self-guided Teacher? | **YES** | On BothValid, CG−SG is +0.0196 AUPRC, +0.0222 IoU, +0.0168 F1; same direction on TargetValid. |
| 2. Under restored convergence, is S2 stably better than S0? | **UNCLEAR** | S2 gains +0.0041 IoU / +0.0032 F1 on BothValid but loses 0.0009 AUPRC; one seed cannot establish repeatability. |
| 3. Does Reliability Distillation transfer reliability judgment and improve segmentation? | **NO** | Reliability map MAE is 0.0173, but all three BothValid segmentation metrics are below S0. |
| 4. Does Stable-Evidence Distillation transfer stable evidence selection and improve segmentation? | **NO** | Stable-evidence alignment is 0.9938 cosine, but all three BothValid segmentation metrics are below S0. |
| 5. Is there at least one physically plausible dual-to-single transfer mechanism supported by this run? | **UNCLEAR** | S2 has small IoU/F1 gains but lower AUPRC; both new mechanisms underperform S0. A single seed does not establish repeatable added capability. |

**Conclusion:** Correct dual-view guidance provides a substantial valid-region Teacher benefit and that benefit collapses under counter shuffle, but this seed does not establish that the gain transfers repeatably to a target-only Student.

This is one matched seed (42), so between-seed variance is unavailable. The prior fixed-30 diagnostic and its scripts are kept separately under `archive/diagnostics/fixed30_masked_20260923/` and are not pooled with these results.

Artifacts: [`comparison.csv`](comparison.csv), [`test_metrics.json`](test_metrics.json), [`protocol_and_geometry.json`](protocol_and_geometry.json), and the per-arm progress JSON / best-validation checkpoints in this directory.
