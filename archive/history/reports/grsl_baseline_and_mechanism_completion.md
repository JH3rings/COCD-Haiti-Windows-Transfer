# GRSL baseline and mechanism completion

## Frozen protocol

All values below use the existing fixed Haiti 80/20 split, the existing 10% internal validation partition, five-channel single-orbit Student input, pixel-wise landslide GT, and fixed 0.5 headline threshold. Geometry is not a Student input.

## Existing-method main results

| method | iou | f1 | precision | recall | auprc | G10_f1 | G10_auprc | validation_best_f1_threshold | test_f1_at_validation_threshold | overall_precision_at_recall_095 | G10_precision_at_recall_095 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.5302 | 0.6930 | 0.5455 | 0.9496 | 0.8391 | 0.7009 | 0.8455 | 0.8800 | 0.7597 | 0.5445 | 0.5726 |
| T | 0.5795 | 0.7338 | 0.5965 | 0.9530 | 0.8710 | 0.7437 | 0.8832 | 0.8700 | 0.7883 | 0.6041 | 0.6390 |
| KD | 0.5541 | 0.7131 | 0.5700 | 0.9522 | 0.8545 | 0.7193 | 0.8590 | 0.8700 | 0.7754 | 0.5756 | 0.5969 |
| Ours | 0.5619 | 0.7195 | 0.5821 | 0.9418 | 0.8589 | 0.7333 | 0.8714 | 0.8800 | 0.7769 | 0.5602 | 0.5972 |

## Four-region geometry analysis

| region | pixels | positive_pixels | prevalence | S0_iou | S0_f1 | S0_precision | S0_recall | S0_auprc | T_iou | T_f1 | T_precision | T_recall | T_auprc | KD_iou | KD_f1 | KD_precision | KD_recall | KD_auprc | Ours_iou | Ours_f1 | Ours_precision | Ours_recall | Ours_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G00 | 9058636 | 480704 | 0.0531 | 0.5189 | 0.6833 | 0.5359 | 0.9425 | 0.8294 | 0.5686 | 0.7250 | 0.5878 | 0.9459 | 0.8611 | 0.5420 | 0.7030 | 0.5593 | 0.9460 | 0.8457 | 0.5496 | 0.7094 | 0.5715 | 0.9348 | 0.8494 |
| G10 | 1039406 | 153884 | 0.1480 | 0.5395 | 0.7009 | 0.5523 | 0.9589 | 0.8455 | 0.5920 | 0.7437 | 0.6047 | 0.9657 | 0.8832 | 0.5616 | 0.7193 | 0.5750 | 0.9602 | 0.8590 | 0.5790 | 0.7333 | 0.5971 | 0.9500 | 0.8714 |
| G01 | 1039406 | 153884 | 0.1480 | 0.5459 | 0.7063 | 0.5575 | 0.9634 | 0.8550 | 0.5897 | 0.7419 | 0.6043 | 0.9607 | 0.8815 | 0.5734 | 0.7289 | 0.5868 | 0.9619 | 0.8680 | 0.5734 | 0.7288 | 0.5892 | 0.9551 | 0.8684 |
| G11 | 101976 | 25704 | 0.2521 | 0.6043 | 0.7533 | 0.6259 | 0.9460 | 0.8817 | 0.6580 | 0.7937 | 0.6743 | 0.9646 | 0.9122 | 0.6365 | 0.7779 | 0.6527 | 0.9626 | 0.9031 | 0.6336 | 0.7757 | 0.6587 | 0.9434 | 0.8972 |

## Gap recovery

| method | metric | gap_recovery |
| --- | --- | --- |
| KD | iou | 0.4851 |
| KD | f1 | 0.4930 |
| KD | auprc | 0.4811 |
| KD | G10_f1 | 0.4290 |
| KD | G10_auprc | 0.3583 |
| Ours | iou | 0.6427 |
| Ours | f1 | 0.6500 |
| Ours | auprc | 0.6202 |
| Ours | G10_f1 | 0.7574 |
| Ours | G10_auprc | 0.6879 |

## PR analysis

PR curves: `experiments/rapid_landslide_cocd/pr_overall_g10.png`. Thresholds were selected on validation only and evaluated once on test.

## Qualitative candidates

Automatically selected candidate locations: `experiments/rapid_landslide_cocd/qualitative_case_candidates.csv`. Full panels will be rendered after DIS2-adapted is available so every requested method appears in one panel.

## Reproduction status

- CDNetE: author thesis and Haiti dataset record located; official code has not yet been identified.
- Boehm SAR U-Net: official `iprapas/landslide-sar-unet` source snapshot downloaded.
- MFEWF: paper points to ChenLifu2022, but the currently public profile does not expose an MFEWF repository; status pending author-code verification.
- DIS2-style adapted: official `nhikieu/DIS2` source snapshot downloaded; adaptation pending.

## Interim strict conclusion

Teacher exceeds S0. Ours exceeds vanilla KD in IoU/F1/AUPRC and G10 F1/AUPRC, but final comparison to CDNetE, Boehm, MFEWF and DIS2 is not yet available; therefore this is not yet a final GO/NO-GO conclusion.

## Unified main comparison (completed)

| Method | Type | Test-time orbit | IoU | F1 | AUPRC | Comp-F1 | Comp-AUPRC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CDNetE Early Fusion | Haiti-native thesis adaptation | Single | 0.493 | 0.660 | 0.771 | 0.655 | 0.776 |
| Boehm SAR U-Net++ | Official-code adaptation | Single | **0.593** | **0.744** | 0.836 | 0.727 | 0.837 |
| MFEWF adapted | Paper adaptation | Single | 0.323 | 0.489 | 0.658 | 0.428 | 0.644 |
| S0 | Our backbone baseline | Single | 0.530 | 0.693 | 0.839 | 0.701 | 0.846 |
| Vanilla KD | KD | Single | 0.554 | 0.713 | 0.854 | 0.719 | 0.859 |
| DIS2-style adapted | Generic missing-modality KD | Single | **0.566** | **0.723** | **0.862** | 0.728 | 0.867 |
| Ours | Geometry-conditioned corrective KD | Single | 0.562 | 0.719 | 0.859 | **0.733** | **0.871** |
| Dual Teacher | Upper bound | Dual | 0.579 | 0.734 | 0.871 | 0.744 | 0.883 |

### Strict conclusion after external baselines

- **Q1:** Yes. The dual Teacher exceeds S0 in every main metric.
- **Q2:** Yes. Ours exceeds vanilla KD in IoU, F1, AUPRC, Comp-F1 and Comp-AUPRC.
- **Q3:** No for overall IoU/F1/AUPRC: DIS2-style adapted is higher. Yes in the mechanism-critical complementary region: Ours has higher Comp-F1 and Comp-AUPRC.
- **Q4:** Conditional. Ours' largest advantage over vanilla KD is G10 (Comp-F1 +0.014, Comp-AUPRC +0.012); full region-wise analysis remains the mechanism evidence.
- **Q5:** Ours exceeds CDNetE and MFEWF, but Boehm U-Net++ has higher fixed-threshold IoU/F1. Ours has higher AUPRC and complementary AUPRC, so the comparison is metric-dependent rather than a universal win.
- **Q6:** **CONDITIONAL GO.** The evidence supports privileged opposite-orbit knowledge and a geometry-conditioned advantage over vanilla KD, but it does not yet support a universal superiority claim over generic DIS2-style compensation or Boehm U-Net++.
