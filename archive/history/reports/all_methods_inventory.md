# Every method and every run in this project — complete inventory

The previous summary (`reports/windows_main_results_so_far.md`) covered only the
seven Windows formal arms. This one is the full inventory: **every method that has
been trained, under every protocol, with its own measurement convention stated.**
A number is only meaningful with the protocol it was measured under, so the four
protocols are separated and never mixed in one ranking.

**Nothing here is a test-set result for the Windows arms. The V1/V2/V3 blocks are
test-set numbers from the Mac-era work.**

---

## The four protocols

| protocol | optimizer | batch | seg loss | epochs | threshold | where |
|---|---|---|---|---|---|---|
| **V1** | AdamW 1e-4, wd 1e-4 | 2 | BCE + 0.2 dice | 50 | fixed 0.5 | `results/rapid_landslide_cocd/` |
| **V2** | AdamW 1e-4, wd 1e-4 | 2 | BCE + 0.2 dice | 50 | fixed 0.5 | `results/ours_v2/` |
| **V3** | AdamW 1e-4, wd 1e-4 | 8 | BCE + 0.2 dice | 20 / 30 / 50, early stop varies | fixed 0.5 | `results/ours_v2/` |
| **Windows unified** | Adam 5e-5, wd 0 | 16 | BCE only | 20 | val-selected max-F1, frozen | `experiments/windows_main/` |

The Windows unified protocol is the one specified for the formal results table. The
earlier three are kept because they produced the numbers already written into
drafts, and because some arms exist **only** under them.

---

## Block 1 — V1 protocol, test set, `results/rapid_landslide_cocd/`

The first round. Single-orbit students, dual-orbit teacher, threshold 0.5.

| method | IoU | F1 | AUPRC | note |
|---|---:|---:|---:|---|
| S0 (single-orbit student, no distillation) | 0.5302 | 0.6930 | 0.8391 | floor |
| T (dual-orbit teacher) | **0.5795** | **0.7338** | **0.8710** | upper reference, **dual-orbit readout** |
| Vanilla KD | 0.5541 | 0.7131 | 0.8545 | |
| Ours-V1 (residual gate) | 0.5619 | 0.7195 | 0.8589 | |
| **old DIS2-style** | **0.5662** | **0.7230** | **0.8620** | our own residual-matching loss, not official DIS2 |

Gap recovery over S0→T (`gap_recovery.csv`), same block:

| method | IoU | F1 | AUPRC | G10 F1 | G10 AUPRC |
|---|---:|---:|---:|---:|---:|
| KD | 0.4851 | 0.4930 | 0.4811 | 0.4290 | 0.3583 |
| **Ours-V1** | **0.6427** | **0.6500** | **0.6202** | **0.7574** | **0.6879** |

This is the one place in the project where a method recovers a clear majority of the
teacher gap. It is V1, batch 2, threshold 0.5, and the teacher is dual-orbit while
the students are single-orbit, so the comparison is not like-for-like.

Four-region detail (S0 / T / KD / Ours): all four methods rank G11 > G01 > G10 > G00.

---

## Block 2 — V3 structure ablation, test set, `results/ours_v2/`

686 orbit-specific test samples, threshold 0.5. Each row differs from the one above
by a single structural factor. Full write-up: `results/ours_v2/ABLATION_R1R2R3_SUMMARY.md`.

| method | IoU | F1 | AUPRC | G10 IoU | G10 AUPRC | epochs |
|---|---:|---:|---:|---:|---:|---:|
| Teacher_self (target track only) | 0.6006 | 0.7505 | 0.8820 | 0.6053 | 0.8865 | 50 |
| Teacher_dual (target + counter) | 0.6250 | 0.7692 | 0.8972 | 0.6320 | 0.9038 | 50 |
| Ours R1 = free residual head | 0.6211 | 0.7663 | 0.8924 | 0.6251 | 0.8978 | 20 (early) |
| Ours R2 = R1 + inherited Omega | 0.6194 | 0.7650 | 0.8932 | 0.6252 | 0.8993 | 30 (early) |
| **Ours R3 = R2 + progressive drive** | **0.6298** | **0.7728** | **0.8972** | **0.6372** | **0.9030** | 50 |
| Ours R3g = R3 + geometry-priority weighting | 0.6260 | 0.7700 | 0.8955 | 0.6292 | 0.8989 | 50 |
| **DIS2-port = official DIS2 rules on R1** | 0.6272 | 0.7709 | 0.8963 | 0.6335 | 0.9023 | 50 |

Attribution inside this block: R1 − Teacher_self = +0.0205 IoU (distillation works on
this skeleton); R2 − R1 = **−0.0017** (inheriting Omega contributes nothing alone);
R3 − R2 = **+0.0104** (the progressive driver carries the gain); R3 − Teacher_dual =
+0.0048.

### External baselines, same V3-era convention

From `results/grsl_external_baselines/metrics.csv` and the ablation table:

| method | IoU | F1 | AUPRC |
|---|---:|---:|---:|
| Boehm SAR U-Net++ adapted | 0.5926 | 0.7442 | 0.8363 |
| CDNetE Early-Fusion adapted | 0.4927 | 0.6601 | 0.7715 |
| MFEWF-light adapted | 0.3232 | 0.4886 | 0.6583 |
| DIS2 official width-50, Haiti-adapted | 0.1829 | 0.3092 | 0.3732 |

The width-50 row is a failed reproduction (never converged: 1/3 of the data, 8
auxiliary heads taking 77 % of the gradient, no pretrained load). **Deleted on
instruction 2026-09-17; the number is kept as a record only and must not appear in a
results table.**

### The two other things called "DIS2"

| variant | skeleton | distillation content | status | IoU |
|---|---|---|---|---:|
| old DIS2-style | our `SingleOrbitStudent(correction=True)` | residual cosine + output MSE, our own formula | kept | 0.5662 |
| official width-50 | official `DLKD_ver4(gf_dim=50)` | official three-stage + orthogonality | deleted | 0.1829 |
| **DIS2-port** | our `OursV3Student('R1')` | official three-stage + orthogonality, line-by-line | kept | 0.6272 |

Two findings from this block worth carrying forward:

- **The orthogonality term is inert on this dataset.** It collapses to 0.0000 within
  3 epochs (official code: 9.31e-05). The three distillation terms do decrease, so the
  student is genuinely moving toward the teacher; only the orthogonality constraint is
  dead. This can be reported positively: "correction degenerating into re-encoding" is
  not the dominant failure mode here.
- **Old DIS2-style matches correction (`r_s` vs `r_t`), official DIS2 matches
  activations.** The old version may have worked partly by accident for this reason,
  and porting to the official rules gave that property up. This is recorded as an
  inference, not a conclusion: the two ran under different protocols and the old one
  used weight 0.1.

---

## Block 3 — geometry-priority weighting, negative result, V3

`R3g` vs `R3`, single variable changed: `w = gain·protect` becomes
`w = gain·protect·(1 + 1_real·G10)`, beta = 1 fixed. Both 50 epochs, no early stop.

| comparison | delta IoU | delta F1 | delta AUPRC |
|---|---:|---:|---:|
| overall | **−0.0038** | −0.0028 | −0.0017 |
| overall / G10 | **−0.0080** | −0.0060 | −0.0041 |
| overall / G00 | −0.0033 | −0.0025 | −0.0015 |
| overall / G11 | +0.0019 | +0.0013 | +0.0008 |

All four pre-registered criteria fail. Verification that the switch was really on:
50/50 epochs ran, `geo_frac` steady at 0.0303, `w_frac` 0.0306 → 0.0142. The
explanation is measured: the KD channel carries only **0.8–1.8 %** of the total
objective, so geometry weighting has a ceiling of about 1 % no matter how well it
targets. The targeting itself is correct — the active set is enriched 10–12x inside
the boosted region, and the selected pixels are ones the teacher genuinely gets wrong
(per-pixel BCE 4.04e-1 on the active set vs 2.84e-2 overall). Full audit:
`reports/g10_geometry_distillation_alignment_audit.md`.

---

## Block 4 — six single-SAR distortion baselines, different task

**Different task, not comparable to any block above.** This set detects geometric
distortion (layover / shadow), not landslides. Input contract: `[VV, VH, orbit_map]`
from **one orbit and one date only**; explicitly forbidden inputs are opposite-orbit
VV/VH, pre/post pairs, geometry mask, optical, DEM. Label = `geometry_mask > 0`.
Split 1199 / 257 / 257 locations, disjoint. Audit:
`results/six_single_sar_distortion_baselines/single_scene_input_contract_audit.json`.

| method | threshold | IoU | F1 | AUPRC | layover F1 | shadow F1 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net++ | 0.78 | **0.8148** | **0.8980** | **0.9606** | 0.8922 | 0.0413 |
| DeepLabV3+ | 0.85 | 0.7204 | 0.8374 | 0.9226 | 0.8435 | 0.0165 |
| wu2021 original layover | 0.61 | 0.4351 | 0.6064 | — | 0.6064 | 0.0011 |
| amplitude baseline | 0.50 | 0.3409 | 0.5084 | 0.3025 | 0.5075 | 0.0148 |
| wu2021 adapted distortion | 0.74 | 0.2764 | 0.4330 | — | 0.4317 | 0.0127 |

Specialist method `wu2021` ranks **below** generic segmentation networks here. Every
method detects layover and essentially misses shadow (shadow F1 ≤ 0.041 across the
board), which reads as a label/geometry property of this dataset rather than a model
difference.

---

## Block 5 — Windows unified protocol, validation, seed 42

The formal set. Seven arms, all re-trained under one protocol. Validation only.

| run | what it is | lam | best val AUPRC | IoU@thr | F1@thr |
|---|---|---:|---:|---:|---:|
| N0 | single-orbit baseline | 0 | 0.64428 | 0.3278 | 0.4938 |
| VKD | Vanilla KD student | — | 0.60333 | 0.3687 | 0.5388 |
| DIS2 | DIS2-port student | — | 0.59898 | 0.2774 | 0.4344 |
| R3 | legacy R3 student | — | 0.61073 | 0.3628 | 0.5324 |
| **N1** | complement predictor, BCE only | 0 | **0.65045** | 0.3231 | 0.4884 |
| N2 | full COCD | 1.0 | 0.55830 | 0.2505 | 0.4006 |
| N2-no-orbit | N2 without orbit embedding | 1.0 | 0.55575 | 0.2777 | 0.4347 |

Teachers under this protocol:

| run | reading | best val AUPRC |
|---|---|---:|
| Legacy teacher (re-trained) | z0 self / z4 / dual | 0.62766 / 0.64957 / 0.66635 |
| New COCD teacher | z_self / **z_dual** | 0.64594 / **0.68967** |

Diagnostic outside the protocol: N2 at lam = 0.1 → 0.63136 (vs 0.55830 at lam = 1.0,
still 0.0191 below N1). Registered in `diagnostic_runs`, does not enter the table.

---

## Cross-protocol: where the same arm name means different numbers

This is the trap in this project. The same nominal method appears in several blocks
with different values because the protocol changed:

| method | V1 test | V3 test | Windows val |
|---|---:|---:|---:|
| Single-orbit baseline | 0.5302 IoU | — (R0 never run) | 0.6443 AUPRC |
| Vanilla KD | 0.5541 IoU | — | 0.6033 AUPRC |
| DIS2-port | — | 0.6272 IoU | 0.5990 AUPRC |
| R3 | — | 0.6298 IoU | 0.6107 AUPRC |
| Teacher | 0.5795 IoU (dual) | 0.6250 IoU (dual) | 0.6664 AUPRC (dual) |

Rules that follow:

1. **A V1 or V3 number may not be placed in a Windows-protocol table**, and vice
   versa. The Windows run set exists precisely to replace the earlier numbers with a
   single-protocol version.
2. **Model selection throughout V1/V3 used fixed threshold 0.5**, whereas the Windows
   protocol selects max-F1 on validation and freezes it. This changes both the
   thresholded metrics and which checkpoint is called best.
3. **Training volume is not comparable even inside V3**: R1 stopped at epoch 20, R2 at
   30, R3 and DIS2-port ran 50. Only R3 vs DIS2-port is a like-for-like comparison,
   and R3 wins it by 0.0026 IoU / 0.0009 AUPRC.

---

## Things this inventory surfaces that were not in the earlier summary

1. **Only V1 has a method with a comfortable margin over its baseline.** Ours-V1
   recovers 0.64 IoU of the S0→T gap, and old DIS2-style beats Vanilla KD by 0.0121
   IoU. Every later block is much tighter or negative. The Windows N1 lead is +0.0062
   AUPRC, below the ±0.01 platform noise.
2. **The strongest documented claim in the project is not in the Windows block.**
   It is R3 − R2 = +0.0104 IoU in V3, i.e. "the progressive driver is the part that
   carries gain", with R2 ≈ R1 showing the inherited-Omega branch contributes nothing.
   The Windows protocol did not re-run R2, so this attribution has **no Windows
   re-measurement**.
3. **Two arms named in the plan have never been run under the Windows protocol**:
   `S0v2` (warm-start baseline) and `R0` (single-orbit floor for the R chain). The
   N0 arm covers the role, but it is not the same configuration.
4. **Three negative results are documented and usable**: geometry-priority weighting
   (channel authority, not targeting), the DIS2 orthogonality term (inert), and the
   official width-50 reproduction (never converged, deleted).
5. **The distortion-detection baselines answer a different question** and should
   never be merged with the landslide table. Their shadow F1 ≤ 0.041 across all five
   methods is itself worth a sentence somewhere.
6. **Block 1's gap-recovery number is the only "how much of the teacher gap is
   recovered" reading in the project.** If the paper wants that framing, the metric
   exists but only under V1.

---

## Where each block lives

| block | metrics | write-up |
|---|---|---|
| V1 | `results/rapid_landslide_cocd/` | `docs/DIS2_THREE_VERSIONS_RESULT.md` |
| V3 structure | `results/ours_v2/` | `results/ours_v2/ABLATION_R1R2R3_SUMMARY.md` |
| V3 geometry | `results/ours_v2/results_R3g_seed42.csv` | `reports/g10_geometry_distillation_alignment_audit.md` |
| External | `results/grsl_external_baselines/` | `reports/grsl_baseline_and_mechanism_completion.md` |
| Distortion | `results/six_single_sar_distortion_baselines/` | contract audit json |
| Windows | `experiments/windows_main/` | `reports/windows_main_results_so_far.md` |
| Protocols | — | `reports/protocol_v2_unification.md`, `cocd/configs/protocol_windows_main.json` |
