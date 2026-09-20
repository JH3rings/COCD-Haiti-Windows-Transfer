# Counter-orbit knowledge transfer audit (2026-09-20)

## Verdict

The current evidence establishes that the New COCD Teacher uses the correct
counter orbit and that its largest regional AUPRC gain is on G10. It also shows
that raw complement reconstruction is not a suitable transfer target: the free
N1 complement is farther from `C_T` than the zero predictor, and both the old
validation-era and current v4 N2 runs are worse than N1 on overall and G10.

On a Train-only diagnostic subset, the Student's decision effect aligns much
more strongly with the Teacher's decision effect than its raw complement aligns
with the Teacher complement. A single decision-effect KD experiment is therefore
scientifically motivated. It was **not trained in this audit**, because the clean
validation-era NewTeacher/N1 checkpoints have been overwritten by v4 checkpoints
selected with test AUPRC, and the `N2_lam0p1` checkpoint is absent. Training only
the new arm now would not leave a matched, test-independent N1 control.

## Evidence levels and protocol conflict

- **Old validation-era evidence:** `WINDOWS_CONTINUATION_LOG.md`,
  `reports/windows_main_results_so_far.md`, and the frozen protocol diagnostic
  record. The corresponding N0/N1/N2/NewTeacher weight files no longer exist as
  that run state, so these numbers are log-verified but cannot all be recomputed
  from the current checkpoint paths.
- **Current v4 evidence:** recomputed directly from the present seed-42
  checkpoints on 343 test locations / 686 orbit cases at fixed threshold 0.5.
  These checkpoints were selected by test AUPRC, as their progress JSON files
  explicitly state. The readings are therefore current and reproducible but
  optimistically biased, and cannot be used to choose a new loss.
- **Predictability evidence:** recomputed without optimisation on the 137 IDs in
  `data/splits/val_ids.csv`. These IDs are part of v4 Train, so this is a
  Train-only mechanism diagnostic, not a held-out result.

## Object and loss audit

| Method | Added structure | Added supervision |
|---|---|---|
| N0 | none; `F_t -> Head` | BCE |
| N1 | orbit-conditioned complement predictor; `F_t + C_S -> Head` | BCE |
| N2 | exactly N1 | BCE + `lambda SmoothL1(C_S, stopgrad(C_T))` on real pairs |
| New COCD Teacher | shared ConvNeXt-Tiny/FPN, DCA + Phi, one p2 injection | mean BCE on shared-head `z_self` and `z_dual` |
| Vanilla KD | legacy R1 Student | BCE + logit KD |
| DIS2-port | legacy R1 Student; no extra pooling parameters in the port | BCE + four feature matches + penultimate match + logit KL + two diversity terms |
| Legacy R3 | legacy R3 correction structure | BCE + detached `gain * protect` selective KD |

Code inspection confirms that N1 and N2 instantiate the same
`ComplementStudent(complement=True, orbit_embedding=True)` and differ only in
whether the one complement distance is added. Every internal arm resets seed 42
before construction and starts from ImageNet; N2 lambda=0.1 used the same arm and
an isolated `lam_override`, with the run log recording otherwise identical
settings. The Teacher routes both readings through the same `backbone.head`; its
`C_T` and the Student `C_S` are both `(N,128,32,32)` and are injected once at p2.

DIS2 has all four named components active. In the v4 log at epoch 1 they were
`feat=0.0089`, `pen=0.0051`, `logit=0.2267`, `div=0.0040`; by epoch 20 they were
`0.0008`, `0.0010`, `0.0418`, and approximately `0`. Thus diversity becomes
negligible, while logit KL remains dominant. R3 computes a detached relative
Teacher error gain, clamps it to `[0,1]`, multiplies it by a protection indicator
that keeps only pixels where the Teacher dual prediction beats the Student, and
uses that weight for selective KD.

## Q1: Teacher availability

Current v4 checkpoint, recomputed test AUPRC:

| Partition/region | Self | Dual | Dual - self |
|---|---:|---:|---:|
| Overall | 0.850081 | 0.881685 | +0.031605 |
| ASC | 0.848577 | 0.881519 | +0.032942 |
| DESC | 0.851605 | 0.881853 | +0.030248 |
| G00 | 0.838784 | 0.870959 | +0.032174 |
| G01 | 0.869058 | 0.893667 | +0.024609 |
| **G10** | **0.854978** | **0.893065** | **+0.038088** |
| G11 | 0.903597 | 0.928849 | +0.025252 |

G10 is the largest of the four regional gains. Correct-counter dual AUPRC is
0.881685; zero-counter is exactly the self reading, 0.850081. Five fixed shuffled
counters give 0.451345, 0.415040, 0.436314, 0.435405, and 0.429042. Therefore
`correct > zero` and `correct > shuffled` hold with large margins.

The old validation-era audit independently had the same qualitative result:
overall `0.68967 - 0.64594 = +0.04373`, with G10 `+0.0522`, the largest region.

## Q2: Predictability

Read-only Train-subset diagnostic using current v4 NewTeacher and N1 weights.
Raw-feature metrics are elementwise at p2 resolution; norms are RMS magnitudes.

| Scope | `C_S,C_T` cosine | SmoothL1 | SmoothL1 of zero | norm ratio `C_S/C_T` |
|---|---:|---:|---:|---:|
| Overall | 0.7010 | 0.9036 | **0.6156** | 1.7855 |
| Foreground | 0.1963 | 0.3526 | **0.0888** | 2.3648 |
| Background | 0.7056 | 0.9444 | **0.6546** | 1.7811 |
| G00 | 0.7066 | 0.9383 | **0.6537** | 1.7764 |
| G10 | 0.6845 | 0.6934 | **0.5193** | 1.6236 |

`C_T` RMS is 1.4275 overall and 1.3059 on G10; `C_S` RMS is 2.5487 overall and
2.1203 on G10. N1's free complement is farther from `C_T` than zero in every
reported scope. Raw `C_T` is therefore not the representation naturally adopted
by the successful free complement branch.

Decision effects use logits: `Delta z_T = z_dual-z_self` and
`Delta z_S = z_corr-z_base`.

| Scope | cosine | Pearson r | Student RMS | Teacher RMS |
|---|---:|---:|---:|---:|
| Overall | 0.8266 | 0.5843 | 17.0242 | 17.8858 |
| Foreground | 0.6826 | 0.5445 | 2.9111 | 1.3634 |
| Background | 0.8270 | 0.5268 | 17.6216 | 18.5282 |
| G00 | 0.8301 | 0.5647 | 17.5594 | 18.5926 |
| **G10** | **0.8241** | **0.6608** | 13.9751 | 15.8823 |

The decision-effect target is diagnostically more aligned than the raw latent
target, especially on foreground and G10. This supports testing exactly one
auxiliary effect loss, but it is not itself evidence that effect KD improves
segmentation.

## Q3: Recoverability

### Old validation-era AUPRC (log-verified)

| Method | Overall | G00 | G01 | G10 | G11 |
|---|---:|---:|---:|---:|---:|
| N0 | 0.6443 | 0.6373 | 0.6680 | 0.6439 | 0.6743 |
| N1 | 0.6504 | 0.6425 | 0.6761 | 0.6541 | 0.6682 |
| N2, lambda=1 | 0.5583 | 0.5472 | 0.6034 | 0.5542 | 0.5778 |
| N2 diagnostic, lambda=0.1 | 0.6314 | 0.6244 | 0.6598 | 0.6314 | 0.6438 |
| NewTeacher self | 0.6459 | null | null | null | null |
| NewTeacher dual | 0.6897 | null | null | null | null |
| Vanilla KD | 0.6033 | 0.5968 | 0.6277 | 0.5961 | 0.6446 |
| DIS2-port | 0.5990 | 0.5901 | 0.6305 | 0.6004 | 0.6385 |
| Legacy R3 | 0.6107 | 0.6023 | 0.6433 | 0.6041 | 0.6535 |

The Teacher's region-specific self/dual values were not retained in the old
summary, only their differences; missing cells remain `null` rather than being
reconstructed.

Old validation causal differences: N1-N0 is +0.0062 overall and +0.0102 on G10;
N2(lambda=1)-N1 is -0.0922 overall and -0.0999 on G10; the lambda=0.1 diagnostic
is still -0.0191 overall and -0.0227 on G10. Raw complement KD did not transfer
G10 capability.

### Current v4 test, fixed threshold 0.5 (checkpoint-recomputed)

Each cell is `AUPRC / IoU / F1`.

| Method | Overall | G00 | G01 | G10 | G11 |
|---|---:|---:|---:|---:|---:|
| N0 | .8796/.6335/.7756 | .8684/.6179/.7638 | .8952/.6575/.7934 | .8883/.6440/.7834 | .9307/.7181/.8359 |
| N1 | .8304/.5695/.7257 | .8170/.5543/.7132 | .8502/.5981/.7485 | .8396/.5734/.7289 | .8946/.6589/.7944 |
| N2, lambda=1 | .8025/.5261/.6895 | .7885/.5089/.6745 | .8268/.5578/.7162 | .8094/.5343/.6965 | .8616/.6048/.7537 |
| NewTeacher self | .8501/.5877/.7403 | .8388/.5734/.7289 | .8691/.6129/.7600 | .8550/.5931/.7446 | .9036/.6711/.8032 |
| NewTeacher dual | .8817/.6230/.7677 | .8710/.6068/.7553 | .8937/.6409/.7811 | .8931/.6404/.7808 | .9288/.7123/.8320 |
| Vanilla KD | .8303/.5818/.7356 | .8175/.5673/.7239 | .8489/.6046/.7536 | .8393/.5919/.7436 | .8894/.6547/.7913 |
| DIS2-port | .8733/.6177/.7637 | .8627/.6012/.7510 | .8898/.6434/.7830 | .8802/.6290/.7723 | .9208/.7012/.8243 |
| Legacy R3 | .7536/.5104/.6758 | .7389/.4961/.6632 | .7786/.5359/.6979 | .7593/.5169/.6815 | .8296/.5818/.7356 |

These v4 causal readings are also negative: N1-N0 is -0.0493 overall and
-0.0486 on G10; N2-N1 is -0.0279 overall and -0.0303 on G10; DIS2-N0 is -0.0063
overall and -0.0081 on G10. Because test selected the stopping epoch, these are
not clean held-out estimates, but they do not supply evidence of G10 transfer.

## Required conclusion table

| Question | Evidence | Conclusion |
|---|---|---|
| Does counter orbit help Teacher? | dual-self; current G10 +0.0381 and old validation G10 +0.0522; correct > zero and 5/5 shuffles | **YES** |
| Does Student architecture help G10? | old N1-N0 +0.0102, but current v4 N1-N0 -0.0486 and only seed 42 | **unclear** |
| Is raw `C_T` predictable? | N1 `C_S` SmoothL1 is worse than zero overall, foreground, background and every G-region | **NO** |
| Does raw complement KD help G10? | old lambda=1: -0.0999; old lambda=0.1: -0.0227; current v4: -0.0303 vs N1 | **NO** |
| Is Teacher decision effect more predictable? | effect cosine/Pearson exceed raw alignment; G10 0.824/0.661 | **YES, diagnostically** |
| Does a new single effect KD help G10? | no clean matched run was executed | **null** |
| Does DIS2 transfer G10 capability? | old DIS2-N0 -0.0435; current v4 DIS2-N0 -0.0081 | **NO** |

## Training decision and frozen criteria

Raw `C_T` matching is stopped; no lambda sweep is justified. The only admissible
new target is `SmoothL1(Delta z_S, stopgrad(Delta z_T))`, added to BCE and nothing
else. Before such a run, lambda must be calibrated once on Train so that the
weighted auxiliary term is approximately 10--20% of initial segmentation loss,
then frozen.

The pre-registered success criterion is effect-KD > matched N1 overall and,
primarily, effect-KD(G10) > matched N1(G10). Overall unchanged with G10 improved
is a useful positive result. If neither improves, the conclusion is that no
transfer has been demonstrated and the line stops.

The minimum valid execution unit is a clean old-development-protocol retrain of
NewTeacher, matched N1, and effect-KD under unique non-overwriting tags, because
the needed clean checkpoints are no longer present. Test must remain untouched
until those models, lambda, checkpoint, and criterion are frozen.

## One-sentence answer

At present, **no**: the Teacher has a verified counter-orbit advantage and the
Student naturally expresses a correlated decision effect, but no existing
distillation run demonstrates that a single-orbit Student gained cross-orbit
complementary decision capability over its matched N1 control.
