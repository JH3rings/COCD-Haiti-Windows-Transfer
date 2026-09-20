# Handoff — Haiti SAR single-orbit landslide COCD

You are picking up a research project whose code, data, splits, checkpoints and
recorded results have just been moved from a macOS machine to this one. Nothing
in this document refers to a conversation you did not see: everything needed to
continue is in this package.

**Read this file first, then `README_WINDOWS.md`. Do not start a training run
before `python verify_transfer.py` prints `TRANSFER CHECK PASSED`.**

---

## 1. The research question

> Rapid single-orbit post-disaster landslide mapping under view-dependent SAR
> geometric distortion.

The setting is operational. After an earthquake, whichever post-event SAR orbit
reaches the ground first is the one that gets used, so the deployed model has to
work from a **single orbit** at inference time even though training may see both.
The difficulty is that the same terrain looks different depending on the viewing
direction: terrain that is imaged well from ascending may be in layover or shadow
from descending, and vice versa. A single-orbit model therefore has to be
evaluated *conditioned on how badly its own view is distorted and whether the
other view would have been usable* — which is what the four evaluation regions
below are for.

**Deployment input** — exactly five channels, nothing else:

```
[VV_pre, VH_pre, VV_post, VH_post, OrbitID]     128 × 128
```

The student is **never** given, at training or test time:

- the counter orbit (the other viewing direction),
- a geometry mask,
- a DEM,
- optical imagery.

The geometry masks exist in the dataset but are **evaluation-only**. They
partition the pixels so results can be reported per region; they never enter a
model, a loss, or a training loop.

## 2. Data protocol

Split by **spatial location**, fixed, already pinned (see `DATASET_MANIFEST.md`):

| partition | locations | role |
|---|---:|---|
| train | 1,233 | training |
| validation | 137 | checkpoint selection, threshold selection |
| test | 343 | one-shot evaluation, never used for any choice |

All four acquisitions plus both masks of a location travel together, so a
location cannot straddle two partitions.

**Training samples** — 2 orbits × 3 temporal modes per train location:

| mode | pair | label | purpose |
|---|---|---|---|
| `pre→post` | pre-event → post-event | the landslide mask | the real task |
| `pre→pre` | pre-event → pre-event | all zero | stops the model reading a single post image |
| `post→post` | post-event → post-event | all zero | symmetric |

3,699 spec items × 2 orbits = **7,398 single-orbit samples per epoch**, of which
2,466 are the real `pre→post` pairs.

**Validation and test never generate the synthetic pairs.** They evaluate the real
`pre→post` pair of both orbits: 274 single-orbit cases for validation, 686 for
test.

**Evaluation partitions.** `gt` = target-orbit geometry mask > 0 (distorted),
`gc` = counter-orbit geometry mask > 0:

| region | meaning |
|---|---|
| `G00` | target view valid, counter view valid |
| `G01` | target view valid, **counter distorted** |
| `G10` | **target distorted**, counter valid |
| `G11` | both distorted |

`G10` is the region the whole study is about: the deployed view is the degraded
one and the other view was usable. Report overall / ASC-target / DESC-target ×
these four regions, always.

## 3. The unified training protocol (frozen)

This is the protocol **every** internal arm must use. It is defined once in
`cocd/configs/protocol_v2.json` and enforced by `cocd/scripts/48_smoke_protocol_v2.py`
(57 assertions, all passing in this package).

| knob | value |
|---|---|
| optimiser | `Adam` |
| learning rate | `5e-5`, **constant** |
| scheduler | **none** |
| betas / eps / weight decay | `(0.9, 0.999)` / `1e-8` / `0` |
| batch | physical 16, no accumulation (`physical × accumulate` must equal 16 or the script exits) |
| segmentation loss | **plain BCE** — no `pos_weight`, no Dice, no Focal |
| epochs | at most 20 |
| validation | every epoch |
| early stop | `patience = 3` on validation AUPRC |
| checkpoint | best validation AUPRC |
| threshold | searched on **validation** by max F1, frozen, then applied to test; AUPRC is threshold-free |
| test | **no** threshold re-search, no checkpoint re-selection, no hyper-parameter choice |
| masks | raw / no-mask in training and in inference; geometry only partitions the evaluation |
| initialisation | ImageNet for every internal arm; nothing is warm-started from another arm |
| seed | 42 |

### Discrepancies between that summary and what the code actually does

Things to know rather than discover:

1. **`R3D` is still in `PLAN`.** `cocd/scripts/23_train_ours_v2.py` currently has
   `PLAN = ['SO', 'VKD', 'DIS2', 'R3', 'R3D']`, where `R3D` is `R3` plus a
   decision-change distillation term (`L_delta`). That term has been **withdrawn
   from the new method** (see §5) and must not be trained as part of it. Either
   drop it from `PLAN` or drive the sweep with an explicit `--stages` list.
2. **The v1 arms did not use this protocol.** The already-reported numbers
   (`R1`/`R2`/`R3`/`R3g`/`DIS2-port`/`teacher.pt`) come from AdamW lr 1e-4 wd 1e-4,
   batch 2 (teacher) and 8 (students), BCE + 0.2·Dice with `pos_weight = 14.356`,
   50 epochs, a warm start from `S0`, and a fixed 0.5 threshold. **v1 and v2
   absolute values are not comparable**: dropping the warm start moves the
   segmentation loss at initialisation from ≈0.01 to ≈0.74.
3. **The 137 validation locations are pinned as explicit id lists**, not
   re-derived from the seed. That is stricter than the JSON says
   (`internal_val_fraction 0.1, seed 42`), and deliberate: `cocd/scripts/21`
   reads `data/splits/*_ids.csv` before falling back to the random draw.
4. **Early stopping was active on the v1 arms but not on the v2 teacher.** The v2
   teacher ran all 20 epochs (its patience never triggered); its learning curve is
   flat by the end (`results/ours_v2/teacher_v2_curve.csv`). When you compare two
   arms, check whether both ran the same number of epochs.
5. **"Two-thirds of the samples are inert" is a v1 statement.** With the v1
   warm start the model saturated to a probability of exactly 0 on the synthetic
   no-change pairs, so those samples produced exactly zero gradient and about
   44 % of the steps were zero-gradient coasting steps. **Under v2 this does not
   hold**: measured on `teacher_v2.pt`, the probability is non-zero at 100 % of
   pixels on those samples, so they train normally. Do not carry the v1 claim
   into a v2 description.
6. **One shared number is protocol-dependent in a way worth remembering.**
   The selective distillation weight `w = gain × protect` is active on about
   12 % of pixels under v1 and about 61 % under v2. The new design does not use
   `w` at all, so this matters only if you evaluate the legacy `R3` arm.
7. **`R3g` is deprecated** and remains in `CONFIGS` with a `deprecated` flag.
   Geometry-weighted distillation is off the table.
8. **External baselines are not part of this protocol.** Boehm / CDNetE /
   FC-Siam / MFEWF keep their own architectures and are not forced onto v2. Their
   result tables are in `results/grsl_external_baselines/`; their weights were not
   shipped.

### How to run it

```bash
python cocd/run_protocol_v2.py plan      # show what would run, change nothing
python cocd/run_protocol_v2.py train     # teacher (if absent) then the arms
python cocd/run_protocol_v2.py test      # the teacher's one-shot test table
```

Everything is also reachable directly, which is the supported entry point:

```bash
python cocd/scripts/23_train_ours_v2.py all --stages SO,VKD,DIS2,R3 --tag _v2
python cocd/scripts/23_train_ours_v2.py teacher --tag _cocd --epochs 20
python cocd/scripts/50_protocol_val_report.py --tag _v2 teacher    # validation only
```

Outputs land in `experiments/ours_v2/` (override with `COCD_OUT_ROOT`):
`<ARM>_seed42.pt`, `teachers`, `results_*.csv`, `thresholds_*.json`, per-arm logs.
The `_v2` tag keeps a new run away from the shipped `teacher_v2.pt`.

## 4. Legacy methods — these are baselines, not the new method

| name | what it is | where |
|---|---|---|
| **Additive Cross-Orbit Teacher** | shared ConvNeXt-Tiny/FPN, counter orbit enters as an additive correction, `Ω(a,c) = op(cat(a,c)) − op(cat(a,0))`, three readings `z0`/`z4`/`z34` off one decoder | `cocd/models/landslide_cocd_v2.py::OursV2Teacher` |
| **R0 / R1 / R2 / R3** | the structural ladder on the single-orbit student: no correction / free residual head / inherited `Ω` + a driver / progressive drive | `OursV3Student` |
| **Vanilla KD** | the same student distilled with no per-pixel selection (`w ≡ 1`) | `CONFIGS['VKD']` |
| **DIS2-port** | the DIS2 rule ported verbatim onto our backbone and taps | `cocd/losses/distill.py::dis2_multilevel_kd` |
| **Single-Orbit Baseline** | the student with no correction and no distillation | `CONFIGS['SO']` |

**`R3` is not the new COCD.** R3 inherits the teacher's `Ω` operator and is
distilled with the selective logit rule. The new design replaces the knowledge
object entirely (a complement representation instead of a correction) and drops
the selective rule.

Optional: a run in which the teacher itself is replaced by an external model.

## 5. The new COCD — final design, not yet implemented

Two claims, one network module each, one distillation term. Anything not written
here is not part of the method.

### 5.1 New Teacher — Deformable Complement Extraction

```
Target SAR  ──► shared Encoder+FPN ──► F_t ─────────────┐
                                                        │
Counter SAR ──► the same Encoder+FPN ──► F_c            │
                                                        │
        H_tc = DCA(F_t, F_c)         ← the only new module
                                                        │
        C_T  = Φ(F_t, H_tc) − Φ(F_t, DCA(F_t, 0))       │
                                                        │
        F_T  = F_t + C_T  ◄──────────────────────────────┘
                 │
            shared decoder D
                 │
        z_dual = D(F_T)      z_self = D(F_t)
```

- `F_t`, `F_c` are the **fused semantic feature** of the FPN (in this codebase,
  the tensor called `p2`: 128 channels at 1/4 resolution, 32×32 for a 128 input).
- `DCA` is a **deformable cross-attention**: for each position of `F_t`, sample a
  few learned offsets from `F_c` and weight them. It exists because ascending and
  descending geometry is not pixel-to-pixel aligned, so a query has to look in a
  neighbourhood of the other orbit rather than at the same coordinate.
- `Φ` is a plain convolutional projection of `(F_t, H_tc)`.
- **`C_T` is defined by a difference**, and the second term is the same expression
  with the counter feature set to zero. When `F_c = 0` the two terms are literally
  the same expression, so **`C_T(F_c=0) ≡ 0` is an algebraic identity** — the zero
  reference is exact by construction, and no constraint has to be placed on `DCA`
  to obtain it.
- **The complement is injected once only**, at the fused semantic feature:
  `F_T = F_t + C_T`. Not at several pyramid levels. The reason to state it in the
  method section: the FPN already does multi-scale modelling, so COCD does not
  repeat multi-scale fusion — it learns what the other orbit contributes.
- Loss, and nothing else:

```
L_T = 0.5 · L_seg(z_self, y) + 0.5 · L_seg(z_dual, y)
```

`C_T` has no separate supervision. Its only route to a gradient is
`F_t + C_T → landslide prediction`, so the landslide ground truth is its task
signal.

**Absent by design**: gate, task-relevance head, sparsity or anchoring term,
`gain × protect`, logit-level KD, multi-level injection, `L_delta`, `L_corr`, any
extra regulariser.

### 5.2 New Student — Orbit-Conditioned Complement Prediction

```
Single SAR ──► shared Encoder+FPN ──► F_t
                                       │
OrbitID ──► e_o ──► MLP ──► (γ_o, β_o) │
                                       ▼
                    F̃_t = (1+γ_o) ⊙ F_t + β_o      ← FiLM, complement branch only
                                       │
                              C_S = P(F̃_t)          ← two or three conv blocks
                                       │
                              F_S = F_t + C_S
                                       │
                                  shared decoder D
                                       │
                                  z_S = D(F_S)
```

- Two learnable embeddings only, `e_ASC` and `e_DESC`. Their meaning is **which
  opposite view is currently missing**, not "this is ASC/DESC".
- **The modulation touches the complement branch only.** The main segmentation
  path stays the unmodulated `F_t`, so the backbone is a clean single-orbit
  baseline and the orbit condition answers one question only.
- `P` is deliberately plain: no token, no mixture of experts, no router, no
  dynamic convolution, no separate spatial gate, no multi-scale predictor.
- Injection point identical to the teacher: `F_S = F_t + C_S`, once.

### 5.3 Distillation — one term

```
L_S = L_seg(z_S, y) + λ · L_dist(C_S, sg(C_T))

L_seg   on all three temporal modes (pre→post, pre→pre, post→post)
L_dist  = D(C_S, sg(C_T))   on the REAL pre→post pairs only
```

- `sg` is stop-gradient: `C_T` comes from the frozen trained teacher.
- **Only on real pairs.** The synthetic `pre→pre` / `post→post` pairs have their
  labels forced to zero; asking the student to match the teacher's complement on
  them is not meaningful. This is a protocol rule, not a loss weight.
- `D` is **not yet chosen** (L1 / MSE / SmoothL1 / cosine). Recommendation:
  a plain spatial mean, since a feature-matching term has no class imbalance and
  does not need foreground/background balancing.

### 5.4 Calibrating λ

On a fixed **train real `pre→post` subset**, with no parameter update, measure
`L_seg⁰` and `L_dist⁰`, then

```
λ = 0.1 · L_seg⁰ / L_dist⁰          so that at the start   λ·L_dist ≈ 0.1 · L_seg
```

and **freeze λ permanently**. Do not look at validation, do not sweep it.
Rationale: the segmentation ground truth is the real task and must dominate; a
term at ~1 % would repeat the earlier failure mode of a distillation term that
exists but has no optimisation authority.

Calibrate **once, at the shared ImageNet initialisation, and reuse the same λ for
every arm** — otherwise the arms have different auxiliary strength and the
comparison stops being controlled.

Treat 10 % as the **initial** ratio, not a running one: `L_seg` falls quickly from
0.74 while `L_dist` also falls as `C_S` approaches `C_T`, so the ratio drifts.
Record the measured ratio at a few epochs as a diagnostic, and report "≈10 %
auxiliary strength at initialisation" in the paper. Do not build a normalisation
system around it.

### 5.5 Ablation — three arms

| arm | content | question |
|---|---|---|
| `N0` | single-orbit baseline, `F_t → D` | floor |
| `N1` | `F_t + C_S → D`, segmentation loss only | does the extra branch help on its own |
| `N2` | `N1` + `L_dist(C_S, C_T)` = Full COCD | does explicit complement distillation help |
| `N2 w/o Orbit Embedding` | Full without the FiLM conditioning | is the orbit condition necessary |

There is **no N3**. Do not reintroduce `L_delta`, `L_corr`, or the selective rule.

## 6. Order of work

1. **Implement the new Teacher** — `DCA` + `Φ` + the difference-defined `C_T`, one
   injection point at the fused feature, `L_T = 0.5·L_seg(z_self) + 0.5·L_seg(z_dual)`.
   Keep `OursV2Teacher` intact; add a new class.
2. **Train it** under the frozen v2 protocol, 20 epochs, seed 42, ImageNet init.
   Ship its learning curve and its validation readings for `z_self` / `z_dual`.
3. **Teacher audit — a gate, not a formality.** On the validation set, with the
   trained teacher:

   | condition | how |
   |---|---|
   | correct counter | the other orbit of the same location |
   | zero counter | the counter feature set to zero |
   | shuffled counter | the other orbit of a **different** location |

   Report: (a) is `dual > self`; (b) is correct clearly better than zero and
   shuffled; (c) does `C_T` change sensibly across the three conditions. Also
   report `dual − self` **per region** — every previous round found `G10` to have
   the largest gain, so reproducing that ordering is the strongest available
   evidence that the counter benefit lands where it should.
   Two cautions: the zero-counter case is degenerate (`C_T ≡ 0` by construction),
   so the informative contrast is **correct vs shuffled**; and the pass threshold
   should be pre-registered using the teacher's own validation curve variability
   as the noise floor.

   **If the audit fails, do not distil a student from this teacher.**
4. **Freeze the teacher** and stop touching it.
5. **Run / confirm `N0`** — the single-orbit baseline under the same protocol.
6. **Train `N1`** (complement predictor, segmentation loss only).
7. **Decide `D`** in `L_dist` and its aggregation.
8. **Calibrate λ** on the fixed train real subset, once, and freeze it.
9. **Train `N2`** (Full COCD).
10. **Train `N2 w/o Orbit Embedding`.**
11. **Complete the legacy comparison** on the same protocol: Vanilla KD,
    DIS2-port, R3. `teacher_v2.pt` is already the v2 teacher, so only the arms
    need training.
12. **Final unified test evaluation** — every arm, one threshold each, selected on
    validation and frozen, reported overall / ASC / DESC × `G00`/`G01`/`G10`/`G11`.

Then a single table where the arms differ only in the mechanism that names them.

## 7. Things that must not happen

- **Do not skip the teacher audit and distil a student anyway.** A teacher that
  did not learn the complement produces a student trained toward noise, and the
  run costs hours.
- **Do not select anything on the test partition** — not a threshold, not a
  checkpoint, not a model, not a hyper-parameter. `cocd/scripts/50_protocol_val_report.py`
  exists so a validation-only readout is always available.
- **Do not re-draw the split.** `data/splits/*_ids.csv` are pinned; read them.
- **Do not warm-start a new arm from a legacy checkpoint.** Everything internal
  starts from ImageNet. `checkpoints/teacher_v2.pt` is read as the *teacher*, not
  as an initialisation.
- **Do not put the geometry mask into training**, in any form, including as a
  distillation weight. That direction was tried and closed (`R3g`).
- **Do not change the protocol to make a result look better.** If an arm needs a
  different protocol, it is a different experiment and must be reported as such.
- **Do not treat `R3` as the new COCD.**
- **Do not carry v1 numbers into a v2 comparison** or the other way round.

## 8. What is in this package

```
README_WINDOWS.md              unpack, environment, verify, what to do first
HANDOFF_FOR_WINDOWS_AI.md      this file
DATASET_MANIFEST.md            data layout, naming, split, integrity, QC
CHECKPOINTS.md                 every shipped checkpoint, its metrics and hashes
ENVIRONMENT.md                 the macOS versions and the Windows recommendation
TRANSFER_AUDIT.md              what was kept, what was dropped, and why
requirements.txt
verify_transfer.py             acceptance check, prints TRANSFER CHECK PASSED

cocd/
  paths.py                     the only place that resolves locations and device
  configs/protocol_v2.json     the frozen protocol
  models/landslide_cocd.py     shared ConvNeXt-Tiny + FPN backbone
  models/landslide_cocd_v2.py  the teacher and the structural student ladder
  losses/landslide_cocd.py     segmentation losses
  losses/distill.py            the DIS2 rule and the selective rule
  manifests/                   the upstream split manifests
  metadata/                    per-file inventory of the raw data (QC reference)
  scripts/21_train_rapid_landslide_cocd.py   dataset loader + frozen split
  scripts/23_train_ours_v2.py                training / evaluation entry point
  scripts/47,48,50_*.py                      batch feasibility, protocol asserts, val readout
  scripts/14_balanced_spatial_split.py       provenance of the frozen split
  scripts/32,36,37,39,40,43,44,45,46_*.py    verification and diagnostics behind the recorded numbers
  third_party/landslide_baselines/dis2/      the DIS2 snapshot the port is diffed against

data/haiti/                    1,372.6 MB, 13,704 files, per-file sha256 in DATA_MANIFEST.csv
data/splits/                   the pinned train / val / test id lists
checkpoints/teacher_v2.pt      the v2 teacher (required by the distillation arms)
legacy_checkpoints/v1_protocol/  the six v1 checkpoints behind the reported tables
results/                       every recorded csv / json / log, by experiment line
docs/, reports/                the design and audit documents, newest last
```

Months of work compressed into this paragraph: the project established that a
single-orbit student can be pushed above its own single-orbit ceiling by
distilling from a model that sees both orbits, that the benefit of the second
orbit is concentrated where the deployed view is degraded (`G10`) and is mostly
*false-positive suppression* rather than new detections, and that a correction
representation learned that way behaves more like counter-orbit feature novelty
than like task-relevant evidence. The last point is why the new design makes the
complement an explicit, zero-referenced object rather than a side effect of an
additive correction.
