# WINDOWS_CONTINUATION_LOG.md

Migration log for the Haiti SAR single-orbit landslide COCD project, moved from
macOS (Apple MPS) to this Windows/CUDA host.

Rules followed throughout: no method change, no protocol change, no loss change,
no network change. Cross-platform repair only. Every phase below records what was
inspected, what was run, what came out, and what is still open.

Environment used for every command in this log:

| item | value |
|---|---|
| interpreter | `D:\deeplearning\anaconda3\python.exe` — Python 3.12.4 |
| torch | 2.6.0+cu126 (CUDA build) |
| torchvision | 0.21.0+cu126 |
| rasterio / GDAL | 1.4.3 (bundled GDAL) |
| numpy / pandas / scikit-learn / tqdm | 1.26.4 / 2.2.2 / 1.4.2 / 4.66.4 |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16,380 MiB, capability 8.9, driver 572.47 |

No `.venv` was created. The machine already carries this interpreter with every
required wheel present and a CUDA build of torch, so a fresh venv would have
re-downloaded ~2.5 GB to reproduce it. Everything in this log is read-only or
validation-only; nothing was installed or upgraded.

---

## Phase 1 — read the handoff, change nothing

**Date** 2026-09-18

Read: `HANDOFF_FOR_WINDOWS_AI.md`, `README_WINDOWS.md`, `CHECKPOINTS.md`,
`ENVIRONMENT.md`, `TRANSFER_AUDIT.md`, `DATASET_MANIFEST.md`,
`cocd/paths.py`, `cocd/configs/protocol_v2.json`, `cocd/run_protocol_v2.py`,
`cocd/models/landslide_cocd.py`, `cocd/models/landslide_cocd_v2.py`,
`cocd/losses/landslide_cocd.py`, `cocd/losses/distill.py`,
`cocd/scripts/21_train_rapid_landslide_cocd.py`,
`cocd/scripts/23_train_ours_v2.py`, `verify_transfer.py`,
`reports/protocol_v2_unification.md`, and the `results/ours_v2/` curves.

**Modified files** none.
**Reason** Phase 1 is reading only; the instruction is to understand before acting.
**Category** n/a.

Findings: see the summary delivered with this run and section "Open conflicts"
below. Nothing in the package was edited at this stage.

**Next** Phase 2.

---

## Phase 2 — Windows / CUDA acceptance

**Date** 2026-09-18
**Category** cross-platform verification (no repair was needed)

Command:

```
D:\deeplearning\anaconda3\python.exe -u verify_transfer.py
```

Result:

```
TRANSFER CHECK PASSED
  2 note(s) above; read them before starting a run.
```

42 checks, 0 failures, 0 warnings, 2 notes. The two notes are informational: the
CUDA device is present with 16.0 GiB (above the 8 GiB threshold that would have
triggered the batch warning), and the v1-protocol checkpoints are baselines to
evaluate rather than initialisation for a new run.

Detail worth keeping:

- `torch.cuda.is_available()` is **True**; device `NVIDIA GeForce RTX 4060 Ti`,
  capability 8.9, 16.0 GiB. The CUDA warning path in section [2] was not taken.
- Device resolution picks `cuda`, i.e. the `CUDA -> MPS -> CPU` order in
  `cocd/paths.py` works and nothing in the training path is MPS-specific.
- Split files present and correct: 1233 / 137 / 343, disjoint, all ids < 1713.
- Dataset on disk: 1713 tif in each of the four acquisition folders, 1713
  `sample_*` directories, 1713 `landslide_mask.tif`.
- Loader: 18 samples for 6 locations (3 temporal modes each), input `(5,128,128)`,
  orbit channel a constant plane (ASC 0 / DESC 1), batch of 16 reaches the device.
- Forward passes: `OursV2Teacher.forward_pair` returns z0/z4/z34 at 128×128 plus
  `r3 (16,128,16,16)` and `r4 (16,128,8,8)`; teacher is 28.1237 M parameters;
  the tap path and the deployment path agree **bitwise**.
- All seven checkpoints load with `strict=True` through the code path that
  consumes them (392/392/392/400/400/392/190 tensors).
- Output root `experiments\` is writable.

**Windows-specific modifications made: none.** The port had already handled
device resolution, path resolution, `map_location`, `num_workers=0`, `__main__`
guards and the removal of `caffeinate` / BSD `stat` / hard-coded macOS paths
(`TRANSFER_AUDIT.md` §5). Nothing additional was required on this machine.

**Next** Phase 3.

---

## Phase 3 — data integrity and split audit

**Date** 2026-09-18
**Category** verification only, read-only

Tool written (new file, Windows-side, outside the shipped package):

```
windows_ops/phase3_data_split_audit.py
```

Command:

```
D:\deeplearning\anaconda3\python.exe -u windows_ops/phase3_data_split_audit.py
```

Result: **PHASE 3 AUDIT PASSED — 78 checks, 0 failures.** Nothing written. The
split was read, never re-drawn.

Confirmed independently:

| claim in the handoff | result |
|---|---|
| `train_ids.csv` / `val_ids.csv` / `test_ids.csv` SHA256 | all three match `DATASET_MANIFEST.md` §3 byte for byte |
| 1713 locations / train 1233 / val 137 / test 343 | exact, all unique |
| the three partitions are pairwise disjoint | yes |
| the union covers exactly `0 … 1712` | yes, nothing missing, nothing outside |
| 1233 + 137 = 1370 = the upstream 80/20 train call | yes |
| `split()` returns the pinned lists in the pinned order | yes |
| train spec = 3 × 1233 = 3699, modes `(0,1,2)` per location | yes, 1233 items per mode |
| val spec = 137, test spec = 343, no synthetic modes | yes |
| val / test single-orbit cases | 274 / 686 |
| same location never crosses a split | structural: `HaitiPairs.__init__` reads all four acquisitions and both masks of a location before any pairing, and the split is on `sample_id` |
| train standardisation stats reused by val and test | yes, same object; stats come from the train rows only (`mean [-8.1613, -14.7997]`, `std [5.2658, 5.078]`) |
| geometry masks never enter training | yes — see below |

Spot check, five locations read end to end (`0, 1, 857, 9, 1712`):

- all four stacks are `(2,128,128)` float32 for every location;
- mask and both geometry rasters are 128×128 with the documented coding
  (`{0,1,2,3}` for the mask, `{0,1,2,3}` for the geometry classes);
- **band order verified empirically**: band 2 is VV and band 1 is VH — the median
  VV−VH gap is 5.87–12.03 dB across the five locations, which is the expected
  separation and the opposite sign would have been visible immediately;
- pre ≠ post of the same orbit, ASC ≠ DESC, and the target and counter geometry
  masks are different rasters — so no location is silently reading one file twice;
- the loader's temporal modes behave exactly as documented: mode 1 feeds pre as
  post (channels 0/1 == 2/3), mode 2 feeds post as pre, and both carry an
  all-zero label; mode 0 has pre ≠ post and carries the real mask;
- the orbit channel is a constant 0 plane for ASC and a constant 1 plane for DESC,
  and it is the one channel the loader leaves unstandardised.

Geometry audit by static inspection of `cocd/scripts/23_train_ours_v2.py`:

- the two live KD modes (`'gain'`, `'all'`) are handed `g10` and never read it;
- only the deprecated `'full'` mode (`R3g`) consumes the geometry boost
  `w = (1 + 1_real·G10) · gain · protect`;
- `R3g` carries `'deprecated': True` and is not in `PLAN`;
- every other use of the geometry masks is inside the evaluation report
  (`report()`, `predict_student`, `predict_teacher_heads`, `teacher_curve_row`),
  which is exactly the "evaluation-only" role.

**One live footgun, not a defect in the protocol:** `R3g` remains launchable by
name, and `cocd/losses/landslide_cocd.py::gain_gate` (the stage-1 loss, used by
`scripts/21`, not by `scripts/23`) takes the two geometry masks as arguments.
Both are dormant under the v2 protocol and neither is reachable from `PLAN`.
Recorded here so it is a known state rather than a discovery later.

**Next** Phase 4.

---

## Phase 4 — legacy / shipped baselines smoke test

**Date** 2026-09-18
**Category** verification only, validation partition only, no training

Tool written (new file, Windows-side):

```
windows_ops/phase4_legacy_smoke.py
```

Command:

```
D:\deeplearning\anaconda3\python.exe -u windows_ops/phase4_legacy_smoke.py
```

Result: **PHASE 4 SMOKE PASSED.** No arm was retrained, no test prediction was
made.

### A. v2 teacher — `checkpoints/teacher_v2.pt`

Loaded through `load_teacher()`, evaluated on the 137 validation locations
(274 single-orbit cases).

| reading | this host | `CHECKPOINTS.md` | delta |
|---|---:|---:|---:|
| `z0` (self) | 0.630433 | 0.630431 | +1.8e-06 |
| `z4` | 0.651205 | 0.651206 | −1.1e-06 |
| `z34` (dual) | 0.667077 | 0.667081 | −4.0e-06 |

The residual 1e-6-scale difference is float reassociation on a different device
plus a different scikit-learn build; the recorded values were produced on MPS.
`average_precision_score` is the only metric in the package sensitive to this.
Treat the migration as reproducing the recorded numbers.

`tap_path_max_abs_diff = 0.0` on CUDA — the tap path that the distillation terms
score and the deployment path that a run reports are still bitwise identical on
this device, which is the property those terms rest on.

**The region ordering of the dual gain is reproduced, and `G10` is the largest:**

| region | dual − self AUPRC |
|---|---:|
| `G10` (target distorted, counter valid) | **+0.0497** |
| `G11` (both distorted) | +0.0384 |
| `G00` (both valid) | +0.0358 |
| `G01` (target valid, counter distorted) | +0.0269 |

This is the ordering the handoff says every previous round found, so the
incumbent v2 teacher is a usable reference for the new teacher's audit.

### B. v1 teacher — `legacy_checkpoints/v1_protocol/teacher.pt`

Loads and runs. `z34` validation AUPRC **0.89626**, exactly the recorded value;
`z0` 0.88091, `z4` 0.88556.

### C. v1 student arms

| checkpoint | this host | recorded | delta |
|---|---:|---:|---:|
| `R1_seed42.pt` | 0.89045 | 0.89040 | +4.9e-05 |
| `R2_seed42.pt` | 0.89090 | 0.89090 | +3.0e-06 |
| `R3_seed42.pt` | 0.89399 | 0.89400 | −1.4e-05 |
| `DIS2_port_seed42.pt` | 0.89320 | 0.89320 | −1.4e-06 |

All four load with `strict=True` and reproduce their recorded validation AUPRC.
The recorded numbers are maxima over the run log, so this confirms the shipped
`best_state` really is the selected epoch.

### D. stage-1 `S0.pt` and the SO / N0 path

- `S0.pt` loads into `SingleOrbitStudent(correction=False)` (its own legacy key
  space, 190 `backbone.*` tensors) and runs — output `(16,1,128,128)`.
- `OursV3Student('R0')`, i.e. the `SO` arm and the `N0` baseline, runs and
  returns a single reading with `z0 == z34` by construction (no correction).
- `CONFIGS['SO']` needs no teacher: `kd=None`, `init='scratch'`.

### E. runnability of the configs the handoff names

`SO`, `VKD`, `DIS2`, `R3`, `R3D` all exist with the expected variant / kd / init;
`R3g` exists and is flagged deprecated; the `--protocol v1` branch is still
reachable and still differs from v2 (AdamW, lr 1e-4, batch 2, 50 epochs); the v2
branch is the default and matches `protocol_v2.json` exactly.

### F. the v1/v2 scale gap

One validation batch through a freshly built `OursV3Student('R1')` at the shared
ImageNet initialisation: `seg = 0.7035`, initial sigmoid mean 0.5056, std 0.0144.
That is the order of magnitude the handoff records (0.742 for the whole
partition) and confirms the v1 and v2 absolute scales are not comparable. It also
confirms the model is not sitting on a constant background at initialisation.

### Legacy arms still missing a v2 run

`TRANSFER_AUDIT.md` is explicit and the code agrees: under the **v2** protocol
only the teacher has been trained. `SO`, `Vanilla KD`, `DIS2-port` and `R3` exist
as configs and as v1 checkpoints, not as v2 checkpoints. The legacy comparison
(HANDOFF step 11) therefore still costs four training runs; nothing was started
here.

**Next** Phase 5 — implement the new Teacher. Not started. See below.

---

## Protocol assertions, re-run

```
D:\deeplearning\anaconda3\python.exe -u cocd/scripts/48_smoke_protocol_v2.py
-> 57 passed, 0 failed
```

This matches `HANDOFF_FOR_WINDOWS_AI.md` §3 and `TRANSFER_AUDIT.md` §7 (57).
`reports/protocol_v2_unification.md` reports 44 then 54 — that report is stale
(see Conflict 4).

---

## Open conflicts — reported, then decided by the user (2026-09-19)

Per the handoff rule, a conflict between the documents and the code is reported
rather than silently settled. All six were reported first; the user then decided.
The decisions are listed after the six conflicts and are what the Windows main
run implements. None of them was resolved by editing a frozen file.

### Conflict 1 — `TEACHER_W` in the code is the rejected T2a allocation

- `cocd/scripts/23_train_ours_v2.py:178` sets `TEACHER_W = (0.25, 0.25, 0.50)`.
- The docstring directly above it (lines 172–179) and `teacher_batch()`
  (lines 328–341) state that **the frozen teacher used `(0.5, 0.25, 0.25)`** and
  that `(0.25, 0.25, 0.50)` is what *Teacher-T2a* used.
- `reports/cocd_final_teacher_student.md` §4 (lines 208–215) records that the
  original teacher loss balances to `z0 : z4 : z34 = 0.5 : 0.25 : 0.25`, and
  records T2a's verdict as **NOT IMPROVED — executed and stopped**.
- So the shipped default is the weighting that the reports say was tried once and
  abandoned, while the text describes the other one as "the frozen Teacher".
- Which allocation trained `checkpoints/teacher_v2.pt` is **not stated anywhere
  in the package**. The curve is suggestive — `z34 > z4 > z0` at every epoch and
  the shipped weights give `z34` twice the weight of either — which is what one
  would see under `(0.25, 0.25, 0.50)`, but the artefacts do not prove it.
- Consequence: `run_protocol_v2.py train --tag <new>` would silently train a
  teacher with the allocation the reports rejected. The new teacher's own loss is
  fixed by `HANDOFF_FOR_WINDOWS_AI.md` §5.1 (`0.5·L_seg(z_self) + 0.5·L_seg(z_dual)`)
  and is unaffected, so this matters for the legacy control and for any retrain.

### Conflict 2 — `PLAN` still contains `R3D`, and the config still funds `L_delta`

- `cocd/scripts/23_train_ours_v2.py:170` → `PLAN = ['SO', 'VKD', 'DIS2', 'R3', 'R3D']`.
- `cocd/configs/protocol_v2.json` still lists `"R3 + L_delta"` in
  `comparison_arms` and still carries a full `corrective_distillation` block.
- `HANDOFF_FOR_WINDOWS_AI.md` §3 item 1 says to drop `R3D` from `PLAN` or drive
  sweeps with an explicit `--stages` list; §5.3 and §7 say `L_delta` is withdrawn
  from the new method and must not be reintroduced.
- This is the documented discrepancy, confirmed in code. It is unresolved because
  fixing it edits the frozen protocol artefacts, which needs a decision.

### Conflict 3 — `R3g` and `gain_gate` remain reachable geometry-in-training paths

- `CONFIGS['R3g']` is launchable by name and `selective_kl(mode='full')` consumes
  `g10`; `cocd/losses/landslide_cocd.py::gain_gate` takes both geometry masks.
- Both are dormant: `R3g` is flagged deprecated, is out of `PLAN`, and assertion
  [H] passes. But `HANDOFF_FOR_WINDOWS_AI.md` §7 forbids geometry in training "in
  any form", and a name-only launch is one typo away.
- Reported rather than removed: deleting a config entry is a change to the frozen
  protocol file.

### Conflict 4 — `reports/protocol_v2_unification.md` contradicts the handoff

Three points, all staleness in the report rather than in the code:

1. **§1.4** asserts the synthetic `pre/pre` and `post/post` samples have
   "损失与梯度精确为 0". `HANDOFF_FOR_WINDOWS_AI.md` §3 item 5 says explicitly
   this is a **v1** statement that does **not** hold under v2 (measured on
   `teacher_v2.pt`, the probability is non-zero at 100 % of those pixels). The
   report must not be quoted for a v2 description.
2. **§5 / §9** give `bash scripts/49_run_protocol_v2.sh train` as the entry point.
   That `.sh` was deliberately dropped from the package (`TRANSFER_AUDIT.md` §4,
   macOS `caffeinate` / BSD `stat` / hard-coded interpreter) and replaced by
   `cocd/run_protocol_v2.py`. Following the report would call a file that is not
   here.
3. **§1 / §8** report the protocol assertions as "44/44" and then "54/54". The
   shipped `48_smoke_protocol_v2.py` reports **57 passed, 0 failed** on this host,
   matching the handoff and `TRANSFER_AUDIT.md`.

### Conflict 5 — the "single injection point at `p2`" has no seam in the shipped decoder

`HANDOFF_FOR_WINDOWS_AI.md` §5.1 defines the injection as `F_T = F_t + C_T` where
`F_t` / `F_c` are "the fused semantic feature of the FPN (in this codebase, the
tensor called `p2`: 128 channels at 1/4 resolution, 32×32 for a 128 input)", and
then `z_dual = D(F_T)` with `D` the shared decoder. But
`FPNStateDecoder.decode(c2, p3, p4)` **recomputes** `p2 = l2(c2) + interpolate(p3)`
internally, so there is no point at which a `p2` can be handed to `D`. Both
statements are true individually; implementing one injection point at `p2` with
one shared decoder requires a choice:

- (a) add a `decode` variant that accepts `p2` as an argument (then `z_self` and
  `z_dual` share `head` and the upsampling, and differ only in `p2`); or
- (b) inject into the `(c2, p3)` pair that produces `p2` instead of into `p2`.

(a) matches the written definition more literally and keeps the injection point
unique; (b) needs no decoder change but moves the injection one level up and
makes the "single injection point" claim harder to state. This changes what
"shared decoder" means in the method section, so it should be settled before
coding rather than chosen silently.

### Conflict 6 — the new teacher and the legacy students are structurally incompatible

`HANDOFF_FOR_WINDOWS_AI.md` §5.1 gives the new teacher exactly two readings,
`z_self = D(F_t)` and `z_dual = D(F_T)`. The legacy teacher has three
(`z0`/`z4`/`z34`) and `selective_kl` distils from `t['z4']` **and** `t['z34']`
(`cocd/scripts/23_train_ours_v2.py:678-679`). The new teacher therefore cannot be
substituted into the legacy distillation path, so the legacy arms keep distilling
from a legacy (z0/z4/z34) teacher — retrained here as `teacher_wm.pt` — and never
from the new one. Expected for a new method, but it means "swap the teacher and
rerun" is not a valid way to compare old and new.

---

## The user's decisions (2026-09-19)

| conflict | decision |
|---|---|
| 1 `TEACHER_W` | `(0.25, 0.25, 0.50)` **is** the current legacy v2 definition. Retrain the legacy teacher once from ImageNet under exactly this allocation. Do not investigate what the macOS `teacher_v2.pt` was trained with; it stays on disk as history. |
| 2 `PLAN` / `R3D` / `L_delta` | `R3D`, `L_delta`, `R3g` and geometry KD are **out of the active plan**. Nothing frozen is edited: the active plan lives in the new protocol file and the new runner, and `scripts/23` keeps its history untouched. |
| 3 `R3g` / `gain_gate` | Same treatment: out of the active plan. Left in place as history, never launched by the Windows runner. |
| 4 stale report | Not edited. The stale statements are recorded here so they are not quoted: the "zero gradient on synthetic samples" claim is v1-only; `scripts/49_run_protocol_v2.sh` does not exist in this package; the assertion count is 57, not 44 or 54. |
| 5 the `p2` injection seam | Implemented literally: the fused feature `p2` is exposed as a first-class tensor (``fused_p2``) and the shared head is ``backbone.head`` plus the same bilinear upsample, so `F_dual = F_t + C_T` is injected once at `p2` and the "shared decoder" is the shared head. Verified bitwise against `OursV3Student('R0')` — see Phase 6 below. |
| 6 teacher incompatibility | Accepted. The legacy arms keep distilling from the retrained legacy teacher; only the new method's N2/N2-no-orbit distil from the new teacher, through the one complement term. |

Two further instructions were given and they supersede the handoff where they
differ:

* the teacher loss is written `L_T = mean(BCE(z_self, y), BCE(z_dual, y))`, so the
  0.5 is not a knob;
* the complement distance is fixed to `SmoothL1(C_S, sg(C_T))` decided by scale
  alone, and λ is chosen once from `{0.1, 1, 10}` by train-loss magnitude only
  (the handoff's `λ = 0.1 · L_seg⁰ / L_dist⁰` rule is **not** used).

---

## Available input for the Phase 8 audit threshold

`HANDOFF_FOR_WINDOWS_AI.md` §6 step 3 asks for the audit's pass threshold to be
pre-registered "using the teacher's own validation curve variability as the noise
floor". The shipped curve can supply it: on `teacher_v2`, the late-plateau
epoch-to-epoch change in `z34` validation AUPRC is +0.009 / +0.003 / +0.017 /
+0.007 over the last few epochs, i.e. a noise floor of roughly ±0.01 on a
274-case validation partition. Candidate pre-registered criterion, for the user
to accept or replace: *correct counter must beat the shuffled counter by more than
the largest late-plateau epoch-to-epoch movement of the incumbent v2 teacher, and
the `dual − self` gain must be largest on `G10`.*

---

## Phase 5 / step 1 — the active protocol

**Date** 2026-09-19
**Category** new configuration; no frozen file edited

New file:

```
cocd/configs/protocol_windows_main.json
```

`cocd/configs/protocol_v2.json` is **untouched** and stays the legacy revision.
The new file carries the same unified protocol numbers (Adam 5e-5 constant, no
scheduler, physical batch 16, plain BCE with no reweighting, 20 epochs,
patience 3, validation-AUPRC checkpoint, validation max-F1 threshold frozen
before test, raw/no-mask, ImageNet initialisation, seed 42) plus:

| field | value |
|---|---|
| `init.legacy_teacher_z_weights` | `[0.25, 0.25, 0.5]` — the legacy teacher's supervision allocation, fixed |
| `complement_distillation.distance` | `smooth_l1`, beta 1.0, spatial mean, teacher stop-grad, real pre→post pairs only |
| `complement_distillation.lambda` | `null`, to be frozen once from `{0.1, 1, 10}` by train-loss scale |
| `active_plan` | the nine methods |
| `excluded_from_active_plan` | `R3D`, `L_delta`, `R3g`, geometry KD, `L_corr` |
| `seeds.extension` | `[42, 123, 2026]` for N0 / DIS2 / R3 / N1 / N2 |
| `comparison_structure.internal_causal_chain` | `N0 -> N1 -> N2` |
| `comparison_structure.internal_ablation` | `N2` vs `N2noorbit` |

The runner refuses to start if any knob in the file disagrees with the intended
protocol, and it refuses any `physical x accumulate != 16`.

## Phase 5 / step 5 — implementation of the new COCD

**Date** 2026-09-19
**Category** new method code; the shipped model, loss and script files are not edited

New files:

```
cocd/windows_main/__init__.py
cocd/windows_main/models_complement.py     the two new networks
cocd/windows_main/main.py                  the runner for every stage
cocd/windows_main/tests_complement.py      the phase 6 unit tests
```

Nothing in `cocd/models/`, `cocd/losses/` or `cocd/scripts/` was modified, so the
legacy arms are still trained by their original code. The runner imports
`scripts/23` and calls `train_teacher` / `train_student` directly for the legacy
arms, which is why their method definitions are not re-implemented and cannot
drift.

What was implemented, exactly as specified:

* `fused_p2(backbone, x)` exposes `F_t = p2` as a first-class tensor — 128
  channels at 1/4 resolution, `l2(c2) + up(p3)`. The prediction path is
  `backbone.head` followed by the same bilinear upsample to 128, so the head is
  literally the one N0 uses.
* `DCA(F_t, F_c)`: 4 learned offsets per position, bounded to ±3 pixels, sampled
  from `F_c` with `grid_sample`, weighted by a softmax over query-key similarity.
  The value path and the output projection carry no bias, so `DCA(F_t, 0) = 0`
  exactly.
* `C_T = Φ(F_t, DCA(F_t, F_c)) − Φ(F_t, DCA(F_t, 0))`, with `Φ` a plain 1×1–3×3–1×1
  convolutional projection. The zero reference is the same expression with the
  counter feature set to zero, so `C_T(F_c = 0) ≡ 0` by construction.
* One injection point: `F_dual = F_t + C_T`. `z_self = Head(F_t)`,
  `z_dual = Head(F_dual)`, both through the one shared head.
* `L_teacher = 0.25 · Σ over (2 directions × 2 readings) BCE`, i.e. the mean of
  `BCE(z_self, y)` and `BCE(z_dual, y)` per direction.
* `ComplementStudent`: `F_mod = (1+γ)⊙F_t + β` from a two-entry orbit embedding
  through a two-layer MLP, then three lightweight conv blocks for `C_S`, then
  `F_S = F_t + C_S` into the same shared head. Two switches give N0
  (`complement=False`), N1, N2, and N2 w/o Orbit Embedding.

Two initialisation choices, both inherited from the existing codebase
(`Correction` and `Driver` zero-start their output layer) rather than invented
here, and both easy to veto:

1. `Φ`'s output layer starts at zero, so a fresh teacher has `C_T ≡ 0` and
   `z_dual == z_self`; the complement grows from nothing.
2. The student's `C_S` output layer and FiLM output layer start at zero, so a
   fresh N1/N2 is numerically its own N0.

Consequence, measured and then guarded in code: at initialisation **both**
complements are exactly zero, so `L_dist⁰ = 0` and a λ rule based on a ratio would
divide by zero. The calibration stage now refuses to run unless the trained
teacher's complement is non-zero, and it records `L_seg⁰`, `L_dist⁰` and the
chosen λ in `experiments/windows_main/lambda_calibration.json`.

Step-0 note: with `Φ`'s output layer at zero, the gradient reaches
that layer and stops there on the very first step; the DCA and the earlier `Φ`
layers begin receiving gradient on step 1. This is a property of the existing
zero-start convention, is asserted in the unit tests at both steps, and is not a
defect.

## Phase 6 / step 6 — unit tests, and the result

**Date** 2026-09-19
**Category** verification

```
D:\deeplearning\anaconda3\python.exe -u cocd/windows_main/tests_complement.py
-> PHASE 6 PASSED -- 76 checks, 0 failures
```

The checks the instruction asked for, and what they returned:

| asked | result |
|---|---|
| ASC target / DESC counter | runs, self + dual at 128×128 |
| DESC target / ASC counter | runs, self + dual at 128×128 |
| shared encoder | one `backbone`; exactly two encoder calls in `forward_pair`; perturbing `l5` moves **both** orbits' `F_t` (`max|Δ|` 3.25e-01 / 3.20e-01) |
| shared prediction head | 4 head tensors, one module; the only 1-channel conv in the model; `z_self` and `z_dual` both go through it |
| fused `p2` single injection | `F_dual == F_t + C_T` bitwise; `z_dual` bitwise equals `Head(F_dual)`; touching `C_T` cannot move `z_self`; one `+ c_t` site in the source |
| `C_T(F_c = 0) ≈ 0` | **exactly 0.0**, both at initialisation and with `Φ` made non-degenerate |
| CUDA backward | real loss at batch 16 on CUDA; grads finite; step 0 reaches the head and backbone and stops at `Φ`; step 1 reaches `dca.q` / `dca.off` / `dca.out` / `phi.net` / backbone; the offset branch is learned (non-zero grad) |
| batch 16 | fits and steps |
| no geometry input | forward signatures take only the target/counter or the 5-channel input; no geometry token anywhere in the code (docstrings stripped before the scan); no geometry parameter |
| **N0 forward equivalent across the refactor** | `ComplementStudent(complement=False)` loads the legacy `backbone.*` key space with no missing or unexpected keys, has the **same 28,080,353 trainable parameters**, and its forward is **bitwise identical** to `OursV3Student('R0')`, `max|diff| = 0.000e+00` |

Also verified: `N2noorbit` differs from `N2` by exactly the orbit embedding and
MLP (parameter counts reconcile), the FiLM reads only channel 4 and does so per
sample, and `SmoothL1` is confirmed by a numerical quadratic-regime test rather
than by reading the code.

## Windows main run — protocol, artefacts and order

**Date** 2026-09-19
**Category** the formal run

Measured throughput on this host, batch 16, real data: **0.116 s/batch**,
232 batches, so **27 s per epoch** — about 6.5× the macOS rate (0.75 s/batch).
One arm at 20 epochs is therefore ≈ 9–10 minutes including the ~20 s data cache,
and the whole seed-42 round is ≈ 1.5 h.

Artefacts, all tagged so nothing shipped is overwritten:

```
experiments/windows_main/                 the new method and the new teacher
    NewTeacher_seed42.pt  NewTeacher_seed42_curve.csv  ...
    N0_seed42.pt  N1_seed42.pt  N2_seed42.pt  N2noorbit_seed42.pt
    val_<arm>_seed42.csv  <arm>_seed42_curve.csv
    audit_NewTeacher_seed42.{csv,json}
    lambda_calibration.json  lambda_frozen.json
    FINAL_test_seed42.csv  FINAL_thresholds_seed42.json
experiments/ours_v2/                      the legacy arms, through their own code
    teacher_wm.pt  SO_wm_seed42.pt  VKD_wm_seed42.pt
    DIS2_wm_seed42.pt  R3_wm_seed42.pt
```

The `_wm` tag is what keeps `teacher_wm.pt` and the legacy arms away from the
shipped `teacher_v2.pt` / `teacher.pt`, which are read-only history.

Order, with the audit as a real gate:

| # | stage | command | gate |
|---|---|---|---|
| 1 | N0 | `main.py N0` | none — the baseline needs no teacher |
| 2 | legacy teacher | `main.py legacy-teacher` | none |
| 3 | Vanilla KD / DIS2-port / R3 | `main.py VKD` `main.py DIS2` `main.py R3` | needs step 2 |
| 4 | new teacher | `main.py new-teacher` | none |
| 5 | teacher audit | `main.py auditteacher` | **if it fails, N1/N2 are not trained** |
| 6 | λ calibration | `main.py calib` | needs step 4; refuses if `L_dist⁰ = 0` |
| 7 | N1 / N2 / N2-no-orbit | `main.py N1` `main.py N2` `main.py N2noorbit` | needs steps 5 and 6 |
| 8 | seed extension | `--seed 123`, `--seed 2026` on N0 / DIS2 / R3 / N1 / N2 | after seed 42 runs clean |
| 9 | test | `main.py test` | **only after everything is frozen** |

`stage_test` is the only code path in the runner that opens the test partition.

---

## Progress — seed 42

**Date** 2026-09-19
**Category** the formal run

Every arm below ran the identical protocol: Adam 5e-5 constant, physical batch 16,
plain BCE, ≤20 epochs, patience 3, ImageNet initialisation, checkpoint on
validation AUPRC. Validation partition only; the test partition was not opened.

### Legacy teacher and the legacy control arms

The legacy teacher was retrained from ImageNet under `z0 : z4 : z34 = 0.25 : 0.25 : 0.50`.

| arm | best validation AUPRC | epochs run | note |
|---|---:|---:|---|
| Legacy teacher, `z0` (self) | 0.6277 | 20 | patience never triggered |
| Legacy teacher, `z4` | 0.6496 | 20 | |
| Legacy teacher, `z34` (dual) | **0.6663** | 20 | IoU 0.3804, F1 0.5511 |
| Vanilla KD | 0.6033 | 20 | |
| DIS2-port | 0.5990 | 20 | |
| Legacy R3 | 0.6107 | 20 | |
| N0 Single-Orbit | 0.6443 | 20 | no teacher needed |

Two facts worth keeping:

1. **Conflict 1 is now settled empirically, in the direction of the code.** The
   retrained teacher with `(0.25, 0.25, 0.50)` reaches 0.66635 and its head
   ordering and gaps are `z0 0.6277 < z4 0.6496 < z34 0.6663`, spread 0.0387. The
   shipped macOS `teacher_v2.pt` recorded `z0 0.63043 < z4 0.65121 < z34 0.66708`,
   spread 0.0367. Same ordering, gap and total to within 0.0007. The allocation in
   the code was the one that teacher was trained with; the docstring and
   `reports/cocd_final_teacher_student.md` §4 were stale, not the code.
2. **All three legacy distillation arms land below the single-orbit baseline**
   under this protocol (0.5990–0.6107 against N0's 0.6443). Their run logs show
   why: by epoch 20 the segmentation term has fallen to 0.036–0.041 while the
   distillation term is still 0.021 (VKD) / 0.0065+0.0007+0.0002 (DIS2) / 0.0028
   (R3), so the auxiliary term drives the objective in the later epochs. This is
   the behaviour `reports/protocol_v2_unification.md` §7 predicted and it means
   the legacy KD arms are weak controls. Recorded as a fact about the controls,
   not as a failure: nothing about them was changed.

### New teacher

| reading | validation AUPRC | epochs |
|---|---:|---:|
| `z_self` | 0.6459 | best at 20 |
| `z_dual` | **0.6897** | best at 20 |
| dual − self | +0.0437 | |

`mean |C_T|` grows to 234.9 by epoch 20 (the legacy teacher's `|r3|` reached
107.2), so the complement is a substantial learned object rather than a
perturbation. The curve was still rising at the last epoch for both readings.

### Teacher audit — PASS

`main.py auditteacher`, validation partition, five fixed counter permutations
(shifts 1, 2, 3, 5, 7):

| condition | dual AUPRC |
|---|---:|
| correct counter | **0.68967** |
| zero counter | 0.64594 |
| shuffled counters | 0.34059, 0.30898, 0.34026, 0.34891, 0.35116 |

| check | result |
|---|---|
| `z_self` identical across the three counter conditions | `0.000e+00` — the injection is single |
| dual > self on the correct counter | +0.04373 |
| correct > zero | +0.04373 (zero returns exactly the self reading, as the identity requires) |
| correct beats shuffles | **5 / 5**, median 0.34059, margin over the best shuffle **+0.33851** |
| `mean |C_T|` responds to the condition | correct 234.9, zero 0.0000, shuffled 282.3 |
| dual − self largest on `G10` | **G10 +0.0522** > G11 +0.0452 > G00 +0.0443 > G01 +0.0308 |

The `G10` ordering matches what the incumbent legacy teacher shows
(`G10 +0.0497` > G11 +0.0384 > G00 +0.0358 > G01 +0.0269), so the counter benefit
lands where the study says it should. No magnitude threshold was imposed; the
margins are reported next to the incumbent's ±0.01 late-plateau variability.

Gate verdict: **PASS** — N1 / N2 / N2-no-orbit may proceed.

### λ — measured once, frozen

`main.py calib`, on the first 64 pinned train locations, real pre→post pairs only,
128 single-orbit samples, at the shared ImageNet initialisation, no parameter
update:

| quantity | value |
|---|---:|
| `L_seg⁰` | 0.7543187439441681 |
| `L_dist⁰` | 0.24027762934565544 |
| ratio `L_seg⁰ / L_dist⁰` | 3.139 |
| **λ** | **1.0** |
| reason | the two terms are already the same order of magnitude |
| initial share of the segmentation loss | 31.9 % |

Recorded in `experiments/windows_main/lambda_calibration.json`, frozen in
`lambda_frozen.json`. Chosen from train-loss scale only; no validation data was
read and no sweep was run. The runner refuses to choose λ a second time.

Note on scale: at initialisation the student's complement is exactly zero, so
`L_dist⁰` measures the magnitude of the trained teacher's complement. That is the
intended reference — it is how large the object being distilled actually is —
and it is why the calibration requires a trained teacher and refuses to run when
`L_dist⁰ = 0`.

### Complement arms — N2 is worse than N0 and N1

| arm | what it adds | best validation AUPRC | ASC | DESC |
|---|---|---:|---:|---:|
| N0 | nothing, `F_t → Head` | 0.6443 | 0.6391 | 0.6495 |
| N1 | `F_t + C_S → Head`, BCE only | **0.6504** | 0.6465 | 0.6543 |
| N2 | N1 + `λ·SmoothL1(C_S, sg(C_T))`, λ = 1.0 | 0.5583 | 0.5577 | 0.5590 |
| N2noorbit | N2 without the FiLM conditioning | 0.5558 | 0.5480 | 0.5631 |

N1 is slightly above N0 and still rising at epoch 20. **N2 is 0.0921 below N1 and
0.0860 below N0.** The internal causal chain N0 → N1 → N2 therefore does not hold
in this run: the first step is positive and the second is strongly negative.

This is an ANOMALY and it was investigated in the order the handoff prescribes —
data, then compatibility, then implementation, then training — before being
reported. Nothing was changed in response: no new loss, no new module, no new
hyper-parameter, and λ was not re-chosen.

### Anomaly investigation, `windows_ops/diag_complement.py` (read-only, validation)

**1. Data.** All four arms read the identical validation tensor `(274, 128, 128)`
through the same loader. N0 and N1 trained normally on that path, and the 78-check
Phase 3 audit already cleared the data. Not the cause.

**2. Compatibility.** No device or dtype issue: all four runs completed 20 epochs
on CUDA with finite losses. Not the cause.

**3. Implementation.** Alignment and behaviour of the term check out.

| check | result |
|---|---|
| `C_S` and `C_T` row counts and shapes | 32 rows each, both `(N, 128, 32, 32)` |
| `L_dist` against the correct counter | 0.082982 |
| `L_dist` against a rolled counter | 0.176490 — the term distinguishes them |
| `L_dist(C_S = 0, C_T)` reference | 0.171852 |
| N2's `C_S` moved toward the teacher | 0.082982 < 0.171852 ✓ the term works |
| N1's free `C_S` against the teacher | **0.309987 > 0.171852** — it points away from `C_T` |

The term is doing exactly what it says: N2's complement is 2.1× closer to the
teacher's than an all-zero complement is. So the mechanism is implemented
correctly; it is the mechanism that does not help.

**4. Training dynamics.** Two independent facts, both measured at the state N2
finished in.

*How much of the objective the term carries.* The calibration measured
`L_seg⁰ = 0.7543`, `L_dist⁰ = 0.2403`, ratio 3.14, giving λ = 1.0 and an initial
share of **31.9 %**. During the run `L_seg` falls about 20× over 20 epochs while
`L_dist` falls only about 9×, so the balance does not hold:

| point | share of the objective carried by `λ·L_dist` |
|---|---:|
| calibration, at initialisation | 31.9 % |
| epoch 1 | **51.6 %** |
| epoch 20 | 40.4 % |

The rule the instruction specifies selects λ from the initialisation ratio. That
ratio is 3.14 and λ = 1.0 follows from it, but the design's target was an
*auxiliary* term of roughly 10 %, and the realised term is 3–5× that for the whole
run. This is a property of the rule as written, not a bug in following it: the
weight was not swept and was not changed.

*The transfer target itself.* The teacher's complement is large — `mean|C_T| =
241.9` against `mean|F_t| = 381.5`, so the correction is 63 % of the magnitude of
the feature it corrects. Two consequences:

- N1's freely learned complement (`mean|C_S| = 300.9`) is *farther* from `C_T`
  than zero is, yet N1 is the better arm. The useful complement is not the
  teacher's complement.
- the student never observes the counter orbit, so the counter-dependent part of
  `C_T` is irreducible noise for it, and `C_T` lives in the teacher's separately
  trained feature space rather than the student's; forcing `C_S` onto it removes
  the freedom that made N1 work. The two terms' gradients at the end of the run
  are near-orthogonal (cosine **+0.0598**, with the distillation gradient 32 % of
  the segmentation one), which is consistent with a persistent shift of the
  solution rather than a direct fight with the segmentation objective.

*Reproduce with:* `python windows_ops/diag_complement.py`.

### Decision needed before the seed extension

The seed extension and the final test are on hold for N2 / N2-no-orbit, because
what N2 *is* may change. Two candidate readings, both of them the user's to make:

- **(a) accept the result.** λ = 1.0 is what the specified rule gives; N2 is a
  negative result and the paper reports that the complement predictor (N1) is the
  part that works while explicit complement distillation from this teacher does
  not, under this calibration.
- **(b) amend the λ rule.** The rule's premise — "the same order of magnitude at
  initialisation is the right balance" — does not survive training, and 0.1 is
  also in the permitted set. Changing it is a change to a frozen protocol number,
  which is why it is not done here.

Either way the anomaly is a result about the method, not about the transfer: the
data, the port, the implementation and the training loop all check out.

---

## Status


| phase | state |
|---|---|
| 1 read the handoff | done |
| 2 Windows / CUDA acceptance | done — `TRANSFER CHECK PASSED`, 0 repairs needed |
| 3 data and split audit | done — 78 checks, 0 failures |
| 4 legacy baseline smoke | done — all 7 shipped checkpoints reproduce their recorded numbers |
| 5 implement the new teacher and student | done — `cocd/windows_main/` |
| 6 unit tests | done — 76 checks, 0 failures |
| 7 legacy teacher + N0 + 3 legacy control arms | done — see the progress section |
| 8 teacher audit | **done — PASS**, correct counter beats zero and 5/5 shuffles |
| 10 complement distance | done — `SmoothL1`, fixed by the protocol |
| 11 λ | done — frozen at 1.0 |
| 12–13 N1 / N2 / N2-no-orbit | done — N1 0.6504 (above N0), **N2 0.5583 and N2-no-orbit 0.5558, an anomaly, investigated and reported** |
| 14 seed extension (123, 2026) | **on hold for N2** pending the decision above; the unaffected arms can start as soon as it is given |
| 15 test | pending, once and only once |

The test partition has not been evaluated with any Windows artefact.

---

## Step 3 follow-up: the single lam = 0.1 diagnostic run

*Decision taken.* Option (a), accept the lam = 1.0 result outright, was declined.
Instead exactly one diagnostic run was commissioned, at **lam = 0.1**, to separate
two explanations that the lam = 1.0 result conflates:

- the auxiliary term is simply too loud — the distillation share of the
  segmentation term was 0.3185 at initialisation and 0.40-0.52 once trained,
  roughly three to five times the weak-auxiliary intent; or
- raw `C_T` is the wrong knowledge object to match `C_S` against, in which case
  the weight is not the problem and lowering it cannot rescue the arm.

*This is a calibration correction, not a sweep.* One value, chosen because it is
one order of magnitude below the frozen one and because it is a member of the
permitted set `{0.1, 1, 10}` that the rule already names. No other value is tried
(not 0.03, 0.05, 0.2, 0.5), the choice reads the train-loss scale only, and no
validation number is consulted to pick it. **lam = 1.0 remains the frozen
protocol value**; the diagnostic run does not replace the N2 protocol arm and
does not enter the reported table.

*How it is isolated.* `cocd/windows_main/main.py` gained two flags, `--lam-override`
and `--tag`, plus a guard that refuses an override without a tag and refuses any
value outside `{0.1, 1, 10}`. The run is written under the tag suffix
(`N2_lam0p1_seed42_*`), so it cannot overwrite `N2_seed42_*`, and its summary
records `lam_source = "explicit override, single diagnostic run"`. No default path
changed: with no flags the runner reproduces the frozen behaviour exactly.

*The run.* `python -m cocd.windows_main.main N2 --seed 42 --lam-override 0.1 --tag _lam0p1` —
same protocol in every other respect: Adam, lr 5e-5 constant, batch 16, plain BCE,
20 epochs, patience 3, seed 42, the same frozen dual-orbit teacher, the same
complement term on the real pre->post pairs only. Confirmed at epoch 1: `seg =
0.1112` against the lam = 1.0 run's `0.1134` and `dist = 0.1284` against `0.1207`,
so the two runs differ in the weight and nothing else.

*Reading the outcome.* Pre-registered in advance, so the interpretation cannot be
fitted to the number afterwards:

| result | reading |
|---|---|
| at or above N1 (0.6504) | the weight was the whole problem; weak complement guidance can work |
| well above lam = 1.0 (0.5583) but below N1 | `C_T` carries some usable information, raw matching is still biased |
| still well below N1 | matching raw `C_T` is the wrong knowledge object; stop this line |

Only the third outcome ends the `C_S -> C_T` direct feature distillation line.

### The lam = 0.1 result

| run | lam | val AUPRC | vs N0 | vs N1 | seg @20 | dist @20 | lam*dist/seg @20 |
|---|---|---|---|---|---|---|---|
| N0 Single-Orbit | 0.0 | 0.64428 | — | −0.0062 | 0.0345 | — | — |
| N1 Complement-Predictor | 0.0 | **0.65045** | +0.0062 | — | 0.0337 | — | — |
| N2 COCD-Full | 1.0 | 0.55830 | −0.0860 | −0.0922 | 0.0395 | 0.0267 | 0.677 |
| N2-no-OrbitEmb | 1.0 | 0.55575 | −0.0885 | −0.0947 | 0.0394 | 0.0263 | 0.667 |
| **N2 diagnostic** | **0.1** | **0.63136** | −0.0129 | −0.0191 | 0.0347 | 0.0652 | 0.188 |

*It falls in the second row of the pre-registered table.* It is far above lam = 1.0
(**+0.0731**), so the weight was doing real damage, but it still sits **0.0191 below
N1** and **0.0129 below N0**. Lowering the weight recovered most of the ground and
none of the advantage.

*Two things moved in the right direction.* The share of the segmentation term at the
end of training falls from 0.677 to 0.188, so the term is now genuinely auxiliary
rather than a near-equal partner. The distillation loss also drops much further,
0.0652 against 0.0267 — but note this is a weaker fit at a lighter weight, not a
better one: the same distance at lam = 0.1 lands at a lower absolute value because
almost nothing is pushing `C_S` onto `C_T`.

*What this rules in and out.* The weight being too large explains about **0.073 of
the 0.092** gap to N1. The remaining **0.019** is the part that the weight cannot
explain, and it is the part that matters: even at a tenth of the strength, with the
distillation term at 0.188 of the segmentation term, `C_S -> C_T` matching still ends
up below both N1 and N0. So for the ~19 % of the deficit that survives the weight
change, the more plausible reading is the one from the diagnosis above — `C_T` is the
teacher's complement in the teacher's own separately trained feature space, computed
from a counter orbit the student never sees, and matching it is a worse objective than
the student's own freely learned complement. The gains N1 gets from the complement
branch are real; the distillation term erodes rather than adds to them.

*Consequence for the protocol.* lam = 1.0 stays the frozen value as specified; it is
the number the rule yields and the N2 protocol row remains the lam = 1.0 run. The
diagnostic run is recorded under `diagnostic_runs` in the protocol file and does not
enter the reported table.

*What this is not.* It is not a sweep, and no third value was or will be tried. The
lam = 0.1 result is not a "better N2" to be promoted.
