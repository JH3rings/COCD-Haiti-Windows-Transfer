# Checkpoints in this package

Seven files, 753.0 MB. Each one is a plain `state_dict` (weights only), so it
loads with `torch.load(path, map_location=..., weights_only=True)`.

**Read this first.** The unified protocol starts **every** internal arm from the
same ImageNet initialisation. None of the checkpoints below is an initialisation
for a new run. `checkpoints/teacher_v2.pt` is the one file a new run actually
needs, and only because the distillation arms read it as the teacher — not
because anything is warm-started from it.

---

## 1. Shipping checkpoints

### `checkpoints/teacher_v2.pt` — **required**

| field | value |
|---|---|
| method | Additive Cross-Orbit Teacher (`models.landslide_cocd_v2.OursV2Teacher`) |
| role | **the teacher of the unified comparison**; the distillation source for Vanilla KD / DIS2-port / R3 |
| protocol | **v2** (Adam 5e-5, constant, batch 16, plain BCE, 20 epochs, patience 3, ImageNet init) |
| training | 20 / 20 epochs, no early stop |
| best validation AUPRC | **0.66708** on the `z34` (dual) reading, at epoch 20 |
| other validation readings | `z0` 0.63043, `z4` 0.65121 |
| validation threshold | **0.206** (max-F1 on the `z34` reading; frozen, not yet applied to test) |
| test | **not run** — no test prediction has been made with this checkpoint |
| may initialise a new arm | **no** |
| sha256 | `85ae7ae090e156f787431db63d559e73e5663050d57071abb4dabb40ed9611a4` |
| size | 107.4 MB |

Its full 20-epoch learning curve is in `results/ours_v2/teacher_v2_curve.csv`.
The architecture is **not** the new design: it is the legacy additive-correction
teacher, kept as the control the new teacher is measured against.

---

## 2. Legacy checkpoints — `legacy_checkpoints/v1_protocol/`

These are the arms behind the numbers already reported. They belong to the
**v1 protocol** (AdamW lr 1e-4 wd 1e-4, batch 2 for the teacher and 8 for the
students, BCE + 0.2·Dice, 50 epochs, a warm start from S0, and a **fixed
threshold of 0.5**). **Absolute values from v1 and v2 are not comparable** — v2
drops the warm start, which moves the segmentation loss at initialisation from
≈0.01 to ≈0.74. They are shipped so the reported tables can be re-evaluated
without retraining, not so they can be mixed into a v2 comparison.

| file | method | best val AUPRC | epoch | test IoU | test F1 | test AUPRC | sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| `teacher.pt` | Additive Cross-Orbit Teacher, v1 protocol | 0.89626 (`z34`) | 50 | self 0.6006 / dual **0.6250** | self 0.7505 / dual 0.7728 | self 0.8820 / dual 0.8972 | `fea1aa6e04a4adcdeacfc94bbeb9e3d16f68493fb4c22212c097c8e1c93daf4a` |
| `R1_seed42.pt` | `OursV3Student('R1')` + selective KD | 0.89040 | 10 (stopped 20) | 0.6211 | 0.7663 | 0.8924 | `5b406c652df7caa1243808beeb34be389bf6427258a136740310ac43400e82ed` |
| `R2_seed42.pt` | `OursV3Student('R2')` + selective KD | 0.89090 | 20 (stopped 30) | 0.6194 | 0.7650 | 0.8932 | `d3b33af791f92946f6143407d6532887d76b19924328871d9ebe2c873c4e537d` |
| `R3_seed42.pt` | `OursV3Student('R3')` + selective KD | 0.89400 | 45 (stopped 50) | **0.6298** | 0.7728 | 0.8972 | `f5b5bfe648732bb8751a1eab1f7e58436b5d6f8aaf9661d907eb2331d8c184eb` |
| `DIS2_port_seed42.pt` | `OursV3Student('R1')` + the DIS2 rule | 0.89320 | 30 (stopped 50) | 0.6272 | 0.7709 | 0.8963 | `54bb4fee3e070e1907c9f9a70f9ccf8832eb5d374e1391886fb7f90b8f7fa3be` |
| `S0.pt` | stage-1 `SingleOrbitStudent(correction=False)` | not recorded | 50 | — | — | — | `2e0ae536b87eb55ed6574bedde2e9b0edf97b3265344b21d3ac11cbd9e039b01` |

Notes:

- **`R3` is not the new COCD.** R3 is `Ω`-inheritance plus a progressive driver,
  distilled with the selective logit rule. The new design replaces the whole
  knowledge object (a complement representation) and drops the selective rule.
  The two must not be described as the same method.
- **The best validation AUPRC for R1/R2/R3 was not written to a progress file**;
  it was recovered from `results/ours_v2/train_log.txt`, which is the only record
  of those runs. The values above are the maxima over that log's
  `validation epoch=N AUPRC=X` lines.
- **R1/R2/R3 stopped early** (`bad >= 2` on the 5-epoch validation). They
  therefore trained for different numbers of epochs (20 / 30 / 50), and R3 saw
  about 2.5× as many real updates as R1. Any difference between them carries that
  confound; this is recorded in `results/ours_v2/ABLATION_R1R2R3_SUMMARY.md`.
- **`S0.pt` is a different model class** — stage-1 `SingleOrbitStudent` with 190
  `backbone.*` tensors, not an `OursV3Student`. It is loaded through
  `OursV2Teacher.load_s0()`, which maps those keys onto the shared encoder. A
  strict `load_state_dict` into an Ours-V3 student fails, as it should.
- `teacher.pt` and `R1/R2/R3/DIS2` share a **single frozen threshold of 0.5**,
  which is what their test tables report. The v1 teacher's validation threshold
  was never searched — the v2 protocol introduced validation-selected thresholds.

---

## 3. Checkpoints deliberately not shipped

| excluded | count | size | why |
|---|---:|---:|---|
| `*_latest.pt` resume states | 9 | 4.7 GB | each is ~537 MB (weights + AdamW moments). A new run starts fresh; nothing needs to resume. |
| `teacher_T2a.pt` | 1 | 107.4 MB | a single-variable reweighting experiment whose verdict was **NOT IMPROVED**. The result is recorded in `reports/cocd_final_teacher_student.md` §13 and its curve is in `results/ours_v2/teacher_T2a_curve.csv`; the weights add nothing. |
| `R3g_seed42.pt` | 1 | 108.0 MB | the geometry-weighted arm, **deprecated** — geometry does not enter training. Its number (0.6260) is a negative result recorded in `reports/`. |
| `experiments/rapid_landslide_cocd/{T,Ours,KD,DIS2}.pt` | 4 | 434 MB | stage-1 models on the older `DualOrbitTeacher` / `SingleOrbitStudent` classes, from a different experiment line. `S0.pt` from the same folder **is** shipped, because the v1 chain is warm-started from it. |
| `experiments/{phase1_b1_b2,grsl_external_baselines,landslide_downstream_reproduction,phase2_distill_app,six_single_sar_distortion_baselines}/` | 45 | 2.4 GB | separate experiment lines (geometry-mask prediction, external baselines, optical downstream, distortion baselines). Their result tables and logs **are** shipped under `results/`; only the weights are left behind. |

---

## 4. Loading them correctly

```python
from paths import DEV, resolve_checkpoint          # cocd/paths.py
import torch
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student

state = torch.load(resolve_checkpoint('teacher_v2.pt'), map_location=DEV, weights_only=True)
teacher = OursV2Teacher().to(DEV)
teacher.load_state_dict(state, strict=True)        # 392 tensors
```

`resolve_checkpoint(name)` searches `experiments/ours_v2/`, then
`checkpoints/`, then `legacy_checkpoints/v1_protocol/`, then
`experiments/rapid_landslide_cocd/`, and returns the canonical output path if the
file is not there yet. That is why a v2 run launched with `--tag _v2` reuses the
shipped `teacher_v2.pt` instead of retraining a teacher.

`map_location` is always passed explicitly, so a checkpoint written on MPS loads
on CUDA and vice versa. `verify_transfer.py` section [7] performs exactly these
loads and reports the tensor count of each.
