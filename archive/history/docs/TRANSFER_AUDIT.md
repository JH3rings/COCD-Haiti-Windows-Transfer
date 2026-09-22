# Transfer audit

What was in the source project, what was carried over, what was left behind, and
which files were changed to make the package run on Windows. Nothing in the
source project was modified: the package was built by copying.

## 1. What was audited

| item | where | size |
|---|---|---|
| project root | `/Users/zhangjiuqi/Desktop/distortion/` | 3.0 GB of data + 11 GB of experiment output |
| code and results | `Haiti_SAR_GRSL_Audit/` | 11.0 GB total |
| experiment output | `Haiti_SAR_GRSL_Audit/experiments/` | 10.0 GB (51 `.pt` = 9.7 GB, 16 `.npz` = 433 MB, 2 logs = 33 MB, 76 figures = 10 MB, results csv/json ≈ 0.3 MB) |
| processed dataset | `Haiti_SAR_GRSL_Audit/processed/` | 629 MB, 1,713 locations × 9 files |
| raw Sentinel-1 | `Pre_event/`, `Post_event/` | 1.31 GB across 4 acquisition folders, 1,713 files each |
| raw Sentinel-2 | `Pre_event/S2_20210804/`, `Post_event/S2_20210814/` | 1.7 GB (not used by this study) |
| landslide source rasters | `Annotations/` | 114 MB, 1,713 files |
| other code | `baselines/`, `third_party/`, `scripts/`, `models/`, `losses/`, `configs/` | 22 MB |
| QC and metadata | `qc/`, `metadata/`, `manifests/` | 24 MB |

## 2. Inventory of the study itself

**Dataset and split.** Loader in `scripts/21_train_rapid_landslide_cocd.py`;
frozen split in `manifests/spatial_80_20_split.csv` (1,370 train / 343 test,
seed 42) plus a seed-42 10 % internal validation draw, giving
**1,233 / 137 / 343 locations**. Same-location rasters are read together, so a
location cannot cross partitions.

**Unified protocol v2.** `configs/protocol_v2.json`, consumed by
`scripts/23_train_ours_v2.py`, enforced by `scripts/48_smoke_protocol_v2.py`
(57 assertions). Adam 5e-5 constant, batch 16, plain BCE, 20 epochs, patience 3,
validation AUPRC checkpoint, validation-selected F1 threshold, raw/no-mask,
ImageNet initialisation.

**Implemented and run.**

| method | code | status |
|---|---|---|
| Single-Orbit baseline | `CONFIGS['SO']` | configured, **not trained** under v2 |
| Vanilla KD | `CONFIGS['VKD']`, `selective_kl(...,'all')` | configured, **not trained** under v2 |
| DIS2-port | `CONFIGS['DIS2']`, `losses/distill.py::dis2_multilevel_kd` | trained under **v1**, test IoU 0.6272 |
| legacy R0–R3 ladder | `models/landslide_cocd_v2.py::OursV3Student` | R1/R2/R3 trained under **v1** (0.6211 / 0.6194 / 0.6298) |
| R3g (geometry-weighted) | `CONFIGS['R3g']` | trained under **v1**, 0.6260, **deprecated** |
| R3D (R3 + `L_delta`) | `CONFIGS['R3D']` | implemented and asserted, **never trained** |
| old Teacher | `OursV2Teacher`, `teacher.pt` | trained under **v1**, val AUPRC 0.89626, test dual IoU 0.6250 |
| Teacher-T2a | tag `_T2a` | trained under **v1**, verdict **NOT IMPROVED** |
| teacher_v2 | `teacher_v2.pt` | trained under **v2**, val AUPRC 0.66708, **no test** |
| stage-1 models | `experiments/rapid_landslide_cocd/` | S0 / T / KD / Ours / DIS2 on the older classes |
| external baselines | `experiments/grsl_external_baselines/` | Boehm / CDNetE / MFEWF trained; FC-Siam not |
| other experiment lines | `experiments/phase1_b1_b2`, `landslide_downstream_reproduction`, `phase2_distill_app`, `six_single_sar_distortion_baselines` | geometry-mask prediction, optical downstream, distortion baselines |

**Not implemented — the new design.** The new teacher (deformable cross-orbit
extraction with a difference-defined complement), the new student
(orbit-conditioned complement prediction), orbit embedding / FiLM, and the
complement distillation term do not exist in the source project. The design is
recorded in `reports/ocdp_complement_design_mapping.md` §10–11 and restated in
`HANDOFF_FOR_WINDOWS_AI.md` §5.

**Deprecated / closed.** Geometry-weighted distillation (`R3g`), `L_corr`
(explicitly never implemented), the teacher-reweighting experiment (T2a,
negative), and the batch-2/batch-8 v1 schedule.

## 3. Kept

| kept | size | why |
|---|---:|---|
| `cocd/scripts/21_...` | 10 KB | owns the dataset loader **and** the frozen split; `scripts/23` imports it by path, so the package does not run without it |
| `cocd/scripts/23_...` | 49 KB | the only training / evaluation entry point: teacher, all arms, DIS2, R3D, threshold selection, geometry partitions, test stage |
| `cocd/scripts/{47,48,50}` | 22 KB | batch feasibility, the 57 protocol assertions, the validation-only readout |
| `cocd/scripts/{32,36,37,39,40,43,44,45,46}` | 76 KB | the verification and diagnostics the recorded numbers rest on |
| `cocd/scripts/14_...` | 1 KB | provenance of the frozen split manifest |
| `cocd/models/*`, `cocd/losses/*` | 36 KB | the backbone, the teacher, the student ladder, the DIS2 rule, the segmentation loss |
| `cocd/configs/protocol_v2.json` | 4 KB | single source of truth for the protocol |
| `cocd/manifests/` | 0.9 MB | upstream split manifests, needed by the fallback path of `split()` |
| `cocd/metadata/` | 6.2 MB | per-file inventory of the raw data, used as the independent QC reference for the copy |
| `cocd/third_party/landslide_baselines/dis2/` | 108 KB | the snapshot the DIS2 port is diffed against |
| pinned splits | 0.05 MB | `data/splits/*_ids.csv`, so the partition can never be re-drawn |
| raw Sentinel-1, 4 folders | 1.31 GB | the model input; nothing can be trained without it |
| `processed/` masks and geometries | 84.7 MB | `landslide_mask.tif`, `asc_pre_geom.tif`, `desc_pre_geom.tif`, `metadata.json` — exactly the files `HaitiPairs` opens |
| `checkpoints/teacher_v2.pt` | 107.4 MB | **required**: the distillation source for the v2 arms |
| `legacy_checkpoints/v1_protocol/` (6 files) | 645 MB | the weights behind every reported table, so those numbers can be re-evaluated without retraining |
| `results/` | 34 MB | all recorded csv / json / logs, including the 32 MB `train_log.txt`, which is the only record of the v1 arms' validation curves |
| `docs/`, `reports/` | 0.3 MB | the design and audit documents |

## 4. Left behind, with the reason

| dropped | size | reason |
|---|---:|---|
| `experiments/**/*_latest.pt` (9) | 4.7 GB | resume states, ~537 MB each (weights + AdamW moments). A new run starts fresh. |
| `experiments/**/*.npz` (16) | 433 MB | prediction dumps from the diagnostics; the numbers derived from them are in the csv/json. |
| `experiments/rapid_landslide_cocd/{T,Ours,KD,DIS2}.pt` | 434 MB | stage-1 models on the older `DualOrbitTeacher`/`SingleOrbitStudent` classes, a different experiment line. `S0.pt` from the same folder **is** shipped (v1 warm-start root). |
| `teacher_T2a.pt`, `R3g_seed42.pt` | 215 MB | a negative-result experiment and a deprecated arm; both verdicts are recorded. |
| `experiments/{phase1_b1_b2, grsl_external_baselines, landslide_downstream_reproduction, phase2_distill_app, six_single_sar_distortion_baselines}/` weights | 2.4 GB | separate experiment lines. Their result tables and logs **are** shipped. |
| `Pre_event/S2_20210804`, `Post_event/S2_20210814` | 1.7 GB | optical. The student is explicitly forbidden to use it. |
| `Annotations/` | 114 MB | upstream inventory rasters; `processed/landslide_mask.tif` is derived from them and is what the loader reads. |
| `processed/sample_*/{asc,desc}_{pre,post}_vv.tif` | 456 MB | single-band VV extracts used only by the geometry-prediction line (`scripts/11`, `scripts/13`). The learner reads VV/VH from the raw stacks. |
| `processed/sample_*/{asc,desc}_post_geom.tif` | 66 MB | the regions are defined from the pre-event masks only. |
| `baselines/` | 21 MB | external baseline implementations (`unetpp`, `segformer`, `deeplabv3plus`, `upsnet2022`, `wu2021`). Not part of the COCD line; several need separately downloaded encoder weights. |
| `qc/`, `raw_archives/`, `extracted/` | 17 MB | QC figures and empty staging folders. |
| `scripts/{00..13, 15..20, 22..31, 33..34, 38, 41, 42, 49, monitor_*, 28_watch_*}` | — | dataset exploration and QC, other experiment lines, macOS-only launchers, and the dashboard. See §6 for the four that matter. |
| `scripts/09_build_processed.py`, `scripts/22_analyze_rapid_landslide_cocd.py` | 25 KB | kept initially, then removed: the first needs `Annotations/` and the Sentinel-2 folders, the second needs the stage-1 checkpoints, and neither ships. Their provenance role is taken by `DATASET_MANIFEST.md`. |
| `.sh` launchers (`15`, `31`, `38`, `41`, `42`, `49`, `monitor_phase2_teacher.sh`) | — | `caffeinate`, BSD `stat -f`, and a hard-coded macOS interpreter path. The sweep order and the concurrency guard are reproduced in `cocd/run_protocol_v2.py`. |
| `__pycache__`, `.DS_Store`, caches | — | build artefacts. |

Nothing was dropped that a v2 training run reads. The check is mechanical: the
only paths `scripts/21` and `scripts/23` open at run time are the four raw
acquisition folders, the four per-location processed files, the pinned split
files, `configs/protocol_v2.json` and the checkpoints — all present.

## 5. Windows / CUDA changes actually made

| change | where | detail |
|---|---|---|
| **device resolution added** | new `cocd/paths.py` | `CUDA → MPS → CPU`, overridable with `COCD_DEVICE`. Replaces the two `torch.device('mps' if ... else 'cpu')` sites (`scripts/21:14`, and the imported `DEV` that `scripts/23` used). |
| **every absolute macOS path removed** | `scripts/21`, `scripts/23`, and 11 auxiliary scripts | `Path('/Users/zhangjiuqi/Desktop/distortion/...')` → a resolution relative to the file. Verified: `grep -r '/Users/'` over `cocd/**/*.py` returns nothing. |
| **dataset and output roots made configurable** | `cocd/paths.py` | `COCD_DATASET_ROOT`, `COCD_SPLIT_DIR`, `COCD_OUT_ROOT`. Defaults match the package layout, so no configuration is needed. |
| **the split is read, not re-drawn** | `scripts/21::split()` | reads `data/splits/*_ids.csv` when present; the original seed-42 draw remains only as a fallback. This is stricter than before, where the validation draw was recomputed from the RNG on every run. |
| **checkpoint lookup made portable** | `cocd/paths.py::resolve_checkpoint` + `scripts/23` | searches `experiments/ours_v2/`, `checkpoints/`, `legacy_checkpoints/v1_protocol/`, `experiments/rapid_landslide_cocd/`. Replaces the two hard-coded `A/'experiments'/'rapid_landslide_cocd'/'S0.pt'` sites, so the shipped layout works. |
| **`map_location` everywhere** | already present | every `torch.load` in the training path passes `map_location=DEV`, so an MPS checkpoint loads on CUDA. No change needed. |
| **loader default batch** | `cocd/paths.py::LOADER_BATCH` | the stage-1 loader hard-coded 2 on MPS / 4 otherwise, which would have been very slow on CUDA. Now 16 off MPS, overridable with `COCD_LOADER_BATCH`. `scripts/23` still takes `--batch`. |
| **`__main__` guards** | already present | both `scripts/21` and `scripts/23` guard with `if __name__ == '__main__':`. This matters on Windows: `scripts/23` imports `scripts/21` by path, and an unguarded module body would execute on import under `spawn`. |
| **DataLoader** | already safe | `num_workers=0` at every construction site, so no Windows spawn/re-pickle issues. Raising it is possible but is not needed — the pipeline is not the bottleneck. |
| **no shell dependency for training** | new `cocd/run_protocol_v2.py` | the sweep order, the `batch × accum == 16` contract check and the "a checkpoint was written moments ago" guard, in Python. Every step is also a direct `python cocd/scripts/23_train_ours_v2.py ...` call. |
| **`caffeinate` removed** | — | macOS sleep prevention has no counterpart and is unnecessary on Windows. |

Deliberately **not** changed: the model, the losses, the protocol numbers, the
evaluation, the split. This is a port, not a revision.

## 6. Things the Windows side still has to handle

1. **CUDA is not installed here.** Section [2] of `verify_transfer.py` prints a
   warning instead of a failure on a machine without an NVIDIA GPU, and the check
   still passes. Training on CPU is not practical; install a CUDA build of torch
   first (`ENVIRONMENT.md` §"Recommended Windows setup").
2. **Re-measure the batch size.** Physical 16 was measured to fit in a 24 GB
   unified-memory Mac. On a smaller GPU use `--batch 8 --accum 2`; the effective
   batch must stay 16 and the script exits if it does not. `cocd/scripts/47_protocol_batch_feasibility.py`
   measures the peak for any `(batch, accum)` pair.
3. **`processed/*/metadata.json` contains macOS absolute paths.** They are
   provenance only — no code reads that file — and the package is therefore
   *not* free of the string `/Users/zhangjiuqi`. Do not rewrite them; see
   `DATASET_MANIFEST.md` §6.
4. **`PLAN` still contains `R3D`.** `L_delta` has been withdrawn from the new
   method, so the sweep should be driven with an explicit `--stages` list until
   `PLAN` is updated.
5. **`D` in `L_dist` is undecided** (L1 / MSE / SmoothL1 / cosine) and λ depends on
   it. Decide before calibrating λ, because `L_dist⁰` is defined by that choice.
6. **`third_party/landslide_baselines/dis2/` is reference code.** It is kept so the
   port in `losses/distill.py` can be diffed against the original. It is not
   imported and it brings its own dataset/training utilities that do not run here.
7. **The `--protocol v1` path is preserved but is not the working path.** A v1 run
   needs `S0.pt` plus the v1 schedule and is only for reproducing old numbers.
   Do not compare v1 and v2 absolute values.
8. **`baselines/` is absent.** If an external baseline has to be retrained, take it
   from its upstream repository; the result tables of the three that were run are
   in `results/grsl_external_baselines/`.

## 7. Verification performed on the package

| check | result |
|---|---|
| every `.py` in `cocd/` parses | 30 files, 0 syntax errors |
| no absolute macOS path left in `cocd/**/*.py` | 0 matches for `/Users/` |
| the package resolves its own paths and device | `cocd/paths.py` prints all four roots plus the device |
| the frozen split loads and has the right sizes | 1,233 / 137 / 343, disjoint, in the pinned order |
| protocol assertions | `cocd/scripts/48_smoke_protocol_v2.py` → **57 passed, 0 failed** |
| one location read end to end, a batch built and pushed through encoder and decoder | `verify_transfer.py` sections [4]–[6] |
| all seven checkpoints load through the code path that consumes them | `verify_transfer.py` section [7] |
| launcher modes and guards | `plan`, `train`, `test`, `--dry-run`, and the `batch × accum ≠ 16` refusal all behave |
| data integrity | 13,704 files hashed individually into `data/haiti/DATA_MANIFEST.csv` |
| acceptance check | `verify_transfer.py` → **TRANSFER CHECK PASSED** (42 checks, 1 CUDA warning, 0 failures) |
