# README — Windows

## 1. Unpack

Unzip anywhere with enough room for about 4 GB free (2.2 GB unpacked, plus what a
run writes). Keep the folder structure; everything inside the package finds its
own files relative to itself.

```
COCD_Haiti_Windows_Transfer\
    README_WINDOWS.md            <- you are here
    HANDOFF_FOR_WINDOWS_AI.md    <- read this next, or hand it to an AI
    DATASET_MANIFEST.md  CHECKPOINTS.md  ENVIRONMENT.md  TRANSFER_AUDIT.md
    requirements.txt  verify_transfer.py
    cocd\        code (data loader, models, losses, training entry points)
    data\haiti\  the dataset, 1,372.6 MB across 13,704 files
    data\splits\ the pinned train / val / test id lists
    checkpoints\ teacher_v2.pt
    legacy_checkpoints\v1_protocol\  six v1 checkpoints
    results\     every recorded csv / json / log
    docs\ reports\
```

## 2. Build the environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Pick the CUDA index that matches your driver (`nvidia-smi` → the CUDA version it
reports). `cu124` is a safe default for a current driver. **Install torch before
`requirements.txt`**, otherwise pip may settle on a CPU-only wheel.

Details and the Windows-specific notes for each package are in `ENVIRONMENT.md`.

## 3. Put the data where it belongs, and verify it

Nothing has to be moved: unpacking restores `data\haiti\` exactly as the loader
expects. If you want the data on another drive, set `COCD_DATASET_ROOT` to that
path instead of moving the folder.

Expected layout and per-file hashes are in `DATASET_MANIFEST.md`, section 8.
Quick totals:

```
data\haiti\Pre_event\S1_ASC_20210805   1713 files   322.0 MB
data\haiti\Pre_event\S1_DESC_20210803  1713 files   322.0 MB
data\haiti\Post_event\S1_ASC_20210817  1713 files   322.0 MB
data\haiti\Post_event\S1_DESC_20210815 1713 files   322.0 MB
data\haiti\processed\                  6852 files    84.7 MB
```

If the copy came over a network or a USB disk, run the per-file SHA256 check in
`DATASET_MANIFEST.md` §8 before training.

## 4. Run the acceptance check

```powershell
python verify_transfer.py
```

It imports everything, reports the device, confirms the pinned split, reads one
training location end to end, builds a batch, pushes it through the encoder and
the decoder, and loads every shipped checkpoint. It trains nothing and writes
nothing.

Success looks like:

```
TRANSFER CHECK PASSED
```

Read the warnings and notes above that line. Expect one warning about CUDA if the
machine has no NVIDIA GPU — that is a warning, not a failure, but do not train
without one.

## 5. Then hand the project to an AI, do not start training on your own

Give `HANDOFF_FOR_WINDOWS_AI.md` to the AI that will continue the work. It
contains the research question, the data protocol, the frozen training protocol,
what is legacy, the final design of the new method, and the order of the next
twelve steps.

**The next step is not "train something".** It is to implement the new teacher and
then run the teacher audit described in §6 of that document. Training a student
before that audit passes wastes hours on a teacher whose complement was never
verified.

If you want to see the plumbing work first, this is safe and cheap — it changes
nothing and trains nothing:

```powershell
python cocd\run_protocol_v2.py plan
```

To actually run the unified comparison later:

```powershell
python cocd\run_protocol_v2.py train                 # teacher if absent, then the arms
python cocd\run_protocol_v2.py test                  # the teacher's one-shot test table
python cocd\run_protocol_v2.py train --stages SO     # just the baseline, no teacher needed
```

## 6. Where a run writes

Everything goes under `experiments\ours_v2\` (change with `COCD_OUT_ROOT`):

```
experiments\ours_v2\
    teacher<tag>.pt                     best-validation teacher
    teacher<tag>_latest.pt              resumable state, one per completed epoch
    teacher<tag>_curve.csv              per-5-epoch learning curve
    <ARM><tag>_seed42.pt                best-validation student arm
    <ARM><tag>_seed42_latest.pt         resumable state
    results_<stage><tag>_seed42.csv     per-stage metrics table
    thresholds_<stage><tag>_seed42.json the frozen validation threshold
    results<tag>_seed42.csv             cumulative table, merged on (method, partition, region)
    protocol_v2_log.txt                 the launcher's log
```

Every artefact carries the `--tag`. The launcher defaults to `_v2`, which is why
it reuses the shipped `checkpoints\teacher_v2.pt` instead of retraining a teacher.
Give a new experiment its own tag so it cannot collide with anything shipped.

## 7. Layout of a result table

One row per (method, partition, region):

- `partition` ∈ `overall`, `asc`, `desc` — which target orbit the cases are from;
- `region` ∈ `all`, `G00`, `G01`, `G10`, `G11` — the geometry partition;
- `pixels`, `iou`, `f1`, `precision`, `recall`, `auprc` — computed at the frozen
  validation-selected threshold; `auprc` is threshold-free;
- `threshold` — the frozen threshold, and `iou@0.5` / `f1@0.5` — the same reading
  at 0.5, kept only as continuity with the v1 tables.

`G10` (target view distorted, counter view valid) is the region the study is
about. Always report all four regions, never only the aggregate.
