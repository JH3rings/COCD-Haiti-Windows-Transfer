# Environment

## What the results in this package were produced with (macOS, Apple silicon)

| item | value |
|---|---|
| machine | Mac17,4, 10 cores, 24 GB unified memory |
| OS | macOS |
| Python | 3.12.14 |
| accelerator | Apple MPS (`torch.backends.mps.is_available()` was used throughout) |
| CUDA | not applicable |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scipy | 1.18.1 (not imported by the code in this package) |
| scikit-learn | 1.9.0 |
| rasterio | 1.5.1 |
| GDAL | not installed as a separate package (bundled with the rasterio wheel) |
| tqdm | 4.70.0 |
| matplotlib | 3.11.1 (not imported by the code in this package) |

Do **not** reproduce this list as a `pip freeze` on the target machine. It is
recorded so a difference in behaviour can be traced back to a version.

## Measured cost on that machine

These numbers describe the Mac; a CUDA host will differ, and the batch size
question should be re-measured there (`cocd/scripts/47_protocol_batch_feasibility.py`).

| quantity | measured |
|---|---|
| loader batch 16, teacher step (forward + backward) | ≈ 0.75 s/batch |
| loader batch 16, student step | ≈ 0.72 s/batch |
| batches per epoch at physical 16 | 232 (3,699 samples) |
| wall clock per arm, 20 epochs | ≈ 1 h |
| peak device memory, student at batch 16 | 1.18 GiB |
| peak device memory, teacher at batch 16 (two orbits) | 2.50–2.83 GiB |
| device memory ceiling reported by torch | 17.8 GiB |

## Recommended Windows setup

1. **Python 3.11 or 3.12.** 3.12 was used here; 3.11 avoids a few wheel gaps
   that can appear for `rasterio` on the newest Python. Avoid 3.13 unless every
   wheel below is confirmed available for it.
2. **Create an isolated environment** so the CUDA wheel choice is not fought
   over by an existing global torch:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install PyTorch from the CUDA index that matches the driver**, before
   anything else, so `pip` does not pull a CPU-only wheel and then refuse to
   replace it:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   `cu124` is a reasonable default for a recent driver; check
   `nvidia-smi` for the driver version and pick the matching index
   (`cu121`, `cu124`, `cu126`, …). Any of them is fine here: the model is a
   plain convnet and uses no version-specific operator.
4. **Install the rest:**
   ```
   pip install -r requirements.txt
   ```
5. **Check the device is actually visible before doing anything else:**
   ```
   python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Then `python verify_transfer.py`, whose section [2] reports the same thing.

### Packages that need attention on Windows

| package | what to expect |
|---|---|
| `torch` / `torchvision` | the only package where the Windows install differs materially. A CPU-only wheel installs happily and silently; check `torch.cuda.is_available()` rather than trusting the install log. |
| `rasterio` | ships its own GDAL. Do not install a separate `GDAL` package on the same environment — the two GDAL builds will conflict. `pip install rasterio` alone is enough, and it is the only geospatial dependency the code needs. |
| `scikit-learn` | used for a single function, `average_precision_score`. Any recent version works. |
| `tqdm` | progress bars only. |
| NumPy 2.x | fine. The loader does plain float32 array arithmetic; nothing in the package depends on removed NumPy 1.x aliases. |

### Device selection in the package

`cocd/paths.py` resolves the device once, in the order **CUDA → MPS → CPU**.
Nothing in the training path hard-codes MPS. Set `COCD_DEVICE=cpu` to force CPU,
or `COCD_DEVICE=cuda:1` to pick a specific GPU.

Two legacy measurements were taken on MPS and are quoted in the reports; they are
Mac-specific and should not be carried over: the throughput findings in
`cocd/scripts/27_bench_ours_v2_throughput.py` (which is not part of this package)
and the batch-size default of 2 that the stage-1 loader used. Both are replaced
here by `paths.LOADER_BATCH`, which defaults to 16 off MPS.

### One number to re-measure on the CUDA host

Physical batch 16 was measured to fit on a 24 GB unified-memory machine. On a
smaller GPU, `--batch 16` may not fit; the protocol requires
`physical × accumulate == 16`, so use `--batch 8 --accum 2` (or `--batch 4
--accum 4`) rather than changing the effective batch. The script refuses to run
if the product is not 16.
