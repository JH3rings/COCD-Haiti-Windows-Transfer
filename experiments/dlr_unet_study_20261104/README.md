# Active DLR U-Net study

This directory supersedes the DLR ConvNeXt training entries. The prior 3+1 study remains traceable on GitHub at `840054e` (`codex/dlr-3plus1-sar`). The earlier Pico, ConvNeXt, search, M-FSAD and FFT/Haar studies remain historical evidence. Keep their dependencies and recorded results together. Haiti training and shared model code are outside this migration.

The active baseline uses the preserved paper-based U-Net body in `unet.py`, the official data reader in `dataset.py`, and the shared pooled scoring logic in `evaluate.py`. The original author training source was unavailable. The four-channel B1 changes the first convolution input width and keeps the five-level U-Net body. B0 uses the frozen six-channel SAR/DEM/slope reference from the equal BCE/Dice run.

Run `python train_baseline.py` to verify B0 and train B1. The progress file is `results/B1/progress.json`. B1 retains the official split and B0 SAR normalization, uses synchronized augmentation and numerical validity masking, and selects checkpoints with validation only.

B2 remains gated on acquisition verification. The preceding audit found 563 structural 3+1 pairs, but full relative-orbit evidence only for 7 training and 2 validation samples and none in test. The earlier eight-channel 3+1 run is a separate historical architecture/input variant. It cannot substitute for the proposed four-channel aggregated B2.

The optical Teacher and D0–D3 are gated on validating the B1 baseline. Existing distillation scores are historical and are not transferred to U-Net.

| Prior path | Status | Dependency decision |
| --- | --- | --- |
| `dlr_optical_attention_3pre_20261104` | Archived in `840054e` | Retain the measured eight-channel U-Net, Tiny and KD comparisons as historical evidence. |
| `dlr_threelevel_frequency_v2_20261103`, `dlr_published_kd_20261103`, `mfsad_v2tiny_20261103` | Retired as active DLR entry points | Keep source dependencies on the historical branch and preserve frozen results. |
| `dlr_bce_dice_equal_20261103/results/unet` | B0 reference | Preserve its initial state, training-only normalization, official manifest and frozen metrics. |
| Haiti and shared COCD modules | Independent scope | Retain all paths and usage. |

Local execution needs the original HDF5 archives and the recorded B0 reference assets in the paths above. Checkpoints and dataset files remain external to GitHub.
