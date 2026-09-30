# Locked architecture specification

- Each orbit input is five channels in the loader's actual order `[pre VV, pre VH, post VV, post VH, orbit_id]`; the encoder weights are shared between target and counter.
- Shared ConvNeXt-Tiny encoder with a three-level FPN (P2/P3/P4; 128 channels).
- T0 applies an independent same-channel 1x1 convolution to `[target, counter]` at each FPN level.
- T1 uses the fused dual context to predict bounded offsets and softmax weights at a 32x32 query grid; there are 3 levels and 4 points per level. All sampled values come from the target FPN memory.
- SearchWrite is a zero-initialized 1x1 projection over target P2 and retrieved target evidence; the real decoder is the existing FPN head.
- T1 supervises the dual path and target-evidence auxiliary path with `L_dual + 0.10 L_aux`. S0 inherits the backbone, target policy, SearchWrite and target projection once, removes counter/fusion, and adapts with target + GT using `L_seg` only.
- Counter is not an inference input to S0. Search KD/S1 is not trained.

Implementation source: `cocd/windows_main/models_da_search.py`.
