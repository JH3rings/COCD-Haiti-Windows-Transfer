# Baseline provenance

All comparison methods consume the same target-only 5-channel pre/post SAR
view.  Raw checkpoints and per-pixel NPZ predictions are local-only.  Compact
metrics and final visual comparisons are in
`experiments/GRSL_complete_paper_package/`.

| Method | Source/status |
|---|---|
| Boehm | Local adapted SAR U-Net++ runner and retained prediction cache. |
| CDNetE | Local adapted early-fusion runner and retained prediction cache. |
| MFEWF | Local adapted implementation and retained prediction cache. |
| DIS2-port | Archived checkpoint re-inferred by `retest_dis2_tgd.py`. |
| Vanilla KD | Re-trained VKD R1 baseline, exported by `retest_vanilla_kd.py`. |
