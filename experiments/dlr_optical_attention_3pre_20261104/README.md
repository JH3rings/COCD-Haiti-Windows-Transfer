# DLR 3+1 optical-to-SAR distillation

The Student uses three pre-event and one dataset-selected post-event VV/VH acquisitions. The optical Teacher receives six pre/post optical bands during training. The dataset uses the official 355 train, 88 validation, and 120 test patches, including 44 in the independent subset.

| Model | Independent 44 IoU | All 120 IoU |
| --- | ---: | ---: |
| ConvNeXtV2-Tiny SAR baseline | 40.41% | 37.45% |
| M-FSAD adaptation | 48.02% | 41.04% |
| Change attention + output KD | 40.97% | 37.62% |
| Change attention + Haar + output KD | 48.08% | 41.34% |
| Paper-based DLR U-Net, eight SAR channels | 71.89% | 59.65% |

The U-Net uses separate input channels for each acquisition. The Tiny Student averages the three encoded pre-event feature sets. The observed gap does not isolate temporal aggregation from other architecture and initialization differences.

Entry points are `train.py`, `train_unet_sar.py`, and `monitor.py`. Running the training entry points requires the local official DLR HDF5 files, the baseline registry assets (including its clean initialization and split manifest), and the referenced COCD initialization. These data and weights are external to this source release. File paths and locked checksums must match the local installation before training.
