# External landslide / missing-modality baselines

This directory keeps unmodified upstream source snapshots separate from Haiti adapters.
Adapters, checkpoints and measured results belong under the project `baselines/` and
`experiments/` directories; they must not silently alter an upstream snapshot.

| Baseline | Paper | Upstream source | Status | Pinned source | License | Haiti adaptation contract |
|---|---|---|---|---|---|---|
| CDNetE / Early Fusion | Antoine Bralet, *Deep Learning for Multimodal Detection of Sudden and Slow Moving Slope Instabilities on Bitemporal Remote Sensing Images* (2024), [HAL thesis record](https://hal.science/tel-05029007v1) | No Haiti/CDNetE official repository found yet after author-page and GitHub search | source verification in progress | n/a | n/a | Faithful implementation only if official source remains unavailable; target-orbit pre/post VV/VH plus orbit indicator only |
| Boehm SAR U-Net | Boehm et al., *Deep Learning for Rapid Landslide Detection using SAR Datacubes* (2022), [arXiv](https://arxiv.org/abs/2211.02869) | [iprapas/landslide-sar-unet](https://github.com/iprapas/landslide-sar-unet) | official source snapshot downloaded | `e35fad9948c7ff27b7a7f751789c5e90763ee7e4` | MIT | Preserve U-Net++ bitemporal SAR segmentation and Dice/CE choices; replace Hokkaido datacube loader/split and use a capacity-matched encoder |
| MFEWF | Chen et al., *Automatic detection of earthquake triggered landslides using Sentinel-1 SAR imagery based on deep learning* (2024), [article](https://doi.org/10.1080/17538947.2024.2393261) | Paper names [ChenLifu2022](https://github.com/ChenLifu2022), whose visible public repository is unrelated Aircraft-detection | official source not currently available | n/a | n/a | Implement MFEWF's documented DRN + AMM + CAASP + MFFRM only after source-status record is complete; capacity-match its backbone |
| DIS2-style adapted | Kieu et al., *DIS2* (WACV Workshops 2026), [paper](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html) | [nhikieu/DIS2](https://github.com/nhikieu/DIS2) | official source snapshot downloaded | `497afc2d86ea6a85271ccace0a052603d918a9b3` | MIT | Keep the full-modality teacher / missing-modality student and multi-level feature + logit distillation; use the frozen Haiti Teacher/Student information budget and call it `DIS2-style adapted` |

The official snapshots are intentionally not imported as project modules. This avoids
mixing their original Hokkaido/Potsdam data protocols with the frozen Haiti split.
