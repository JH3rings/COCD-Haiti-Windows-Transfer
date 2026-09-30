# Related multi-view privileged-information methods

Literature context only. Neither method was trained or included in the Haiti numerical comparison.

## MVCD (Neurocomputing, 2026)

**Learning from multi-view fragments: An adaptive consistency distillation framework for occluded person re-identification** introduces Multi-View Consistency Distillation under a learning-using-privileged-information setup. Its teacher uses training-only multiple views and identity labels, with saliency-guided feature purification, cross-view patch alignment, and reliability-guided aggregation. It distills consistency to a single-view student and removes its multi-view components at inference.

This is close in paradigm to training with privileged views and deploying a single-view model. Its task is occluded-person re-identification, where aligned identity fragments and person-part correspondence are central. It is not dense remote-sensing segmentation, so it is not a directly comparable Haiti baseline.

Source: [ScienceDirect article page](https://www.sciencedirect.com/science/article/pii/S0925231226004121), DOI [10.1016/j.neucom.2026.133015](https://doi.org/10.1016/j.neucom.2026.133015), accessed 2026-09-28.

## EpiDistill (ECCV 2026)

**Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues for Monocular Depth** transfers multi-view geometric behavior into a monocular depth model. The multi-view branch uses depth-guided epipolar attention over source views. At inference, the single-view model uses learned Rectified Stereo Tokens to preserve that attention path without source images.

The shared idea is to transfer multi-view search/attention behavior into a single-view deployment model. EpiDistill depends on calibrated rectified stereo geometry and predicts monocular depth; Haiti uses opposite-orbit SAR segmentation and does not have the same camera epipolar construction. It is conceptual related work, not a directly comparable baseline.

Sources: [ECCV 2026 project page](https://jungheekim29.github.io/EpiDistill/), [arXiv paper](https://arxiv.org/abs/2607.15600), and [author code](https://github.com/JungHeeKim29/EpiDistill), accessed 2026-09-28.
