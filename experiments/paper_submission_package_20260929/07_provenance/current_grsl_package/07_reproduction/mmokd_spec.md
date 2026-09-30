# MMOKD-Port reproduction specification

Status: pre-implementation specification, 2026-09-28. This is a mechanism-faithful Haiti port, not a numerical reproduction of the published land-cover classification experiments.

## Source audit

| Item | Finding | Evidence/status |
|---|---|---|
| Paper | Xiao Liu et al., “Multimodal Online Knowledge Distillation Framework for Land Use/Cover Classification Using Full or Missing Modalities,” IEEE TGRS 62 (2024), DOI 10.1109/TGRS.2024.3388604. | [IEEE Xplore DOI page](https://doi.org/10.1109/TGRS.2024.3388604); [NASA ADS/SciX record](https://scixplorer.org/abs/2024ITGRS..6288604L/abstract). The paper abstract describes an end-to-end fusion network and two modality-specific networks, MFF, SFG, and online JD-MFK. |
| Full text | The publisher page returned the IEEE JavaScript/robot gate in this environment; DOI metadata identifies a 17-page paper but did not expose its PDF. ResearchGate reports no full text. | Exact layer equations and optimization details could not be independently inspected here. |
| Supplement | Searches of the publisher record and indexed supplementary-material sources found no accessible supplement. | No supplement URL/file located. |
| Public technical/patent material | Searches for MMOKD-specific patent/technical documentation did not locate a public implementation description beyond the paper abstract. | No exact MFF/SFG equations or α/β values verified from a patent. |
| Author code | The authors' public repository contains only a README, which says the code will be released soon. | [Xiao-RS/MMOKD](https://github.com/Xiao-RS/MMOKD), checked 2026-09-28. |

## Published method elements retained

The experiment brief requires the following method elements, which define this port:

1. **Online three-branch learning:** one ASC/DESC fusion Teacher and two modality-specific branches are optimized synchronously; the fusion branch is not frozen.
2. **MFF:** combine paired ASC/DESC encoder features symmetrically into fusion features at multiple encoder levels.
3. **SFG:** transform each single-orbit encoder feature into a pseudo-fusion feature. Each deployable branch uses only its own orbit feature.
4. **JD-MFK encoding distillation:** MSE from fusion Teacher features to the corresponding pseudo-fusion Student features at a low and a high encoder level.
5. **JD-MFK decoding-relation distillation:** match full pixel-to-pixel cosine relation matrices from decoder features, with relation entries `S_ij = cosine(f_i, f_j)`.
6. **JD-MFK response distillation:** two-class background/landslide logits with KL at temperature `T=4`.
7. **Segmentation supervision:** CE plus Dice on each of ASC, DESC, and fusion outputs.

The `T=4` response temperature and these loss/module requirements are specified in the present experiment brief. Because the article body was not accessible, this audit does not claim independent full-text verification of the temperature or formula notation.

## Haiti adaptation and implementation choices

| Component | Original task/context | Haiti port |
|---|---|---|
| Modalities | Optical + SAR | ASC + DESC SAR; direction roles are symmetric. |
| Branches | Fusion model plus two modality-specific models | Two single-orbit ConvNeXt-Tiny/FPN branches and one paired fusion Teacher. Train all together; export either Student independently. |
| Initialization | Paper-specific backbones and data | All encoder/FPN branches start from the same existing seed-42 Haiti `SO_backbone_seed42.pt` initialization. The two-class classifier is initialized by splitting the existing one-logit classifier into equivalent `[-z/2,+z/2]` class logits. |
| MFF | Published module details unavailable in accessible sources | Symmetric per-level concatenation followed by a channel projection; no preferred orbit or cross-view-only shortcut. Exact operator is a necessary port choice, not asserted as the source's exact MFF. |
| SFG | Published module details unavailable in accessible sources | One learned 1x1 channel projection per level, separately for ASC and DESC, producing pseudo-fusion maps from only that branch's own encoder maps. No counter activation enters either SFG or decoder. Exact operator is a necessary port choice. |
| Feature loss | Multi-type feature distillation | Equal mean of low-level `c2` and high-level `c5` mean-squared errors, fusion features as targets. |
| Relation loss | Pixel-to-pixel decoder cosine relations | Full spatial relation matrix over the decoder's 32x32 feature grid. Student relation matrices match the fusion Teacher matrix by mean-squared error; no local crop/window. |
| Response loss | Temperature-scaled KL | Symmetric ASC and DESC two-class KL to the live fusion Teacher at `T=4`, with `T^2` multiplier. |
| Supervision | Original task-specific supervised loss | Per-branch `CE + soft Dice`, then the arithmetic mean over ASC, DESC, and fusion outputs. |
| Total weights | Paper coefficients could not be recovered from accessible full text/supplement/code | One symmetric, unswept choice: `alpha=1`, `beta=1`; `L_total = beta*L_sup + alpha*mean(L_feature, L_relation, L_response)`. Each constituent is averaged over its specified levels/branches before entering the joint distillation mean. No 10% controller or metric-driven coefficient selection. This is a documented adaptation, not a verified published coefficient. |
| Geometry | Not part of the requested method | Geometry codes are loaded only for post-hoc reporting. They do not enter network input, CE/Dice, MFF/SFG, or any KD term. |
| Protocol | The published paper's classification data/training schedule | Reuse the locked Haiti train/test split (1,370/0/343 locations), existing three training modes (4,110 paired rows), seed 42, batch 16, Adam `lr=5e-5`, betas `(0.9,0.999)`, `eps=1e-8`, weight decay 0, max 100 epochs, patience 2, and project test-Overall-AUPRC checkpoint monitor. This inherits the project's test-monitored-development limitation. |

## Faithfulness boundary

**Retained:** the online fusion Teacher/two single-modality branches, symmetric opposite-view treatment, MFF and SFG roles, the three requested JD-MFK knowledge types, `T=4`, and CE+Dice supervision on all branches.

**Necessary task adaptations:** changing land-cover classification into dense binary segmentation; mapping optical/SAR to ASC/DESC; using the Haiti ConvNeXt-Tiny/FPN; using two-class segmentation logits; and reporting each target-only Student separately.

**Unresolved / not independently verifiable from public materials:** original exact MFF and SFG layer equations; exact feature layers and normalization for all three JD-MFK terms; published α/β values and their normalization convention; original optimizer and schedule; and whether an online supplement exists outside the indexed sources checked here. The fixed port choices above are recorded without a hyperparameter sweep.

No code had been changed when this specification was first written. The implementation and the required 32-pair sanity gate are recorded separately in the run outputs.
