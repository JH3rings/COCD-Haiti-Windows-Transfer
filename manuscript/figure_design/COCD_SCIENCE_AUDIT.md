# COCD / DLR F_SSP diagram truth checklist (2026-11-13)

**Use only for this landslide paper**, not the universal drawing engine. Live scientific authority (check latest main branch and commit again before rendering):

- `experiments/dlr_official6_faithful_20261110/common.py`: `Inputs`, `UNet`, `segmentation`.
- `experiments/dlr_official6_faithful_20261110/final_method.py`: `FinalNet`, `SharedEPM`, `cew_terms`, `tokens`, `objective`, `LEVELS`, `LOSS_LEVELS`.
- `experiments/dlr_official6_faithful_20261110/pr.py`: `EPRM`, `cross_event_wasserstein`, `PROTOS`, `PDIM`, `TAU`.
- `experiments/dlr_official6_faithful_20261110/methods.py`, `run.py`: teacher assembly and actual input.
- `manuscript/main.tex`: actual method narrative and figure caption.
- `manuscript/tools/draw_method.py`, `source/method.svg`: existing, scientifically grounded diagram baseline.
- `manuscript/figures_dlr/qualitative.png`: reference real outputs.

## Mandatory nodes and relationships

1. **Student input 6 channels**: PRE1/POST1 gVV/gVH (4) + DEM/slope (2); deployment contains **no optical**.
2. **Teacher input 18 channels**: same six SAR/terrain channels plus 12 PRE1/POST1 Sentinel-2 bands. The pretrained teacher is frozen during student distillation.
3. **Matched 5-stage U-Nets**: enc spatial resolution 128→64→32→16→8, widths 64/128/256/512/1024; 4 decoder stages 16→32→64→128 with skip concatenation; student segmentation head outputs binary logits.
4. **Student EPM after decoder indices 1/2/3** = resolutions 32/64/128. All three use **one learned bank** of K=16 prototypes, embedding dim 64, but **separate query, value, and output projections** per stage; zero-initialized output projection and residual addition. EPM stays in inference.
5. **CEW relation loss training-only**, at **decoder index 1 (32 px)** after EPM, not at all three scales in final F_SSP. Both teacher and student corresponding features make 4×4 pooled region tokens; cover >=0.75; cross-event relationships only.
6. **Cross-event Wasserstein semantics**: teacher and student compute within-modality similarity matrices over region tokens, exclude same-event pairs, subtract each modality's minimum valid cross-event similarity, **sort independently per anchor**, compare valid sorted values via mean squared difference = squared 1D Wasserstein. **Not** hard pairwise student-teacher similarity matching; **not** a Sinkhorn solver.
7. **Student segmentation loss**: valid-mask BCE + Dice; final training combines segmentation with weighted CEW (`LAMBDA_W=30` in final fixed run). GT supports segmentation supervision; teacher output mask is not directly transferred to student as the target.
8. **Real figure assets**: use same known test case across optical, SAR, GT and teacher/student predictions from `qualitative.png`. Do not identify a probability map as binary mask or vice versa without checking the plot legend.
9. **A/B/C inset meanings**: (a) actual network, (b) EPM prototype/refinement with shared bank, (c) cross-event relation distribution loss with independent sorting.

## Prohibited diagram errors observed in past ImageGen concepts

- Calling EPM "Event Prototype Mining" when the manuscript calls it event-shared prototype module.
- Calling CEW "Cross-Event Weighting" instead of cross-event Wasserstein relation distillation.
- Adding pink bidirectional arrows between all teacher/student encoder stages; no such transfer exists.
- Adding EPM to teacher or placing just one student EPM at the bottleneck.
- Showing loss on the teacher's segmentation prediction directly supervised into the student.
- Drawing CEW at all 3 EPM scales in the final F_SSP.
- Making the training graph look identical to the inference-only student graph.
- Adding invented optical inputs to student inference.
- Presenting synthetic heatmaps, ranks or confusion grids as measured experiment values.

## Final image graph must pass
For each node/edge: record code source, exact stage, input tensor(s), output and whether the arrow means forward flow, shared weights, loss or supervision. If any of these are unsupported by source, mark it unverified and leave it out.
