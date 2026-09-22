# Counter-guided Target Evidence Search for target-only SAR segmentation

This repository is the curated technical record for the Haiti SAR GRSL study.
It asks whether complementary-orbit information available during training can
improve target-only deployment by transferring a counter-guided evidence-search
strategy.

## FINAL METHOD

Counter-Guided Target Evidence Search with Target-Only Search Distillation.

The final **CG10 Teacher** receives a target and counter orbit.  The counter
orbit is used only to generate the search policy; values retrieved by
deformable search come only from target features, and the decoder receives only
the searched target representation.  The final **S2 Student** receives only a
single target-orbit stack `[VV_pre, VH_pre, VV_post, VH_post, orbit_id]`.  It
learns a target-side policy and searched evidence through policy and evidence
distillation from CG10.

The final objectives are:

`L_teacher = L_seg + 0.1 L_TGD`

`L_student = L_seg + lambda_KD [0.5 L_policy + 0.5 L_evidence]`, calibrated
so the KD contribution is 10% of segmentation loss at initialization.

## Read in this order

1. [Model description](docs/model_description.md)
2. [Experiment summary](docs/experiment_summary.md)
3. [Naming dictionary](docs/naming_dictionary.md)
4. [Cleanup audit](cleanup_audit.md) and [cleanup report](cleanup_report.md)
5. [Audit and retention policy](project_audit.md)
6. [Complete paper package](experiments/GRSL_complete_paper_package/README.md)

## Canonical code

| Purpose | Entrypoint |
|---|---|
| Final Teacher and SG/CG ablation | `cocd/windows_main/teacher_target_search_v2.py` |
| Final Teacher model | `cocd/windows_main/models_teacher_target_search_v2.py` |
| Final Student S0/S1/S2 | `cocd/windows_main/student_search_kd_v2.py` |
| Final Student model | `cocd/windows_main/models_student_target_search_v2.py` |
| KD evaluation | `cocd/windows_main/evaluate_student_search_kd_v2.py` |
| KD composition/strength ablations | `cocd/windows_main/run_student_kd_weight_sweep.py`, `run_student_kd_strength_sweep.py` |
| Paper-package builders | `cocd/windows_main/build_grsl_final_package.py`, `build_grsl_qualitative_package.py`, `assemble_grsl_complete_paper_package.py` |

## Results and figures

All paper-ready compact artefacts are under
`experiments/GRSL_complete_paper_package/`.  In particular,
`qualitative/figures/fig_overall_external_comparison.png` compares Boehm,
CDNetE, MFEWF, DIS2, Vanilla KD, SO and S2; the matching TGD local comparison
is `fig_TGD_external_comparison.png`.

Raw data, checkpoints, NPZ prediction tensors and full logs are intentionally
excluded from Git.  See `dataset/README.md` and `project_audit.md`.
