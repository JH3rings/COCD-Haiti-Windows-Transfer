# COCD Haiti code package

This repository snapshot is intentionally code-only. Raw Haiti imagery,
processed samples, checkpoints, optimizer states, predictions, full experiment
directories, and large generated result folders are excluded by `.gitignore`.

## Recommended reading order

1. `README.md` — final method and canonical entry points.
2. `docs/final_architecture.md` — locked input, Teacher, Student, and KD
   contract.
3. `docs/model_description.md` and `docs/experiment_summary.md` — method
   semantics and retained paper evidence.
4. `cocd/windows_main/data.py` — active Haiti pair/split loading.
5. `cocd/windows_main/models_teacher_target_search_v2.py` and
   `models_student_target_search_v2.py` — final models.
6. `cocd/windows_main/teacher_target_search_v2.py` and
   `student_search_kd_v2.py` — training and distillation entry points.
7. `experiments/GRSL_complete_paper_package/README.md` — paper-ready results.
8. `baselines/README.md` and `cocd/third_party/landslide_baselines/` — external
   comparison methods and provenance.

## Important evidence boundary

The source code and small text/CSV/JSON documentation in this repository can be
inspected directly. Trained weights and raw data are not included, so numerical
claims must be treated as recorded experiment evidence rather than independently
recomputed results unless the user supplies the corresponding private artifacts.

The repository also contains archived research provenance. Anything under
`archive/history/` is not part of the final method and must not be imported by
the final-v2 entry points. Historical result packages remain available for
audit, but their values are evidence records rather than newly recomputed
claims.
