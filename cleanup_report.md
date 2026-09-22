# GRSL final-repository cleanup report

Date: 2026-09-22  
Branch: `cleanup/grsl-final`  
Rollback branch: `archive/pre-grsl-final-cleanup` at committed `HEAD` `3664fb2`  
Status: **PASS — cleanup completed without training or result regeneration**

## What changed

1. Superseded self-owned architectures and their runners/tests were moved
   unchanged into `archive/history/code/`.
2. Old handoffs, checkpoint ledgers, planning documents, protocols, and
   architecture reports were moved unchanged into `archive/history/docs/`,
   `archive/history/protocols/`, and `archive/history/reports/`.
3. `cocd/windows_main/models_da_search.py` was reduced to the final search
   primitives plus `TargetOnlyBaseline`:
   `CHANNELS`, `LEVELS`, `POINTS`, `pyramid`, `pyramid_from_encoded`,
   `SearchContext`, and `DeformableSearch`.
4. `cocd/models/landslide_cocd.py` now exposes only the shared final
   `ConvNeXtTinyFPN` backbone/FPN.
5. The external-baseline runner now imports the shared final data loader and
   metric path instead of importing the archived internal model runner.
6. Root reading guides, `docs/final_architecture.md`, `reports/README.md`, and
   the final-only `verify_transfer.py` were added or updated.

## Archived source groups

The archive contains 54 moved files in this pass, including:

- Additive Cross-Orbit Teacher and old correction networks;
- old DualOrbitTeacher, SingleOrbitStudent, and R0/R1/R2/R3/R3g code;
- complement-prediction, CGSearch, EAR, and old DA T0/T1/DAStudent code;
- old distillation/corrective losses and protocol runners;
- old handoff/protocol/report material and the legacy package integration helper.

The archive is recoverable from the working tree and Git history. Nothing in
the archive is imported by the final-v2 entry points.

## Preserved research assets

The following were not deleted, regenerated, or numerically edited:

- `experiments/GRSL_complete_paper_package/` — 110 files, approximately 4.11
  MiB;
- `results/tables/` — 7 compact CSV tables;
- local `experiments/**` and `results/**` payloads already present in the
  worktree;
- `checkpoints/` and `legacy_checkpoints/` weights;
- `baselines/` and all `cocd/third_party/landslide_baselines/` source;
- external baseline and historical comparison documentation.

Evidence-presence checks passed for SG0/SG10/CG0/CG10, Normal/Center/Random
offset/Counter shuffle, S0/S1/S2, all five loss-composition settings, and all
four KD-strength settings. `git diff` reported no modified tracked files under
the consolidated paper package or `results/tables/`.

## Verification executed

| Check | Result |
|---|---|
| AST syntax parse for all active final code | PASS |
| `python -B verify_transfer.py` | PASS |
| Final Teacher/Student CPU instantiation and dummy forward | PASS |
| Counter affects Teacher policy context while target value memory is fixed | PASS |
| Student has no counter input; decoder accepts retrieved `R` | PASS |
| Final shared-module retired-symbol scan | PASS |
| External baseline runner import | PASS (`cuda`, batch 16; Boehm/CDNetE/MFEWF seats discovered) |
| Evidence CSV presence/schema audit | PASS |
| `git diff --check` | PASS |
| Training or checkpoint/result rewrite | NOT RUN |

## Final repository tree

```text
README.md
README_WINDOWS.md
GITHUB_CODE_PACKAGE.md
cleanup_audit.md
cleanup_report.md
project_audit.md
docs/
  final_architecture.md
  model_description.md
  experiment_summary.md
  naming_dictionary.md
reports/
  README.md
  final_haiti_dataset_audit.md
  directory_structure.md
  handoff_baseline_coverage_audit.md
  external_baselines_and_kd_cards.md
cocd/
  models/landslide_cocd.py
  windows_main/
    data.py
    models_da_search.py
    models_teacher_target_search_v2.py
    models_student_target_search_v2.py
    teacher_target_search_v2.py
    student_search_kd_v2.py
    evaluate_student_search_kd_v2.py
    run_student_kd_weight_sweep.py
    run_student_kd_strength_sweep.py
    build_grsl_final_package.py
    build_grsl_qualitative_package.py
    build_external_method_qualitative.py
    assemble_grsl_complete_paper_package.py
    retest_dis2_tgd.py
    retest_vanilla_kd.py
    retest_external_tgd.py
  scripts/23_train_external_baselines.py
  scripts/24_external_seat_table.py
  third_party/landslide_baselines/
baselines/
experiments/GRSL_complete_paper_package/
results/tables/
archive/history/
  code/
  docs/
  protocols/
  reports/
```

Runtime data, checkpoints, and local full experiment payloads remain in their
existing locations and are intentionally omitted from this compact tree.
