# GRSL final-repository cleanup audit

Audit date: 2026-09-22  
Working branch: `cleanup/grsl-final`  
Rollback branch: `archive/pre-grsl-final-cleanup` at committed `HEAD` `3664fb2`  
Audit status: **inventory completed; approved archive/refactor pass executed**.
The pre-mutation inventory was read-only; the execution outcome is recorded in
`cleanup_report.md`.

This audit separates the executable final-v2 method from historical self-owned
architectures and from external comparison methods. The action boundary and
execution outcome are recorded below and in `cleanup_report.md`.

## 1. Non-negotiable retention rules

- The final method is Counter-Guided Target Evidence Search Teacher plus
  Target-Only Search Student, Policy Distillation, and Shared-Memory Evidence
  Distillation.
- `README.md`, `docs/model_description.md`, and
  `docs/experiment_summary.md` are the authoritative scientific description.
- Existing experiment values are immutable. No training or result regeneration
  is part of this cleanup.
- `experiments/GRSL_complete_paper_package/` and all of its quantitative,
  qualitative, and historical-external evidence are **KEEP**.
- `baselines/`, `cocd/third_party/`, external-baseline runners, compact tables,
  prediction figures, and source/provenance notes are **KEEP** unless a later
  audit proves a particular file is unrelated.
- Current uncommitted files were present before this audit and are treated as
  user-owned assets. They are not overwritten by the cleanup.

## 2. Final dependency graph

The static local import graph for the final entry points is:

```text
data.py
  -> paths.py

models_teacher_target_search_v2.py
  -> models_da_search.py
       CHANNELS, LEVELS, POINTS
       pyramid, pyramid_from_encoded
       SearchContext, DeformableSearch
  -> models/landslide_cocd.py
       ConvNeXtTinyFPN

models_student_target_search_v2.py
  -> models_da_search.py
       CHANNELS, LEVELS, POINTS
       pyramid, SearchContext, DeformableSearch
  -> models_teacher_target_search_v2.py
       AnchoredDeformablePolicy
  -> models/landslide_cocd.py
       ConvNeXtTinyFPN

teacher_target_search_v2.py
  -> data.py
  -> models_da_search.py (LEVELS, POINTS)
  -> models_teacher_target_search_v2.py

student_search_kd_v2.py
  -> data.py
  -> models_teacher_target_search_v2.py
  -> models_student_target_search_v2.py

evaluate_student_search_kd_v2.py
  -> data.py
  -> models_da_search.py (TargetOnlyBaseline)
  -> models_teacher_target_search_v2.py
  -> models_student_target_search_v2.py
  -> student_search_kd_v2.py (KD/evaluation helpers)

run_student_kd_weight_sweep.py / run_student_kd_strength_sweep.py
  -> student_search_kd_v2.py
```

The paper-package builders additionally use `TargetOnlyBaseline` as the
ordinary target-only reference. This is why that small baseline helper remains
part of the final support surface even though it is not the Student model.

## 3. Shared-module dependency audit

| File | Current role | Final dependency | Proposed action | Reason |
|---|---|---|---|---|
| `cocd/windows_main/models_da_search.py` | Mixed search primitives and the older T0/T1/DAStudent network family | `CHANNELS`, `LEVELS`, `POINTS`, `pyramid`, `pyramid_from_encoded`, `SearchContext`, `DeformableSearch`; `TargetOnlyBaseline` is used by final evaluation/package builders | **REFACTOR** | Keep final primitives and the ordinary baseline helper; remove old `FusionPyramid`, `DeformablePolicy`, `SearchWrite`, `NormalDualTeacher`, `DASearchTeacher`, `DAStudent`, and their initialization helpers after all old callers are archived. Tensor behavior of retained primitives must remain unchanged. |
| `cocd/models/landslide_cocd.py` | Shared ConvNeXt-Tiny/FPN plus old correction, dual-teacher, and single-student classes | `ConvNeXtTinyFPN` | **REFACTOR** | Keep only the backbone/FPN implementation used by final Teacher/Student and external final package paths; archive old `TeacherCorrection`, `StudentCorrection`, `DualOrbitTeacher`, and `SingleOrbitStudent` callers first. |

## 4. File-level proposed actions

### KEEP: final method and paper assembly

| File or directory | Current role / final dependency | Action |
|---|---|---|
| `README.md` | Primary final-method entry point | KEEP; update first section to the explicit FINAL METHOD wording |
| `docs/model_description.md` | Final Teacher, Student, and KD semantics | KEEP |
| `docs/experiment_summary.md` | Locked ablations and evidence map | KEEP |
| `docs/naming_dictionary.md` | Final names SG0/SG10/CG0/CG10 and S0/S1/S2 | KEEP |
| `docs/final_architecture.md` | Required compact input/model/KD contract | ADD and KEEP |
| `cocd/windows_main/data.py` | Final dataset loader and split contract | KEEP |
| `cocd/windows_main/models_teacher_target_search_v2.py` | `TargetSearchTeacherV2`, counter-guided policy, target-only value memory | KEEP |
| `cocd/windows_main/models_student_target_search_v2.py` | `StudentTargetSearchV2`, target-only search | KEEP |
| `cocd/windows_main/teacher_target_search_v2.py` | SG/CG Teacher training and ablations | KEEP |
| `cocd/windows_main/student_search_kd_v2.py` | S0/S1/S2 training and policy/evidence KD | KEEP |
| `cocd/windows_main/evaluate_student_search_kd_v2.py` | Final Student evaluation | KEEP |
| `cocd/windows_main/run_student_kd_weight_sweep.py` | KD composition sweep | KEEP |
| `cocd/windows_main/run_student_kd_strength_sweep.py` | KD strength sweep | KEEP |
| `cocd/windows_main/build_grsl_final_package.py` | Compact quantitative paper package builder | KEEP |
| `cocd/windows_main/build_grsl_qualitative_package.py` | Final-v2 qualitative builder | KEEP |
| `cocd/windows_main/build_external_method_qualitative.py` | Final-v2 plus external qualitative comparison | KEEP |
| `cocd/windows_main/assemble_grsl_complete_paper_package.py` | Non-destructive paper-package assembly | KEEP |
| `cocd/windows_main/retest_dis2_tgd.py` | Inference-only DIS2 external comparison | KEEP |
| `cocd/windows_main/retest_vanilla_kd.py` | Inference-only Vanilla KD comparison | KEEP |
| `cocd/windows_main/retest_external_tgd.py` | External result re-evaluation helper | KEEP |
| `experiments/GRSL_complete_paper_package/` | Quantitative CSV/JSON/TEX, qualitative figures/crops/overlays, historical comparison | KEEP byte-for-byte; never regenerate during cleanup |
| `results/tables/` and local `results/**` | Existing compact and local result evidence | KEEP; values are immutable |
| `baselines/` | Baseline provenance notes | KEEP |
| `cocd/external_mfewf.py` | MFEWF external baseline implementation | KEEP |
| `cocd/third_party/landslide_baselines/` | Boehm, CDNetE, MFEWF, DIS2 and related source | KEEP, including untracked source currently present |
| `cocd/scripts/23_train_external_baselines.py` | External baseline training source | KEEP; remove its unnecessary dependency on archived internal model code only if a smoke/import check requires it |
| `cocd/scripts/24_external_seat_table.py` | External baseline result table assembly | KEEP |
| `cocd/configs/protocol_windows_main.json` | External baseline protocol provenance | KEEP |

### ARCHIVE: superseded self-owned architecture and its support code

The following files are historical code or historical method-specific runners.
They should be moved under `archive/history/code/` (preserving relative paths
and adding an archive README), not silently deleted:

```text
cocd/models/landslide_cocd_v2.py
cocd/losses/landslide_cocd.py
cocd/losses/distill.py
cocd/windows_main/models_complement.py
cocd/windows_main/models_cgsearch.py
cocd/windows_main/main.py
cocd/windows_main/ear_main.py
cocd/windows_main/cgsearch_main.py
cocd/windows_main/da_search_main.py
cocd/windows_main/g10_mechanism_analysis.py
cocd/windows_main/teacher_search_ablation.py
cocd/windows_main/tests_complement.py
cocd/windows_main/tests_ear.py
cocd/windows_main/tests_cgsearch.py
cocd/scripts/21_train_rapid_landslide_cocd.py
cocd/scripts/23_train_ours_v2.py
cocd/scripts/45_stage1_region_analysis.py
cocd/scripts/97_run_v4_chain.py
cocd/scripts/99_smoke_early_stop.py
cocd/_removed_pre_v3/50_protocol_val_report.py
cocd/_removed_pre_v3/run_protocol_v2.py
windows_ops/diag_complement.py
windows_ops/phase3_data_split_audit.py
windows_ops/phase4_legacy_smoke.py
```

These files contain the Additive Cross-Orbit Teacher, old DualOrbitTeacher and
SingleOrbitStudent, complement prediction, CGSearch/EAR, R0-R3/R3g, old DA
T0/T1/DAStudent, and old corrective/distillation losses. Their checkpoints and
results remain retained as historical evidence; the code is not part of final
v2 execution.

### ARCHIVE: historical documents and old protocols

The following are provenance, handoff, or superseded-plan documents that should
be moved under `archive/history/docs/` or `archive/history/protocols/` with no
content edits:

```text
HANDOFF_FOR_WINDOWS_AI.md
CHECKPOINTS.md
TRANSFER_AUDIT.md
WINDOWS_CONTINUATION_LOG.md
COCD_Evidence_Reallocation_Implementation_Prompt.md
DA_Search_Experiment_Summary.md
progress.html
cocd/configs/protocol_v2.json
cocd/configs/protocol_da_search.json
cocd/configs/protocol_cgsearch.json
cocd/configs/protocol_ear.json
cocd/scripts/14_balanced_spatial_split.py
docs/FROZEN_EXPERIMENT_PROTOCOL_v3.md
docs/FROZEN_EXPERIMENT_PROTOCOL_v4.md
```

Reports that describe retired architectures were moved unchanged to
`archive/history/reports/`; the remaining `reports/README.md` points readers to
the final method and retained external/dataset provenance.

`GITHUB_CODE_PACKAGE.md` and `README_WINDOWS.md` should remain at the root only
after their reading order is updated to point to the final-v2 documents rather
than the archived handoff.

## 5. Evidence retention audit

The following evidence classes are explicitly **KEEP** and must not be edited,
renamed in a way that breaks references, or regenerated:

- Teacher ablations: SG0, SG10, CG0, CG10.
- Search interventions: Normal, Center, Random offset, Counter shuffle.
- Student ablations: S0, S1, S2.
- KD composition: policy-only, evidence-only, 25/75, and 50/50.
- KD strength: 0%, 5%, 10%, and 20%.
- External baselines: Boehm, CDNetE, MFEWF, DIS2-port, and Vanilla KD.
- Historical baseline comparisons, quantitative CSV/JSON/TEX, qualitative
  figures, prediction figures, captions, and experiment summaries.

No numeric field has been changed by this audit.

## 6. Pre-cleanup verification already executed

These checks were read-only and did not write experiment outputs:

- AST syntax parsing passed for the final data, shared, Teacher, Student, KD,
  evaluation, sweep, and backbone files.
- Final module imports passed for `TargetSearchTeacherV2`,
  `StudentTargetSearchV2`, and `models_da_search`.
- CPU dummy forward passed for both final models with output shape `(1, 1, 128,
  128)` and retrieved evidence shape `(1, 128, 32, 32)`.
- The Teacher output exposed target/counter tensors only as policy-side context
  and exposed retrieved `R`; the Student forward accepted only target input and
  exposed retrieved `R`.

## 7. Post-cleanup verification executed

The following checks were executed after the archive/refactor pass; detailed
results are in `cleanup_report.md`:

1. AST syntax parsing and imports passed for the active final code.
2. Final Teacher and Student CPU dummy forward passed.
3. Counter policy-context, target-value memory, decoder-`R`, and Student
   no-counter checks passed.
4. `verify_transfer.py` passed without importing archived legacy classes.
5. External baseline import and evidence CSV presence/schema checks passed.
6. No training, checkpoint rewrite, or result regeneration was performed.

## 8. Execution boundary

The user approved the archive/refactor pass. Archive moves and shared-module
edits were performed on `cleanup/grsl-final`; no training, checkpoint rewrite,
result regeneration, or numeric result edit was performed. The backup branch
captures committed `HEAD`; the pre-existing uncommitted worktree files remain
outside that branch, while moved legacy files are preserved under
`archive/history/`.
