# GRSL project audit

Audit date: 2026-09-22.  This document is the retention decision for the
Haiti SAR Counter-guided Target Evidence Search study.  It is deliberately
source-oriented: weights, raw rasters, NPZ prediction tensors and verbose
training logs remain local-only and are excluded from Git.

## Keep: final paper path

| Area | Canonical source or artefact | Role |
|---|---|---|
| Data | `cocd/windows_main/data.py` | Fixed asc/desc target views, 5-channel target input, G10/TGD construction. |
| Teacher | `models_teacher_target_search_v2.py`, `teacher_target_search_v2.py` | CG10 counter-guided target-value search Teacher and its four-arm ablation. |
| Student | `models_student_target_search_v2.py`, `student_search_kd_v2.py`, `evaluate_student_search_kd_v2.py` | Target-only searched-evidence Student and S0/S1/S2 evaluation. |
| Search primitives | `models_da_search.py` | `SearchContext`, deformable policy/search and pyramid utilities imported by final v2 models. |
| KD sweeps | `run_student_kd_weight_sweep.py`, `run_student_kd_strength_sweep.py` | Composition and total-strength ablations. |
| Re-evaluation | `retest_dis2_tgd.py`, `retest_vanilla_kd.py`, `retest_external_tgd.py` | Inference-only baseline exports. |
| Paper materials | `experiments/GRSL_complete_paper_package/` | Retained tables, JSON provenance, captions and PDF/PNG figures. |
| External baselines | `cocd/scripts/23_train_external_baselines.py`, `cocd/third_party/landslide_baselines/` | Boehm, CDNetE, MFEWF and DIS2 provenance. |

## Archive: retained local provenance, not the paper implementation

| Area | Why archived |
|---|---|
| `archive/history/code/` | Historical self-owned architectures and protocol runners, preserved byte-for-byte. |
| `archive/history/docs/` and `archive/history/protocols/` | Old handoffs, checkpoint ledgers, migration logs, plans, and protocol JSON. |
| `experiments/windows_main/da_search/grsl_assets_v1/` | Historical raw-feature-bypass package; never cite as final S2. |
| `experiments/windows_main/ear_reallocation*`, `cgsearch/`, old `loss_ablation*` | Exploratory or replaced development branches. |
| `legacy_checkpoints/` | Local reproducibility cache; DIS2 recovery provenance only. |

## Delete: safe completed removals

The tracked smoke/debug scripts listed as deleted by Git (`32`, `36`, `37`,
`39`, `40`, `43`, `44`, `46`, `47`, `48`, `98`) remain deleted and must not be
restored.  They were one-off checks, not paper entry points.  No large local
experiment directory is deleted by this archival pass: every one requires a
separate size-and-reference review before irreversible removal.

## Git publication policy

Publish code, compact configurations, CSV/JSON tables, Markdown documentation
and final PNG/PDF figures.  Do not publish raw data, checkpoints, NPZ tensors,
full logs or cached training samples.  The final package is approximately 4 MB
without those excluded payloads.
