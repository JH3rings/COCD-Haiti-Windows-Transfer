# README — Windows final repository

This checkout contains the final GRSL method and its retained paper evidence.
Start with [`README.md`](README.md), then
[`docs/final_architecture.md`](docs/final_architecture.md).

## Final method

```text
Counter-Guided Target Evidence Search Teacher
    + Target-Only Search Student
    + Policy Distillation
    + Shared-Memory Evidence Distillation
```

The canonical code is:

```text
cocd/windows_main/data.py
cocd/windows_main/models_teacher_target_search_v2.py
cocd/windows_main/models_student_target_search_v2.py
cocd/windows_main/teacher_target_search_v2.py
cocd/windows_main/student_search_kd_v2.py
cocd/windows_main/evaluate_student_search_kd_v2.py
cocd/windows_main/run_student_kd_weight_sweep.py
cocd/windows_main/run_student_kd_strength_sweep.py
```

`archive/history/` is provenance only. Its code is not part of the final
method and is not imported by the active entry points.

## Environment

Create a Windows environment with the CUDA build appropriate for the installed
driver, then install the repository requirements:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for package and platform notes. Set
`COCD_DATASET_ROOT` or `COCD_OUT_ROOT` when data or outputs live elsewhere.

## Read-only verification

From the repository root, run:

```powershell
python verify_transfer.py
```

The verifier checks the active split, one loader batch, final Teacher/Student
instantiation, dummy forwards, policy/value separation, decoder dependence on
retrieved evidence `R`, and final KD imports. It does not train or rewrite
checkpoints/results. A successful run ends with:

```text
TRANSFER CHECK PASSED
```

## Results and baselines

All experiment data is collected under `experiments/`. Start with
`experiments/README.md`; it links the current paper package, master run catalog,
unified metrics, comparison charts, full figure inventory, dataset audit,
checkpoints, and retained historical runs. The current paper tables are under
`experiments/GRSL_complete_paper_package/04_tables/`; qualitative and method
figures are under `02_qualitative/` and `01_method/`. External comparison
source code and provenance remain under `baselines/` and
`cocd/third_party/landslide_baselines/`.
