# Reproduction entrypoints

Run from the repository root with the dataset path configured as described in
`ENVIRONMENT.md`.

```powershell
python cocd/windows_main/teacher_target_search_v2.py --stage all
python cocd/windows_main/student_search_kd_v2.py --stage all
python cocd/windows_main/run_student_kd_weight_sweep.py
python cocd/windows_main/run_student_kd_strength_sweep.py
```

The commands above are the canonical runners.  No wrapper is provided because
the recorded scripts resolve paths and checkpoint names relative to their
current locations.
