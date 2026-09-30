# Windows execution

The final research contract is [paper_release/README.md](paper_release/README.md), not the archived one-stage or geometry-weighted method. Run from this repository root with the existing Python environment and local dataset/checkpoints:

```powershell
python paper_release/scripts/verify_release.py
python paper_release/scripts/evaluate_final.py --method student --smoke
```

Optional full frozen inference: `python paper_release/scripts/evaluate_final.py --method all`. It writes only ignored `paper_release/local_predictions/` outputs. No retraining is required. Dataset paths default to `data/haiti`, splits to `data/splits_v3`; `COCD_DATASET_ROOT`, `COCD_SPLIT_DIR`, and `COCD_DEVICE` overrides are resolved by `cocd/paths.py`. Override splits only for explicitly historical reproduction, never to substitute final results.

The pre-freeze Windows notes are preserved under `archive/history/reports/pre-freeze-20260930/README_WINDOWS.md`.
