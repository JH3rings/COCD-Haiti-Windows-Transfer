# EAR train-only continuation — 2026-09-20

## Protocol actually run

- Original train locations: `1233`.
- Original pinned validation locations merged into train: `137`.
- Effective training set: `1370` locations.
- Validation partition: not used; no validation-based model selection or early stopping.
- Test locations: `343`, not opened during training or readout.
- Seed: `42`.
- Batch size: `16`.
- Requested epochs: `20`; the last requested epoch is the saved model.
- Device: NVIDIA GeForce RTX 4060 Ti, CUDA (`torch 2.6.0+cu126`).

## Saved code and protocol

- `cocd/windows_main/ear_main.py`
- `cocd/windows_main/models_complement.py`
- `cocd/windows_main/tests_ear.py`
- `cocd/configs/protocol_ear.json`

The structural EAR tests passed before training. The generated checkpoints and
raw experiment directory remain local and are excluded by `.gitignore`.

## Frozen lambda calibration

The static calibration used eight real train-only batches and did not read
validation or test labels:

| quantity | recorded value |
|---|---:|
| `Lseg0` | `0.04431732231751084` |
| `LEAR0` | `0.046242188196629286` |
| `lambda_EAR` | `0.04791870201440948` |

## Last-epoch training records

These are training losses from the final requested epoch, not validation
metrics:

| arm | `train_loss` | `Lseg` | `LEAR` |
|---|---:|---:|---:|
| EARTeacher | `0.0334611820690662` | — | — |
| EAR-GT | `0.0242662096295528` | `0.0242662096295528` | `0` |
| EAR-Full | `0.0256551742913643` | `0.0244472881983821` | `0.0252069868799661` |
| EAR-noGeo | `0.0256129287825219` | `0.0244151378958214` | `0.0249963132628323` |

## Train-only readout

The file `train_only_results_seed42.csv` was written from the same 1370
locations. It is an in-sample readout and must not be described as independent
validation or test evidence. At the fixed threshold `0.5`, the recorded
overall rows were:

| method | AUPRC | F1 |
|---|---:|---:|
| EAR-Teacher-self | `0.7357796323819299` | `0.6142961262057892` |
| EAR-Teacher-dual | `0.7582115620050482` | `0.6491658083284754` |
| EAR-GT | `0.8619886712158824` | `0.7596333341245032` |
| EAR-Full | `0.8593544366035126` | `0.753850890484042` |
| EAR-noGeo | `0.859738070869799` | `0.7547958602230875` |

The current DIS2 checkpoint was included only as a labelled contaminated
re-readout in the local CSV; it did not drive EAR selection and should not be
used as a clean baseline.
