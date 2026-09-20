# COCD Haiti code package

This repository snapshot is intentionally code-only. Raw Haiti imagery,
processed samples, checkpoints, optimizer states, predictions, full experiment
directories, and large generated result folders are excluded by `.gitignore`.

## Recommended reading order

1. `README_WINDOWS.md` — environment and Windows entry points.
2. `HANDOFF_FOR_WINDOWS_AI.md` — original project handoff and scientific scope.
3. `cocd/configs/protocol_windows_main.json` — current frozen protocol and its
   disclosed test-monitoring risk.
4. `cocd/windows_main/models_complement.py` — New COCD Teacher and Student.
5. `cocd/windows_main/main.py` — training, distillation, evaluation, and
   checkpoint-selection paths.
6. `cocd/losses/distill.py` — Vanilla KD, DIS2-port, and selective R3 losses.
7. `cocd/windows_main/tests_complement.py` — architecture/protocol assertions.
8. `reports/counter_orbit_transfer_audit_20260920.md` — latest evidence audit
   and the boundary between verified results and proposed effect distillation.

## Important evidence boundary

The source code and small text/CSV/JSON documentation in this repository can be
inspected directly. Trained weights and raw data are not included, so numerical
claims must be treated as recorded experiment evidence rather than independently
recomputed results unless the user supplies the corresponding private artifacts.

The current task is not to add more modules or multiple losses. The unresolved
scientific question is whether a single-orbit Student can learn any predictable
part of the counter-orbit Teacher's decision effect. Raw `C_T` matching is a
negative result; the only admissible proposed follow-up is a single
decision-effect auxiliary loss, subject to a clean matched protocol.
