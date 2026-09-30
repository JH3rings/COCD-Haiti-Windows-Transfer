# Reproduction interface

Five channels: pre-event VV/VH, post-event VV/VH and a constant orbit indicator. All observations from one spatial location share a partition. Train-only radiometric statistics normalize both time points and orbits. The loader additionally uses pre/pre and post/post zero-change examples.

Teacher segmentation: binary cross-entropy, averaged over both target directions. Student: five segmentation-only epochs, epoch-five loss calibration, then equal normalized Policy and Evidence coefficients with a cosine multiplier. The frozen source uses seed 42, Adam at 5e-5, batch 16, ceiling 100 epochs and patience 2; checkpoints are selected by test Overall AUPRC. Threshold 0.5 is fixed. These settings are recorded so reproduction does not silently substitute a different selection rule.

Entrypoints: `cocd/windows_main/train_teacher.py`, `train_student.py`, `evaluate.py`, and `smoke.py`. Training requires the local dataset, pinned splits and initialization weights. Evaluation requires a Student checkpoint. Data, checkpoints, comparisons and manuscript assets are local-only. External comparison adapters are excluded from this code repository.
