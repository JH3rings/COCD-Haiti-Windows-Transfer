# Final Teacher source map

The executable final Teacher is intentionally retained at
`cocd/windows_main/models_teacher_target_search_v2.py`, because its imports
are used by the recorded experiment runners.  `counter_guided_teacher.py`,
`search_policy.py` and `deformable_search.py` are not duplicated here: their
canonical implementations are `TargetSearchTeacherV2`,
`AnchoredDeformablePolicy`, and `DeformableSearch` in that source tree.

This map avoids a cosmetic file move that would change import resolution and
potentially invalidate the reproducibility path.
