# Final architecture

This file is the compact implementation contract for the GRSL final
repository. It describes the locked final-v2 method only; earlier architectures
and their source are retained under `archive/history/` for provenance.

## Input

Each orbit is a five-channel tensor:

```text
[VV_pre, VH_pre, VV_post, VH_post, OrbitID]
```

The Teacher receives a target orbit and a counter orbit during training. The
deployed Student receives only the target orbit.

## Teacher

```text
Target + Counter
    -> target/counter policy context
    -> counter-guided search policy
    -> target-value-only deformable search
    -> retrieved target evidence R
    -> decoder
```

Counter features are allowed to affect the policy context only. Search values
come from the target memory pyramid. The decoder is reachable through the
retrieved representation `R`, not through a raw target/counter or dual-fusion
skip path.

Canonical implementation:
`cocd/windows_main/models_teacher_target_search_v2.py`.

## Student

```text
Target only
    -> student policy
    -> target-memory search
    -> retrieved target evidence R
    -> decoder
```

The Student has no counter input at inference or training-time forward. S0,
S1, and S2 are training/evidence settings of the same
`StudentTargetSearchV2` implementation, not separate network definitions.

Canonical implementation:
`cocd/windows_main/models_student_target_search_v2.py`.

## Distillation

```text
Policy Distillation
    + Shared-Memory Evidence Distillation
```

Policy KD matches Teacher and Student search distributions. Evidence KD matches
the evidence retrieved from the same Student target memory under the Teacher
and Student policies. The training and evaluation entry point is
`cocd/windows_main/student_search_kd_v2.py`.

## Dependency boundary

The final models use `ConvNeXtTinyFPN` from
`cocd/models/landslide_cocd.py` and the retained search primitives from
`cocd/windows_main/models_da_search.py`: `CHANNELS`, `LEVELS`, `POINTS`,
`pyramid`, `pyramid_from_encoded`, `SearchContext`, and
`DeformableSearch`. The ordinary `TargetOnlyBaseline` helper remains for the
paper comparison and qualitative package builders.
