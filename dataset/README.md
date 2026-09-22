# Haiti SAR dataset protocol

The dataset is not distributed with this repository.  Each location contains
ascending and descending Sentinel-1 SAR stacks at pre-event and post-event
times.  A target deployment view is one orbit only:
`[VV_pre, VH_pre, VV_post, VH_post, orbit_id]`.  The opposite orbit is the
counter view and is available to the Teacher only.

The fixed train/test locations are resolved by `cocd/windows_main/data.py`.
TGD (Target-Orbit Geometric Distortion) is the existing G10 mask: geometry is
positive in the target orbit and absent in the counter orbit.  OmegaTR is the
fixed subset `TGD AND SO-wrong AND CG10-Teacher-correct`.  Neither the counter
image nor its geometry mask is provided to deployed target-only methods.
