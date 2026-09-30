# Captions

**Figure 1.** Locked dual-orbit Teacher and target-only Student pipeline. Target and counter are encoded by shared ConvNeXt-Tiny/FPN weights; T1 policy context uses target and dual fused features, while deformable sampling reads target memory only. S0 receives a one-time parameter transfer from T1 and is adapted with target and GT only.

**Figure 2.** Qualitative test examples with a common crop and scale. Yellow contours mark TGD. The SAR panel is the displayed mean absolute pre/post difference over the two input bands; it is a visualization, not a calibrated physical unit.

**Figure 3.** TGD mechanism evidence from locked predictions. (a) S0 minus SO in percentage points. (b) T1 correct, Target-FPN bypass and shuffled-counter interventions. (c) S0 correctness inside the fixed SO-wrong/Teacher-correct Union subset.

**Figure 4.** Candidate DA behavior summary in common feature-coordinate units; whiskers show p10–p90. No physical ground displacement or geometry-aware outside-region claim is made.
