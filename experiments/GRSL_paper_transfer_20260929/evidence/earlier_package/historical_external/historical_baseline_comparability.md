# Baseline comparability

The current primary comparison is SO-reference vs S0 under the locked da-search evaluation. T0 and T1 are dual-orbit references, not single-orbit deployment competitors. Historical DIS2 and Vanilla KD numbers come from the retained local CGSearch CSV and no longer have a retained current checkpoint after the q3/q4 cleanup; they are labeled historical/missing-checkpoint. Boehm, CDNetE and MFEWF have local prediction caches and are retained as completed external/local baselines, but their training/architecture details come from their own local records. Old CGSearch is kept only as a historical index and is not mixed into the current DA-search method claim.

T1 changes DA and target-evidence auxiliary supervision jointly relative to T0; no strict single-factor DA-only or auxiliary-only checkpoint is available. S0 is Teacher-initialized target-only adaptation, not a scratch GT-only model.
