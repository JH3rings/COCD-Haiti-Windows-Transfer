# Experiment summary

## 1. Teacher counter-guidance ablation

**Purpose:** establish that counter-guided policy and TGD supervision improve
target-side evidence retrieval.  **Models:** SG0, SG10, CG0, CG10.
**Conclusion:** CG10 is the final Teacher (Overall AUPRC 0.9012; TGD AUPRC
0.9158, IoU 0.6884).  Source:
`experiments/GRSL_complete_paper_package/quantitative/csv/table_teacher_ablation.csv`.

## 2. Search mechanism interventions

**Purpose:** test whether policy positions and counter context matter.
**Conditions:** normal, centered offsets, random offsets, counter shuffle.
Random and counter-shuffle degrade CG10, demonstrating non-decorative search
and counter dependence. Source: `table_search_mechanism.csv`.

## 3. Student knowledge transfer

**Purpose:** separate no KD (S0), policy-only KD (S1) and policy+evidence KD
(S2). **Conclusion:** S1 aligns policy but has limited task gain; S2 reduces
evidence distance and improves S2-S1 TGD IoU by 0.0225 and OmegaTR F1 by
0.1080. Sources: `table_student_distillation.csv` and
`table_student_mechanism.csv`.

## 4. Loss ablations

**Composition:** P100E0, P50E50, P25E75, P0E100 at fixed initial KD/Seg=10%.
**Strength:** 0%, 5%, 10%, 20% with alpha=0.5.  P50E50 and 10% are the final
setting. Sources: `table_loss_composition.csv`, `table_loss_strength.csv`.

## 5. External comparison and robustness

The curated package retains SO, Boehm SAR U-Net++, CDNetE, MFEWF, DIS2-port,
Vanilla KD and S2 visual comparisons.  DIS2-port and Vanilla KD are
test-monitored development evidence and are explicitly separated from final-v2
primary claims.  Sources: `quantitative/csv/` and `qualitative/figures/`.
