# GRSL final experiment summary

The final Teacher is CG10 and the final deployable Student is S2 (policy plus evidence KD, alpha=0.5, initial KD/Seg=10%).

- **Teacher:** CG10 yields the strongest four-arm Teacher performance, and counter shuffle/random offset interventions degrade TGD performance.
- **Search:** random locations and incorrect counter context degrade TGD IoU, supporting learned, counter-dependent policy use.
- **Student:** S1 aligns policy JS but has limited task gain; S2 reduces shared-memory evidence distance and improves prediction, with the strongest gains on OmegaTR.
- **Robustness:** S2 improves TGD IoU/F1 over S1; the AUPRC TGD gain is similar to Remaining, so claims should emphasize IoU/F1 and OmegaTR rather than assert an AUPRC-only TGD concentration.
- **Loss:** P50E50 is the best completed composition setting; 10% is the best completed total strength.

All results are test-monitored selection evidence, not an independent held-out blind test.
