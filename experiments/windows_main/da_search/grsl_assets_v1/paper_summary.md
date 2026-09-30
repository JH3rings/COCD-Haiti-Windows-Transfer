# Haiti SAR GRSL material summary

## Main locked evidence

S0 is the only active target-only Student. Relative to the SO-reference, S0 improves Overall AUPRC by 1.6906 pp, IoU by 2.1677 pp and F1 by 1.6035 pp. In TGD the gains are 1.5286, 1.9364 and 1.4162 pp; in Remaining they are 1.7301, 2.2220 and 1.6481 pp. The TGD gain is therefore not larger than the Remaining gain.

T1 with the correct counter reaches TGD AUPRC 0.920059. Shuffled counter drops TGD AUPRC to 0.487625 and the target-FPN bypass to 0.887063. These are analysis interventions, not retrained target-only models.

The fixed Union diagnostic contains 26,019 TGD pixels (2.503% of TGD). By definition SO is wrong on the subset; S0 is correct on 47.6613% of those pixels. This is a fixed diagnostic correction fraction, not a recovery rate for all TGD pixels or all landslide pixels. Single-DA scratch is unavailable.

## Boundaries

The geometry-aware “sample outside distorted region” statistic is skipped because no invertible mask-to-feature coordinate contract is declared. DA behavior is reported only in feature-coordinate units. Search KD/S1 was not trained. The test partition selected checkpoints, so all locked values are development-monitor evidence. No claim is made that S0 recovered physically unavailable raw counter observations.
