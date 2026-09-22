# Naming dictionary

| Name | Meaning |
|---|---|
| SO | Target-only no-search single-orbit baseline; letter O. |
| SG0 / SG10 | Target-self-policy Teacher without / with `gamma=0.10` TGD supervision. |
| CG0 / CG10 | Counter-guided-policy Teacher without / with TGD supervision; CG10 is final Teacher. |
| S0 | Target-only search Student with no KD. |
| S1 | Student with policy KD only. |
| S2 | Final Student with policy + evidence KD. |
| TGD | Target-Orbit Geometric Distortion region; legacy G10. |
| OmegaTR | Fixed Teacher-resolvable TGD subset: SO wrong, CG10 correct. |
| DIS2-port | Retained adapted DIS2 baseline re-inferred from archived checkpoint. |
| Vanilla KD / VKD | R1 target-only baseline with ordinary full-pixel logit KD. |
