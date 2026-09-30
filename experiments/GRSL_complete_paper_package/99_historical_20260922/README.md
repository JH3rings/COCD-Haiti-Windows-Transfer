# GRSL complete paper package

This is the single entry point for the completed Haiti SAR GRSL materials.

- `quantitative/`: final-v2 Teacher/Student tables, loss ablations, figure data, source hashes, and final-protocol checklist.
- `qualitative/`: reproducible Overall, TGD-local, OmegaTR, and external-method comparison figures, crops, overlays, captions, and scene/crop metrics.  The external panels include Boehm SAR U-Net++, CDNetE, MFEWF, recovered DIS2-port, recovered Vanilla KD, SO, and final S2.
- `historical_external/`: retained external-method metric records with explicit comparability status. These rows must not be pooled into the final-v2 main ranking.

Use `quantitative/paper_summary.md` and `qualitative/qualitative_summary.md` when drafting. This consolidated directory is the retained, self-contained paper package.

DIS2-port was recovered from its archived checkpoint and re-inferred without training on the locked test set. Vanilla KD was retrained from the recorded VKD protocol, then exported by a separate inference-only pass. Both runs are test-monitored development evidence and are kept distinct from the final-v2 primary claim.
