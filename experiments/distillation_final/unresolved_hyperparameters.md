# External-method source gaps and fixed Haiti-port choices

## OGP-Net-Port — source gaps and sanity gate failure

The [AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/32743) specifies the two-stage training, DMC, DUR/DIST, GSU, and semi-hard pixel selection, but does not give numeric contrastive temperature or a gamma cutoff for feature exchange. No author code was located. Fixed once at `tau=0.2` (the cited PCD implementation's value, explicitly an adaptation) and `gamma_threshold=1e-3` (fixed near-zero cutoff). The OGP paper's source architecture is DeepLabV3+/EfficientNet-B3; this Haiti port uses the project's ConvNeXt-Tiny/FPN and explicit two-class logits. SGD uses the paper's initial `5e-3`, momentum `0.9`, weight decay 0, and one fixed global gradient norm clip of 1.0 after the unbounded Eq.12 gradient overflowed in the 32-row preflight.

The 32-row preflight completed 100 finite steps and all three KD terms were nonzero and differentiated, but deployed single-orbit IoU/F1 declined from `0.8143/0.8977` to `0.5388/0.7003`. This failed the required sanity gate, so formal OGP Stage-2 training was not started. The existing S0-Raw checkpoint was reused as the frozen Stage-1 target-only teacher; no historical checkpoint was altered. The row in the benchmark table is a failed preflight with no formal result, not a trained model.

## Multi2Uni-Port — mechanism-faithful choices because source material is unavailable

The [publisher record](https://doi.org/10.1016/j.inffus.2026.104696) confirms MEF, low-level DWT-based FAD, and high-level CAD, but the accessible full text/supplement/author-code search did not resolve:

- MEF exact equation, placement, and channel configuration;
- FAD levels, wavelet family, and exact detail-loss definition;
- CAD exact affinity formula and high-level layer;
- published loss coefficients and optimization schedule.

One fixed port was used, with no sweep and no reuse of retired CO-FAD/CO-CAD results: bidirectional gated residual MEF at c2-c5; one-level orthonormal Haar DWT details LH/HL/HH at c2/c3, MSE FAD; global c5 teacher cross-view affinity supervising target-only student's self-affinity with L1 CAD. One initial batch set `lambda_FAD=0.2943639` and `lambda_CAD=0.0325334`, calibrating each weighted auxiliary term to 10% of the segmentation sum. The locked project protocol supplies seed 42, batch 16, Adam `5e-5`, weight decay 0, max 100 epochs, patience 2, and the existing test Overall AUPRC monitor.

Accordingly, Multi2Uni-Port is reported as a mechanism-faithful Haiti port, not an official hyperparameter reproduction. Its sanity test passed and its formal best checkpoint is recorded in `multi2uni_port/summary.json`.

## Existing DIS2-Port adaptation note

The checked-in official `DLKD_ver4.py` duplicates the accumulated class-0 segmentation maps when appending class-1 maps in three decoder stages. The Haiti port makes the narrow correction to append the current class-1 map, preserving the published two-class classwise decoder intent. The upstream snapshot is left untouched. This source correction is recorded so the run is not misrepresented as byte-identical official execution.
