# DLR optical-guided SAR retrieval study

## Scientific question

Can optical PRE/POST observations teach a SAR-only U-Net to select and combine
evidence already encoded in the same-orbit PRE1/PRE2/PRE3/POST1 SAR inputs?

All spatial skips and all retrieved values come from SAR. Optical conditions
the Teacher policy only. Every SAR encoder and decoder parameter remains
trainable. A fixed, previously completed B1 reference is reused.

## Audit of existing code

| Item | Verified existing implementation | This experiment |
| --- | --- | --- |
| Original policy transfer | cocd/windows_main/train_student.py spatial_policy splats all levels onto P2 and sums level identities | M0 uses an equivalent bounded sparse splat, regression-checked against the dense original |
| Original evidence transfer | kd_terms reads a detached Student value bank using Teacher and Student policies | Retained in M0–M4. Teacher rule and target result are detached. KD updates the Student context, policy and optional channel controller; segmentation updates the encoder and decoder |
| Learned retrieval | models_da_search.py uses bounded offsets, bilinear grid_sample and jointly normalized scale/point weights | Reused directly, three memory scales and four samples per scale |
| Point symmetry | Original three learned offset outputs start identically at zero | Three distinct small initial directions remove exact initialization symmetry. The center remains fixed. No diversity loss is added |
| Original decoder | Original manuscript decodes retrieved evidence without raw skips | New task explicitly permits SAR-only U-Net skips; the retrieval residual is injected at stride 4 |
| Old DLR SAR inputs | Historical Pico work could read pre optical in the Student | New Student has exactly eight SAR channels |
| Old DLR U-Net search | optical_search_swap_20261105 freezes SAR encoder and uses local single-level rules | New encoder, searcher and decoder all train; memory combines strides 4, 8 and 16 |
| Old fusion teacher | train_fusion_then_output.py adds optical-conditioned feature residual at stride 4 | Preserved as history. The new optical branch cannot add features directly to the decoder |
| Old frequency and channel work | Multiscale decoder interventions, spectrum/modulation/output losses together | M3 changes only the SAR policy context; M4 adds SAR channel selection. Neither aligns optical and SAR frequency coefficients |
| Earlier mechanism gate | Historical optical_search_swap status is SWAP_GATE_NOT_PASSED | Treated as negative evidence, not a successful prior model |

## Architecture

SAR encoder widths 64/128/256/512/1024 and the four-stage decoder are the
existing DLR paper-style U-Net. Each encoder scale sees the joint eight-channel
SAR input. Projections map stride 4/8/16 features to 128 channels. Teacher and
Student use the original deformable sampler. Optical PRE and POST share a
light optical encoder. Their before, after, difference and absolute difference
features condition the Teacher query. Optical features supply no sampled values.

The decoder receives SAR skips plus a residual projection of retrieved SAR
evidence at stride 4. This is explicitly a U-Net adaptation of the original
COCD evidence path, not a replica of the original skip-free decoder.

## Controlled variants

| Method | Sole change relative to M0 | Matched no-KD control |
| --- | --- | --- |
| M0 | Original merged spatial Policy JS and shared-memory cosine | M0 SAR search |
| M1 | Add learned projections of mean-PRE, POST, signed and absolute SAR differences to each memory scale | M1 SAR search |
| M2 | Preserve scale identity in the bilinear-splatted policy distribution | Reuse M0 architecture and Teacher |
| M3 | Fixed one-level Haar LL/LH/HL/HH of SAR POST minus mean-PRE condition the policy query | M3 SAR search |
| M4 | ECA-style channel selector uses policy context to generate bounded multiplicative gains on retrieved SAR evidence | M4 SAR search |

M1 is a lightweight explicit input-change residual into existing joint SAR
memory, not a new temporal encoder. M3 retains all four bands and changes
query generation only. No band is declared noise or a true landslide boundary.
M4 evidence KD also supervises the effect of the multiplicative gains.

For all Students:
L = BCE + foreground Dice + 0.05 JS + 0.05 cosine.
The first five epochs use segmentation alone. The same scheduled validation
epochs (5,10,...,100) are eligible for every model. If a Student selects a
segmentation-only warmup checkpoint, that selection cannot establish KD benefit.

## Training and evaluation lock

- Official 355 train / 88 validation / 120 test positions; independent44 is a
  subset of all120. Existing overlap limitations are retained.
- SAR PRE1/PRE2/PRE3/POST1 VV/VH, eight channels. Data fields and observation
  dates retain their original audit limitations; POST1 is not a certified
  satellite-first acquisition and relative track membership is not fully
  recorded in the release.
- Locked train-only SAR and optical normalization, synchronized spatial
  augmentation, identical numerical-validity masks and threshold 0.5.
- Random seed42 common initialization; baseline and KD Student have exactly
  matching state-dict shapes. All SAR backbone layers train.
- 100 epochs per arm, batch16, validation every5; AdamW1e-4, wd.01 excluding
  norm/bias, clip1, first50 constant and last50 cosine to1%.
- Original B1 frozen 3+1 reference is reused; new search controls are necessary
  capacity-matched controls authorized by the present task.

Mechanism A fixes Policy while changing optical input and verifies exact
invariance of SAR memory, skips and predictions. Mechanism B perturbs offsets
and verifies actual sampled evidence changes, and checks finite task/KD
gradients. The sparse M0 JS is tested against canonical dense COCD JS. M2
is additionally tested to distinguish scale-mass changes that M0 merges.

Mechanism C trains M0 SAR control and M0 Teacher, freezes both, and evaluates
both policies on each of the same fixed host memories and decoders. To pass,
Teacher native validation IoU must exceed the control, the mean rule-swap gain
across the two hosts must be positive, and one host must gain >=0.2 percentage
points. These fixed host comparisons isolate policy changes on those hosts;
cross-model policy application still has a host-distribution limitation.

If M0 fails this prerequisite, Student variants are NOT_RUN and the queue
reports TEACHER_SEARCH_GATE_NOT_PASSED. No test images are opened to rescue
a failed validation prerequisite. M1/M3/M4 use corresponding Teacher checks
before their Student runs.

Validation gains >=0.2pp over each matched control trigger one seed43 paired
repeat. This is a screening rule, not statistical significance. All eligible
models and repeats freeze before test. The final report includes negative
results, per-event scores, small-component recall and corrected/added FP/FN,
fixed-Student-memory policy swaps, scale weights, parameter counts and
convolution MAC estimates. GPU timing is omitted while unrelated work is active.

## Mechanism and novelty sources

- Original COCD: manuscript/main.tex, equations for retrieval, policy JS and
  shared Student-memory evidence cosine; and the retained source files above.
- Zhu et al., Deformable DETR (ICLR2021),
  https://arxiv.org/abs/2010.04159 .
- Wang et al., ECA-Net (CVPR2020),
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html .
- Wolter and Garcke, Adaptive Wavelet Pooling (AISTATS2021),
  https://proceedings.mlr.press/v130/wolter21a.html .

Deformable retrieval, cosine KD, JS KD, Haar bands and channel attention have
published precedents. A possible contribution would require measured evidence
that optical-conditioned SAR-change selection transfers to a SAR-only Student
beyond a matched no-KD network. The current code and passed structural checks
alone establish no performance or novelty claim.

## Execution

Run python run.py from this directory. The queue resumes completed arms with
checkpoint-hash verification. Logs and status are written beneath results/.
Source files and compact non-image experiment records are suitable for the
private repository. Dataset tensors, checkpoints, prediction arrays and raw
run logs stay local.
