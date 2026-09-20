# FROZEN EXPERIMENT PROTOCOL v4 — Haiti SAR Landslide / COCD

> **Status: AUTHORITATIVE.** From this point on, every formal experiment in the Haiti SAR
> Landslide / COCD project must obey this document. Unless the user explicitly revises the
> protocol, do not change the training strategy, the data split, the model-selection rule,
> or the evaluation rule on your own initiative.
>
> Frozen: 2026-09-19 (Windows continuation)
v4 revision: 2026-09-19 — the hard 20-epoch cap is replaced by patience-2 early stopping.

## Core objective

> All methods are retrained under the **same data, the same training budget, the same
> optimizer, and the same evaluation protocol**. Between methods, the only things allowed
> to differ are the network structure and the method-specific loss / distillation mechanism
> that the method's own definition requires.
>
> **Do not distinguish any more between a "unified protocol for internal methods" and an
> "exempt protocol for external baselines."** There is one protocol.

This document supersedes the split / selection / threshold rules of
`cocd/configs/protocol_windows_main.json` (which relied on a 137-location validation set)
and the external-baseline exemption recorded in `protocol_v2.json` §`external_baselines`
and `HANDOFF_FOR_WINDOWS_AI.md` item 8.

---

# 1. Data: Train / Test only. Validation is abolished.

The validation set is **permanently cancelled**.

Original spatial partition:

- train = 1233 locations
- validation = 137 locations
- test = 343 locations

From now on:

- **Train = original train + original validation = 1370 spatial locations**
- **Test = original test = 343 spatial locations**

Must hold:

- the original 343 test locations are **completely unchanged**;
- all of ASC / DESC / pre / post for one spatial location always move together;
- the test split is **not** re-randomised;
- validation is **not** carved out of test;
- the data is **not** re-split for any method.

Generate and freeze a new train/test manifest, and record its **SHA256**.

---

# 2. Uniform construction of training samples

Every single-orbit method uses the **identical** training-data construction.

For each train location, 2 orbits × 3 temporal modes:

### Mode 0
`pre → post`

GT = landslide mask

### Mode 1
`pre → pre`

GT = all-zero mask

### Mode 2
`post → post`

GT = all-zero mask

Hence each train location yields

`2 × 3 = 6`

single-orbit training cases.

Train locations = 1370, so each epoch there are

`1370 × 6 = 8220`

single-orbit training cases.

**All methods must use these 8220 cases.** External baselines are **no longer** permitted to
train on real `pre→post` only.

---

# 3. Test evaluates the real task only

Test locations = 343.

Only these are evaluated:

- ASC `pre→post`
- DESC `pre→post`

Hence

`343 × 2 = 686`

single-orbit test cases.

Test does **not** generate:

- `pre→pre`
- `post→post`

---

# 4. All single-orbit methods have identical deployment privilege

At deployment a formal single-orbit method may use only

`[VV_pre, VH_pre, VV_post, VH_post, OrbitID]`

Forbidden:

- counter orbit
- geometry mask
- DEM
- optical
- test-time privileged information

ASC / DESC must be handled by the **same model**. Training a separate ASC model and a
separate DESC model is not allowed. `OrbitID` is information every single-orbit method is
permitted to use.

If some external architecture cannot directly accept a fifth channel, a **minimal interface
adaptation** is allowed, but the adaptation must be recorded and must not alter the method's
core structure.

---

# 5. Uniform training protocol

Every formal method uses:

- Optimizer: Adam
- learning rate: `5e-5`
- betas: `(0.9, 0.999)`
- eps: `1e-8`
- weight decay: `0`
- scheduler: None
- constant LR
- physical batch: `16`
- gradient accumulation: None, unless the model genuinely cannot fit batch 16 in memory; if
  that happens, effective batch must still be 16 and it must be explicitly recorded
- maximum epoch: **200** — a **ceiling**, not a plan
- **early stopping: ON**
- patience: **2** — after an epoch completes, if the next two consecutive epochs both fail
  to beat the best score seen so far, training stops
- the monitored score: **test AUPRC**, threshold-free, read once per epoch
- the formal checkpoint is the **best-monitored epoch**, not the last epoch
- learning rate, loss, threshold, split and architecture are unchanged from v3

The ceiling exists only so that a run cannot go on forever if the monitored score keeps
inching upward. In expectation every method stops on the patience rule, well before 200.

**Each arm monitors its own curve.** Two arms may therefore stop at different epochs. That
is the intended behaviour, not an inconsistency: the rule is "train until this method stops
improving", applied identically to every method.

Whatever the stopping epoch, every method is still given the same *rule*, the same data and
the same optimizer. What v4 gives up is the guarantee that all methods received the same
number of gradient steps.

---

# 5b. ⚠️ The monitored score is read on TEST — this is an acknowledged bias

**This is a deliberate, user-authorised departure from the v3 rule that test never changes a
model.** It is recorded here, in `cocd/configs/protocol_windows_main.json` under
`monitor_risk`, in every run log header and in every `progress.json`.

- User decision, 2026-09-19: *"仍然用 test 监控，但标注风险"*.
- Consequence: the reported test AUPRC is the **maximum over the epochs visited**. It is
  optimistically biased by an amount that cannot be measured from this run alone.
- Cross-method comparability survives only because every method is selected under the
  **identical** rule. A v4 number and a v3 number are **not** comparable.
- Still protected: the threshold is fixed at 0.5 and never re-searched; no architecture, loss
  or lambda is chosen from the monitored score; only the stopping epoch is.
- The cost-free alternative — restoring the 137-location validation partition as a monitoring
  set — was verified to be arithmetically available (`v3 train == v2 train ∪ v2 val`,
  `v3 test == v2 test`, zero overlap) and was **offered and declined**.
- When this work is written up, the table caption must state that model selection used the
  test partition.

---

# 6. No validation ⇒ the monitored score selects the stopping epoch

There is still no validation partition, so no validation-based selection of any kind exists.
What v4 adds in its place, and explicitly:

- **allowed**: choosing the stopping epoch by the monitored test AUPRC, under the fixed
  patience-2 rule applied to every method alike.

Still cancelled:

- validation-AUPRC checkpoint selection (there is no validation set)
- validation threshold search
- validation-based hyper-parameter selection

Still not allowed:

- picking an arbitrary earlier epoch because it "looks better" outside the patience rule;
- selecting the checkpoint on any metric other than the declared monitored score;
- re-searching the threshold on test.

The checkpoint is whichever epoch the rule selects, and it is written down together with
`selected_epoch`, `monitor`, `monitor_value` and `stopped_by` in the run's `summary.json`.

---

# 7. Binary threshold permanently fixed at 0.5

Since there is no validation, IoU / F1 / Precision / Recall all use

`sigmoid(logits) >= 0.5`

as the binary prediction. Identical for every method.

Not allowed:

- searching for the best threshold on test
- a different test-optimal threshold per method
- changing the threshold according to test F1

AUPRC remains the threshold-free metric. It is recommended that formal results report AUPRC
as one of the primary metrics.

---

# 8. Uniform segmentation supervision

The basic segmentation supervision for every internal ConvNeXt-FPN method is

`BCEWithLogitsLoss`

i.e. plain BCE:

- no `pos_weight`
- no Dice
- no Focal

If a published external method's special loss is an **inseparable part of that method's
definition**, it may be kept, but:

1. it must be explicitly labelled a *method-specific objective*;
2. its weight must not be tuned on test;
3. all other training conditions stay identical.

Do not delete a method's core loss in the name of "fairness" — that would mean it is no
longer that method.

---

# 9. Initialisation rules

Internal ConvNeXt-Tiny methods: all use the **same** ImageNet initialisation.

Forbidden:

- warm-starting from another Student checkpoint
- choosing an initialisation by performance
- deciding an initialisation from test

Where the method definition itself requires inheriting Teacher parameters — e.g. the Ω
operator of legacy R3 — this is allowed, because it is part of the method and is not a
training advantage.

External architectures:

- if the official backbone has ImageNet pretrained weights, use the official ImageNet
  initialisation;
- if not, use its official initialisation;
- record it explicitly.

Architectures are **not** required to have the same parameter count.

---

# 10. The formal comparison method set

## A. Single-Orbit supervised baselines

At least:

- N0 Single-Orbit ConvNeXt-FPN
- Boehm
- CDNetE
- FC-Siam
- MFEWF

All retrained under this unified train/test protocol.

## B. Knowledge-distillation baselines

Including:

- Vanilla KD
- DIS2-port
- Legacy R3

They keep their own already-defined distillation rules. But:

- identical data
- identical epochs
- identical optimizer / lr / batch
- epoch-20 checkpoint
- test threshold 0.5

Legacy KD methods use the **same retrained Legacy Teacher**.

## C. New COCD

Including:

- New Teacher
- N1 Complement Predictor
- N2 Full COCD
- N2 w/o Orbit Embedding

The New COCD method definition stays at the currently frozen version; **no further modules
are added**.

---

# 11. The status of the Teacher

The Teacher is a **privileged-information reference**, not a single-orbit deployment
competitor. The Teacher may see, at training time, target orbit **and** counter orbit.

But its training-management rules remain uniform:

- same Train split
- same temporal modes
- Adam 5e-5
- batch 16
- 20 full epochs
- epoch-20 checkpoint
- no validation selection
- test only after freezing

New Teacher:

`z_self = Head(F_t)`

`z_dual = Head(F_t + C_T)`

loss:

`L_teacher = mean(BCE(z_self,y), BCE(z_dual,y))`

The 0.5 / 0.5 here is merely the arithmetic mean of two BCEs, not a tunable hyper-parameter.

---

# 12. New COCD Student

N1:

`F_t -> Orbit-conditioned Complement Predictor -> C_S`

`F_student = F_t + C_S`

uses only:

`L_seg`

N2:

adds, on top of N1:

`L_dist(C_S, stopgrad(C_T))`

full loss:

`L = L_seg + lambda * L_dist`

**Only one `lambda` is allowed.**

Not added:

- gain×protect
- logit KD
- L_delta
- gate loss
- multilevel KD
- geometry weighting

---

# 13. The lambda rule for New COCD

`lambda` must not be tuned on test. It may be determined only from the **numeric scale of
the losses on Train data**.

The default candidate set is only:

`0.1 / 1 / 10`

Selection principle: look only at the magnitude of `L_seg` and `L_dist` at the start of
training, and keep distillation an auxiliary supervision rather than the driver of training.

Not allowed:

- comparing the three lambdas on test and picking the best
- changing lambda afterwards based on test AUPRC
- running a large continuous lambda sweep

Once fixed, all seeds use the same lambda.

If there is already clear evidence that `lambda = 1` makes distillation occupy roughly
40–50 % of the training objective for a long time, then — **without looking at test** —
`lambda = 0.1` may be retrained as the weak-auxiliary version, with the reason recorded.

---

# 14. The New Teacher audit is a diagnostic, not model selection

After New Teacher training, the following may be run on diagnostic samples reserved from
Train, or on training samples outside Test:

- self
- correct counter
- zero counter
- shuffled counter

This audit is used only to confirm whether the Teacher really uses the correct counter.

Test audit results must not be used to tune the network.

If the audit is to be formally reported, prefer a fixed diagnostics subset drawn from Train.
Do not re-establish a validation split.

---

# 15. Geometry mask never participates in training

The geometry mask is used only for explanatory evaluation on Test:

- G00
- G01
- G10
- G11

Forbidden:

- model input
- loss weighting
- hard gate
- pixel removal
- GT masking
- geometry KD

Legacy R3g may be kept as historical code, but must not enter the new active experiment plan.

---

# 16. Random seeds

First full pipeline:

`seed = 42`

to confirm every method can complete training normally.

The formal paper's core results use at least:

`42 / 123 / 2026`

three seeds.

At least covering:

- N0
- DIS2-port
- Legacy R3
- N1
- N2

If compute allows, also cover:

- New Teacher
- Vanilla KD
- major external baselines

Report mean ± std. Do not decide whether to report one seed on the basis of another seed.

---

# 17. Rules for using Test

Test is the final one-shot evaluation data.

Allowed to compute:

- AUPRC
- IoU @ 0.5
- F1 @ 0.5
- Precision @ 0.5
- Recall @ 0.5
- ASC
- DESC
- G00
- G01
- G10
- G11

Forbidden:

- test threshold tuning
- test lambda selection
- test architecture selection
- looking at test, then going back to redesign the method while still presenting the same
  test as the final unbiased result

**The one exception, declared rather than hidden (v4 §5b):** test AUPRC is read every epoch
and decides the stopping epoch under the patience-2 rule. `test checkpoint selection` and
`test epoch selection` are therefore *permitted only in the form of that single declared
rule*, applied identically to every method. They remain forbidden in every other form — in
particular, hand-picking an epoch outside the rule.

If the network is redesigned on the basis of test results, that test has participated in
development and this fact must be carefully explained in the final paper.

---

# 18. Fairness principle

The formal main table must obey:

> Same spatial split, same temporal-pair construction, same deployment information, same
> optimizer, same learning rate, same batch size, same *stopping rule*, same test threshold
> and same evaluation protocol.

Under v4 "same epoch budget" is replaced by "same stopping rule": every method trains under
patience-2 early stopping on the same monitored quantity, and may stop at a different epoch.
The number of epochs each method actually ran, and the epoch that was selected, must be
reported alongside its metrics — a reader must be able to see that the budgets differed in
realised length even though the rule was identical.

The only things allowed to differ are:

- the architecture itself
- the published method-specific mechanism
- the published method-specific KD / auxiliary loss
- privileged Teacher information (training time only)

Not required:

- same parameter count
- same FLOPs
- same network structure
- same loss-coefficient values

But important differences must be reported.

---

# 19. Organisation of the formal experiment table

The main table is recommended in three groups:

### Supervised single-orbit
N0 / Boehm / CDNetE / FC-Siam / MFEWF

### Distillation baselines
Vanilla KD / DIS2-port / Legacy R3

### Proposed
N1 / N2 Full COCD / N2 w/o Orbit Embedding

Listed separately:

New Dual-Orbit Teacher

as a privileged reference / upper reference, **not** mixed in as a same-privilege model with
the single-orbit deployment methods.

---

# 20. Do not change these rules on your own from now on

If some method performs badly, it is not permitted to "rescue" it by:

- adding epochs
- changing LR
- changing batch
- changing threshold
- changing train modes
- switching checkpoint selection
- adding augmentation for one method only
- peeking at test and tuning

Execute the unified protocol completely first.

If some published method plainly cannot train normally under the unified protocol, first
report the concrete evidence, then decide whether to additionally provide an
"official-training-setting reproduction" as a supplementary experiment. But:

**the paper's main fairness table still follows this unified protocol.**
