# External baselines and the KD reference — full cards

The five methods named in the request: **Boehm SAR U-Net++, CDNetE Early-Fusion,
MFEWF, DIS2, and Vanilla KD.** Each gets its own card: provenance, adaptation
status, code availability, where its numbers live, and what it means for our claim.

All five are measured on the **same V1 frozen protocol** as the first-round table:
686 orbit-specific test samples, fixed threshold 0.5, five-channel single-orbit
Student input, pixel-wise landslide GT, geometry not a Student input. They are
**not** under the Windows unified protocol and must never be mixed into that table.

---

## Card 1 — Boehm SAR U-Net++

| field | value |
|---|---|
| reference | Boehm et al., SAR landslide segmentation with U-Net++ |
| official code | `iprapas/landslide-sar-unet` — **source snapshot downloaded** |
| type | official-code adaptation |
| test-time orbit | single |
| params / architecture | keeps its own architecture, not forced onto v2 |

| metric | value |
|---|---:|
| IoU | 0.5926 |
| F1 | 0.7442 |
| AUPRC | 0.8363 |
| complementary F1 | 0.7274 |
| complementary AUPRC | 0.8368 |
| G10 F1 (from ablation table) | 0.7274 |
| G10 AUPRC | 0.8368 |

**Reading.** This is the only external baseline that beats us on the fixed-threshold
metrics: its IoU 0.5926 / F1 0.7442 exceed Ours-V1 (0.5619 / 0.7195) and even exceed
DIS2-style (0.5662 / 0.7230). We do win on the threshold-free AUPRC: 0.8589 vs 0.8363,
and on complementary AUPRC 0.8714 vs 0.8368. So the comparison is **metric-dependent,
not a win**: at the fixed 0.5 threshold a generic SAR segmentation network with the
same single-orbit input is stronger, while we are stronger when the operating point
is chosen properly. This is one of the two results that forced the CONDITIONAL GO.

---

## Card 2 — CDNetE Early-Fusion

| field | value |
|---|---|
| reference | CDNetE, change detection, early-fusion configuration |
| official code | **not identified** — author thesis and the Haiti dataset record were located, but no repository |
| type | Haiti-native thesis adaptation |
| test-time orbit | single |

| metric | value |
|---|---:|
| IoU | 0.4927 |
| F1 | 0.6601 |
| AUPRC | 0.7715 |
| complementary F1 | 0.6550 |
| complementary AUPRC | 0.7760 |
| G10 F1 | 0.6550 |
| G10 AUPRC | 0.7760 |

**Reading.** Weakest of the three external methods across every column, and clearly
below S0 (IoU 0.5302). We beat it on everything. The caveat is provenance: with no
official code, this is a thesis-derived adaptation, so it is the least defensible
number of the three. If a reviewer asks for the implementation, there is none to
point at — worth deciding whether it stays in the table.

---

## Card 3 — MFEWF

| field | value |
|---|---|
| reference | MFEWF; paper points to ChenLifu2022 |
| official code | **not found** — the public profile does not expose an MFEWF repository; status pending author-code verification |
| type | paper adaptation |
| test-time orbit | single |
| artifact | `results/grsl_external_baselines/` — **this directory contains MFEWF only** |

| metric | value |
|---|---:|
| IoU | 0.3232 |
| F1 | 0.4886 |
| AUPRC | 0.6583 |
| complementary F1 | 0.4278 |
| complementary AUPRC | 0.6442 |
| saved val AUPRC (`progress.json`) | 0.6525 |
| asc IoU / desc IoU | 0.3168 / 0.3297 |

**Reading.** Far below S0 and below every other method here, with a large
precision/recall asymmetry (precision 0.7905, recall 0.3535 — it predicts very few
positives and is usually right when it does). A near-identical pattern appears in the
distortion line, where a paper adaptation also collapsed (0.2764). Two adaptations
of published methods landing this low is more likely an adaptation-fidelity problem
than two independent method failures, and it should be described that way rather than
as "the published method performs poorly".

**Directory caveat.** `results/grsl_external_baselines/` contains `metrics.csv` and
`progress.json` for MFEWF only. The Boehm and CDNetE numbers exist in
`reports/grsl_baseline_and_mechanism_completion.md` and in the ablation table, but
**not** as separate per-method artefacts in that directory. Their weights were never
shipped.

---

## Card 4 — DIS2

There are **three different things called DIS2** in this project and only one of them
is a baseline for us. Full write-up: `docs/DIS2_THREE_VERSIONS_RESULT.md`.

| variant | skeleton | distillation content | protocol | verdict | IoU | AUPRC |
|---|---|---|---|---:|---:|---:|
| **old DIS2-style** | our `SingleOrbitStudent(correction=True)` | residual cosine + output MSE, **our own formula** | V1, batch 2 | kept | **0.5662** | 0.8620 |
| official width-50 | official `DLKD_ver4(gf_dim=50)` | official three-stage + orthogonality | V2, batch 2 | **deleted** | 0.1829 | 0.3732 |
| **DIS2-port** | our `OursV3Student('R1')` | official three-stage + orthogonality, line-by-line | V3, batch 8 | kept | **0.6272** | 0.8963 |

### 4a. old DIS2-style — the V1 baseline

| metric | value |
|---|---:|
| IoU | 0.5662 |
| F1 | 0.7230 |
| AUPRC | 0.8620 |
| complementary F1 | 0.7278 |
| complementary AUPRC | 0.8669 |

Versus the V1 reference points: above Vanilla KD by **+0.0121 IoU**, above Ours-V1 by
+0.0043, below the dual teacher by 0.0133. This is the strongest generic-KD baseline in
the V1 block.

**Q3 of the frozen checklist is answered No because of this row:** on overall
IoU/F1/AUPRC, DIS2-style adapted is higher than Ours-V1. The reply is that we win in
the mechanism-critical complementary region (Comp-F1 0.7333 vs 0.7278, Comp-AUPRC
0.8714 vs 0.8669). That is a region-specific claim, not a global one.

An unresolved observation worth keeping: old DIS2-style matches **correction**
(`r_s` vs `r_t`) while official DIS2 matches **activations**. The old version may have
been effective partly by accident for this reason, and the line-by-line port to the
official rules gave that property up. Recorded as an inference, not a conclusion —
the two ran under different protocols and the old one used weight 0.1.

### 4b. official width-50 — failed reproduction, deleted

Never converged: one third of the data, eight auxiliary heads consuming 77 % of the
gradient, no pretrained weight load. The implementation, scripts and model definition
were **deleted on instruction 2026-09-17**. The number is retained here as a record
only and **must not appear in any results table.**

### 4c. DIS2-port — the V3 port

| metric | value |
|---|---:|
| IoU | 0.6272 |
| F1 | 0.7709 |
| AUPRC | 0.8963 |
| G10 IoU | 0.6335 |
| G10 AUPRC | 0.9023 |

Under V3 this is the only arm with a like-for-like training budget against R3 (both
50 epochs, batch 8, no early stop), and **R3 wins by 0.0026 IoU / 0.0009 AUPRC**.
Region by region the two differ by only 0.0004-0.0047 and alternate leads, so the
difference between output-level selective KD and official multi-level distillation is
**not established** by this comparison.

And DIS2-port is a baseline in the Windows protocol too: 0.59898 AUPRC, **below** the
N0 baseline 0.64428.

### 4d. The orthogonality term — a usable negative result

Official DIS2 carries `orthogonality_loss` at weight 1.0. Ported here, it collapses to
**0.0000 within 3 epochs** (official code: 9.31e-05). The three distillation terms do
decrease in the same run (feat 0.0007 to 0.0002, logit 0.0050 to 0.0028), so the
student genuinely moves toward the teacher and only the orthogonality constraint is
inert. This can be stated positively: the "correction degenerating into re-encoding"
failure mode is not the dominant one on this dataset.

---

## Card 5 — Vanilla KD

| field | value |
|---|---|
| what it is | output-level knowledge distillation from the dual-orbit teacher, no selection rule, no correction matching |
| code | `CONFIGS['VKD']`, `selective_kl(..., 'all')` |
| role | the reference the mechanism claim is measured against |

Two independent measurements, from two protocols:

| protocol | IoU | F1 | AUPRC | gap recovery (IoU) |
|---|---:|---:|---:|---:|
| V1 test | 0.5541 | 0.7131 | 0.8545 | 0.4851 |
| Windows val | — | — | **0.60333** | — |

Gap recovery over S0→T in V1, the one place the metric exists:

| method | IoU | F1 | AUPRC | G10 F1 | G10 AUPRC |
|---|---:|---:|---:|---:|---:|
| Vanilla KD | 0.4851 | 0.4930 | 0.4811 | 0.4290 | 0.3583 |
| **Ours-V1** | **0.6427** | **0.6500** | **0.6202** | **0.7574** | **0.6879** |

**Reading.** This is the cleanest supporting result in the project. Vanilla KD
recovers under half the S0→T gap; Ours-V1 recovers about two thirds overall and
**three quarters on G10**, the region where the target orbit is distorted and the
counter orbit is reliable. The separation is large relative to any noise estimate
we have. It is also V1, batch 2, fixed threshold 0.5 — so if this is going to carry a
mechanism claim, it needs either a Windows re-measurement or an explicit statement of
the protocol.

**Windows protocol.** VKD lands at 0.60333, i.e. **below the single-orbit baseline
N0 (0.64428) by 0.0410**. Its training log shows why: segmentation settles at
0.036-0.041 while the KD term is still 0.021/0.007/0.003, so the auxiliary term
dominates the late objective. That is the same signature as DIS2 and R3 under the
Windows protocol — all three distillation controls land below their own baseline. So
under the unified protocol, distillation from this teacher *hurts* every controlled
arm, which is a result about the protocol as much as about the methods.

---

## Housekeeping items this surfaces

1. **`results/grsl_external_baselines/` holds MFEWF only.** Boehm and CDNetE numbers
   live in the report and the ablation table but have no per-method artefact file in
   that directory. If a table is regenerated from files, those two rows will vanish.
2. **Boehm's weights were never shipped**, and CDNetE / MFEWF have no official code
   located. Neither can be re-evaluated on the Windows protocol without re-obtaining
   the upstream sources.
3. **FC-Siam was named in the handoff but is not trained** — `TRANSFER_AUDIT.md` records
   "Boehm / CDNetE / MFEWF trained; FC-Siam not".
4. **The handoff is explicit that external baselines are outside the v2 protocol**:
   "Boehm / CDNetE / FC-Siam / MFEWF keep their own architectures and are not forced
   onto v2." So the current formal table and these five rows live under different
   protocols by design, and any merged table must say so.
5. **Boehm is the load-bearing caveat on any superiority claim.** It beats us at
   threshold 0.5 on the same single-orbit input. The honest framing is that we win on
   threshold-free AUPRC and on the complementary region, and that the fixed-threshold
   comparison goes the other way.
