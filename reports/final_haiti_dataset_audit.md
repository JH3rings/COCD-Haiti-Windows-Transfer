# Haiti SAR Dataset — GRSL Feasibility Audit (final)

**Audit date:** 2026-09-12 · **Data:** *Multimodal Remote Sensing Dataset for Landslide Change Detection in Haiti*, Bralet, Trouvé, Chanussot, Atto — IEEE Dataport 2024, doi [10.21227/4heb-7h07](https://dx.doi.org/10.21227/4heb-7h07)
**Source path (read-only):** `/Users/zhangjiuqi/Desktop/distortion/{Pre_event,Post_event,Annotations}`
**Derived workspace:** `Haiti_SAR_GRSL_Audit/` (≈653 MB added; source 3.1 GB untouched)
**Scope:** data/provenance/science validation only — **no deep model was trained**.

> The task expected `*.tar.gz` in `~/Downloads`; Downloads held only an unrelated `sar-ship-dataset-master.zip`.
> The dataset was found **already extracted** in the selected project folder. All findings below come from
> reading the actual rasters; documentation claims (Bralet PhD thesis, HAL tel-05029007, §5.2) are used only to
> corroborate, never to substitute for measurement.

---

## 1. Data scale (requested summary item 1)

| Metric | Value |
|---|---|
| Total samples | **1713** (IDs `0…1712`, contiguous, identical in every folder) |
| Complete samples | **1713 / 1713 = PASS** (no missing ASC/DESC/pre/post/optical/annotation/geometry) |
| SAR files | **6852** (4 per sample: ASC/DESC × pre/post) |
| Optical files | 3426 (S2 pre + post) |
| Annotation files | 1713 |
| Total files | **11991 GeoTIFF**, only `.tif` present (no README/XML/JSON/sidecar shipped) |
| Image size | **128 × 128 px**, uniform; EPSG:4326; ≈ **10 m/px** (9.48 × 9.93 m); float32 SAR / float64 S2 |
| Total extracted size | **3.26 GB** logical (≈3.1 GB on disk): Pre 1.5 GB, Post 1.5 GB, Annotations 114 MB |
| Patch construction (thesis) | 128×128 sliding window, **stride 64 (≈50% overlap, 4×)**, kept if >100 landslide px |

## 2. Actual sample schema (requested item 2)

Each integer `sample_id` maps 1:1 to **7 files** (verified on all 1713, inspected in depth on dozens):

```
sample_id = k
├── Pre_event/S1_DESC_20210803/k.tif   SAR DESC pre  (3×float32)  band1 VH(dB), band2 VV(dB), band3 GEOM mask
├── Pre_event/S1_ASC_20210805/k.tif    SAR ASC  pre  (3×float32)  band1 VH(dB), band2 VV(dB), band3 GEOM mask
├── Pre_event/S2_20210804/k.tif        Optical pre    (4×float64)  band1-3 RGB reflectance, band4 cloud mask
├── Post_event/S1_DESC_20210815/k.tif  SAR DESC post (3×float32)  same band layout
├── Post_event/S1_ASC_20210817/k.tif   SAR ASC  post (3×float32)  same band layout
├── Post_event/S2_20210814/k.tif       Optical post   (4×float64) band1-3 RGB, band4 cloud mask
└── Annotations/k.tif                  Landslide label(1×float32) {0,1,2,3}, NOT georeferenced (identity tfm)
```

- **VV/VH are separate bands, not files.** Order is **band1 = VH, band2 = VV** (assigned from data: VV is 6–7 dB
  stronger with a higher noise floor; the author's thesis also writes the dual-pol pair as "(VH, VV)").
  Values are **σ0 in dB (log scale)**, σ0-calibrated and SRTM-orthorectified (thesis §5.2); no NaN, no zeros,
  no declared nodata. Global medians: VH ≈ −15 dB, VV ≈ −8.3 dB (see `qc/tables/sar_band_stats.csv`).
- **The geometry mask is bundled as SAR band 3** — there is no separate mask file/folder.
- S2 band 4 is a binary **cloud/cloud-shadow mask** (Sen2Cor medium+high cloud ∪ cloud shadow).
- Annotations are the **NASA landslide** label, not geometry (§10).

## 3. ASC/DESC authenticity (requested item 3) — **genuine, not relabelled duplicates**

The patched GeoTIFFs carry **no embedded Sentinel-1 product ID, relative orbit, flight-direction or acquisition
time-of-day** (only `AREA_OR_POINT=Area`; the "log-amplitude" tag is an auto-generated GDAL virtual subdataset).
So orbit direction cannot be read from a metadata tag. It is instead established by four independent, mutually
consistent lines of evidence (**all measured, not assumed from the `ASC`/`DESC` string**):

1. **Different acquisition dates** in the folder names, corroborated by thesis §5.2 (ASC pre 08-05 / DESC pre
   08-03; ASC post 08-17 / DESC post 08-15) — 48 h apart, i.e. distinct satellite passes.
2. **Pixel content is never identical.** Across all 1713 samples, both polarisations, both dates:
   **`array_equal = 0`, identical-MD5 = 0** out of 6852 comparisons (`qc/tables/ad_independence.csv`).
3. **Large radiometric disagreement with *negative* pixel correlation** — median |ASC−DESC| ≈ **6.4–7.1 dB**,
   max ≈ 30–41 dB, Pearson corr **−0.38 (VH) / −0.46 (VV)**. Opposite-view backscatter over steep terrain is
   anti-correlated because radar-facing slopes are bright in one view and dark in the other — the physical
   slope-facing signature, impossible for a duplicated acquisition.
4. **View-specific geometry masks differ** (band 3 ASC vs DESC disagree on ~33% of distorted pixels; §7) and are
   tied to the documented ascending/descending layover/shadow computation.

**Verdict Q1 — true independent ASCENDING + DESCENDING: YES.** (Caveat for reviewers: original Sentinel product
IDs were stripped during patch extraction; provenance rests on the above evidence + the published thesis. The
mask is re-derivable from SRTM + orbit geometry via Meier et al. 1993 if a from-scratch reproduction is demanded.)

## 4. Registration / co-registration (requested item 4)

- **Common grid, exactly.** For every one of 1713 samples all six georeferenced rasters share identical
  **CRS (EPSG:4326), width/height (128), affine transform and bounds** (`qc/tables/grid_consistency.csv`:
  0 samples with any mismatch). Thesis: all images are orthorectified with SRTM 1-sec DEM and co-registered to
  the 2021-08-03 DESC SAR.
- **Same-view temporal residual is sub-pixel.** ASC pre↔post / DESC pre↔post gradient phase-correlation shift
  median **0.22 px**, P90 **0.41 px**; checkerboards are seamless and edge overlaps are yellow/coincident.
- **Opposite-view ASC↔DESC shows no constant global offset.** On both-good (undistorted) pixels the masked
  phase-correlation shift is scattered (std ≈42 px ≫ median, no dominant direction) and the normalised-cross-
  correlation peak is only ≈0.07. This is **not mis-registration** — the map frame is identical and the same-view
  control proves the geocoding is accurate; it is the genuine view-dependent radiometry/geometry of steep
  terrain (foreshortening/layover/shadow rearrange content between looks). 12 checkerboard/edge/diff panels in
  `qc/figures/registration/` confirm no uniform translation.

**Verdict Q2 — coregistered for pixel-level / local-region comparison: YES.** The shared map grid makes pixel-
level **mask** comparison valid (this is what the complementarity analysis uses). Opposite-view **intensity**
fusion must account for physical view differences; per instructions, **no non-rigid "correction" was applied**
(the local discrepancy is the phenomenon under study).

## 5. Geometry labels — layover / shadow (requested item 5) — **PRESENT (bundled as SAR band 3)**

Recursive search found no separate mask file, but **band 3 of every SAR GeoTIFF is the aggregated layover +
radar-shadow mask**, explicitly documented in thesis §5.2 ("the layover and radar shadow masks computed
according to [Meier et al., 1993] are aggregated and provided for the ascending and descending passes").
Coding validated empirically (brightness/value structure) and consistent with the standard additive aggregation:

| value | meaning | empirical signature |
|---|---|---|
| 0 | valid / visible | mid intensity |
| **1** | **layover** | **brightest** class (mixed backscatter): VV −2.5 dB vs −9 dB valid |
| **2** | **radar shadow** | **darkest** class (no signal): VV −13.6 to −16.6 dB |
| 3 | layover + shadow | rarest (intersection), intermediate-low |

Pooled class proportions (pre; post identical by construction), `qc/tables/geometry_class_percentages.csv`:

| view | valid | layover | shadow | both | total "bad" |
|---|---|---|---|---|---|
| ASC  | 86.25% | **13.68%** | 0.054% | 0.017% | **13.75%** |
| DESC | 94.02% | **5.33%** | 0.638% | 0.011% | **5.98%** |

**Important asymmetry:** distortion is dominated by **layover**; **shadow is sparse**, almost absent in ASC
(0.05%). ASC carries ≈2.3× more flagged terrain than DESC. This drives the recommended target choice (§Verdict).

**Verdict Q3 — true layover/shadow geometric-distortion mask: YES** (available in-band; also DERIVABLE from
DEM+orbit). It is independent of the landslide labels (different file, different spatial pattern, static).

## 6. Time information (requested item 6)

`metadata/temporal_pairs.csv` (uniform for all 1713; date-level only — **time-of-day = NA**, not in the patches):

| pair | ASC | DESC | Δ(ASC−DESC) |
|---|---|---|---|
| pre  | 2021-08-05 | 2021-08-03 | **48 h** |
| post | 2021-08-17 | 2021-08-15 | **48 h** |
| pre→post (ASC) | 12 days | pre→post (DESC) 12 days | |
S2 pre 2021-08-04, S2 post 2021-08-14; **earthquake D-day 2021-08-14**.
Bucket for |t_ASC − t_DESC|: **all 1713 fall in ≤3 days (48 h)**; none >3 days. A/D within each event window are
close in time; the 12-day pre/post gap brackets the earthquake.

## 7. Complementarity (requested item 7) — **the physical basis holds strongly**

Per-pixel 2×2 contingency of "bad = mask>0" (pooled over all 1713; pre and post give identical numbers because
the mask is static), `qc/tables/complementarity_statistics.csv`:

| stratum | pooled pixel % |
|---|---|
| **both good** (A=0,D=0) | **81.05%** |
| **ASC bad / DESC good** | **12.97%** |
| **ASC good / DESC bad** | **5.20%** |
| **both bad** (A=1,D=1) | **0.78%** |

**ComplementarityRatio = (A_bad·D_good + A_good·D_bad) / (A_bad ∪ D_bad):**
mean **0.972**, median **0.987**, std 0.040, P25 0.961, P75 1.000. **420/1713 samples have CR = 1.000** (every
distorted pixel is rescued by the opposite view); **0 samples have CR < 0.5**, only 11 are < 0.8.
Interpretation: of all terrain that is unobservable in at least one view, **≈97% is unobservable in only one
view and recoverable from the other**; irrecoverable "both-bad" terrain is <1%. See Figure C (orange/blue on
opposite-facing slopes, almost no overlap).

**Cross-view disagreement tracks geometry (validation, not label construction; §16):** mean |VV_ASC − VV_DESC|
rises from **5.79 dB (both-good) → 11.06 (ASC-bad/DESC-good) → 12.92 (ASC-good/DESC-bad) → 13.39 dB (both-bad)**;
gradient difference rises likewise (`qc/tables/crossview_disagreement_by_stratum.csv`).

**Verdict Q5 — A/D geometric complementarity: STRONG.**

## 8. Label-leakage audit (requested item 8) — **PASS**

Geometry-bad pixels are **not** encoded as NaN / zero / constant fill: there is no nodata, NaN = 0%, zero = 0%,
and floor-fill pixels ≈ 0.006%. Overlap between the geometry mask and any invalid/fill mask (`leakage_audit.csv`):
IoU median **0.000**, F1 median **0.0016** (VV), P99 0.085, **max 0.22 on one tiny patch, 0 samples > 0.3**,
0 samples > 0.85. Distorted regions contain real (if unreliable) backscatter, so a network cannot "cheat" by
detecting fill values. **Verdict Q4: PASS.**

## 9. Pre/post repeatability (requested item 9)

The band-3 geometry mask is **byte-identical pre↔post for every sample and both views** (IoU = Dice = 1.000,
pixel agreement 1.000, total differing pixels = 0; `geometry_prepost_repeatability.csv`). This is expected — the
mask is computed once from static DEM + orbit geometry and reused for both dates. It confirms the "viewing
geometry is temporally stable" assumption, but it is **the same duplicated mask, not an independent second
observation**, so it cannot independently re-validate geometry across time (the SAR *intensity* does change
pre/post, e.g. landslides). **Verdict Q6: PARTIAL** — perfect repeatability by construction, not an independent
temporal replicate; use pre/post as extra intensity realisations, not as independent geometry labels.

## 10. Annotations decoded (§10/§21) — landslide GT, **not** geometry

`Annotations/{id}.tif` is single-band float32, **128×128 but ungeoreferenced (identity transform)**. Global
values {0,1,2,3}: 0 = 26.19M px; 1 = 0.77M; 2 = 0.46M; 3 = 0.64M. These are the **rasterized NASA co-seismic
landslide inventory (14,482 polygons)**; visually they are irregular landslide scars (Figure F), wholly unlike
the slope-following geometry masks. **`ann > 0` = landslide**; 1700/1713 patches contain >100 landslide px
(min 90, median 503), matching the documented >100-px extraction rule. The 1/2/3 sub-codes reflect overlapping
rasterised polygon instances (not geometry). **Y_landslide (Annotations) and Y_geometry (SAR band 3) are
distinct and must not be swapped.**

## 11. Single-view predictability without any deep net (§17)

A trivial classical baseline (logistic regression on VV local mean/std, gradient, entropy, local range;
train/test split by sample; `singleview_baseline.csv`) already separates good vs geometry-bad pixels:

| view | bad-pixel rate | ROC-AUC | AP | best single feature (local mean) AUC |
|---|---|---|---|---|
| ASC  | 13.2% | **0.921** | **0.725** | 0.914 |
| DESC | 6.1%  | **0.872** | 0.540 | 0.864 |

The single-image task is therefore **far from unidentifiable** — a CNN has a clear, non-trivial signal to learn.

## 12. Master data-structure table (§25)

| Content | present? | count | truly independent? | aligned? | usable for our geometry task? |
|---|---|---|---|---|---|
| ASC pre (SAR)  | ✅ | 1713 | ✅ independent acquisition (neg-corr, MD5 unique) | ✅ common grid | ✅ |
| DESC pre (SAR) | ✅ | 1713 | ✅ | ✅ | ✅ |
| ASC post (SAR)  | ✅ | 1713 | ✅ | ✅ | ✅ (extra realisation) |
| DESC post (SAR) | ✅ | 1713 | ✅ | ✅ | ✅ (extra realisation) |
| VV polarization | ✅ band2 | 6852 | — | ✅ | ✅ recommended stage-1 input |
| VH polarization | ✅ band1 | 6852 | — | ✅ | ✅ keep (source ref) |
| Landslide mask | ✅ Annotations | 1713 | independent label | pixel grid (no CRS) | ❌ not geometry (separate task) |
| **Layover mask** | ✅ band3=1 | per-view | ✅ view-specific | ✅ | ✅ |
| **Shadow mask**  | ✅ band3=2 | per-view (sparse) | ✅ | ✅ | ⚠️ usable but very sparse (ASC 0.05%) |
| Acquisition time | ⚠️ date only | folder name + thesis | dates distinct A/D | — | ✅ (time-of-day NA) |
| Optical (S2 RGB+cloud) | ✅ | 3426 | — | ✅ | auxiliary only |
| DEM | ❌ not shipped | — | — | — | mask is DERIVABLE from SRTM if needed |

---

## 13. Final scientific verdicts

| Q | Question | Answer |
|---|---|---|
| Q1 | True independent ASCENDING + DESCENDING? | **YES** |
| Q2 | Sufficiently coregistered for pixel/local comparison? | **YES** (exact common grid; sub-pixel same-view; opposite-view differences are physical) |
| Q3 | Real layover/shadow geometric-distortion mask? | **YES** (SAR band 3, Meier-1993; also derivable from DEM) |
| Q4 | Geometry-label leakage (NaN/0/fill)? | **PASS** (F1≈0.002, none >0.3) |
| Q5 | A/D geometric complementarity? | **STRONG** — CR median 0.987; Abad/Dgood 12.97%, Agood/Dbad 5.20%, both-bad 0.78%, both-good 81.05% |
| Q6 | Pre/post usable as repeat-validation axis? | **PARTIAL** (mask byte-identical by construction; perfect but not independent) |

## 14. GRSL data-suitability verdict — **GO** (with noted conditions)

The required causal chain is satisfied end-to-end:
**True ASC/DESC (Q1=YES) → independent geometry labels (Q3=YES, Q4=PASS) → A/D complementarity (Q5=STRONG) →
single-view predictability (AUC 0.87–0.92 with trivial features).** The dataset can support the proposed
"train-time dual-view privileged information → test-time single-view geometry-distortion prediction" paper.

**Recommended prediction target (stage 1): binary "geometrically unreliable" region = `band3 > 0`
(layover ∪ shadow ∪ both).** Reasons: best class balance (ASC 13.8% / DESC 6.0%), directly matches the
observability framing, and avoids the near-absent ASC shadow class. A secondary **3-class head
(valid / layover / shadow)** is feasible for DESC but shadow is too sparse in ASC (0.05%) — report per-class and
per-view, and do not build the headline claim on shadow-only. Continuous reliability scoring is a later extension.

**Conditions / caveats to address in the paper (not blockers):**
1. **Class & view imbalance:** ASC ≈2.3× more distortion than DESC; report metrics per view and use balanced
   sampling/loss (BCE+Dice or focal).
2. **Patch overlap leakage:** windows were extracted with stride 64 (≈50% overlap, 4×). A pure sample-ID split
   (delivered) can still put overlapping pixels across train/test and inflate metrics. If parent-scene/footprint
   IDs cannot be recovered, quantify overlap and prefer a spatially-blocked / footprint-grouped split for the
   headline numbers; the delivered 70/15/15 by-sample split (seed 42) is the reproducible baseline.
3. **Sampling bias:** patches are landslide-centred mountainous terrain (good for distortion frequency, but not a
   random landscape); state the operating domain.
4. **Provenance on demand:** no DEM or Sentinel product IDs are shipped. The mask is documented and matches the
   Meier-1993 algorithm; budget a small experiment re-deriving masks from SRTM to answer reviewers.
5. **Static geometry:** do not claim temporal generalisation of the geometry mask from pre/post (identical); use
   post-event imagery only as an intensity-robustness check.

**OPERA / self-build route:** not required. Return to it only if (a) you need non-landslide-centred or broader
terrain, or (b) reviewers reject the in-band mask and the DEM re-derivation (which is straightforward) is deemed
insufficient.

## 15. Minimal network experiments (design only — nothing trained here)

Backbone: **ConvNeXt-Tiny + FPN is sufficient** (128×128, binary segmentation); include a plain U-Net sanity
baseline. Inputs = VV (dB), optionally add VH; target = binary `band3>0`; loss = BCE+Dice; metrics IoU/Dice/
ROC-AUC/AP, reported per view and per class; the held-out **257 test samples** are fixed by `split_manifest.csv`.

- **Baseline 1 — Single-view:** `I_A → M_A` (and symmetric `I_D → M_D`).
- **Baseline 2 — Dual-view teacher:** `(I_A, I_D) → M_A` (concatenate two VV channels; measure the ceiling the
  opposite view provides).
- **Proposed — Privileged / distillation:** train teacher on `(I_A, I_D)`, distil to a student that at **test
  time sees only `I_A`** (feature/logit mimicking or LUPI-style auxiliary loss). The scientific test is whether
  train-time opposite-view information lifts single-view reliability above Baseline 1, quantified on the
  ASC-bad/DESC-good and ASC-good/DESC-bad strata specifically (where privileged info should matter most).
- Ablations: VV vs VV+VH; binary vs 3-class head; with/without the 35 no-distortion patches; pre-only vs
  pre+post training.

## 16. Delivered artefacts (in `Haiti_SAR_GRSL_Audit/`)

- `metadata/`: `file_inventory.csv` (11,991 rows), `acquisition_metadata.csv` (6852 SAR rows),
  `temporal_pairs.csv`, `global_categorical_summary.json`.
- `qc/tables/`: grid_consistency, ad_independence, leakage_audit, complementarity_statistics,
  crossview_disagreement_by_stratum, geometry_class_percentages, sar_band_stats, registration_offsets,
  registration_masked_offsets, singleview_baseline, sample_completeness, geometry_prepost_repeatability,
  annotation_check, band3_persample_counts.
- `qc/figures/`: Figure A (SAR montage), B (geometry masks), C (complementarity), D (pre/post repeatability),
  E (leakage), F (annotation semantics) + `registration/` (12 checkerboard/edge panels).
- `processed/sample_000000…001712/`: aligned standalone GeoTIFFs `{asc,desc}_{pre,post}_{vv,geom}.tif` +
  `landslide_mask.tif` + `metadata.json` (629 MB; VH/optical kept as source references). Read-back verified
  pixel-identical to source bands with CRS preserved.
- `manifests/`: `dataset_manifest.csv` (all source paths + per-view class counts + split),
  `split_manifest.csv` (train 1199 / val 257 / test 257, by sample, seed 42).
- `scripts/`: `00…10` fully reproducible pipeline; `reports/`: this file + `directory_structure.md`.
