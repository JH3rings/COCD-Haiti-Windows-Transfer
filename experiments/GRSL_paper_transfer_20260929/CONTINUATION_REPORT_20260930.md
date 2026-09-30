# GRSL continuation delivery (2026-09-30)

- Formal source: `manuscript/main.tex`; project PDF: `preview/GRSL_updated_20260930.pdf` (5 pages).
- Main comparison preserves approximately five representative baselines and Ours. Historical DIS2 87.33% AUPRC / 61.77% IoU is unchanged; unavailable region scores remain blank.
- New inference controls: CG0 paired, fixed center, random offsets, no-search bypass, global counter derangement, frozen self context; independently trained SG0 re-evaluated. S2 learned, center, random, and no-search are also measured.
- Each control has Overall, TargetValid, BothValid, G10 AUPRC/IoU/F1 and saved full probabilities. G01 is retained in CSV as an additional region.
- Current CG0 alignment replaces the older CG10-derived plot. The statistic and weighting definitions are written in the source protocol and manuscript.
- Three native scene figures plus a genuine unmasked G10 detail are provided from recorded probabilities and original SAR; full scene layers, IDs, crop coordinates, and shared display ranges are saved.
- Actual Teacher/Student sampling and weights are saved for every candidate at a selected query. Offset-head parameter symmetry is verified: three learned points coincide per level in both current checkpoints. No sampling locations are visually jittered.
- Full editable network is preserved with TGD off in CG0 and shared Student memory labels. PPTX package checks passed and it was rendered through PowerPoint; native original objects were retained to avoid importer style loss.
- The historical weight curve and standalone alignment plot remain in figure resources and evidence; they are omitted from the five-page main paper. No current weight scan is claimed.

## Matched current loss controls

The new Seg/SegP/SegE arms use canonical initialization, seed 42, batch 16, optimizer, pixel support, two-stage calibration/schedule, stop-gradient KD, and frozen CG0. Removed component coefficients are not reassigned. The combined row is the existing current S2-2Stage checkpoint, not relabeled historical S2.

| Arm | Completion | Best epoch | Overall AUPRC | TargetValid AUPRC | G10 IoU |
|---|---|---:|---:|---:|---:|
| Seg | COMPLETE; patience 2 | 30 | 88.6897% | 88.2783% | 65.1003% |
| SegP | COMPLETE; patience 2 | 38 | 88.9613% | 88.5525% | 65.4371% |
| SegE | COMPLETE; patience 2 | 30 | 88.6826% | 88.2782% | 64.9085% |

E alone is essentially unchanged from the new segmentation-only control (88.68% versus 88.69% Overall AUPRC); P gives 88.96%, and the frozen combined model gives 89.16%. These are single-seed component-removal results, not statistical superiority.

## Source correspondence and reproduction

- `new_evidence/20260930/paper_source_correspondence.csv`: table/figure to model, checkpoint, code, and original file; recorded SHA256 and existence.
- `new_evidence/20260930/version_boundaries.md`: current/historical separation and claim limits.
- `new_evidence/20260930/figure_sources.json`, `current_query_samples.csv`, and native `layers_*.npz`: editable scientific figure inputs.
- `scripts/current_mechanisms.py`, `train_matched_loss.py`, `build_figures.py`, `render_current_search.py`: computations and plots.
- `scripts/finalize_delivery.py`: completed-row update, compact layout, compilation, and page rendering. `scripts/project_compiler/` is a project-scoped portable compiler, not a replacement for the native editor.
- Primary Hinton probabilities were recovered from retained weights and agree with recorded metrics. Boehm cache order was checked against actual labels, geometry, and orbit arrays. Original cache order and new canonical batch order differ; join using sample ID and orbit.

## Minimal missing originals

Only historical DIS2 regional recomputation still needs originals: the probability arrays corresponding to the 87.33% / 61.77% record, plus sample ID / orbit order and original label / geometry identity. A single NPZ with p, y, target_geometry, counter_geometry, sample_id, orbit and model/source metadata is sufficient. Alternatively the exact historical checkpoint and preprocessing/source metadata can regenerate it. Current low-score DIS2-Port and recovered different-score re-test are not substitutes.

## Claim that needs the strongest further support

The necessity of nonzero adaptive spatial sampling is not established. Fixed-center S2 is nearly unchanged, and three learned points collapse to one per level. Current evidence supports counter-guidance sensitivity and scale/weight allocation, not a claim that diverse spatial sampling causes the gains. Matched loss controls must be interpreted as recorded, including any null or negative result; a single run does not establish statistical superiority.

## Submission limits

This is a complete evidence-oriented revised draft, not a declaration of submission readiness. The formal PDF is strictly limited to five pages, including references, tables, and figures, with an enforced pre-delivery assertion. Author/affiliation/funding information and manual scientific image adjudication remain outside this computational update. Source checks verify resources and labels, not every bibliographic statement.

## Publication-reader revision

The formal paper remains five pages. The task schematic starts page 2; only the detailed architecture spans both columns. All 23 bibliography entries are now cited in substantive related work and dataset discussion. The former PDF contained 16 actual references. All six core equations and Teacher/Student paths are retained.

The masked comparison shows both orbit backscatter images, both distortion regions, landslide reference, support, and five methods. The full-pixel comparison retains distorted predictions and shows five methods plus four matched objective variants. Table II now has seven rows in two groups. Published names are consistent across text, figures, and tables. Internal model/sample/region codes are confined to provenance files. All measured controls and four-region numerical records remain in the evidence CSVs.

Retained online-distillation branch checkpoints reproduce the cached probabilities exactly, with zero threshold prediction disagreements. Recomputed summary metrics differ slightly from the recorded source summary, but all displayed table cells agree at two-decimal percent precision; recorded main-table values were preserved. Sources: reader_figure_sources.json, online_prediction_verification.json, and reader_revision_manifest.json.

The bibliography was restored from the existing 23-entry library. Selected metadata and method summaries were checked on primary publisher pages; this is not a claim that all article full texts were audited. Human scientific image adjudication remains pending.
