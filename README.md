# Cross-event optical relation distillation for SAR landslide mapping

Optical imagery is used **only during training** to improve a landslide mapper
that receives SAR backscatter and terrain at inference (DLR Sentinel-1/2
benchmark, official split).

This is the current manuscript and experiment repository. The latest Haiti paper is preserved separately in [cocd-haiti-grsl-paper](https://github.com/JH3rings/cocd-haiti-grsl-paper).

- **Paper:** [manuscript/main.tex](manuscript/main.tex).
- **Five-page PDF:** [manuscript/GRSL_final.pdf](manuscript/GRSL_final.pdf).
- **Editable diagram:** [manuscript/source/method.svg](manuscript/source/method.svg).
- **Build and figure style:** [manuscript/README.md](manuscript/README.md).
- **Experiment records:** [experiments/dlr_official6_faithful_20261110](experiments/dlr_official6_faithful_20261110).

## Problem

Rapid landslide mapping may require SAR when optical coverage is unavailable. The student uses pre-/post-event SAR, elevation, and slope; an optical–SAR–terrain teacher supplies training-time guidance. The task is evaluated on the official DLR split, which contains both independent and dependent test events.

The method matches cross-event similarity distributions without forcing correspondence between individual partner regions. The observed association between partner agreement and test IoU is diagnostic evidence, not proof that pair matching is generally ineffective or that event memorization causes the gap.

## Method (`final_method.py`, arm `F_SSP`)

| Part | Where | What it does |
|---|---|---|
| Scale- and event-shared prototypes (structure, deployed) | After U-Net decoder stages 2–4 (feature maps 32/64/128 px) | One bank of K = 16 learnable prototypes shared by all events and all three stages. Each pixel is refined by its soft assignment to the prototypes: `f ← f + W_o Σ_k softmax_k(cos(W_q f, P_k)/τ) W_v normalize(P_k)`. `W_o` is zero-initialised, so training starts from the plain U-Net. |
| Cross-event Wasserstein relation distillation, L_CEW (loss, training only) | Stage 2 (32×32 map); 4×4 regions of 320 m per patch | For each region, teacher and student each compute its similarities to all regions of **other** events. Each side sorts its own list, and the loss is the squared 1-D Wasserstein distance between the two sorted lists. Anchor correspondence is retained; partner correspondence is relaxed. There is no top-K and no solver. |

The total loss is `BCE + Dice + 30 · L_CEW`.

Deployment uses SAR + DEM + slope only and costs +0.14 M parameters and
+0.36 GMACs (about 3% more MACs).

## Setting

- **Data:** the official DLR split, 355 / 88 / 120 patches of 128×128 pixels at
  10 m. The 120 test patches are 44 independent-event patches (ind44) and 76
  dependent-event patches (spt76).
- **Inputs:**
  - Student: PRE1/POST1 γ⁰ VV/VH, DEM and slope (6 channels).
  - Teacher: the same 6 channels plus Sentinel-2 PRE1/POST1 B2/B3/B4/B8/B11/B12 (18 channels).
- **Training:**
  - BCE + soft Dice, AdamW at 1e-4 with weight decay 0.01, 200 epochs;
  - validation IoU every 20 epochs selects the checkpoint;
  - every model is tested once after it is frozen, with seed 42.

## Results (test, %, seed 42)

| Model | AP | IoU | F1 | IoU ind44 | IoU spt76 |
|---|---:|---:|---:|---:|---:|
| U-Net (B0) | 72.91 | 52.20 | 68.59 | 63.24 | 46.03 |
| UNet++ (ResNet-34) | 65.96 | 38.69 | 55.79 | 39.61 | 38.16 |
| DeepLabv3 (ResNet-34) | 71.13 | 44.20 | 61.30 | 54.76 | 38.30 |
| Vanilla KD / AT / Multi-to-Uni / ACAM-KD | | 53.45 / 53.12 / 53.28 / 54.59 | | | |
| PKDOT | 74.75 | 55.74 | 71.58 | 70.52 | 47.80 |
| DIS2 | 75.85 | 55.97 | 71.77 | 69.71 | 48.06 |
| **Ours (F_SSP)** | **77.59** | **56.45** | **72.17** | 69.46 | **49.88** |
| Optical–SAR teacher (optical at inference) | 85.09 | 70.38 | 82.62 | 81.90 | 64.88 |

Ablations and sensitivity (IoU):

- **Components:**
  - structure only: 52.07;
  - L_CEW only: 55.57.
- **Stage of L_CEW:**
  - 16 px: 55.57;
  - **32 px: 56.45**;
  - 64 px: 55.18;
  - 128 px: 52.44;
  - all three: 54.31.
- **λ:**
  - 10: 54.06;
  - **30: 56.45**;
  - 100: 52.18.
- **K:**
  - 8: 54.91;
  - **16: 56.45**;
  - 32: 54.99.

Event-averaged F1, as reported in the DLR paper: Ours 59.1, U-Net 58.8, DIS2
62.8. Ours is weak on the small Irish peat slides, which the teacher also
misses.

All results are single runs. An exact rerun of one configuration changed IoU by
0.65 points; single-run differences should therefore be interpreted cautiously.

## Run

The data are the DLR S1/S2 reference HDF5 files (Zenodo record 17007637),
listed in `experiments/dlr_unet_study_20261104/split_manifest.csv`
(`h5_path`). They are not included in this repository.

```powershell
pip install -r requirements.txt
cd experiments/dlr_official6_faithful_20261110
python run.py --arms B0 T_fusion              # SAR U-Net baseline and the optical-SAR teacher
python final_method.py --check                # identity at init, gradient ratio
python final_method.py --arms F_SSP           # our method (needs the frozen T_fusion)
python final_method.py --test F_SSP           # one test after freezing
python direct.py --arms UNETPP DLV3           # direct segmentation baselines
python figures/make_tables.py                 # paper tables + sensitivity figure
python figures/make_figures.py                # current method figure + analysis figure
python figures/make_qualitative.py            # qualitative figure (needs the HDF5 data)
```

Result records (`config.json`, `history.json`, `selection.json`,
`test_metrics.json`) for every model are committed under `results/`,
`results_rel/` and `results_direct/`. Model weights are not committed.

## Compile the manuscript

With a local TeX distribution and Python `pypdf` installed:

```sh
python manuscript/tools/build_paper.py
```

`draw_method.py` additionally needs PyMuPDF. Rebuilding tables and result plots uses the saved result records; rebuilding input-image comparisons also requires the original DLR HDF5 files. The manuscript revision does not change models, saved metrics, or training outcomes.
