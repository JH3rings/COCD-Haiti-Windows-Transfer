# Cross-Event Optical Relation Distillation for SAR Landslide Mapping

Current DLR optical-guided manuscript, based on the 12 November 2026 release (F_SSP).

- `main.tex`: the only manuscript entry point.
- `GRSL_final.pdf`: compiled five-page paper.
- `figures_dlr/`, `tables_dlr/`, `references_dlr.bib`, `IEEEtran.cls`, `template/`: compilation assets.
- `source/method.svg`: editable vector network diagram.
- `tools/draw_method.py`: regenerate the diagram with PyMuPDF.
- `tools/build_paper.py`: compile with a local TeX distribution and pypdf; check unresolved references, overflow, and the five-page limit.

The folder name is retained so the existing editor continues to use the same file. Its contents describe the DLR task, not the old Haiti task.

Training uses an 18-channel optical–SAR–terrain teacher. Deployment uses a 6-channel SAR–terrain student with shared prototypes; no optical observation is required. All reported numbers come from the supplied experiments. No training was rerun during manuscript editing.

Current code and result records: https://github.com/JH3rings/COCD-Haiti-Windows-Transfer

Preserved Haiti paper: https://github.com/JH3rings/cocd-haiti-grsl-paper

## Figure style (reference palette, high-density original layout)

The method figure keeps the original three-panel vector layout, U-Net stage locations and actual F_SSP computation. The temporary alternative redraw has been abandoned; this branch revises the original `source/method.svg` and `tools/draw_method.py`. The main branch is unchanged.

The **only colored block fills** are from the ten explicitly supplied reference swatches:

| HEX | Recommended diagram role |
| --- | --- |
| `#B1AFD4` | Frozen teacher features and prototype-linked cool side surfaces |
| `#DAE2F5` | Trainable SAR student features |
| `#F8CDAC` | EPM/prototype computation and connectors |
| `#FAE4D5` | Light EPM/prototype-panel background |
| `#F5B7BE` | Training-only cross-event relation loss |
| `#F7EAE9` | Pale teacher/relation-panel background |
| `#E1EFD9` | Ground-truth/segmentation supervision |
| `#ECC1C0` | Prototype/teacher side faces and borders |
| `#FDE694` | Matrix/assignment highlights |
| `#E7E6E6` | Neutral regions and same-event excluded entries |

Dark ink (`#303035`) and white are used for typography and linework. Colors represent branches/functions, not quantitative model outputs. The modified design does **not** introduce decorative dashed boxes or extra network modules.

In panel (a) the feature labels include their true spatial sizes and channels:
`128²/64ch → 64²/128ch → 32²/256ch → 16²/512ch → 8²/1024ch → 16²/512ch → 32²/256ch → 64²/128ch → 128²/64ch`.
The optical–SAR–terrain teacher has 18 channels, is frozen, and is training-only. The student uses 6 channels (pre/post S1 VV/VH and elevation/slope), plus EPMs after the 32/64/128-pixel decoder stages, with a single K=16 prototype bank. The stage-2 (32-pixel) branch alone receives the cross-event relation objective.

**Real examples:** The pre/post optical and SAR thumbnails, the first-case ground-truth thumbnail, and the Teacher/Student prediction-overlay thumbnails all derive from the same published DLR test sample (CN0001) in `figures_dlr/qualitative.pdf`. The prediction thumbnails display error overlays, not plain binary masks: green = TP, red = FP, blue = FN; white = invalid; gray = TN. The sample is illustrative and is **not** intended to compare numerical accuracy by visual inspection. Segmentation masks and the EPM/relation matrices in other diagram regions remain schematic as identified in the caption. The relation stage uses 4×4 valid-pooled regions (at least 75% coverage), masks same-event pairs, and independently sorts teacher/student cross-event similarity sets. No paired match is required.

To regenerate SVG, PDF, and PNG once this branch is checked out:

```bash
cd manuscript
python tools/draw_method.py
```

This reads the already committed `figures_dlr/qualitative.pdf`. The generated files are `source/method.svg`, `figures_dlr/method.pdf`, and `figures_dlr/method_preview.png`. **Only the SVG and renderer source are committed in this design branch; the existing PDF/PNG in the branch still represent the prior rendered figure until regeneration.**
