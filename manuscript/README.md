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

## Figure style

The network figure follows the archived paper's Times-family serif typography and restrained, solid-color feature prisms. Each prism has a lighter top and a darker side; the depth is illustrative, while the spatial sizes beneath stages follow the code. Lavender (`#B1AFD4`) identifies the frozen teacher, blue (`#8EB9DD`) the student, peach (`#E9BE99`) the shared-prototype modules, green (`#B8CDA7`) the segmentation outputs/supervision, and rose (`#B36876`) the training-only relation path. Colors do not encode measured performance.

The final F_SSP code has five encoder resolutions (128, 64, 32, 16, 8), four skip-connected decoder stages (16, 32, 64, 128), EPMs after the last three stages, and relation supervision only after the 32-pixel EPM. The teacher is frozen. All three EPMs share one prototype bank but have separate projections. These are the paths illustrated in the diagram; no model module was added by the drawing revision.

Input thumbnails reuse the first published qualitative example. Similarity matrices and sorted bars in the architecture inset are schematic and are identified as such in the caption. The diagram is an editable SVG, with a vector PDF for LaTeX and a PNG preview.
