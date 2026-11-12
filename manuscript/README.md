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

Times-family serif text, white background, dark-gray arrows, muted blue (`#DAE2F5`) for the student, green (`#E1EFD9`) for the teacher, peach (`#FAE4D5`) for prototype operations, and rose (`#ECC1C0`) for the training loss. Colors identify information paths and operations; they do not encode measured performance.
