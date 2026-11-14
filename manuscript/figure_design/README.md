# Figure Design System — COCD / DLR GRSL

**Status (2026-11-13): documentation and reference-source index. This folder does not replace the validated manuscript figure.** The accepted scientific truth remains `manuscript/main.tex` and the F_SSP implementation, not any generated image or earlier V1–V5 render.

## Start here (for a new AI session)

**Current main execution contract: [MASTER_FIGURE_PROMPT_V3.md](MASTER_FIGURE_PROMPT_V3.md).** Its storyboard/wireframe gate and local high-fidelity PPT component test supersede the older V2.1 execution order. Use this universal contract for *any new network or paper*; topology is never copied from COCD.

1. Read [MASTER_FIGURE_PROMPT_V3.md](MASTER_FIGURE_PROMPT_V3.md) first, then [WORKFLOW.md](WORKFLOW.md) for legacy background. This is a **general-purpose** method: styles are reusable; geometry is derived anew from each model's topology.
2. Read [REFERENCES.md](REFERENCES.md) for the four supplied reference sources (two screenshot collages containing four figures, two scientific PDF papers), the authoritative paper links, and what to borrow from each.
3. Read [STYLE_CONTRACT.md](STYLE_CONTRACT.md) for the **one selected palette**, spatial composition, typography, border, line, arrow, and evaluation rules.
4. For this specific DLR landslide paper, read [COCD_SCIENCE_AUDIT.md](COCD_SCIENCE_AUDIT.md), then inspect the actual source files linked there.
5. Run the **visual-first, science-gated** sequence: source audit → image design candidate(s) → science correction → spatial lock → editable PPT → rasterized parity review → regression check.

## Current live paper (do not overwrite without explicit request)
- `../main.tex` — manuscript and figure caption.
- `../source/method.svg` — original editable diagram.
- `../tools/draw_method.py` — original SVG generation logic.
- `../figures_dlr/method.pdf` and `../figures_dlr/method_preview.png` — submitted figure artifacts.
- `../../experiments/dlr_official6_faithful_20261110/final_method.py`, `common.py`, `pr.py`, `methods.py`, `run.py` — executable model, loss and teacher sources.
- `../figures_dlr/qualitative.png` — real visual results; sample IDs and panel identity must be verified before cropping.

## Visual goal
The user rated the editable V4/V5 results about **4–4.5/10**, while judging the AI-image design candidate around **6/10**. Do not assume a successful PPT export equals visual success. Prioritize visual fidelity in the PPT rebuilding phase; user target is an actual **6–7/10** paper-usable result. Do not claim this score without a user assessment.

## Source and distribution note
The original reference screenshots and full PDFs were uploaded in the ChatGPT conversation. The source register in REFERENCES.md provides verifiable paper URLs and page/figure coordinates. **A public GitHub repository must not redistribute publisher-controlled full PDFs without permission.** In particular, the Elsevier Information Fusion paper is publisher-copyrighted. Keep the originals in a local/private asset folder when licensed; refer to the official DOI in public documentation. A separate asset transfer package may accompany this directory when the conversation's uploaded files are available.

## File roles
- `MASTER_FIGURE_PROMPT_V3.md` — **current general-purpose figure creation contract**: science graph, storyboard, wireframe, references, image design, component-level PPT fidelity test, full reconstruction and QA.
- `REFERENCES.md` — four source groups, paper DOI and exact style usage.
- `STYLE_CONTRACT.md` — reusable, topology-independent visual grammar.
- `WORKFLOW.md` — two-engine workflow + hard gates + automatic/manual checks.
- `COCD_SCIENCE_AUDIT.md` — project-specific F_SSP diagram invariants.
- `NEW_CHAT_START.md` — short prompt to start a fresh assistant/agent on this repository.
- `ORIGINAL_SOURCE_CHECKSUMS.md` — exact hash/size inventory for four user-uploaded originals and the private local transfer ZIP.

Do not ingest old V1–V5 visual concepts as scientific ground truth; some image-generated diagrams mislabeled EPM/CEW and fabricated direct teacher-to-student skip connections.

## Reference assets now included in this directory

- [Screenshot inspiration R1](assets/R1_video_text_reference_preview.jpg) (cropped preview)
- [Screenshot inspiration R2](assets/R2_rs_kvcache_reference_preview.jpg) (cropped preview)
- [Single palette](assets/R2_palette.svg)
- [Information Fusion annotated figure-study extract](assets/R3_InformationFusion_figure_study_excerpt.pdf) (**not** the copyrighted full publisher PDF)
- [Jiang IJGI 2026 CC BY article, compressed full PDF](assets/R4_Jiang_IJGI_CC_BY_compact.pdf)

The original unmodified user uploads are packaged privately for local agents. If a new agent needs small text in the screenshots, follow the verified paper DOI links and inspect the publisher PDF rather than relying on reduced screenshot pixels.
