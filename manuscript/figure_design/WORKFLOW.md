# Visual-first + Scientific-Truth Figure Workflow — v2.1

## Why the previous V4/V5 failed

The GPT image design concept was judged visually better (~6/10) than the later editable rendering (~4–4.5/10). The step `image → scientific correction → editable PPT` lost subtle layering, group backdrops, compact composition, typography and rhythm. **Never treat object count, no-overflow or PPT export success as a substitute for visual quality.** Worse, the image model hallucinated labels (e.g. interpreting EPM as "Event Prototype Mining" or CEW as "Cross-Event Weighting") and fabricated some cross-network connections. The transformation must preserve visual intent while replacing scientifically invalid content and edges.

## Required artifacts and gates

### 0. Protect source of truth
Save pristine `main.tex`, original diagram and generator. Start in a new directory/branch; do not overwrite `manuscript/source/method.svg` until accepted.

### 1. Scientific topology compilation (science-first; no styling)
Read actual code, paper and model configurations. Produce `method_graph.json` with stable node IDs, ports, typed edges (data/skip/shared-weights/supervision/loss), train/inference flags, input/GT/output provenance and optional decompositions. Check against source lines. If ambiguous, record unresolved and **do not invent**. Freeze this graph.

**Gate S:** no invented module, no wrong student/teacher flow, no conflation of GT and model predictions, no CEW direct teacher mask transfer.

### 2. Reference-style calibration (visual grammar, not copied algorithm)
Look at screenshots R1–R2 and paper figure pages R3–R4, not only paragraphs. Create `style_tokens.json` and visual analysis: figure size, normalized region rectangles, fonts (measured or estimated), color roles, prisms, arrows, grouping, label density, visual hierarchy. Use only the R2 ten-color palette. Do not borrow specific video, Mamba, KV-cache or cloud-removal operators unless the actual target method uses them.

### 3. Generate image concepts — visual engine
Supply *all four* sources, a simplified **locked model graph** and concise list of scientific invariants to ImageGen. Produce **two alternative composed image drafts**, not V1/V5 as a visual template. Favor pastel functional backgrounds, expressive 2.5D feature stacks, well-defined inset mechanisms, dense structured grids and real data panels. Ask explicitly for short labels rather than long paragraphs.

Treat generated text and arrows as unverified. The goal is visual composition, not final science. Do not claim ImageGen output is vector or editable.

**Gate V:** human or vision-review chooses the strongest; assess relative to reference figures. If it is below ~6/10, iterate in image phase, not PPT phase.

### 4. Correct the chosen concept without destroying its look
Overlay the locked `method_graph.json` and annotate the image concept: valid modules, invalid labels/edges, each group bounding box, region backdrops, connection lanes, text zones, line weights, prism perspective. Make a corrected **visual master**. If a science correction changes geometry, reflow the smallest affected subgraph; keep unaffected colors, spacing, backgrounds and grouping.

Deliver `visual_master.png` plus `visual_master_annotations.json`, not just a prose description.

### 5. Geometry specification
Create `layout.json`: canvas, relative region boxes, node bounding rectangles, exact named ports, connector polylines/waypoints, z-order, text boxes, font metrics. This is the contract for actual PPT generation. Route each edge to a scientific port. Optimize crossings, collisions, region compactness and label wrapping before drawing.

### 6. Editable PowerPoint reconstruction
Use native PPT shapes for rectangles, prisms, matrices, arrows and text; real image crops stay pictures. Group by semantics. Do **not** place a flat concept image as the final single slide picture, or replace the approved art direction with an unrelated geometric drawing. Build SVG/PDF/PNG previews from the exact same PPT master. Keep generator code and assets.

### 7. Two-sided QA and regression
**Science audit:** compare each rendered node/edge label and direction with `method_graph.json`. Verify train-only vs inference; sample image provenance; no hallucination.

**Geometry QA:** check text box overlap, object overflow, line crossing module interiors, port endpoint correctness, type sizes at target journal width and missing fonts.

**Visual parity QA:** render the PPT and compare side by side / semi-transparent overlay against `visual_master.png`, at equal aspect and dimensions. Inspect region colors/area, alignment, prism depth, negative space, typography and visual weight. Do not use per-pixel comparison blindly (AI render and native type differ). Document every qualitative loss and fix **locally**. The subjective goal is no dramatic design downgrade relative to accepted image concept.

**Stop condition:** science and geometry gates pass and the user accepts visual quality (target 6–7/10). Do not self-assign a subjective pass.

### 8. Publication hand-off
Provide `.pptx` (primary, editable), `.svg`, `.pdf`, `.png`, generator source, input provenance, science audit and before/after comparison. No hidden unverified data. Do not alter main manuscript figure without explicit instruction.

## Short operational order

`inspect → freeze graph → analyze references → ImageGen concepts → select → correct concept visually → layout/route → editable PPT → render → compare → local fix → accept`

If ImageGen concept beats PPT by >=1 point according to the user, **stop and fix the reconstruction stage**. Do not iterate an unrelated new design or add more textual Prompt instructions.
