# Visual-first, science-locked academic figure workflow — v3.0

Status: 2026-11-14. This is the operative workflow. The approved COCD visual target is the **LOWER** three-panel GPT image (A left, B upper right, C lower right). Executable code and main.tex control the model truth; an image model controls appearance only.

## Why v2.1 failed
Previous runs converted a good generated image into a mechanical PPT, enlarged front faces, created tilted feature arrows, lost background and label hierarchy, then claimed success because object count and zero-overflow tests passed. The Mac desktop's same-named PPT and SVG could also be regenerated independently, making file identity ambiguous. These are process failures, not prompts to produce a new decorative concept.

## Source-of-truth order
1. Scientific evidence: pinned GitHub commit, final_method.py, common.py, pr.py, methods.py, run.py, main.tex; record every model node and typed edge.
2. Accepted visual master: the lower A/B/C user-approved GPT composition, plus inspected R1/R2 screenshot figures and actual R3/R4 PDF figure pages.
3. Frozen geometry: one method_graph.json and one layout.json. Stable IDs, named ports, bounding boxes, text boxes, z-order, edge waypoints.
4. Final reality: **native PPTX as actually rendered**, plus SVG built from the same scene.
5. Release evidence: hashes of the exact Mac files, rendered screenshots, source commit and saved accepted backup.

## Step 0 — protect and pin
Read originals; record SHA of main.tex and original method.svg/draw_method.py; never overwrite them. Work on a branch and in a staging folder. Preserve the user's approved master and previous accepted revision as immutable files.

## Step 1 — science graph (hard Gate S)
Compile method_graph.json from *forward* implementations and loss, not guessed module titles. For each node record id, tensors, stage, train/infer status, source lines. For each edge record source/target ports and one of: feature, skip_concat, residual, shared_parameter, segmentation_supervision, relation_supervision. Freeze the graph before visual work. No invented teacher-to-student feature arrows; teacher only provides 32px relation supervision. Input provenance, GT and predictions must agree.

## Step 2 — source-pixel inspection (hard Gate R)
Open both R1 and R2 **actual figure screenshots** and R3/R4 **actual PDF diagram pages**. Log normalized rectangles and meaning of background tint, feature prisms, symbols and reference labels. R2 supplies the only color palette; R1 dense multilevel organization; R3 honest feature and dual-stream proportions; R4 main/inset/legend hierarchy. Reference algorithms are NOT imported. Measure the user-selected lower A/B/C image, not merely the written descriptions.

## Step 3 — GPT image master (Gate V)
If no approved master exists, generate exactly two strong compositions from the frozen graph and references, compare, and let the user select. Generated text/arrows are **untrusted**. If the user has already selected a master, DO NOT regenerate or replace its style without authorization. Record visual_master.png and its immutable identifier.

## Step 4 — correct science without redesign
Mark every invalid ImageGen node, label and arrow. Correct the smallest affected area of the master; preserve layout masses, negative space, 2.5D style, palette and short text density. Output annotated_visual_master.json and one corrected image. Any geometry change must be traceable to a graph correction.

## Step 5 — geometry/ports (hard Gate G)
Build layout.json with every A/B/C region; each box has its own content bbox and text bbox, named ports, z-order, 2.5D prism width/height/depth. The main U-Net feature-flow ports MUST be at the same y position; fix module positions first, then draw an exactly horizontal straight arrow. Orthogonal skips are reserved in upper rails. Place CEW supervised links outside the ordinary data spine. A feature block size change automatically changes its arrow endpoint. No manually guessed endpoints.

## Step 6 — single-scene editable reconstruction
One renderer-independent scene generates native PPTX objects and matching SVG. Labels, boxes, prisms, matrices, symbols and arrows are editable; verified remote-sensing crops alone can be raster pictures. Do not paste a whole generated figure as a slide picture. Respect the lower master A-left/B-upper-right/C-lower-right composition, all color roles and text occupancy. Deliver ONLY PPTX and SVG (user preference); keep preview files internal.

## Step 7 — ACTUAL dual-format rendering (hard Gate P)
Render PPTX using PowerPoint/Keynote/LibreOffice or a real slide thumbnail renderer; separately render SVG. Test full figure and high-resolution crops A, B, C. Compare side-by-side to the accepted master, not just PPT to SVG. Check panel sizes, massing, cubic depth, font substitution, label:box ratios, 1px white slivers, centered arrowheads, input/output identity and collision. Object count/no overflow is necessary but not sufficient. When editable PPT loses visual quality, STOP and fix reconstruction locally; do NOT start a new image concept.

## Step 8 — anti-regression and Mac release (hard Gate D)
Run scientific graph diff, text-text and text-border tests, line-text/line-shape tests, endpoint checks, out-of-slide checks and physical journal-scale readability. Render from the FINAL saved PPTX again, not a stale memory preview.
Build in staging, backup accepted prior files in hidden/.work folder; save new PPTX/SVG atomically to Desktop. Compute SHA-256 and size from the exact saved bytes; verify the reopened presentation's properties and shape graph. A cloud backup and a locally regenerated file are NOT byte-identical by assumption. If a PowerPoint window remained open, close it fully and reopen the exact Desktop path; a recent-files entry may point elsewhere. Report which renderer was used and what cannot be confirmed.

## Gate summary
- S: science graph matches code and manuscript, correct 18/6 channels, 5+4 U-Net, EPM 32/64/128, CEW 32 only, GT separate.
- R: real R1–R4 pixels inspected; chosen master identified, not replaced.
- G: no sloping main spine, centered ports, no collisions, no clipping, text safety margins.
- P: rendered PPT and SVG checked independently and visually close to the locked master at A/B/C detail scale.
- D: exact Mac paths, checksums, backup, user-confirmed visual quality.

Stop only when gates pass **AND** the user accepts the image. Never self-award 6/7/8 points.

Concise flow: source pin -> typed graph -> reference measurement -> image master -> minimal science correction -> layout/ports -> editable PPT + same-scene SVG -> real render -> comparison -> local repair -> hash/backup/reopen -> user score.
