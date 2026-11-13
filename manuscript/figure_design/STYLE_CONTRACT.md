# Universal Academic Figure Style Contract — v2.1

This contract applies to **any** scientific neural-network figure. It fixes a *visual language*, **not** a teacher–student, U-Net, transformer or diffusion-model layout. Derive geometry anew from the directed computation graph.

## One palette — no hybrid schemes

Use the ten colors extracted from R2; select 4–7 role colors per figure, not all ten mechanically:

| semantic role | fill |
|---|---|
| primary track / teacher or first modality | `#B1AFD4` |
| secondary track / student or second modality | `#DAE2F5` |
| main proposed module | `#F8CDAC` |
| relation / training-only mechanism | `#F5B7BE` |
| emphasized operation | `#FDE694` |
| output / result / segmentation | `#E1EFD9` |
| secondary mechanism | `#ECC1C0` |
| pale functional backdrop | `#FAE4D5`, `#F7EAE9` |
| neutral utilities | `#E7E6E6` |

White negative space; near-black text and connector ink. To correct a washed-out result, first **reduce background area**, improve border/text contrast and differentiate neighboring roles—do **not** introduce a second color palette. Genuine RGB images and scientific heatmaps retain their data colors. Consistent 2.5D prisms may use lighter/darker shades derived from role fill; this is not a new functional palette.

## Layout is a constraint problem, not prose

Prepare a coordinate-based `layout.json` with all modules: id, normalized bounding box [x,y,w,h], class, containing region, label, text box, visual priority, z-order and ports: left/right/top/bottom or named explicit ports. Every edge: source port, target port, scientific meaning, route waypoints, line style, head/tail, z-order and whether it is train-only. Do not choose arrow endpoints by visual guessing.

Select topology-driven grammar:
- Chain / sequence: primary left-to-right or top-to-bottom spine.
- Parallel branches: aligned lanes, shared stages, controlled merge.
- Encoder–decoder: compression/extraction arc with corresponding skip alignment.
- Recurrence / diffusion: cycle or time axis where needed.
- Multiscale pyramid: rank/scale hierarchy.
- Teacher–student: two truthful paths + explicitly routed supervision, NOT arbitrary matching arrows.
- Multi-method figure: main pipeline plus one or two detail insets.
Do not force the layout used by one reference paper onto a different method.

## Spatial quality and alignment

- Group elements in real semantic regions. A colored backdrop is useful only for a meaningful group; never remove all backdrops and never fill every gap with tint.
- Use grid, common baselines, edge alignment, equal-size rules for equal-stage boxes and consistent gutters; region bounds fit contents closely with controlled inner padding.
- Give the main flow a clear first reading; reserve side lanes for skip, shared weights or losses.
- Avoid unexplained wide gaps and useless sprawling horizontal arrows; compress standard repeated operations, *not* the method's novel mechanism.
- Long arrows are permitted when they express a genuine cross-area link; do not use line thickness simply because an arrow is long.
- For a large method, show main structure at L1, core mechanisms at L2 and a few indispensable parameters at L3. Do not repeat all low-level projections in L1 and L2.

## Lines, frames, arrows — semantic syntax

| relation | default display |
|---|---|
| primary feature/data flow | dark solid directional arrow, slightly stronger line |
| short internal operation | thin solid directional arrow |
| skip/identity | fine orthogonal solid route, clean elbows |
| cross-branch training supervision | colored supervised route, named, never fake a forward feature-transfer arrow |
| shared parameter bank | a shared source with independent connectors to actual consumers |
| group outline | fine solid border or light tint |
| training-only area / optional subgraph | dashed *boundary* only if semantics and legend support it |
| a scientific comparison/association | distinguish from data flow, with label or legend |

Never insert decorative dashed arrows. For every connector verify: exact origin/destination ports, direction, no line through unrelated shapes, no ambiguous crossings, visible arrowhead. Use explicit waypoints, edge/lane reservation and rerouting—not hand-drawn random polyline guessing.

## Shapes and typography

- Feature maps: repeatable 2.5D stacked prisms with consistent top/side angles, solid front, modest depth and no strong shadow or glossy effects.
- Weights, tokens, affinity: meaningful grids, matrices, slices or compact columns; schematic numbers must be labeled as schematic.
- Operation blocks: compact rectangle, limited rounding; not all elements need their own box.
- Equation symbols: +, ×, ⊙, cat, pool and normalized arrows only if actual model supports them.
- Real input/output figures: same sample, verified label and sensible crops, no synthetic result disguised as experiment.
- Text: Times-family serif if compatible with target journal; headings 9–11 pt, modules 7.5–9 pt, secondary >=6.5 pt **at final paper physical size** (recommended ranges, not source-paper measurements).
- Standard label 1–3 technical words; no long sentence inside a box; novel modules can have a concise abbreviation and a small meaningful caption. Prefer mathematically faithful arrows/symbols to explanatory prose.
- Distinguish module title, operator name, tensor scale with three coherent type levels.

## Hard gates (fail regardless of aesthetic grade)
1. Every module and edge matches the actual code/paper; teacher freeze and inference-only paths are accurate.
2. Real inputs, GT and prediction labels agree with the actual stored sample.
3. Text is correct and legible at journal column width; no misspelled ImageGen words transferred blindly.
4. Editable PPT objects exist for labels, blocks, arrows and matrices; image files may be embedded for real remote-sensing data.
5. Text does not collide, connectors do not pass through unrelated modules, all objects stay on slide.

## Preferred measurable soft checks
Object overlap, crossings, edge length, region compactness, alignment gaps, lower-priority label count, color consistency, rendered size and parity to a chosen image-generated art-direction master. Numerical thresholds are **checks**, not guarantees of beauty.

## Quality bar
The end-user target is **6–7/10** for publishable usefulness. Earlier code-generated V4/V5 ~4–4.5/10; image concept ~6/10. No one may self-certify a target subjective grade. Favor restoring the image master’s background, massing, spatial rhythm and feature-prism depth when the PPT reconstruction looks more mechanical. 
