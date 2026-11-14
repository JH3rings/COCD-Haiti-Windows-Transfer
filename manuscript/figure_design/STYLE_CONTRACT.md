# Academic figure visual contract — v3.0

Status: 2026-11-14. Universal rules plus a COCD-specific locked reference. Figure style may be learned from reference papers; scientific components and arrows MUST come from code.

## 1. Only the selected R2 palette
Teacher #B1AFD4; Student #DAE2F5; EPM #F8CDAC; CEW/training #F5B7BE; operations #FDE694; output #E1EFD9; secondary #ECC1C0; backdrops #FAE4D5/#F7EAE9; neutral #E7E6E6. Ink #27232B (or similar near-black). Permitted: tints of these fills for true 2.5D facets, authentic SAR/optical data color. Prohibited: extra decorative palettes, washed-out gray explanatory text, strong glossy effects.

## 2. The lower GPT A/B/C master controls layout
For COCD, preserve the user's approved LOWER image: A left (~58% canvas width), B upper right (~40%), C lower right (~40%). A should show two horizontal U-Net streams with compact training relation and prototype bank; B compact EPM micro-mechanism; C balanced Teacher/Student similarity distribution computation. At normalized image coordinates, measure every region and glyph against the actual master. The former 2400×560 scene's approximate starts are A x=14, B/C x=1422; use measurements rather than hardcoded magic numbers after cropping the approved image.
Preserve panel hierarchy, overall negative space and short labels. Never use the upper full-width figure instead of the selected lower figure.

## 3. Feature-map prism grammar
All feature maps are repeatable 2.5D stacks built as editable FRONT plane, TOP plane, RIGHT side plane and 1–2 restrained back slices. Fronts are solid role fills, not white; no artificial **white strip at left**. All depths share one perspective direction. Don't enlarge the entire face merely to suggest 3D. Keep width/height/depth separately parameterized; current user-approved trajectory is narrow fronts with clearer depth (historical reductions are not hardcoded global rules). Center every prism on its forward-flow baseline and recompute named ports after any size change.
Space and arrows may not invade the next prism. Feature resolution text appears below each block, not overlapping top facets or skip wires. Decoder has extra room for 32/64/128 EPM; encoder remains comparatively compact.

## 4. Port-true arrows, not pictorial guesses
Each block exposes named ports left_mid/right_mid/top/bottom. Primary U-Net flow is a STRAIGHT horizontal solid line between same-y ports, arrowhead centered and ending immediately before next element. If blocks are at different levels, change the layout before drawing. Four genuine skip concatenations use thin orthogonal lanes overhead; shared prototypes have independent links to the three actual EPM inputs. Distinguish residual from shared parameters and training loss by color/legend. CEW originates from 32px stage Teacher and **post-EPM Student** feature, never from Teacher mask. Avoid diagonal arrows, long dotted captions, stray arrowheads, arrows through text and spurious Teacher–Student connections.
The line/symbol check treats arrows and +/× tokens as objects; each needs a logical port and z-order, including the final arrow into a + circle.

## 5. COCD A/B/C data fidelity
A: Teacher 18 channels (12 optical S2 plus 6 SAR/terrain), Student 6 SAR/terrain only, both five encoder levels and four decoder levels. EPM on Student decoder 32px/64px/128px only; GT to BCE+Dice, relation loss training-only. Real sample input, GT and output crops can be embedded as pictures, but do not add giant rectangular white INPUT cards.
B: Q 1×1 -> cosine similarity to K=16 normalized 64-d prototypes -> tau=0.1 softmax -> stage-specific V projection / weighted sum -> zero-init output 1×1 -> residual addition f'=f+correction. Shared prototype bank, but separate Q/V/O per stage. Prototype rows must not overlap text or Value Projection.
C: Teacher and Student 32px outputs -> valid-masked 4×4 regional pooling -> their own normalized tokens -> separate within-model cross-event cosine similarity -> same-event pair masking -> minimum shift separately -> per-anchor independent sort -> squared 1-D Wasserstein. A SINGLE DRAWN matrix must not falsely suggest a shared computed matrix; distinguish T/S paths and label a common central matrix explicitly 'schematic comparison only' if visually required. Real similarity matrix dimension depends on number of valid tokens in the batch, not always 32×32. Min shift + Sort boxes compact; sorting bars labeled illustrative and clearly connected to the appropriate side. Same-event excluded mark must be visible, dark and away from bars.

## 6. Typography: readable and contained
Use a serif compatible with manuscript; Times New Roman when present, with measured font metrics for both SVG and PPT. Three levels: A/B/C title; module/operator label; small tensor/scale/formula. Prefer one to three short words in operation boxes and remove repetitive gray micro-paragraphs. Avoid light-gray text; text should remain high contrast against pastel regions and legible at final journal two-column width (minimum text usually >=6.5pt physical). Give each text glyph a visible internal safe area, not just a non-overflowing text frame. Check four distinct failures: text-text, text-border, text-image, line-text. “Zero shapes outside slide” does NOT mean “no overlap”.
Greek/italic/special math characters must have confirmed installed font coverage in both PPT and SVG. Don't render Unicode blackboard characters as empty squares.

## 7. Precise visual parity validation
Record for each A/B/C region a normalized rectangle, text frames and artwork masks from the chosen GPT image. Generate PPTX and SVG from the SAME scene graph and type data. Render **actual PPTX** and **actual SVG** separately, crop aligned A/B/C, and compare directly to the fixed master:
- A/B/C panel bounds and visual area, negative space, tinted backgrounds;
- perspective, facet width and depth, front area and repetition;
- operation-box width, text occupancy and nearby gutters;
- true line ports, arrowhead centers, crossings and marker shape;
- font weight/contrast and local legibility;
- C panel’s two model-specific token/similarity/sort paths.
Use optional metrics (normalized boundary error, overlap count, font safety, line slope, image difference) only as diagnostics. Report qualitative deviations and repair affected modules in place. Do not replace the style with a new design.

## 8. Release contract
Deliver editable PPTX and corresponding SVG, not a flattened GPT screenshot. Show a screenshot **rendered from the saved PPTX**, not merely the SVG. Keep source script, SHA-256 manifest, master image, audit logs and old approved copies in a backup folder; on Mac Desktop there should be one visible final PPTX/SVG pair. Verify file content after atomic replacement and ask the user to reopen exact path. User alone scores visual quality. Any failure in science, typography, ports, PPT rendering, or provenance blocks the release.
