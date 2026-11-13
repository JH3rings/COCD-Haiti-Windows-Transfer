# Original user-uploaded source checksums (local/private transfer)

Checked on 2026-11-13 against the four uploaded files in the current conversation. A local ZIP named `COCD_Figure_Reference_Originals_Private.zip` contains these byte-identical originals plus `SHA256_MANIFEST.json`. **It is a private/local delivery, not a public GitHub attachment.**

| Local filename | Original size | SHA-256 |
|---|---:|---|
| `R1_screenshot_original.jpg` | 328,188 B | `24a3f61b8348e19d418cc5f4117c8f3541f6af935e0069314a66e5398dfeb84b` |
| `R2_screenshot_original.jpg` | 423,295 B | `d19dfa430fc6ea6ff387b687268ef5dc8fd5395927985e5a74ac54fbae1e8651` |
| `R3_InformationFusion_2025.pdf` | 4,298,679 B | `ca353fe6ca86c9ff406bedbadf4933e5ec71631fa1b53c3e2e33a3df3a1ebe1f` |
| `R4_Jiang_IJGI_2026.pdf` | 6,743,632 B | `1fa6f55c682518883a6887948a2984462006a06fce599a38f4ad1bccd9204393` |

The two images under `assets/R1_*.jpg` and `assets/R2_*.jpg` in this public repo are **cropped, resized study previews**, not byte-identical uploads. R3's public excerpt is **not** the complete copyrighted article; it is only a short figure-study extract. R4 is a compressed complete openly licensed version; it does not match the original upload byte-for-byte.

## Local setup

1. Download the private ZIP supplied in the ChatGPT conversation.
2. Place it on the same computer as a clone of this repo.
3. Run: `python manuscript/figure_design/sync_local_assets.py /path/to/COCD_Figure_Reference_Originals_Private.zip`.
4. Use the four originals for high-resolution visual analysis. Do not `git add` the local reference directory or publicly redistribute R3's full PDF without permission.
5. Verify exact originals against the SHA-256 values above if transfer integrity is important.

The PDF author links, page references, palette ownership, and transfer limitations are documented in [REFERENCES.md](REFERENCES.md). All scientific truth must continue to come from the current F_SSP source code and paper.
