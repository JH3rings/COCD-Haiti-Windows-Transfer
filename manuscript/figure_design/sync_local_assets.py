#!/usr/bin/env python3
"""Stage user-supplied figure references in a *local clone* of the repo.
Never push controlled publisher PDFs to a public repository without rights.
Usage: python manuscript/figure_design/sync_local_assets.py /path/to/figure_reference_bundle.zip
This copies only the two original screenshots and the openly licensed MDPI PDF
under a private/untracked staging location by default; pass --public to stage
just low-resolution preview screenshots in manuscript/figure_design/assets/.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse,shutil,tempfile
parser=argparse.ArgumentParser()
parser.add_argument("bundle",type=Path)
parser.add_argument("--public",action="store_true",help="stage public low-resolution preview JPEGs only")
args=parser.parse_args()
repo=Path(__file__).resolve().parents[2]
target=repo/"manuscript"/"figure_design"
with ZipFile(args.bundle) as z:
    names={Path(n).name:n for n in z.namelist()}
    if args.public:
        out=target/"assets";out.mkdir(parents=True,exist_ok=True)
        for key,name in [("R1_reference_preview.jpg","R1_video_text_reference_preview.jpg"),("R2_reference_preview.jpg","R2_rs_kvcache_reference_preview.jpg")]:
            if key not in names: raise RuntimeError(f"missing {key}")
            (out/name).write_bytes(z.read(names[key]))
        print("Staged public preview images; review attribution before git add/commit.")
    else:
        out=target/"local_reference_sources";out.mkdir(parents=True,exist_ok=True)
        for key in ("R1_screenshot_original.jpg","R2_screenshot_original.jpg",
                    "R3_InformationFusion_2025.pdf","R4_Jiang_IJGI_2026.pdf"):
            if key not in names:raise RuntimeError(f"missing {key}")
            (out/key).write_bytes(z.read(names[key]))
        (out/".gitignore").write_text("*\n!.gitignore\n",encoding="utf-8")
        print("Extracted four files to LOCAL staging. DO NOT git add full copyrighted PDFs.")
