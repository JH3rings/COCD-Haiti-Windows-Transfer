"""Add an explicitly exploratory raster GT-boundary / SAR-change co-location proxy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/haiti"
OUT = Path(__file__).resolve().parent
N = 1713
ORBITS = {
    "asc": ("Pre_event/S1_ASC_20210805", "Post_event/S1_ASC_20210817"),
    "desc": ("Pre_event/S1_DESC_20210803", "Post_event/S1_DESC_20210815"),
}

def read_stack(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

rows = []
for sid in range(N):
    gt_path = DATA / "processed" / f"sample_{sid:06d}" / "landslide_mask.tif"
    with rasterio.open(gt_path) as src:
        gt = src.read(1) > 0
    geo = {o: read_stack(DATA / pre / f"{sid}.tif")[2].astype(np.uint8) for o, (pre, _) in ORBITS.items()}
    bothvalid = (geo["asc"] == 0) & (geo["desc"] == 0)
    boundary = gt & ~ndimage.binary_erosion(gt, structure=np.ones((3, 3), dtype=bool), border_value=0)
    interior = gt & ~boundary
    joint_orbit = []
    for orbit, (pre, post) in ORBITS.items():
        a = read_stack(DATA / pre / f"{sid}.tif")
        b = read_stack(DATA / post / f"{sid}.tif")
        # Stored stack convention in the local data reader: band 1 VH, band 2 VV.
        dvv = np.abs(b[1] - a[1])
        dvh = np.abs(b[0] - a[0])
        joint_orbit.append(np.hypot(dvv, dvh))
    joint_change = 0.5 * (joint_orbit[0] + joint_orbit[1])
    masks = {
        "boundary": boundary & bothvalid,
        "interior": interior & bothvalid,
        "background": (~gt) & bothvalid,
    }
    means = {k: float(joint_change[m].mean()) if m.any() else None for k, m in masks.items()}
    n_boundary = int(masks["boundary"].sum())
    bg = means["background"]
    bd = means["boundary"]
    rows.append({
        "sample_id": sid,
        "gt_boundary_bothvalid_pixels": n_boundary,
        "gt_boundary_joint_change_bothvalid": bd,
        "gt_interior_joint_change_bothvalid": means["interior"],
        "bothvalid_background_joint_change": bg,
        "gt_boundary_minus_background_joint_change": (bd - bg) if bd is not None and bg is not None else None,
        "gt_boundary_over_background_joint_change": (bd / bg) if bd is not None and bg is not None and bg > 0 else None,
    })

new = pd.DataFrame(rows)
new.to_csv(OUT / "gt_boundary_change_alignment.csv", index=False)

patch_path = OUT / "dataset_patch_audit.csv"
patches = pd.read_csv(patch_path)
patches = patches.drop(columns=[c for c in new.columns if c != "sample_id" and c in patches.columns], errors="ignore")
patches = patches.merge(new, on="sample_id", how="left", validate="one_to_one")
patches.to_csv(patch_path, index=False)

anomaly_path = OUT / "anomaly_topk.csv"
anomaly = pd.read_csv(anomaly_path)
anomaly = anomaly.drop(columns=[c for c in new.columns if c != "sample_id" and c in anomaly.columns], errors="ignore")
anomaly = anomaly.merge(new, on="sample_id", how="left", validate="many_to_one")
anomaly.to_csv(anomaly_path, index=False)

valid = new.dropna(subset=["gt_boundary_minus_background_joint_change"])
positive = valid["gt_boundary_minus_background_joint_change"]
summary_path = OUT / "audit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["gt_boundary_change_alignment"] = {
    "method": "inside BothValid: 1-pixel inner GT contour = GT minus 3x3 binary erosion; average hypot(|ΔVV|,|ΔVH|) over ASC and DESC; compare contour with all BothValid background",
    "n_patches_with_valid_boundary_and_background": int(len(valid)),
    "mean_boundary_minus_background": float(positive.mean()),
    "median_boundary_minus_background": float(positive.median()),
    "fraction_patches_boundary_change_exceeds_background": float((positive > 0).mean()),
    "mean_boundary_to_background_ratio": float(valid["gt_boundary_over_background_joint_change"].mean()),
    "limitation": "rasterized GT contour proxy only; no original vectors or optical source images locally; not a boundary registration validation and not terrain matched",
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary["gt_boundary_change_alignment"], indent=2))
