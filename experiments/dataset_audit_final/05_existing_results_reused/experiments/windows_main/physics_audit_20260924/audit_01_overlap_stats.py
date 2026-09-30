# -*- coding: utf-8 -*-
"""Physics audit pass 1: file semantics, geometry mask verification, GT x geometry
overlap statistics over all 1713 locations, and per-location candidate metrics CSV.

No training. Read-only over data/haiti.
"""
import json
import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(r"D:/deeplearning/COCD_Haiti_Windows_Transfer")
DATA = ROOT / "data" / "haiti"
OUT = ROOT / "experiments" / "windows_main" / "physics_audit_20260924"
OUT.mkdir(parents=True, exist_ok=True)

RAW = {("asc", "pre"): "Pre_event/S1_ASC_20210805",
       ("desc", "pre"): "Pre_event/S1_DESC_20210803",
       ("asc", "post"): "Post_event/S1_ASC_20210817",
       ("desc", "post"): "Post_event/S1_DESC_20210815"}

N = 1713

# accumulate
geom_counts = {o: np.zeros(4, dtype=np.int64) for o in ("asc", "desc")}
gt_counts = np.zeros(4, dtype=np.int64)

# GT x mask contingency per orbit: rows = mask class (0,1,2,3), cols = gt(0,1)
cont = {o: np.zeros((4, 2), dtype=np.int64) for o in ("asc", "desc")}

# pre vs post band3 identity check
prepost_diff_pixels = {o: 0 for o in ("asc", "desc")}
processed_match_fail = {o: 0 for o in ("asc", "desc")}

# spatial extent
xmin, ymin, xmax, ymax = 1e9, 1e9, -1e9, -1e9

# per-location candidate metrics
rows = []

for sid in range(N):
    bands = {}
    for o in ("asc", "desc"):
        for e in ("pre", "post"):
            with rasterio.open(DATA / RAW[(o, e)] / f"{sid}.tif") as s:
                a = s.read()
            bands[(o, e)] = a
            if e == "pre":
                with rasterio.open(DATA / RAW[(o, e)] / f"{sid}.tif") as s:
                    b = s.bounds
                xmin, ymin = min(xmin, b.left), min(ymin, b.bottom)
                xmax, ymax = max(xmax, b.right), max(ymax, b.top)

    gt = np.zeros((128, 128), dtype=np.uint8)
    with rasterio.open(DATA / "processed" / f"sample_{sid:06d}" / "landslide_mask.tif") as s:
        gt_raw = s.read(1)
    gt = (gt_raw > 0).astype(np.uint8)

    geoms = {}
    for o in ("asc", "desc"):
        g_pre = bands[(o, "pre")][2].astype(np.int64)
        g_post = bands[(o, "post")][2].astype(np.int64)
        # clip to {0..3} in case of float artifacts
        g_pre = np.clip(g_pre, 0, 3)
        g_post = np.clip(g_post, 0, 3)
        # processed file match check (pre only; post processed not shipped)
        with rasterio.open(DATA / "processed" / f"sample_{sid:06d}" / f"{o}_pre_geom.tif") as s:
            g_proc = s.read(1).astype(np.int64)
        if not np.array_equal(g_pre, g_proc):
            processed_match_fail[o] += 1
        prepost_diff_pixels[o] += int((g_pre != g_post).sum())
        geoms[o] = g_pre  # audit uses pre mask (as the pipeline does)

        for c in range(4):
            geom_counts[o][c] += int((g_pre == c).sum())
            cont[o][c, 0] += int(((g_pre == c) & (gt == 0)).sum())
            cont[o][c, 1] += int(((g_pre == c) & (gt == 1)).sum())

    for c in range(4):
        gt_counts[c] += int((gt_raw == c).sum())

    vv = {k: v[1] for k, v in bands.items()}  # band2 = VV
    chg = {o: vv[(o, "post")] - vv[(o, "pre")] for o in ("asc", "desc")}

    row = {"sample_id": sid}
    for o in ("asc", "desc"):
        g = geoms[o]
        row[f"{o}_layover_frac"] = float((g == 1).mean())
        row[f"{o}_shadow_frac"] = float((g == 2).mean())
        row[f"{o}_both_frac"] = float((g == 3).mean())
        row[f"{o}_bad_frac"] = float((g > 0).mean())
        m = g > 0
        row[f"{o}_gt_in_bad_frac"] = float(gt[m].mean()) if m.any() else 0.0  # P(GT=1|M>0) local
        row[f"{o}_gt_mask_overlap"] = float((gt[m].sum()) / max(gt.sum(), 1))  # frac of GT in bad
        gm = float(np.abs(chg[o][gt == 1]).mean()) if (gt == 1).any() else np.nan
        bm = float(np.abs(chg[o][gt == 0]).mean())
        row[f"{o}_gt_change_mean"] = gm
        row[f"{o}_bg_change_mean"] = bm
        row[f"{o}_change_contrast"] = gm / bm if bm > 0 else np.nan
        row[f"{o}_gt_change_signed_mean"] = float(chg[o][gt == 1].mean()) if (gt == 1).any() else np.nan
    row["gt_frac"] = float(gt.mean())
    row["bothvalid_frac"] = float(((geoms["asc"] == 0) & (geoms["desc"] == 0)).mean())
    row["gt_in_bothvalid_frac"] = float(
        (gt[(geoms["asc"] == 0) & (geoms["desc"] == 0)]).sum() / max(gt.sum(), 1))
    rows.append(row)

# ---- geometry coding sanity: unique values ----
uniq_report = {}
for o in ("asc", "desc"):
    with rasterio.open(DATA / RAW[(o, "pre")] / "0.tif") as s:
        u = np.unique(s.read(2)[2].astype(np.int64))  # unused var guard
    uniq_report[o] = None

# ---- aggregate statistics ----
res = {"n_locations": N, "extent_lonlat": [xmin, ymin, xmax, ymax],
       "prepost_band3_diff_pixels": prepost_diff_pixels,
       "processed_geom_mismatch_locations": processed_match_fail}

for o in ("asc", "desc"):
    tot = geom_counts[o].sum()
    res[f"{o}_class_counts"] = geom_counts[o].tolist()
    res[f"{o}_class_pct"] = (100.0 * geom_counts[o] / tot).round(4).tolist()

    C = cont[o]
    gt1 = C[:, 1].sum(); gt0 = C[:, 0].sum()
    bad = C[1:].sum(); good = C[0].sum()
    res[f"{o}_contingency_maskclass_x_gt"] = C.tolist()
    res[f"{o}_P_bad_given_gt1"] = float(C[1:, 1].sum() / gt1)
    res[f"{o}_P_bad_given_gt0"] = float(C[1:, 0].sum() / gt0)
    res[f"{o}_P_gt1_given_bad"] = float(C[1:, 1].sum() / bad)
    res[f"{o}_P_gt1_given_good"] = float(C[0, 1].sum() / good)
    # IoU / Dice between GT and M>0
    inter = C[1:, 1].sum()
    union = gt1 + bad - inter
    res[f"{o}_IoU"] = float(inter / union)
    res[f"{o}_Dice"] = float(2 * inter / (gt1 + bad))
    # odds ratio
    a = inter                      # gt1 & bad
    b = gt1 - inter                # gt1 & good
    c = bad - inter                # gt0 & bad
    d = gt0 - c                    # gt0 & good
    res[f"{o}_odds_ratio"] = float((a * d) / (b * c)) if b * c > 0 else None
    res[f"{o}_relative_risk"] = float((a / (a + b)) / (c / (c + d))) if (c + d) > 0 else None
    # per class
    for ci, cname in ((1, "layover"), (2, "shadow"), (3, "both")):
        res[f"{o}_P_gt1_given_{cname}"] = float(C[ci, 1].sum() / max(C[ci].sum(), 1))
        res[f"{o}_frac_of_{cname}_that_is_gt"] = float(C[ci, 1].sum() / max(C[ci].sum(), 1))
    res[f"{o}_P_bad_given_gt1_by_class"] = {
        cname: float(C[ci, 1].sum() / max(gt1, 1)) for ci, cname in ((1, "layover"), (2, "shadow"), (3, "both"))}

res["gt_class_counts_raw"] = gt_counts.tolist()

with open(OUT / "audit_summary.json", "w") as f:
    json.dump(res, f, indent=2)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(OUT / "candidate_metrics.csv", index=False)

print(json.dumps(res, indent=2)[:4000])
print("\nrows:", len(df))
