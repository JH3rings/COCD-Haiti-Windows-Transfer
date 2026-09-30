# -*- coding: utf-8 -*-
"""Physics audit pass 2: DEM-based terrain control.

Mosaic Copernicus GLO-30 tiles, compute slope at native 30 m, resample to each
patch grid, and stratify P(GT=1 | M>0) vs P(GT=1 | M=0) by slope bin for ASC
and DESC. Also saves per-patch mean slope for case selection.
"""
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

ROOT = Path(r"D:/deeplearning/COCD_Haiti_Windows_Transfer")
DATA = ROOT / "data" / "haiti"
OUT = ROOT / "experiments" / "windows_main" / "physics_audit_20260924"
DEM_DIR = OUT / "dem"

RAW = {("asc", "pre"): "Pre_event/S1_ASC_20210805",
       ("desc", "pre"): "Pre_event/S1_DESC_20210803"}

# ---- mosaic two tiles (both EPSG:4326, same grid offset by 1 deg lon) ----
tiles = []
for f in ("N18W075.tif", "N18W074.tif"):
    with rasterio.open(DEM_DIR / f) as s:
        tiles.append(s.read(1).astype(np.float32))
        prof_x = s.transform.a
        west = s.bounds.left
dem = np.concatenate(tiles, axis=1)  # west-to-east
top = 19.000138888888888888
H, W = dem.shape
lon0 = -75.00013888888888
lat_top = top
ddeg = 0.0002777777777777778

# ---- slope at native 30 m (Horn-style central difference, per-row metric scaling) ----
lats = lat_top + ddeg * (np.arange(H) + 0.5)          # row lat (deg), y decreases downward
dy_m = 111320.0 * ddeg * np.ones(H)
dx_m = 111320.0 * np.cos(np.deg2rad(lats)) * ddeg

z = dem
zp = np.pad(z, 1, mode="edge")
dzdy = (zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2 * dy_m[:, None])   # dZ/dY (north positive)
dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2 * dx_m[:, None])   # dZ/dX (east positive)
slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))).astype(np.float32)
elev = z

def sample_patch(arr, bounds, out=128):
    """Nearest-neighbour sample of a native-grid array at patch bounds."""
    c = arr.shape[1] * (bounds[0] - lon0) / (ddeg * arr.shape[1] * 1.0)
    # column = (lon - lon0)/ddeg ; row = (lat_top - lat)/ddeg
    x0 = (bounds[0] - lon0) / ddeg
    x1 = (bounds[2] - lon0) / ddeg
    y0 = (lat_top - bounds[3]) / ddeg
    y1 = (lat_top - bounds[1]) / ddeg
    xs = np.clip(np.linspace(x0 + 0.5, x1 - 0.5, out).astype(int), 0, arr.shape[1] - 1)
    ys = np.clip(np.linspace(y0 + 0.5, y1 - 0.5, out).astype(int), 0, arr.shape[0] - 1)
    return arr[np.ix_(ys, xs)]

BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 90)]
NB = len(BINS)

# contingency[orbit, bin, mask01, gt01]
cont = np.zeros((2, NB, 2, 2), dtype=np.int64)
patch_slope_rows = []

N = 1713
for sid in range(N):
    with rasterio.open(DATA / RAW[("asc", "pre")] / f"{sid}.tif") as s:
        b = s.bounds
        vv_pre = s.read(2)
    with rasterio.open(DATA / RAW[("desc", "pre")] / f"{sid}.tif") as s:
        g_a = np.clip(s.read(3)[2].astype(np.int64) if False else 0, 0, 0)  # placeholder no-op
    # read masks and GT
    with rasterio.open(DATA / RAW[("asc", "pre")] / f"{sid}.tif") as s:
        ga = np.clip(s.read(3).astype(np.int64), 0, 3)
    with rasterio.open(DATA / RAW[("desc", "pre")] / f"{sid}.tif") as s:
        gd = np.clip(s.read(3).astype(np.int64), 0, 3)
    with rasterio.open(DATA / "processed" / f"sample_{sid:06d}" / "landslide_mask.tif") as s:
        gt = (s.read(1) > 0).astype(np.uint8)

    sl = sample_patch(slope, (b.left, b.bottom, b.right, b.top))
    el = sample_patch(elev, (b.left, b.bottom, b.right, b.top))
    bi = np.digitize(sl.ravel(), [10, 20, 30, 40])  # 0..4
    gt_f = gt.ravel()
    ga_f = (ga > 0).astype(np.uint8).ravel()
    gd_f = (gd > 0).astype(np.uint8).ravel()

    for oi, mf in ((0, ga_f), (1, gd_f)):
        for k in range(NB):
            sel = bi == k
            cont[oi, k, 0, 0] += int(((mf == 0) & (gt_f == 0) & sel).sum())
            cont[oi, k, 0, 1] += int(((mf == 0) & (gt_f == 1) & sel).sum())
            cont[oi, k, 1, 0] += int(((mf == 1) & (gt_f == 0) & sel).sum())
            cont[oi, k, 1, 1] += int(((mf == 1) & (gt_f == 1) & sel).sum())

    patch_slope_rows.append({
        "sample_id": sid,
        "slope_mean": float(sl.mean()),
        "slope_p90": float(np.percentile(sl, 90)),
        "elev_mean": float(el.mean()),
    })

res = {"bins": [f"{a}-{b}" if b < 90 else f">{a}" for a, b in BINS]}
for oi, oname in ((0, "asc"), (1, "desc")):
    res[oname] = []
    for k in range(NB):
        c = cont[oi, k]
        tot = c.sum()
        if tot == 0:
            res[oname].append({"bin": res["bins"][k], "pixels": 0})
            continue
        p_gt_given_bad = c[1, 1].sum() / max(c[1].sum(), 1)
        p_gt_given_good = c[0, 1].sum() / max(c[0].sum(), 1)
        res[oname].append({
            "bin": res["bins"][k], "pixels": int(tot),
            "bad_frac": float(c[1].sum() / tot),
            "P_gt1_given_M0": float(p_gt_given_good),
            "P_gt1_given_Mbad": float(p_gt_given_bad),
            "risk_ratio": float(p_gt_given_bad / max(p_gt_given_good, 1e-12)),
        })

with open(OUT / "slope_stratified.json", "w") as f:
    json.dump(res, f, indent=2)

import pandas as pd
pd.DataFrame(patch_slope_rows).to_csv(OUT / "patch_slope.csv", index=False)
print(json.dumps(res, indent=2))
