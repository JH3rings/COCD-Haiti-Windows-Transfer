"""Extend the all-patch Haiti provenance/geometry audit with terrain matching,
cross-orbit change, temporal-proxy, registration-shift and anomaly analyses.

No data, split, labels, or model files are modified. Existing 2026-09-24 audit
tables are read as historical source evidence; all new output is isolated here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/haiti"
OLD = ROOT / "experiments/windows_main/physics_audit_20260924"
DEM_DIR = OLD / "dem"
OUT = Path(__file__).resolve().parent
N = 1713
ORBITS = ("asc", "desc")
RAW = {
    ("asc", "pre"): "Pre_event/S1_ASC_20210805",
    ("asc", "post"): "Post_event/S1_ASC_20210817",
    ("desc", "pre"): "Pre_event/S1_DESC_20210803",
    ("desc", "post"): "Post_event/S1_DESC_20210815",
}
SEED = 20260927
METRICS = ("dvv_signed", "dvv_abs", "dvh_signed", "dvh_abs", "joint_abs",
           "vv_texture_change", "vh_texture_change", "vv_contrast_change",
           "vh_contrast_change")
SLOPE_EDGES = [10, 20, 30, 40]
ELEV_EDGES = [0, 100, 200, 300, 500, 800, 1200]
ASPECT_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "flat/<3deg"]
ASPECT_EDGES = np.arange(-22.5, 360, 45)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def boot_ci(values, seed=SEED, n=5000):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(x), size=(n, len(x)))
    means = x[ids].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_inference(values, seed=SEED):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n_patches": 0, "mean": None, "median": None, "bootstrap_95ci_mean": None,
                "cohen_dz": None, "wilcoxon_p": None, "sign_flip_p": None}
    try:
        wp = float(stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        wp = 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(x.mean()))
    flips = rng.choice(np.array([-1.0, 1.0]), size=(10000, x.size), replace=True)
    pperm = float((1 + (np.abs((flips * x).mean(axis=1)) >= observed).sum()) / 10001)
    return {"n_patches": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "bootstrap_95ci_mean": boot_ci(x, seed),
            "cohen_dz": float(x.mean() / (x.std(ddof=1) + 1e-12)) if x.size > 1 else None,
            "wilcoxon_p": wp, "sign_flip_p": pperm}


def sample_patch_balanced(pool, arr, mask, rng, per_patch=64):
    idx = np.flatnonzero(mask)
    if idx.size > per_patch:
        idx = rng.choice(idx, per_patch, replace=False)
    if idx.size:
        pool.extend(arr.reshape(-1)[idx].astype(np.float32).tolist())


def calc_local_change(pre, post):
    # Local 5x5 standard-deviation and dynamic-range changes in dB-like scale.
    pre_m = ndimage.uniform_filter(pre, size=5, mode="reflect")
    post_m = ndimage.uniform_filter(post, size=5, mode="reflect")
    pre_v = np.maximum(ndimage.uniform_filter(pre * pre, size=5, mode="reflect") - pre_m * pre_m, 0)
    post_v = np.maximum(ndimage.uniform_filter(post * post, size=5, mode="reflect") - post_m * post_m, 0)
    tchange = np.abs(np.sqrt(post_v) - np.sqrt(pre_v)).astype(np.float32)
    pre_range = ndimage.maximum_filter(pre, size=5, mode="reflect") - ndimage.minimum_filter(pre, size=5, mode="reflect")
    post_range = ndimage.maximum_filter(post, size=5, mode="reflect") - ndimage.minimum_filter(post, size=5, mode="reflect")
    cchange = np.abs(post_range - pre_range).astype(np.float32)
    return tchange, cchange


def load_dem_terrain():
    files = [DEM_DIR / "N18W075.tif", DEM_DIR / "N18W074.tif"]
    arrays, profiles = [], []
    for path in files:
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype(np.float32))
            profiles.append({"transform": src.transform, "crs": str(src.crs), "tags": src.tags(),
                             "bounds": list(src.bounds), "width": src.width, "height": src.height,
                             "nodata": src.nodata, "dtype": src.dtypes[0]})
    west = profiles[0]["transform"]
    dem = np.concatenate(arrays, axis=1)
    # Derivatives use the native angular grid with latitude-dependent east-west metres.
    dlon, dlat = abs(west.a), abs(west.e)
    rows = np.arange(dem.shape[0], dtype=np.float64)
    lats = west.f - (rows + 0.5) * dlat
    dx = (111320.0 * np.cos(np.deg2rad(lats)) * dlon)[:, None]
    dy = 111320.0 * dlat
    dz_south = np.gradient(dem, axis=0) / dy
    dz_east = np.gradient(dem, axis=1) / dx
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_south))).astype(np.float32)
    # Downhill aspect, clockwise from north. Native row-positive is southward.
    aspect = np.mod(np.degrees(np.arctan2(-dz_east, dz_south)), 360).astype(np.float32)
    return dem, slope, aspect, west, profiles


def resample_dem(arr, src_transform, dst_transform, shape=(128, 128), resampling=Resampling.bilinear):
    # Direct coordinate sampling avoids repeatedly warping the 7200x3600 DEM
    # mosaic for each small patch. Pixel-center offsets honor both affines.
    h, w = shape
    rr, cc = np.mgrid[0:h, 0:w]
    x = dst_transform.c + (cc + .5) * dst_transform.a
    y = dst_transform.f + (rr + .5) * dst_transform.e
    col = (x - src_transform.c) / src_transform.a - .5
    row = (y - src_transform.f) / src_transform.e - .5
    order = 0 if resampling == Resampling.nearest else 1
    return ndimage.map_coordinates(arr, [row, col], order=order, mode="nearest").astype(np.float32)


def shift_without_wrap(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def percentile_rank(series, ascending=True):
    return series.rank(pct=True, ascending=ascending, method="average")


def select_cases(df):
    df = df.copy()
    for col in ("gt_frac", "gt_bothvalid_frac", "asc_bad_frac", "desc_bad_frac", "asc_gt_in_bad",
                "desc_gt_in_bad", "asc_gt_layover_share", "desc_gt_layover_share",
                "asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid",
                "gt_slope_under10_frac", "asc_slope_mask_spearman", "desc_slope_mask_spearman"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    rules = {}
    # A: landslide-rich and nearly wholly in BothValid; among clear cases choose
    # the one nearest the eligible pool median, matching the prior deterministic rule.
    pool_a = df[(df.gt_frac >= 0.03) & (df.gt_bothvalid_frac >= 0.90) &
                (df.asc_bad_frac < 0.05) & (df.desc_bad_frac < 0.05)].copy()
    pool_a["max_contrast"] = pool_a[["asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"]].max(axis=1)
    threshold = float(pool_a.max_contrast.quantile(.75)) if len(pool_a) else None
    clear_a = pool_a[pool_a.max_contrast >= threshold] if len(pool_a) else pool_a
    if len(clear_a):
        vals = clear_a[["gt_frac", "max_contrast"]]
        dist = (vals - vals.median()).abs().sum(axis=1)
        sid_a = int(clear_a.loc[dist.idxmin(), "sample_id"])
    else:
        sid_a = int(df.sort_values("gt_bothvalid_frac", ascending=False).sample_id.iloc[0])
    rules["Case_A"] = {"rule": "gt_frac>=0.03, >=90% GT BothValid, both orbit invalid<5%; retain >=P75 max GT/BG |ΔVV| contrast; nearest eligible median",
                       "eligible_n": int(len(pool_a)), "clear_subset_n": int(len(clear_a)), "id": sid_a}

    pool_b = df[(df.gt_frac >= .02) & (df.asc_gt_in_bad >= .70) & (df.desc_gt_in_bad <= .30) &
                (df.asc_gt_layover_share >= .50)]
    if len(pool_b):
        v = pool_b[["gt_frac", "asc_layover_frac"]]
        dist = (v - v.median()).abs().sum(axis=1)
        sid_b = int(pool_b.loc[dist.idxmin(), "sample_id"])
    else:
        sid_b = int(df.sort_values("asc_gt_in_bad", ascending=False).sample_id.iloc[0])
    rules["Case_B"] = {"rule": "GT>=2%; ASC invalid overlap>=70% and >=50% of ASC-invalid GT is layover; DESC invalid overlap<=30%; nearest pool median",
                       "eligible_n": int(len(pool_b)), "id": sid_b}

    pool_c = df[(df.gt_frac >= .02) & (df.desc_gt_in_bad >= .50) & (df.asc_gt_in_bad <= .30) &
                (df.desc_gt_layover_share >= .25)]
    if len(pool_c):
        v = pool_c[["gt_frac", "desc_layover_frac"]]
        dist = (v - v.median()).abs().sum(axis=1)
        sid_c = int(pool_c.loc[dist.idxmin(), "sample_id"])
    else:
        sid_c = int(df.sort_values("desc_gt_in_bad", ascending=False).sample_id.iloc[0])
    rules["Case_C"] = {"rule": "GT>=2%; DESC invalid overlap>=50% and >=25% of DESC-invalid GT is layover; ASC invalid overlap<=30%; nearest pool median",
                       "eligible_n": int(len(pool_c)), "id": sid_c}

    cand = df[df.gt_frac >= .02].copy()
    cand["score_weak_change"] = percentile_rank(-cand[["asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"]].min(axis=1))
    cand["score_gt_geom_overlap"] = percentile_rank(cand[["asc_gt_in_bad", "desc_gt_in_bad"]].min(axis=1))
    cand["score_low_slope"] = percentile_rank(cand.gt_slope_under10_frac)
    cand["score_geom_slope"] = percentile_rank(-cand[["asc_slope_mask_spearman", "desc_slope_mask_spearman"]].mean(axis=1))
    cand["score_orbit_disagreement"] = percentile_rank(cand.orbit_contrast_log_disagreement)
    score_cols = ["score_weak_change", "score_gt_geom_overlap", "score_low_slope",
                  "score_geom_slope", "score_orbit_disagreement"]
    cand["suspicion_score"] = cand[score_cols].mean(axis=1)
    sid_d = int(cand.sort_values("suspicion_score", ascending=False).sample_id.iloc[0])
    rules["Case_D"] = {"rule": "GT>=2%; highest mean percentile rank across bilateral weak event contrast, bilateral GT-mask overlap, low-slope GT, inverse DEM-slope/mask association, ASC/DESC contrast discordance; flagged candidate only, not an error finding",
                       "eligible_n": int(len(cand)), "id": sid_d,
                       "score": float(cand.loc[cand.sample_id == sid_d, "suspicion_score"].iloc[0])}
    return rules, cand


def make_case_figure(sid, tag, output, slopes, aspects):
    d = {}
    transform = None
    for key, rel in RAW.items():
        with rasterio.open(DATA / rel / f"{sid}.tif") as src:
            d[key] = src.read()
            if transform is None:
                transform = src.transform
    with rasterio.open(DATA / "processed" / f"sample_{sid:06d}" / "landslide_mask.tif") as src:
        gt = src.read(1) > 0
    ga, gd = (d[(o, "pre")][2].astype(int) for o in ORBITS)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    fig.suptitle(f"{tag} — sample {sid} | GT red; layover orange; shadow blue; both purple", fontsize=13)
    vv = {k: d[k][1] for k in d}
    vh = {k: d[k][0] for k in d}
    panels = [("ASC pre VV", vv[("asc", "pre")]), ("ASC post VV", vv[("asc", "post")]),
              ("DESC pre VV", vv[("desc", "pre")]), ("DESC post VV", vv[("desc", "post")])]
    for ax, (title, image) in zip(axes[0], panels):
        lo, hi = np.percentile(image, [2, 98])
        ax.imshow(np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
        ax.contour(gt, [0.5], colors="red", linewidths=.9)
        ax.set_title(title)
    for ax, (title, image) in zip(axes[1], [
        ("ASC |ΔVV|", np.abs(vv[("asc", "post")] - vv[("asc", "pre")])),
        ("DESC |ΔVV|", np.abs(vv[("desc", "post")] - vv[("desc", "pre")])),
        ("ASC |ΔVH|", np.abs(vh[("asc", "post")] - vh[("asc", "pre")])),
        ("DESC |ΔVH|", np.abs(vh[("desc", "post")] - vh[("desc", "pre")])),
    ]):
        ax.imshow(image, cmap="magma", vmin=0, vmax=np.percentile(image, 99))
        ax.contour(gt, [0.5], colors="red", linewidths=.9)
        ax.set_title(title)
    sl, asp = slopes[sid], aspects[sid]
    axes[2, 0].imshow(sl, cmap="terrain", vmin=0, vmax=50)
    axes[2, 0].contour(gt, [0.5], colors="red", linewidths=.8)
    axes[2, 0].set_title("GLO-30 slope (deg), GT red")
    axes[2, 1].imshow(asp, cmap="twilight", vmin=0, vmax=360)
    axes[2, 1].contour(gt, [0.5], colors="red", linewidths=.8)
    axes[2, 1].set_title("Downhill aspect, GT red")
    geom_cmap = matplotlib.colors.ListedColormap(["#444444", "#ff8c00", "#1676c2", "#9b27a7"])
    for ax, g, title in ((axes[2, 2], ga, "ASC geometry code"), (axes[2, 3], gd, "DESC geometry code")):
        ax.imshow(g, cmap=geom_cmap, vmin=0, vmax=3)
        ax.contour(gt, [0.5], colors="red", linewidths=.8)
        ax.set_title(title + " (GT red)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / output, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dem, dem_slope, dem_aspect, dem_transform, dem_profiles = load_dem_terrain()
    rng = np.random.default_rng(SEED)

    disk_inventory = {}
    for sub in ["Pre_event/S1_ASC_20210805", "Pre_event/S1_DESC_20210803",
                "Post_event/S1_ASC_20210817", "Post_event/S1_DESC_20210815",
                "Pre_event/S2_20210804", "Post_event/S2_20210814", "processed"]:
        path = DATA / sub
        disk_inventory[sub] = {"exists": path.exists(), "tif_count": len(list(path.rglob("*.tif"))) if path.exists() else 0}
    vector_files = [str(p.relative_to(DATA)) for p in DATA.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".shp", ".gpkg", ".geojson", ".kml")]
    metadata_fields = {}
    with rasterio.open(DATA / RAW[("asc", "pre")] / "0.tif") as src:
        sample_profile = {"count": src.count, "width": src.width, "height": src.height,
                          "crs": str(src.crs), "transform": list(src.transform), "res": list(src.res),
                          "bounds": list(src.bounds), "nodata": src.nodata, "dtypes": list(src.dtypes),
                          "band_descriptions": list(src.descriptions), "tags": src.tags()}
    metadata_path = DATA / "processed/sample_000000/metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata_fields = json.load(f)
    geom_counts = {o: np.zeros(4, np.int64) for o in ORBITS}
    joint_geometry = np.zeros((2, 2), np.int64)
    slope_mask = {o: np.zeros((5, 4), np.int64) for o in ORBITS}
    slope_gtmask = {o: np.zeros((5, 2, 2), np.int64) for o in ORBITS}
    aspect_mask = {o: np.zeros((9, 4), np.int64) for o in ORBITS}
    gt_slope_counts = np.zeros(5, np.int64)
    bg_slope_counts = np.zeros(5, np.int64)
    gt_aspect_counts = np.zeros(9, np.int64)
    bg_aspect_counts = np.zeros(9, np.int64)
    overlap_cont = {o: np.zeros((2, 2), np.int64) for o in ORBITS}  # bad,gt
    gt_code_counts = np.zeros(256, np.int64)
    align_mismatch = {"shape": 0, "crs": 0, "transform": 0, "nonfinite": 0,
                      "geom_prepost_pixels": {o: 0 for o in ORBITS},
                      "geom_processed_locations": {o: 0 for o in ORBITS},
                      "geom_codes_outside_0_3": {o: 0 for o in ORBITS}}
    raster_inventory_seen = {"complete_locations": 0, "gt_raw_values": set(), "geom_raw_values": {o: set() for o in ORBITS}}
    pixel_sums = {o: {c: {m: 0.0 for m in METRICS} for c in ("gt", "bg")} for o in ORBITS}
    pixel_counts = {o: {c: 0 for c in ("gt", "bg")} for o in ORBITS}
    q_samples = {o: {c: {m: [] for m in METRICS} for c in ("gt", "bg")} for o in ORBITS}
    slope_samples = {"gt": [], "bg": []}
    elevation_samples = {"gt": [], "bg": []}
    patch_rows, bothvalid_by_orbit = [], []
    matched_patch = {o: {m: [] for m in METRICS} for o in ORBITS}
    matched_coverage = []
    matched_gt_pixel_total, bothvalid_gt_pixel_total = 0, 0
    g10_patch, g01_patch = [], []
    g10_pixel_wins, g10_pixel_n = 0, 0
    g01_pixel_wins, g01_pixel_n = 0, 0
    shift_patch_diffs = {(dy, dx): [] for dy in range(-3, 4) for dx in range(-3, 4)}
    shift_patch_contrasts = {(dy, dx): [] for dy in range(-3, 4) for dx in range(-3, 4)}
    gt_flat = 0

    for sid in range(N):
        stacks, profiles = {}, {}
        for (orbit, event), rel in RAW.items():
            path = DATA / rel / f"{sid}.tif"
            with rasterio.open(path) as src:
                stacks[(orbit, event)] = src.read().astype(np.float32)
                profiles[(orbit, event)] = {"shape": (src.height, src.width), "crs": str(src.crs),
                                             "transform": src.transform, "nodata": src.nodata,
                                             "dtypes": src.dtypes}
        base = profiles[("asc", "pre")]
        for prof in profiles.values():
            if prof["shape"] != (128, 128): align_mismatch["shape"] += 1
            if prof["crs"] != base["crs"]: align_mismatch["crs"] += 1
            if prof["transform"] != base["transform"]: align_mismatch["transform"] += 1
        pdir = DATA / "processed" / f"sample_{sid:06d}"
        with rasterio.open(pdir / "landslide_mask.tif") as src:
            gt_raw = src.read(1)
            gt_prof = {"shape": (src.height, src.width), "crs": str(src.crs), "transform": src.transform}
        gt = gt_raw > 0
        raster_inventory_seen["gt_raw_values"].update(np.unique(gt_raw).tolist())
        for prof in profiles.values():
            if prof["shape"] != gt_prof["shape"]: align_mismatch["shape"] += 1
            if prof["crs"] != gt_prof["crs"]: align_mismatch["crs"] += 1
            if prof["transform"] != gt_prof["transform"]: align_mismatch["transform"] += 1

        geoms = {}
        for orbit in ORBITS:
            gpre_raw = stacks[(orbit, "pre")][2]
            gpost_raw = stacks[(orbit, "post")][2]
            align_mismatch["geom_codes_outside_0_3"][orbit] += int(
                ((gpre_raw < 0) | (gpre_raw > 3) | (gpre_raw != np.round(gpre_raw))).sum())
            gpre = np.clip(np.round(gpre_raw).astype(np.int16), 0, 3)
            gpost = np.clip(np.round(gpost_raw).astype(np.int16), 0, 3)
            align_mismatch["geom_prepost_pixels"][orbit] += int(np.count_nonzero(gpre != gpost))
            raster_inventory_seen["geom_raw_values"][orbit].update(np.unique(gpre).tolist())
            with rasterio.open(pdir / f"{orbit}_pre_geom.tif") as src:
                gp = src.read(1).astype(np.int16)
                if not np.array_equal(gpre, gp):
                    align_mismatch["geom_processed_locations"][orbit] += 1
            geoms[orbit] = gpre
            geom_counts[orbit] += np.bincount(gpre.ravel(), minlength=4)[:4]
            bad, pos = (gpre > 0), gt
            overlap_cont[orbit] += np.array([[np.count_nonzero(~bad & ~pos), np.count_nonzero(~bad & pos)],
                                             [np.count_nonzero(bad & ~pos), np.count_nonzero(bad & pos)]])
        raster_inventory_seen["complete_locations"] += 1
        ga, gd = geoms["asc"], geoms["desc"]
        joint_geometry += np.array([[np.count_nonzero((ga == 0) & (gd == 0)),
                                     np.count_nonzero((ga == 0) & (gd > 0))],
                                    [np.count_nonzero((ga > 0) & (gd == 0)),
                                     np.count_nonzero((ga > 0) & (gd > 0))]])
        for code in np.unique(gt_raw):
            gt_code_counts[int(code)] += int((gt_raw == code).sum())

        elev = resample_dem(dem, dem_transform, base["transform"])
        sl = resample_dem(dem_slope, dem_transform, base["transform"])
        asp = resample_dem(dem_aspect, dem_transform, base["transform"], resampling=Resampling.nearest)
        valid_terrain = np.isfinite(elev) & np.isfinite(sl) & np.isfinite(asp)
        align_mismatch["nonfinite"] += int((~valid_terrain).sum())
        sl_bin = np.digitize(sl, SLOPE_EDGES).clip(0, 4)
        asp_bin = np.full((128, 128), 8, np.int8)
        oriented = sl >= 3
        asp_bin[oriented] = (np.floor(((asp[oriented] + 22.5) % 360) / 45).astype(np.int8)).clip(0, 7)
        elev_bin = np.digitize(elev, ELEV_EDGES).clip(0, len(ELEV_EDGES))
        gt_slope_counts += np.bincount(sl_bin[gt & valid_terrain], minlength=5)[:5]
        bg_slope_counts += np.bincount(sl_bin[(~gt) & valid_terrain], minlength=5)[:5]
        gt_aspect_counts += np.bincount(asp_bin[gt & valid_terrain], minlength=9)[:9]
        bg_aspect_counts += np.bincount(asp_bin[(~gt) & valid_terrain], minlength=9)[:9]
        gt_flat += int((gt & valid_terrain & (sl < 3)).sum())
        for orbit, geom in (("asc", ga), ("desc", gd)):
            for sb in range(5):
                select = valid_terrain & (sl_bin == sb)
                slope_mask[orbit][sb] += np.bincount(geom[select].ravel(), minlength=4)[:4]
                for bad_code in (0, 1):
                    for gt_code in (0, 1):
                        slope_gtmask[orbit][sb, bad_code, gt_code] += int(
                            (select & ((geom > 0) == bool(bad_code)) & (gt == bool(gt_code))).sum())
            for ab in range(9):
                select = valid_terrain & (asp_bin == ab)
                aspect_mask[orbit][ab] += np.bincount(geom[select].ravel(), minlength=4)[:4]

        # Per-patch slope association is a screening statistic only: local
        # relief and resampled DEM resolution make it unsuitable as physics proof.
        corr = {}
        for orbit, geom in (("asc", ga), ("desc", gd)):
            good = valid_terrain
            corr[orbit] = float(stats.spearmanr(sl[good], (geom[good] > 0).astype(np.uint8)).statistic) if good.sum() > 5 else np.nan

        bothvalid = (ga == 0) & (gd == 0)
        gt_bv = gt & bothvalid
        bg_bv = (~gt) & bothvalid
        patch = {"sample_id": sid, "gt_frac": float(gt.mean()),
                 "gt_pixels": int(gt.sum()), "gt_bothvalid_pixels": int(gt_bv.sum()),
                 "gt_bothvalid_frac": float(gt_bv.sum() / max(gt.sum(), 1)),
                 "bothvalid_frac": float(bothvalid.mean()),
                 "asc_bad_frac": float((ga > 0).mean()), "desc_bad_frac": float((gd > 0).mean()),
                 "asc_layover_frac": float((ga == 1).mean()), "desc_layover_frac": float((gd == 1).mean()),
                 "asc_shadow_frac": float((ga == 2).mean()), "desc_shadow_frac": float((gd == 2).mean()),
                 "asc_gt_in_bad": float((ga[gt] > 0).mean()) if gt.any() else np.nan,
                 "desc_gt_in_bad": float((gd[gt] > 0).mean()) if gt.any() else np.nan,
                 "asc_gt_layover_share": float((ga[gt] == 1).mean()) if gt.any() else np.nan,
                 "desc_gt_layover_share": float((gd[gt] == 1).mean()) if gt.any() else np.nan,
                 "gt_mean_slope": float(np.mean(sl[gt & valid_terrain])) if (gt & valid_terrain).any() else np.nan,
                 "gt_slope_under10_frac": float(((gt & valid_terrain) & (sl < 10)).sum() / max((gt & valid_terrain).sum(), 1)),
                 "gt_mean_elevation": float(np.mean(elev[gt & valid_terrain])) if (gt & valid_terrain).any() else np.nan,
                 "asc_slope_mask_spearman": corr["asc"], "desc_slope_mask_spearman": corr["desc"]}
        means = {}
        changes = {}
        for orbit in ORBITS:
            pre, post = stacks[(orbit, "pre")], stacks[(orbit, "post")]
            # The raw local stack is documented as band 1=VH and band 2=VV.
            dvv = post[1] - pre[1]
            dvh = post[0] - pre[0]
            tvv, cvv = calc_local_change(pre[1], post[1])
            tvh, cvh = calc_local_change(pre[0], post[0])
            change = {"dvv_signed": dvv, "dvv_abs": np.abs(dvv),
                      "dvh_signed": dvh, "dvh_abs": np.abs(dvh),
                      "joint_abs": np.sqrt(dvv * dvv + dvh * dvh),
                      "vv_texture_change": tvv, "vh_texture_change": tvh,
                      "vv_contrast_change": cvv, "vh_contrast_change": cvh}
            changes[orbit] = change
            for cls, mask in (("gt", gt_bv), ("bg", bg_bv)):
                n = int(mask.sum())
                pixel_counts[orbit][cls] += n
                for m in METRICS:
                    arr = change[m]
                    pixel_sums[orbit][cls][m] += float(arr[mask].sum(dtype=np.float64))
                    sample_patch_balanced(q_samples[orbit][cls][m], arr, mask, rng)
                if n:
                    means[(orbit, cls)] = {m: float(change[m][mask].mean()) for m in METRICS}
            for m in METRICS:
                patch[f"{orbit}_{m}_gt_bothvalid"] = means.get((orbit, "gt"), {}).get(m, np.nan)
                patch[f"{orbit}_{m}_bg_bothvalid"] = means.get((orbit, "bg"), {}).get(m, np.nan)
                gm = patch[f"{orbit}_{m}_gt_bothvalid"]
                bm = patch[f"{orbit}_{m}_bg_bothvalid"]
                patch[f"{orbit}_{m}_diff_bothvalid"] = gm - bm if np.isfinite(gm) and np.isfinite(bm) else np.nan
                patch[f"{orbit}_{m}_contrast_bothvalid"] = gm / (bm + 1e-6) if np.isfinite(gm) and np.isfinite(bm) else np.nan

        # Within-patch exact-stratum terrain-matched background. Strata are
        # BothValid x slope bin x 8 aspect sectors (or flat) x elevation band.
        key = ((sl_bin * 9 + asp_bin) * (len(ELEV_EDGES) + 1) + elev_bin).astype(np.int32)
        nkeys = (5 * 9 * (len(ELEV_EDGES) + 1))
        gt_bv_terrain, bg_bv_terrain = gt_bv & valid_terrain, bg_bv & valid_terrain
        gtc = np.bincount(key[gt_bv_terrain].ravel(), minlength=nkeys)
        bgc = np.bincount(key[bg_bv_terrain].ravel(), minlength=nkeys)
        matched_cells = (gtc > 0) & (bgc > 0)
        matched_n = int(gtc[matched_cells].sum())
        matched_coverage.append(matched_n / max(int(gt_bv.sum()), 1))
        matched_gt_pixel_total += matched_n
        bothvalid_gt_pixel_total += int(gt_bv.sum())
        for orbit in ORBITS:
            for m in METRICS:
                arr = changes[orbit][m]
                gsum = np.bincount(key[gt_bv_terrain].ravel(), weights=arr[gt_bv_terrain].ravel(), minlength=nkeys)
                bsum = np.bincount(key[bg_bv_terrain].ravel(), weights=arr[bg_bv_terrain].ravel(), minlength=nkeys)
                if matched_n:
                    diff = (gsum[matched_cells] / gtc[matched_cells] - bsum[matched_cells] / bgc[matched_cells])
                    matched_diff = float(np.average(diff, weights=gtc[matched_cells]))
                    matched_patch[orbit][m].append(matched_diff)
                    patch[f"{orbit}_{m}_matched_diff"] = matched_diff
                    gt_matched_mean = float(np.average(gsum[matched_cells] / gtc[matched_cells], weights=gtc[matched_cells]))
                    bg_matched_mean = float(np.average(bsum[matched_cells] / bgc[matched_cells], weights=gtc[matched_cells]))
                    patch[f"{orbit}_{m}_matched_gt_mean"] = gt_matched_mean
                    patch[f"{orbit}_{m}_matched_bg_mean"] = bg_matched_mean
                    patch[f"{orbit}_{m}_matched_ratio"] = gt_matched_mean / (bg_matched_mean + 1e-6)
                else:
                    matched_patch[orbit][m].append(np.nan)
                    patch[f"{orbit}_{m}_matched_diff"] = np.nan
                    patch[f"{orbit}_{m}_matched_gt_mean"] = np.nan
                    patch[f"{orbit}_{m}_matched_bg_mean"] = np.nan
                    patch[f"{orbit}_{m}_matched_ratio"] = np.nan
            # Correctly define targeted temporal-proxy contrasts from matched
            # BothValid change (unmatched values are used only as diagnostics).
            patch[f"{orbit}_vv_contrast_bothvalid"] = patch[f"{orbit}_dvv_abs_contrast_bothvalid"]
        patch["orbit_contrast_log_disagreement"] = float(abs(np.log(
            max(patch["asc_vv_contrast_bothvalid"], .01) / max(patch["desc_vv_contrast_bothvalid"], .01))))

        # G10 / G01 paired changes on GT pixels. Positive diff means the
        # geometry-valid view has stronger absolute channel-vector change.
        joint_a, joint_d = changes["asc"]["joint_abs"], changes["desc"]["joint_abs"]
        g10 = gt & (ga > 0) & (gd == 0)
        g01 = gt & (ga == 0) & (gd > 0)
        if g10.any():
            diff = float((joint_d[g10] - joint_a[g10]).mean())
            g10_patch.append(diff)
            g10_pixel_wins += int((joint_d[g10] > joint_a[g10]).sum())
            g10_pixel_n += int(g10.sum())
        if g01.any():
            diff = float((joint_a[g01] - joint_d[g01]).mean())
            g01_patch.append(diff)
            g01_pixel_wins += int((joint_a[g01] > joint_d[g01]).sum())
            g01_pixel_n += int(g01.sum())

        # Patch-cluster shift sensitivity on the fixed BothValid footprint.
        # Boundary padding is dropped; no wraparound pixels are introduced.
        base_abs = changes["asc"]["dvv_abs"]
        base_gt = gt & bothvalid
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                shifted = shift_without_wrap(gt.astype(np.uint8), dy, dx).astype(bool) & bothvalid
                bgshift = (~shift_without_wrap(gt.astype(np.uint8), dy, dx).astype(bool)) & bothvalid
                if shifted.sum() and bgshift.sum():
                    diff = float(base_abs[shifted].mean() - base_abs[bgshift].mean())
                    ratio = float(base_abs[shifted].mean() / (base_abs[bgshift].mean() + 1e-6))
                    shift_patch_diffs[(dy, dx)].append(diff)
                    shift_patch_contrasts[(dy, dx)].append(ratio)
        patch_rows.append(patch)
        bothvalid_by_orbit.append({o: means.get((o, "gt"), {}).get("dvv_abs", np.nan) for o in ORBITS})
        if sid % 100 == 0:
            print(f"audited patches: {sid + 1}/{N}", flush=True)

    # Matched background summary and paired uncertainty at the patch level.
    matched_summary = {}
    for orbit in ORBITS:
        matched_summary[orbit] = {}
        for m in METRICS:
            matched_summary[orbit][m] = paired_inference(matched_patch[orbit][m], SEED + (1 if orbit == "asc" else 2))
            matched_summary[orbit][m]["matched_gt_pixel_coverage"] = float(np.mean(matched_coverage))

    pdf = pd.DataFrame(patch_rows)
    # Orbit difference stability on independent patches (not pixels).
    complete = pdf.dropna(subset=["asc_dvv_abs_gt_bothvalid", "desc_dvv_abs_gt_bothvalid",
                                 "asc_dvv_abs_diff_bothvalid", "desc_dvv_abs_diff_bothvalid"])
    cross_orbit = {}
    for name, a, b in (
        ("GT mean |ΔVV|", "asc_dvv_abs_gt_bothvalid", "desc_dvv_abs_gt_bothvalid"),
        ("GT mean |ΔVH|", "asc_dvh_abs_gt_bothvalid", "desc_dvh_abs_gt_bothvalid"),
        ("GT mean joint change", "asc_joint_abs_gt_bothvalid", "desc_joint_abs_gt_bothvalid"),
        ("GT-background |ΔVV| contrast", "asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"),
        ("GT-background joint-change contrast", "asc_joint_abs_contrast_bothvalid", "desc_joint_abs_contrast_bothvalid"),
    ):
        z = pdf[[a, b]].dropna()
        cross_orbit[name] = {"n_patches": int(len(z)), "pearson_r": float(stats.pearsonr(z[a], z[b]).statistic),
                             "spearman_r": float(stats.spearmanr(z[a], z[b]).statistic)} if len(z) > 2 else {}
    both_gain = ((pdf.asc_joint_abs_diff_bothvalid > 0) & (pdf.desc_joint_abs_diff_bothvalid > 0)).dropna()
    cross_orbit["fraction_patches_both_orbits_GT_change_exceeds_BG"] = {
        "n_patches": int(both_gain.size), "fraction": float(both_gain.mean()) if both_gain.size else None,
        "bootstrap_95ci": boot_ci(both_gain.astype(float).to_numpy(), SEED + 3) if both_gain.size else None}

    # Slope-conditioned GT/mask association.
    slope_assoc = {}
    slope_names = ["0-10", "10-20", "20-30", "30-40", ">40"]
    for orbit in ORBITS:
        out = []
        for i, name in enumerate(slope_names):
            c = slope_mask[orbit][i]
            bad_gt, bad_bg = int(slope_gtmask[orbit][i, 1, 1]), int(slope_gtmask[orbit][i, 1, 0])
            good_gt, good_bg = int(slope_gtmask[orbit][i, 0, 1]), int(slope_gtmask[orbit][i, 0, 0])
            p_bad = bad_gt / max(bad_gt + bad_bg, 1)
            p_good = good_gt / max(good_gt + good_bg, 1)
            out.append({"slope_bin_deg": name, "pixels": int(c.sum()),
                        "invalid_fraction": float(c[1:].sum() / max(c.sum(), 1)),
                        "layover_pct": float(c[1] / max(c.sum(), 1) * 100),
                        "shadow_pct": float(c[2] / max(c.sum(), 1) * 100),
                        "both_pct": float(c[3] / max(c.sum(), 1) * 100),
                        "P_GT_given_invalid": p_bad, "P_GT_given_valid": p_good,
                        "invalid_vs_valid_GT_risk_ratio": p_bad / max(p_good, 1e-12),
                        "invalid_GT_bg_counts": [bad_gt, bad_bg], "valid_GT_bg_counts": [good_gt, good_bg]})
        slope_assoc[orbit] = out

    # Geometry x GT odds/risk; exact invalid=code>0 categories.
    overlap_stats = {}
    for orbit in ORBITS:
        c = overlap_cont[orbit]
        a, b, cc, d = int(c[1, 1]), int(c[0, 1]), int(c[1, 0]), int(c[0, 0])
        overlap_stats[orbit] = {"P_invalid_given_GT": a / max(a + b, 1),
                                "P_invalid_given_BG": cc / max(cc + d, 1),
                                "P_GT_given_invalid": a / max(a + cc, 1),
                                "P_GT_given_valid": b / max(b + d, 1),
                                "IoU_GT_vs_invalid": a / max(a + b + cc, 1),
                                "Dice": 2 * a / max(2 * a + b + cc, 1),
                                "odds_ratio": (a * d / (b * cc)) if b * cc else None,
                                "relative_risk": ((a / max(a + b, 1)) / (cc / max(cc + d, 1))) if cc + d else None,
                                "counts_bad_gt_bg": [a, cc], "counts_valid_gt_bg": [b, d]}

    # Aspect distribution by geometry class on non-flat pixels. Do not map
    # compass sectors to radar-facing/back-facing without acquisition azimuth.
    aspect_table = {}
    for orbit in ORBITS:
        aspect_table[orbit] = []
        for i, name in enumerate(ASPECT_NAMES):
            c = aspect_mask[orbit][i]
            denom = max(int(c.sum()), 1)
            aspect_table[orbit].append({"aspect_bin": name, "pixels": int(c.sum()),
                                        "valid_pct": float(c[0] / denom * 100),
                                        "layover_pct": float(c[1] / denom * 100),
                                        "shadow_pct": float(c[2] / denom * 100),
                                        "both_pct": float(c[3] / denom * 100),
                                        "invalid_pct": float(c[1:].sum() / denom * 100)})

    # Exact-sample alignment / inventory checks are merged with source metadata.
    tif_inventory = {k: v for k, v in disk_inventory.items()}
    geom_pct = {o: (100 * geom_counts[o] / geom_counts[o].sum()).round(6).tolist() for o in ORBITS}
    contig = (joint_geometry / max(joint_geometry.sum(), 1)).tolist()
    geom_summary = {"class_counts_by_code": {o: geom_counts[o].tolist() for o in ORBITS},
                    "class_pct_code0_valid_code1_layover_code2_shadow_code3_both": geom_pct,
                    "binary_joint_counts_rows_ASC_invalid_0_1_cols_DESC_invalid_0_1": joint_geometry.tolist(),
                    "binary_joint_fraction": contig,
                    "overlap_gt": overlap_stats,
                    "gt_raw_code_counts": {str(i): int(v) for i, v in enumerate(gt_code_counts) if v},
                    "slope_geometry_rates": slope_assoc,
                    "aspect_geometry_rates": aspect_table,
                    "global_gt_slope_hist_counts": gt_slope_counts.tolist(),
                    "global_bg_slope_hist_counts": bg_slope_counts.tolist(),
                    "global_gt_aspect_hist_counts": gt_aspect_counts.tolist(),
                    "global_bg_aspect_hist_counts": bg_aspect_counts.tolist(),
                    "gt_fraction_slope_below_3deg": float(gt_flat / max(gt_slope_counts.sum(), 1)),
                    "gt_slope_bins": slope_names,
                    "gt_aspect_bins": ASPECT_NAMES}

    # Pixel means exact; quantiles are patch-balanced samples (max 256/class/patch).
    change_summary = {o: {} for o in ORBITS}
    for o in ORBITS:
        for m in METRICS:
            change_summary[o][m] = {}
            for cls in ("gt", "bg"):
                n = pixel_counts[o][cls]
                vals = np.asarray(q_samples[o][cls][m], dtype=np.float64)
                change_summary[o][m][cls] = {
                    "n_pixels_bothvalid": int(n),
                    "mean_exact_pixel_weighted": float(pixel_sums[o][cls][m] / max(n, 1)),
                    "patch_balanced_sample_n": int(vals.size),
                    "sample_median": float(np.median(vals)) if vals.size else None,
                    "sample_q25_q75": [float(np.quantile(vals, .25)), float(np.quantile(vals, .75))] if vals.size else None,
                }
            col = f"{o}_{m}_diff_bothvalid"
            change_summary[o][m]["patch_paired_gt_minus_bg"] = paired_inference(pdf[col].to_numpy(), SEED + 10 + len(m))

    # Terrain-match summaries are a finer confounder control than the prior
    # slope-only table, but coarse DEM / repeated windows remain limitations.
    terrain = {"method": "within each patch, BothValid positives matched to background within exact 5-bin slope x 9-bin aspect (flat separate) x 8-bin elevation strata; GT-count weighted",
              "elevation_edges_m": ELEV_EDGES, "slope_edges_deg": SLOPE_EDGES,
              "aspect_edges_deg": [float(x) for x in ASPECT_EDGES],
              "mean_patch_positive_pixel_coverage": float(np.mean(matched_coverage)),
              "pooled_positive_pixel_coverage": float(matched_gt_pixel_total / max(bothvalid_gt_pixel_total, 1)),
              "matched_bothvalid_gt_pixels": matched_gt_pixel_total,
              "all_bothvalid_gt_pixels": bothvalid_gt_pixel_total,
              "matched_background_results_by_orbit_and_metric": matched_summary}

    # Paired valid-vs-invalid view evidence in G10/G01.
    g10_summary = paired_inference(g10_patch, SEED + 30)
    g01_summary = paired_inference(g01_patch, SEED + 31)
    g10_summary.update({"valid_view_wins_pixel_fraction": g10_pixel_wins / max(g10_pixel_n, 1),
                        "positive_gt_pixels": g10_pixel_n})
    g01_summary.update({"valid_view_wins_pixel_fraction": g01_pixel_wins / max(g01_pixel_n, 1),
                        "positive_gt_pixels": g01_pixel_n})

    # Time mismatch candidates: only a proxy from orbit-specific GT/BG |ΔVV|
    # ratios in BothValid. It cannot attribute individual mapped landslides.
    ratio_a = pdf.asc_vv_contrast_bothvalid.to_numpy()
    ratio_d = pdf.desc_vv_contrast_bothvalid.to_numpy()
    proxy_a_stronger = (ratio_a >= 1.20) & (ratio_d <= 1.05) & (ratio_a / np.maximum(ratio_d, .01) >= 1.5)
    proxy_d_stronger = (ratio_d >= 1.20) & (ratio_a <= 1.05) & (ratio_d / np.maximum(ratio_a, .01) >= 1.5)
    temporal_candidates = {"candidate_rule": "GT>=2% implicitly via valid GT; one orbit GT/BG |ΔVV| contrast>=1.20, other<=1.05, ratio>=1.5",
                           "candidate_count_ASC_only": int(proxy_a_stronger.sum()),
                           "candidate_count_DESC_only": int(proxy_d_stronger.sum()),
                           "candidate_total_union": int((proxy_a_stronger | proxy_d_stronger).sum()),
                           "eligible_with_bothvalid_GT": int(pdf.dropna(subset=["asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"]).shape[0]),
                           "candidate_sample_ids_ASC_only": pdf.loc[proxy_a_stronger, "sample_id"].head(50).astype(int).tolist(),
                           "candidate_sample_ids_DESC_only": pdf.loc[proxy_d_stronger, "sample_id"].head(50).astype(int).tolist(),
                           "interpretation": "possible temporal/orbit-view discordance; not a mislabeled-polygon determination"}

    # Shift test on GT boundary relative to a fixed BothValid SAR change map.
    shift_results = []
    for (dy, dx), vals in shift_patch_diffs.items():
        if vals:
            shift_results.append({"dy_pixels": dy, "dx_pixels": dx, "n_patches": int(len(vals)),
                                  "mean": float(np.mean(vals)), "median_patch_GT_minus_BG_abs_dVV": float(np.median(vals)),
                                  "mean_gt_bg_ratio_patch": float(np.nanmean(shift_patch_contrasts[(dy, dx)]))})
    shift_df = pd.DataFrame(shift_results)
    base_shift = shift_df[(shift_df.dy_pixels == 0) & (shift_df.dx_pixels == 0)].iloc[0]
    best_shift = shift_df.sort_values("mean", ascending=False).iloc[0]
    base_inf = paired_inference(shift_patch_diffs[(0, 0)], SEED + 99)
    best_inf = paired_inference(shift_patch_diffs[(int(best_shift.dy_pixels), int(best_shift.dx_pixels))], SEED + 100)
    shift_sensitivity = {"shift_grid_pixels": [-3, 3], "fixed_mask": "original unshifted BothValid footprint",
                         "metric": "ASC |ΔVV|, patch-cluster GT-minus-BG mean; GT translated with clipped no-wrap border",
                         "baseline_offset": {"dy_pixels": 0, "dx_pixels": 0, "mean": float(base_shift["mean"]), **base_inf},
                         "best_grid_offset": {"dy_pixels": int(best_shift.dy_pixels), "dx_pixels": int(best_shift.dx_pixels),
                                              "mean": float(best_shift["mean"]), **best_inf},
                         "shift_table_csv": "registration_shift_sensitivity.csv"}
    shift_df.to_csv(OUT / "registration_shift_sensitivity.csv", index=False)

    # Merge legacy complete 1713-row patch stats with new terrain/paired metrics.
    old_patch = pd.read_csv(OLD / "candidate_metrics.csv")
    old_slope = pd.read_csv(OLD / "patch_slope.csv")
    old_patch = old_patch.rename(columns={c: f"legacy_{c}" for c in old_patch.columns if c != "sample_id"})
    old_patch = old_patch.merge(old_slope, on="sample_id", how="left", validate="one_to_one")
    patch_audit = old_patch.merge(pdf, on="sample_id", how="left", validate="one_to_one")
    patch_audit.to_csv(OUT / "dataset_patch_audit.csv", index=False)

    # Per-type top-10 anomaly table, based only on rules defined before listing.
    anomalies = []
    df = patch_audit
    type_candidates = {
        "Type1_large_GT_bothvalid_low_change": df[(df.gt_frac >= .03) & (df.gt_bothvalid_frac >= .80)].sort_values(
            ["asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"], ascending=True),
        "Type2_GT_geometry_extreme_overlap": df[df.gt_frac >= .02].assign(
            both_overlap=lambda x: np.minimum(x.asc_gt_in_bad, x.desc_gt_in_bad)).sort_values("both_overlap", ascending=False),
        "Type3_GT_low_slope": df[df.gt_frac >= .02].sort_values("gt_slope_under10_frac", ascending=False),
        "Type4_geometry_slope_inconsistency": df.assign(
            geom_slope_avg=lambda x: np.mean(np.column_stack([x.asc_slope_mask_spearman, x.desc_slope_mask_spearman]), axis=1)
        ).sort_values("geom_slope_avg", ascending=True),
        # Keep the minimum positive-inventory eligibility rule used by the
        # temporal proxy summary; otherwise tiny/no-GT patches dominate the
        # rank despite not being candidates for polygon-level event mismatch.
        "Type5_possible_temporal_mismatch_proxy": df[df.gt_frac >= .02].assign(
            temporal_discordance=lambda x: np.abs(np.log(np.maximum(x.asc_vv_contrast_bothvalid, .01) /
                                                        np.maximum(x.desc_vv_contrast_bothvalid, .01)))
        ).sort_values("temporal_discordance", ascending=False),
    }
    for typ, sub in type_candidates.items():
        for rank, (_, r) in enumerate(sub.head(10).iterrows(), 1):
            anomalies.append({"anomaly_type": typ, "rank": rank, "sample_id": int(r.sample_id),
                              "gt_frac": float(r.gt_frac), "gt_bothvalid_frac": float(r.gt_bothvalid_frac),
                              "asc_gt_in_bad": float(r.asc_gt_in_bad), "desc_gt_in_bad": float(r.desc_gt_in_bad),
                              "gt_slope_under10_frac": float(r.gt_slope_under10_frac),
                              "asc_slope_mask_spearman": float(r.asc_slope_mask_spearman),
                              "desc_slope_mask_spearman": float(r.desc_slope_mask_spearman),
                              "asc_vv_contrast_bothvalid": float(r.asc_vv_contrast_bothvalid),
                              "desc_vv_contrast_bothvalid": float(r.desc_vv_contrast_bothvalid),
                              "selection_is_review_candidate_not_error": True})
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df.to_csv(OUT / "anomaly_topk.csv", index=False)

    # The final figure panel's selection is deterministic and recorded.
    all_for_select = patch_audit.copy()
    for col in ("asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid"):
        all_for_select[col] = pd.to_numeric(all_for_select[col], errors="coerce").fillna(0)
    case_rules, case_ranked = select_cases(all_for_select)
    case_ranked.sort_values("suspicion_score", ascending=False).head(20).to_csv(OUT / "case_D_ranked_candidates.csv", index=False)
    write_json(OUT / "case_selection_log.json", {"seed": SEED, "selection_rules": case_rules,
                                                   "selected": {k: v["id"] for k, v in case_rules.items()},
                                                   "optical_panels": "unavailable: optical source references exist in metadata but actual image directories are absent"})
    def case_terrain(sid):
        with rasterio.open(DATA / RAW[("asc", "pre")] / f"{sid}.tif") as src:
            transform = src.transform
        return (resample_dem(dem_slope, dem_transform, transform),
                resample_dem(dem_aspect, dem_transform, transform, resampling=Resampling.nearest))
    for key, tag, filename in (
        ("Case_A", "Case A: BothValid landslide, median-representative visible event", "case_A_bothvalid.png"),
        ("Case_B", "Case B: ASC invalid / DESC valid; layover-led example", "case_B_asc_invalid_desc_valid.png"),
        ("Case_C", "Case C: ASC valid / DESC invalid; layover-led example", "case_C_asc_valid_desc_invalid.png"),
        ("Case_D", "Case D: highest-ranked joint anomaly review candidate (not adjudicated error)", "case_D_anomaly_candidate.png"),
    ):
        sid = case_rules[key]["id"]
        sl_case, asp_case = case_terrain(sid)
        make_case_figure(sid, tag, filename, {sid: sl_case}, {sid: asp_case})

    global_summary = {"n_locations": N, "patch_pixels_with_repetition": int(N * 128 * 128),
                      "window_overlap_note": "neighboring patches overlap; patch-cluster intervals still share underlying ground and are not fully independent",
                      "source_inventory_on_disk": tif_inventory, "vector_files_within_dataset_root": vector_files,
                      "sample_geotiff_profile": sample_profile, "sample_processed_metadata": metadata_fields,
                      "dem_local_profiles": dem_profiles,
                      "dem_derivatives": "slope from 30 m elevation gradient; downhill aspect; reprojected to each SAR patch; GLO-30 provenance is the existing audit record, DEM file tags themselves do not identify source/version",
                      "alignment_checks": align_mismatch,
                      "processed_geom_equal_to_raw_pre_locations": {o: N - align_mismatch["geom_processed_locations"][o] for o in ORBITS},
                      "raw_prepost_geom_pixel_mismatches": align_mismatch["geom_prepost_pixels"],
                      "geom_code_values": {o: sorted(int(x) for x in raster_inventory_seen["geom_raw_values"][o]) for o in ORBITS},
                      "gt_code_values": sorted(int(x) for x in raster_inventory_seen["gt_raw_values"]),
                      "geometry": geom_summary, "bothvalid_sar_change": change_summary,
                      "terrain_matched_change": terrain, "cross_orbit": cross_orbit,
                      "G10_valid_vs_invalid_change": g10_summary,
                      "G01_valid_vs_invalid_change": g01_summary,
                      "possible_temporal_mismatch_proxy": temporal_candidates,
                      "registration_shift_sensitivity": shift_sensitivity,
                      "case_selection": case_rules,
                      "event_calendar": {"earthquake_date_utc": "2021-08-14 12:29 UTC (verified USGS)",
                                         "local_acquisition_dates": {"ASC_pre": "2021-08-05", "DESC_pre": "2021-08-03",
                                                                      "Sentinel2_pre_reference": "2021-08-04", "Sentinel2_post_reference": "2021-08-14",
                                                                      "DESC_post": "2021-08-15", "ASC_post": "2021-08-17"},
                                         "ASC_post_time_of_day_unavailable": True,
                                         "per_polygon_event_source_date_unavailable": True}}
    write_json(OUT / "audit_summary.json", global_summary)
    print(json.dumps({"out": str(OUT), "alignment": align_mismatch,
                      "geometry_pct": geom_pct, "cases": {k: v["id"] for k, v in case_rules.items()},
                      "matched_coverage": terrain["pooled_positive_pixel_coverage"],
                      "A_joint": cross_orbit.get("fraction_patches_both_orbits_GT_change_exceeds_BG")}, indent=2))


if __name__ == "__main__":
    main()
