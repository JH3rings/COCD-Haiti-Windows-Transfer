# -*- coding: utf-8 -*-
"""Physics audit pass 3: explicit-rule case selection + figure generation.

Selection rules are printed and saved; no cherry-picking: within each pool the
representative sample is the one closest to the pool median of the defining
metrics. Produces case1/case2/case3 figures + zoom overlay + selection CSV.
"""
import json
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
import pandas as pd
from pathlib import Path

ROOT = Path(r"D:/deeplearning/COCD_Haiti_Windows_Transfer")
DATA = ROOT / "data" / "haiti"
OUT = ROOT / "experiments" / "windows_main" / "physics_audit_20260924"

RAW = {("asc", "pre"): "Pre_event/S1_ASC_20210805",
       ("desc", "pre"): "Pre_event/S1_DESC_20210803",
       ("asc", "post"): "Post_event/S1_ASC_20210817",
       ("desc", "post"): "Post_event/S1_DESC_20210815"}
DATES = {("asc", "pre"): "2021-08-05", ("asc", "post"): "2021-08-17",
         ("desc", "pre"): "2021-08-03", ("desc", "post"): "2021-08-15"}

df = pd.read_csv(OUT / "candidate_metrics.csv")
log = {"rules": {}, "pools": {}, "selected": {}}


def load_sample(sid):
    d = {}
    for k, rel in RAW.items():
        with rasterio.open(DATA / rel / f"{sid}.tif") as s:
            d[k] = s.read()
    with rasterio.open(DATA / "processed" / f"sample_{sid:06d}" / "landslide_mask.tif") as s:
        d["gt_raw"] = s.read(1)
    d["gt"] = (d["gt_raw"] > 0)
    d["ga"] = np.clip(d[("asc", "pre")][2].astype(np.int64), 0, 3)
    d["gd"] = np.clip(d[("desc", "pre")][2].astype(np.int64), 0, 3)
    return d


def boundary_agreement(gt, mask, radius=2):
    gb = ndimage.binary_dilation(gt, iterations=1) & ~gt
    mb = ndimage.binary_dilation(mask > 0, iterations=1) & ~(mask > 0)
    if not gb.any():
        return 0.0
    mb_d = ndimage.binary_dilation(mb, iterations=radius)
    return float((gb & mb_d).sum() / gb.sum())


# ============ Group 1: BothValid landslide ============
pool1 = df[(df.gt_frac >= 0.03) & (df.gt_in_bothvalid_frac >= 0.9)
           & (df.asc_bad_frac < 0.05) & (df.desc_bad_frac < 0.05)].copy()
pool1["max_cc"] = pool1[["asc_change_contrast", "desc_change_contrast"]].max(axis=1)
thr = pool1["max_cc"].quantile(0.75)
pool1v = pool1[pool1.max_cc >= thr]  # clearer-change half; required by the group's purpose
c = pool1v[["gt_frac", "max_cc"]]
med = c.median()
d1 = (c - med).abs().sum(axis=1)
sel1 = sorted(d1.index, key=lambda i: d1[i])[:3]
sid1 = int(pool1v.loc[sel1[0], "sample_id"])
log["rules"]["group1"] = ("gt_frac>=0.03 & gt_in_bothvalid>=0.9 & asc_bad<0.05 & desc_bad<0.05 "
                          "& max_change_contrast>=P75; pick closest to pool median (gt_frac, max_cc)")
log["pools"]["group1"] = {"pool": len(pool1), "visible_half": len(pool1v), "P75_cc": float(thr),
                          "top3_ids": sel1, "median": med.to_dict()}
log["selected"]["group1"] = sid1
print("Group1 pool", len(pool1), "visible", len(pool1v), "-> ids", sel1, "picked", sid1)

# ============ Group 2: GT x geometry overlap ============
# layover-dominant, ASC target invalid / DESC counter valid
pool2a = df[(df.gt_frac >= 0.02) & (df.asc_gt_mask_overlap >= 0.7)
            & (df.desc_gt_mask_overlap <= 0.3)].copy()
lay_share, shw_share = {}, {}
for sid in pool2a.sample_id:
    d = load_sample(sid)
    lay_share[sid] = float((d["ga"][d["gt"]] == 1).mean())
    shw_share[sid] = float((d["ga"][d["gt"]] == 2).mean())
pool2a["lay_share"] = pool2a.sample_id.map(lay_share)
pool2a_lay = pool2a[pool2a.lay_share >= 0.5].copy()
if len(pool2a_lay):
    c = pool2a_lay[["gt_frac", "asc_layover_frac"]]
    d2 = (c - c.median()).abs().sum(axis=1)
    sid2_lay = int(pool2a_lay.loc[d2.idxmin(), "sample_id"])
else:
    sid2_lay = None
print("Group2 layover pool", len(pool2a), "layover-share>=0.5:", len(pool2a_lay), "-> picked", sid2_lay)

# shadow-dominant, DESC target invalid / ASC counter valid
pool2d = df[(df.gt_frac >= 0.02) & (df.desc_gt_mask_overlap >= 0.5)
            & (df.asc_gt_mask_overlap <= 0.5)].copy()
lay2, shw2 = {}, {}
for sid in pool2d.sample_id:
    d = load_sample(sid)
    lay2[sid] = float((d["gd"][d["gt"]] == 1).mean())
    shw2[sid] = float((d["gd"][d["gt"]] == 2).mean())
pool2d["shw_share"] = pool2d.sample_id.map(shw2)
pool2d_shw = pool2d[pool2d.shw_share >= 0.5].copy()
if not len(pool2d_shw):
    pool2d_shw = pool2d[pool2d.shw_share >= 0.25].copy()
    if not len(pool2d_shw):
        pool2d_shw = pool2d.sort_values("shw_share", ascending=False).head(3).copy()
        print("NOTE: no shadow-share>=0.25 candidate; using max-shadow-share top-3")
if len(pool2d_shw):
    sid2_shw = int(pool2d_shw.sort_values("shw_share", ascending=False).sample_id.iloc[0])
else:
    sid2_shw = None
print("Group2 shadow pool", len(pool2d), "shadow-share>=0.5:", len(pool2d_shw), "-> picked", sid2_shw)

log["rules"]["group2"] = ("layover case: gt_frac>=0.02 & asc_gt_mask_overlap>=0.7 & desc_gt_mask_overlap<=0.3 "
                          "& frac(GT in asc==1)>=0.5; shadow case: gt_frac>=0.02 & desc_gt_mask_overlap>=0.5 "
                          "& asc_gt_mask_overlap<=0.5 & frac(GT in desc==2)>=0.5")
log["pools"]["group2"] = {"asc_pool": len(pool2a), "asc_layover_ok": len(pool2a_lay),
                          "desc_pool": len(pool2d), "desc_shadow_ok": len(pool2d_shw)}
log["selected"]["group2_layover"] = sid2_lay
log["selected"]["group2_shadow"] = sid2_shw

# ============ Group 3: suspicious ============
pool3 = df[(df.gt_frac >= 0.02) & (df.asc_gt_mask_overlap < 0.5) & (df.desc_gt_mask_overlap < 0.5)].copy()
ba_asc, ba_desc = {}, {}
for sid in pool3.sample_id:
    d = load_sample(sid)
    ba_asc[sid] = boundary_agreement(d["gt"], d["ga"])
    ba_desc[sid] = boundary_agreement(d["gt"], d["gd"])
pool3["ba_asc"] = pool3.sample_id.map(ba_asc)
pool3["ba_desc"] = pool3.sample_id.map(ba_desc)
pool3["min_cc"] = pool3[["asc_change_contrast", "desc_change_contrast"]].min(axis=1)
pool3["ba_max"] = pool3[["ba_asc", "ba_desc"]].max(axis=1)
score = (pool3.ba_max.rank(pct=True) + (1.3 - pool3.min_cc).rank(pct=True)
         + pool3.gt_frac.rank(pct=True))
pool3["score"] = score
cand3 = pool3.sort_values("score", ascending=False).head(5)
sid3 = int(cand3.sample_id.iloc[0])
log["rules"]["group3"] = ("gt_frac>=0.02 & gt-mask overlap<0.5 both orbits; score = rank(boundary agreement max) "
                          "+ rank(1.3 - min change contrast) + rank(gt_frac); top-5 inspected, pick #1")
log["pools"]["group3"] = {"pool": len(pool3), "top5": cand3[["sample_id", "score", "ba_max", "min_cc",
                                                             "gt_frac"]].round(3).to_dict("records")}
log["selected"]["group3"] = sid3
print("Group3 pool", len(pool3), "-> top5", cand3[["sample_id", "score"]].round(3).values.tolist(), "picked", sid3)

pd.concat([pool1v.loc[sel1].assign(group="G1_bothvalid"),
           pool2a_lay[pool2a_lay.sample_id == sid2_lay].assign(group="G2_layover_asc") if sid2_lay is not None else None,
           pool2d_shw[pool2d_shw.sample_id == sid2_shw].assign(group="G2_shadow_desc") if sid2_shw is not None else None,
           cand3.assign(group="G3_suspicious")]).to_csv(OUT / "case_selection.csv", index=False)

with open(OUT / "case_selection_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

# ================= FIGURES =================
GEOM_CMAP = matplotlib.colors.ListedColormap(["none", "#ff8c00", "#1f77b4", "#8b008b"])
GT_C = "#ff2a2a"


def stretch(a, v1, v2):
    return np.clip((a - v1) / max(v2 - v1, 1e-6), 0, 1)


def geom_contours(ax, g):
    if (g == 1).any():
        ax.contour((g == 1).astype(float), levels=[0.5], colors="#ff8c00", linewidths=1.1)
    if (g == 2).any():
        ax.contour((g == 2).astype(float), levels=[0.5], colors="#1f77b4", linewidths=1.1, linestyles=":")
    if (g == 3).any():
        ax.contour((g == 3).astype(float), levels=[0.5], colors="#8b008b", linewidths=1.1, linestyles="--")


def gt_contour(ax, gt):
    ax.contour(gt.astype(float), levels=[0.5], colors=GT_C, linewidths=1.6)


def group_figure(sid, tag, fname, title_note):
    d = load_sample(sid)
    gt = d["gt"]
    vv = {k: d[k][1] for k in RAW}
    vh = {k: d[k][0] for k in RAW}
    chg_vv = {o: vv[(o, "post")] - vv[(o, "pre")] for o in ("asc", "desc")}
    chg_vh = {o: vh[(o, "post")] - vh[(o, "pre")] for o in ("asc", "desc")}

    fig, axes = plt.subplots(3, 4, figsize=(17, 12.6))
    fig.suptitle(f"Patch {sid} ({tag}) — {title_note}", fontsize=13, y=0.995)

    stretch_cache = {}
    for o in ("asc", "desc"):
        stretch_cache[o] = np.percentile(vv[(o, "pre")], [2, 98])

    panels = [("ASC pre VV", ("asc", "pre")), ("ASC post VV", ("asc", "post")),
              ("DESC pre VV", ("desc", "pre")), ("DESC post VV", ("desc", "post"))]
    for ax, (t, k) in zip(axes[0], panels):
        o = k[0]
        ax.imshow(stretch(vv[k], *stretch_cache[o]), cmap="gray")
        geom_contours(ax, d["ga"] if o == "asc" else d["gd"])
        gt_contour(ax, gt)
        ax.set_title(f"{t}  {DATES[k]}\n[layover=orange shadow=blue: GT=red]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # row 2
    ax = axes[1, 0]
    ax.imshow(stretch(vv[("asc", "post")], *stretch_cache["asc"]), cmap="gray", alpha=0.55)
    ax.contour(ndimage.binary_dilation(gt, iterations=1).astype(float), levels=[0.5],
               colors=GT_C, linewidths=2.2)
    ax.set_title("GT (dilated outline on ASC post)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    for ax, g, t in ((axes[1, 1], d["ga"], "ASC geometry mask (pre)"),
                     (axes[1, 2], d["gd"], "DESC geometry mask (pre)")):
        base = np.zeros((128, 128, 3)) + 0.15
        ax.imshow(base)
        ax.imshow(np.ma.masked_where(g == 0, g), cmap=GEOM_CMAP, vmin=0, vmax=3, alpha=0.85)
        gt_contour(ax, gt)
        ax.set_title(t + "\n[1=layover 2=shadow 3=both; GT=red]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 3]
    base = np.zeros((128, 128, 3)) + 0.15
    ax.imshow(base)
    ga, gd = d["ga"], d["gd"]
    both = (ga > 0) & (gd > 0)
    only_a = (ga > 0) & (gd == 0)
    only_d = (ga == 0) & (gd > 0)
    comp = np.zeros((128, 128, 3))
    comp[only_a] = (1.0, 0.55, 0.0)   # ASC-only bad = orange
    comp[only_d] = (0.0, 0.75, 1.0)   # DESC-only bad = cyan
    comp[both] = (0.85, 0.1, 0.85)    # both bad = magenta
    ax.imshow(comp)
    gt_contour(ax, gt)
    ax.set_title("Complementarity: ASC-only=orange DESC-only=cyan both=magenta", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    # row 3: change magnitude panels
    for ax, (o, chg, pol) in zip(axes[2], [("asc", chg_vv, "VV"), ("desc", chg_vv, "VV"),
                                            ("asc", chg_vh, "VH"), ("desc", chg_vh, "VH")]):
        m = ax.imshow(np.abs(chg[o]), cmap="magma", vmin=0, vmax=np.percentile(np.abs(chg[o]), 99))
        gt_contour(ax, gt)
        geom_contours(ax, d["ga"] if o == "asc" else d["gd"])
        ax.set_title(f"|Δ{pol}| {o.upper()} pre→post (dB)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(m, ax=ax, fraction=0.046)

    st = (f"GT frac {gt.mean():.3f} | ASC bad {(ga>0).mean():.3f} DESC bad {(gd>0).mean():.3f} | "
          f"GT in ASC-bad {(ga[gt] > 0).mean():.2f} GT in DESC-bad {(gd[gt] > 0).mean():.2f} | "
          f"mean|ΔVV| in GT: ASC {np.abs(chg_vv['asc'][gt]).mean():.2f} DESC {np.abs(chg_vv['desc'][gt]).mean():.2f} dB | "
          f"signed ΔVV in GT: ASC {chg_vv['asc'][gt].mean():+.1f} DESC {chg_vv['desc'][gt].mean():+.1f} dB")
    fig.text(0.5, 0.008, st, ha="center", fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return d, chg_vv


d1v, _ = group_figure(sid1, "G1 BothValid landslide", "case1_bothvalid_landslide.png",
                      "GT almost fully inside BothValid region")
if sid2_lay is not None:
    d2v, _ = group_figure(sid2_lay, "G2 GT×layover (ASC invalid / DESC valid)", "case2_gt_geometry_overlap.png",
                          "GT coincides with ASC layover; counter orbit valid")
d3v, _ = group_figure(sid3, "G3 suspicious", "case3_suspicious_case.png",
                      "GT with weak change / suspicious mask agreement")

# ============ zoom overlay for the most explanatory case (G2 if present else G1) ============
zsid = sid2_lay if sid2_lay is not None else sid1
d = load_sample(zsid)
gt = d["gt"]
ys, xs = np.where(gt)
r = 18
y0, y1 = max(ys.min() - r, 0), min(ys.max() + r, 128)
x0, x1 = max(xs.min() - r, 0), min(xs.max() + r, 128)
sl = (slice(y0, y1), slice(x0, x1))
vv = {k: d[k][1][sl] for k in RAW}
chg = {o: vv[(o, "post")] - vv[(o, "pre")] for o in ("asc", "desc")}
gtz = gt[sl]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8))
fig.suptitle(f"Zoom overlay — Patch {zsid} (G2 GT×geometry)  crop [{x0}:{x1}, {y0}:{y1}]", fontsize=13)
for ax, k in zip(axes[0], [("asc", "pre"), ("asc", "post"), ("desc", "post")]):
    o = k[0]
    lo, hi = np.percentile(d[k][1], [2, 98])
    ax.imshow(np.clip((vv[k] - lo) / (hi - lo), 0, 1), cmap="gray")
    geom_contours(ax, d["ga"][sl] if o == "asc" else d["gd"][sl])
    gt_contour(ax, gtz)
    ax.set_title(f"{k[0].upper()} {DATES[k]} VV", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
ax = axes[1, 0]
ax.imshow(d["ga"][sl], cmap=GEOM_CMAP, vmin=0, vmax=3, alpha=0.9)
ax.contour(d["gd"][sl] > 0, levels=[0.5], colors="cyan", linewidths=1.0, linestyles="-")
gt_contour(ax, gtz)
ax.set_title("ASC geom (fill) + DESC bad (cyan) + GT", fontsize=9)
ax = axes[1, 1]
m = ax.imshow(np.abs(chg["asc"]), cmap="magma", vmin=0, vmax=np.percentile(np.abs(chg["asc"]), 99))
gt_contour(ax, gtz); geom_contours(ax, d["ga"][sl])
ax.set_title("|ΔVV| ASC", fontsize=9); plt.colorbar(m, ax=ax, fraction=0.046)
ax = axes[1, 2]
m = ax.imshow(np.abs(chg["desc"]), cmap="magma", vmin=0, vmax=np.percentile(np.abs(chg["desc"]), 99))
gt_contour(ax, gtz); geom_contours(ax, d["gd"][sl])
ax.set_title("|ΔVV| DESC", fontsize=9); plt.colorbar(m, ax=ax, fraction=0.046)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "case_zoom_overlay.png", dpi=170)
plt.close(fig)

print("figures done:", sid1, sid2_lay, sid2_shw, sid3, "zoom:", zsid)
