"""Qualitative test-set figure for manuscript/main_dlr.tex.

Uses only saved test predictions (results*/<arm>/test_predictions.npz); no model
is re-run.  Candidate patches: both PRE1 and POST1 Sentinel-2 scenes cloud-free,
measured as <1% of pixels with B02, B03 and B04 reflectance all > 0.2 (the
dataset has no SCL cloud mask), and landslide fraction > 5%.  Rows: two cases
where the method changes the map (CN0001, KG0002), a neutral case (BR0001) and
the failure case (IE0001, small peat slides).
"""
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

OUT = C.EXP.parent / "manuscript" / "figures_dlr"
CASES = ["EID_CN0001__SID_00066", "EID_KG0002__SID_00618", "EID_BR0001__SID_00377", "EID_IE0001__SID_00110"]
ARMS = [("U-Net", C.HERE / "results" / "B0"), ("PKDOT", C.HERE / "results" / "PKDOT"),
        ("DIS2", C.HERE / "results" / "DIS2"), ("Ours", C.HERE / "results_rel" / "F_SSP"),
        ("Teacher", C.HERE / "results" / "T_fusion")]
COLS = ["S2 pre", "S2 post", "S1 pre VV", "S1 post VV", "GT"] + [a for a, _ in ARMS]


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def rgb(f, i, t):
    return np.dstack([stretch(f[f"{t}_{b}"][i, 0].astype(np.float32)) for b in ("B04", "B03", "B02")])


def sar(f, i, t):
    v = f[f"{t}_gVV"][i, 0].astype(np.float32)
    v = 10 * np.log10(np.maximum(np.where(np.isfinite(v), v, 0), 1e-8))
    return stretch(np.where(v > -70, v, np.nan))


def errors(y, p, valid):
    img = np.full(y.shape + (3,), .9)
    img[p & y] = (.18, .63, .17)        # true positive, green
    img[p & ~y] = (.84, .15, .16)       # false positive, red
    img[~p & y] = (.12, .47, .71)       # false negative, blue
    img[~valid] = 1.
    return img


def main():
    split = pd.read_csv(C.SPLIT).set_index("sample_id")
    preds = {a: np.load(p / "test_predictions.npz") for a, p in ARMS}
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, axes = plt.subplots(len(CASES), len(COLS), figsize=(7.16, 0.76 * len(CASES) + .2))
    for r, sid in enumerate(CASES):
        row = split.loc[sid]
        with h5py.File(row.h5_path, "r") as f:
            i = int(row.h5_index)
            y = f["None_MASK"][i].astype(bool).squeeze()
            panels = [rgb(f, i, "PRE1"), rgb(f, i, "POST1"), sar(f, i, "PRE1"), sar(f, i, "POST1")]
        gt = np.full(y.shape + (3,), .9)
        gt[y] = (.18, .63, .17)
        panels.append(gt)
        for a, _ in ARMS:
            d = preds[a]
            j = list(d["sample_ids"]).index(sid)
            p, v = d["probability"][j].astype(np.float32) >= .5, d["valid"][j]
            panels.append(errors(y, p, v))
            iou = (p & y & v).sum() / max(((p | y) & v).sum(), 1)
            axes[r, len(panels) - 1].text(.5, -.04, f"{100 * iou:.1f}", transform=axes[r, len(panels) - 1].transAxes,
                                          ha="center", va="top", fontsize=6)
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(.3)
            if r == 0:
                ax.set_title(COLS[c], fontsize=7, pad=2)
        axes[r, 0].set_ylabel(f"({'abcd'[r]}) {row.event_id}\n{'Ind.' if row.official_split == 'testind' else 'Dep.'}",
                              fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=.045, right=.995, top=.95, bottom=.04, wspace=.04, hspace=.22)
    fig.savefig(OUT / "qualitative.pdf")
    fig.savefig(OUT / "qualitative.png", dpi=250)


if __name__ == "__main__":
    main()
