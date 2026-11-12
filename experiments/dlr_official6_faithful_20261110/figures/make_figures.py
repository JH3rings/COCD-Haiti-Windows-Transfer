"""Paper figures for manuscript/main.tex (data: results_rel/diag_learned.json)."""
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

OUT = C.EXP.parent / "manuscript" / "figures_dlr"
OUT.mkdir(exist_ok=True)

# Regenerate the current compact vector diagram from its single source.
import runpy
runpy.run_path(str(C.EXP.parent / "manuscript" / "tools" / "draw_method.py"))

# agreement vs test IoU
d = C.read(C.HERE / "results_rel" / "diag_learned.json")
names = {"B0": "U-Net", "PKDOT": "PKDOT", "PKDOT_LBL": "", "OursP1": "",
         "EPRM": "", "EPRM_CEraw": "", "EPRM_CEraw_repro": "", "EPRM_RD": "",
         "EPRM_STR": "", "EPRM_LC": "", "EPRM_SEPnb": "", "EPRM_SEP": "", "EPRM_CC": ""}
pair = {"EPRM_STR", "EPRM_CC", "EPRM_SEP", "EPRM_SEPnb"}
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
xs, ys = [], []
for a, lab in names.items():
    x, y = d["overlap"]["val"][a]["fg-bg"], 100 * d["test_IoU"][a]
    xs.append(x); ys.append(y)
    c = "#c8553d" if a in pair else "#4a78b0" if a != "B0" else "#555555"
    ax.scatter(x, y, s=18, color=c, zorder=3,
               label=("pair-matching losses" if a == "EPRM_CC" else "other students" if a == "PKDOT" else None))
    if lab:
        ax.annotate(lab, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=6.5)
k, b = np.polyfit(xs, ys, 1)
xx = np.linspace(min(xs) - .02, max(xs) + .02, 10)
ax.plot(xx, k * xx + b, color="#999999", lw=.8, ls="--", zorder=1)
r = np.corrcoef(xs, ys)[0, 1]
ax.text(.97, .95, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
ax.set_xlabel("Agreement with teacher's 30 least-similar regions")
ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), fontsize=6.5, frameon=False)
ax.set_ylabel("Test IoU (%)")
ax.grid(alpha=.3, lw=.5)
fig.tight_layout()
fig.savefig(OUT / "overlap.pdf")
fig.savefig(OUT / "overlap.png", dpi=200)
print("r =", r)
