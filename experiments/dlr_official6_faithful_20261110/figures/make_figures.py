"""Paper figures for manuscript/main_dlr.tex (data: results_rel/diag_learned.json)."""
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

# method figure: SVG -> PDF (vector).  MuPDF ignores SVG markers, so explicit
# arrowhead polygons are added, and the size is fixed to the viewBox.
import re


def with_arrowheads(text):
    heads = []
    for m in re.finditer(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"[^>]*marker-end', text):
        heads.append(tuple(map(float, m.groups())))
    for m in re.finditer(r'<path d="([^"]+)"[^>]*marker-end', text):
        pts = [tuple(map(float, q)) for q in re.findall(r"([\d.]+) ([\d.]+)", m.group(1))]
        heads.append(pts[-2] + pts[-1])
    polys = []
    for x1, y1, x2, y2 in heads:
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** .5 or 1
        ux, uy = dx / n, dy / n
        bx, by = x2 - 6 * ux, y2 - 6 * uy
        polys.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx - 3 * uy:.1f},{by + 3 * ux:.1f} {bx + 3 * uy:.1f},{by - 3 * ux:.1f}" fill="#444"/>')
    text = text.replace('width="100%" viewBox="0 0 680 540"', 'width="680" height="540" viewBox="0 0 680 540"')
    return text.replace("</svg>", "\n".join(polys) + "\n</svg>")


src = (Path(__file__).parent / "method.svg").read_text(encoding="utf-8")
tmp = OUT / "_method_print.svg"
tmp.write_text(with_arrowheads(src), encoding="utf-8")
svg = fitz.open(str(tmp))
pdf = fitz.open("pdf", svg.convert_to_pdf())
pdf.save(str(OUT / "method.pdf"))
pdf[0].get_pixmap(dpi=150).save(str(OUT / "method_preview.png"))
tmp.unlink()

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
