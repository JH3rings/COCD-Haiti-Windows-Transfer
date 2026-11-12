"""LaTeX tables and the sensitivity figure for manuscript/main_dlr.tex, generated
only from saved test_metrics.json files (one test per frozen model, seed 42).
Missing runs print as '--'.  Output: manuscript/tables_dlr/*.tex and
manuscript/figures_dlr/sensitivity.pdf/png."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

H = C.HERE
TAB = C.EXP.parent / "manuscript" / "tables_dlr"
FIG = C.EXP.parent / "manuscript" / "figures_dlr"
TAB.mkdir(exist_ok=True)
R, RR, RD = H / "results", H / "results_rel", H / "results_direct"
IND, DEP = ("CN0001", "CO0001", "IN0004", "PH0003"), ("BR0001", "CA0002", "IE0001", "KG0002")


def m(path):
    p = Path(path) / "test_metrics.json"
    return C.read(p) if p.exists() else None


def row(r):
    if r is None:
        return [None] * 5
    s = r["per_official_subset"]
    return [100 * r["AUPRC"], 100 * r["IoU"], 100 * r["F1"], 100 * s["testind"]["IoU"], 100 * s["testspt"]["IoU"]]


def fmt(v, bold=False, nd=2):
    if v is None:
        return "--"
    t = f"{v:.{nd}f}"
    return f"\\textbf{{{t}}}" if bold else t


# ---------------------------------------------------------------- main table
MAIN = [("Direct segmentation", [
            ("U-Net~\\cite{Orynbaikyzy2025}", R / "B0"),
            ("UNet++~\\cite{UNetPlus2018}", RD / "UNETPP"),
            ("DeepLabv3~\\cite{DeepLab2017}", RD / "DLV3")]),
        ("Distillation from the optical--SAR teacher", [
            ("Vanilla KD~\\cite{Hinton2015}", R / "KD"),
            ("AT~\\cite{Zagoruyko2017}", R / "AT"),
            ("Multi-to-Uni~\\cite{Sun2027}", R / "M2U"),
            ("ACAM-KD~\\cite{Lan2025ACAM}", R / "ACAM"),
            ("PKDOT~\\cite{Aslam2024PKDOT}", R / "PKDOT"),
            ("DIS2~\\cite{Kieu2026DIS2}", R / "DIS2"),
            ("\\textbf{Ours}", RR / "F_SSP")])]
REF = [("Optical U-Net", R / "T_optical"), ("Optical--SAR U-Net (teacher)", R / "T_fusion")]


def main_table():
    vals = {n: row(m(p)) for _, g in MAIN for n, p in g}
    best = [max(v[j] for v in vals.values() if v[j] is not None) for j in range(5)]
    out = []
    for title, g in MAIN:
        out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for n, _ in g:
            v = vals[n]
            out.append(n + " & " + " & ".join(fmt(x, x is not None and abs(x - best[j]) < 5e-3) for j, x in enumerate(v)) + " \\\\")
        out.append("\\midrule")
    out.append("\\multicolumn{6}{@{}l}{\\emph{Optical input at inference (reference only)}} \\\\")
    for n, p in REF:
        out.append(n + " & " + " & ".join(fmt(x) for x in row(m(p))) + " \\\\")
    (TAB / "main_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ----------------------------------------------------------- per-event table
EVENT = [("U-Net", R / "B0"), ("DIS2", R / "DIS2"), ("Ours", RR / "F_SSP"), ("Teacher", R / "T_fusion")]


def event_table():
    out = []
    for n, p in EVENT:
        e = m(p)["per_event"]
        f = [100 * e[k]["F1"] for k in IND + DEP]
        mi, md = sum(f[:4]) / 4, sum(f[4:]) / 4
        out.append(n + " & " + " & ".join(f"{x:.0f}" for x in f[:4]) + f" & {mi:.1f} & "
                   + " & ".join(f"{x:.0f}" for x in f[4:]) + f" & {md:.1f} \\\\")
        if n == "Ours":
            out.append("\\midrule")
    (TAB / "event_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------- ablation table
ABL = [("Components", [
           ("U-Net", R / "B0"),
           ("+ shared prototypes (no optical)", RR / "F_SSP0"),
           ("+ $L_{\\mathrm{CEW}}$ (no prototypes)", RR / "UNET_W"),
           ("+ prototypes at 32\\,px + $L_{\\mathrm{CEW}}$", RR / "EPRM_W"),
           ("+ stage-wise prototypes, $L_{\\mathrm{CEW}}$ at 32/64\\,px", RR / "F_MS"),
           ("+ shared prototypes + $L_{\\mathrm{CEW}}$ (Ours)", RR / "F_SSP")]),
       ("Stage of $L_{\\mathrm{CEW}}$ (shared prototypes)", [
           ("16\\,px", RR / "F_SSP_S0"),
           ("32\\,px (Ours)", RR / "F_SSP"),
           ("64\\,px", RR / "F_SSP_S2"),
           ("128\\,px", RR / "F_SSP_S3"),
           ("32, 64, and 128\\,px", RR / "F_SSP_SALL")]),
       ("Relation target (prototypes at 32\\,px)", [
           ("Pairs, rank-stratified partners", RR / "EPRM_STR"),
           ("Pairs, soft correspondence (KL)", RR / "EPRM_CC"),
           ("Pairs, 30 least similar, OT$^\\dagger$", RR / "EPRM_CEraw"),
           ("Distribution, 32 quantiles, OT", RR / "EPRM_RD"),
           ("Distribution, Wasserstein", RR / "EPRM_W")]),
       ("Relation source (PKDOT loss, no prototypes)", [
           ("Label-defined relations", RR / "PKDOT_LBL"),
           ("Optical-teacher relations", R / "PKDOT"),
           ("Optical, cross-event pairs only", RR / "OursP1")])]


def ablation_table():
    out = []
    for i, (title, g) in enumerate(ABL):
        if i:
            out.append("\\midrule")
        out.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for n, p in g:
            v = row(m(p))
            out.append(f"{n} & {fmt(v[0])} & {fmt(v[1])} & {fmt(v[3])} \\\\")
    (TAB / "ablation_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- cost table
def cost_table():
    c = C.read(RR / "deployment_cost.json")
    rows = [("U-Net and KD students", "U-Net (B0, KD, AT, DIS2, ACAM, M2U, PKDOT)"),
            ("UNet++ (ResNet-34)", "UNet++ (ResNet-34)"), ("DeepLabv3 (ResNet-34)", "DeepLabv3 (ResNet-34)"),
            ("Ours", "Ours, scale-shared prototypes (F_SSP)"),
            ("Teacher (training only)", "Optical teacher (training only)")]
    out = [f"{n} & {c[k]['params_M']:.2f} & {c[k]['MACs_G']:.2f} \\\\" for n, k in rows]
    (TAB / "cost_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------- sensitivity figure
def sensitivity():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), sharey=True)
    for ax, (xlabel, pts) in zip(axes, [
            ("$\\lambda$", [(10, "F_SSP_L10"), (30, "F_SSP"), (100, "F_SSP_L100")]),
            ("$K$ (prototypes)", [(8, "F_SSP_K8"), (16, "F_SSP"), (32, "F_SSP_K32")])]):
        xs, iou, ap = [], [], []
        for x, a in pts:
            r = m(RR / a)
            if r is not None:
                xs.append(x); iou.append(100 * r["IoU"]); ap.append(100 * r["AUPRC"])
        ax.plot(xs, iou, "o-", color="#c8553d", ms=4, lw=1, label="IoU")
        for x, y in zip(xs, iou):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5)
        ax.axhline(52.20, color="#777777", lw=.8, ls="--")
        ax.set_xscale("log", base=2 if "K" in xlabel else 10)
        ax.set_xticks([x for x, _ in pts]); ax.set_xticklabels([str(x) for x, _ in pts])
        ax.minorticks_off()
        ax.margins(x=.2)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=.3, lw=.5)
    axes[0].set_ylabel("Test IoU (%)")
    axes[0].text(.03, .06, "U-Net", transform=axes[0].transAxes, fontsize=6.5, color="#555555")
    axes[0].set_ylim(51.5, 58.5)
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "sensitivity.pdf")
    fig.savefig(FIG / "sensitivity.png", dpi=250)


if __name__ == "__main__":
    main_table(); event_table(); ablation_table(); cost_table(); sensitivity()
    for f in sorted(TAB.glob("*.tex")):
        print("==", f.name); print(f.read_text(encoding="utf-8"))
