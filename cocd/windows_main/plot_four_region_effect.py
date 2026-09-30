"""Plot descriptive fixed-threshold region effects from saved inference counts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/GRSL_complete_paper_package/99_historical_qualitative/four_region_rescue"


def main() -> None:
    with (OUT / "regional_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    lookup = {(r["target_orbit"], r["region"], r["method"]): r for r in records}
    labels = ["ASC target / G10", "DESC target / G10", "ASC target / G00", "DESC target / G00"]
    keys = [("asc", "G10"), ("desc", "G10"), ("asc", "G00"), ("desc", "G00")]
    s0 = np.array([float(lookup[o, g, "S0"]["F1"]) for o, g in keys])
    s2 = np.array([float(lookup[o, g, "S2"]["F1"]) for o, g in keys])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.2, 5.1), layout="constrained")
    ax.barh(y + .18, s0, height=.32, color="#677486", label="S0 target")
    ax.barh(y - .18, s2, height=.32, color="#20a46a", label="S2 target after KD")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, .93)
    ax.set_xlabel("Pixel-pooled F1 at the same fixed threshold (0.5)")
    ax.set_title("343 test locations: method change within each geometry region")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    for i, (a, b) in enumerate(zip(s0, s2)):
        ax.text(b + .008, i - .18, f"+{(b-a)*100:.2f} pp", va="center", fontsize=9, color="#116344")
    fig.supxlabel("G10 = target distorted, counter valid; G00 = both valid. GT and geometry are separate masks.\n"
                  "These are test-monitored descriptive results, not independent held-out evidence; overlapping patches are not independent.",
                  fontsize=9)
    fig.savefig(OUT / "regional_effect.png", dpi=220)
    fig.savefig(OUT / "regional_effect.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
