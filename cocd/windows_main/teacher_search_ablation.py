#!/usr/bin/env python3
"""Experiment 1, inference-only: T1 versus the same T1 with SearchWrite disabled.

This is a causal *inference intervention*: all learned T1 weights, inputs,
and dual fusion stay fixed.  Only the learned retrieval/write correction is
removed, by reading the model's pre-existing ``z_dual_noda`` branch.  It is not
presented as a separately trained no-search architecture.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from windows_main import da_search_main as da  # noqa: E402
from windows_main.models_da_search import DASearchTeacher  # noqa: E402

OUT = ROOT / "experiments" / "tgd_robustness_analysis"
CKPT = ROOT / "experiments" / "windows_main" / "da_search" / "T1_seed42.pt"
THRESHOLD = 0.5


def calc(p, y, mask):
    p, y = p[mask], y[mask].astype(bool)
    m = da.metric(p, y)
    h = p >= THRESHOLD
    tp, fp, fn = int((h & y).sum()), int((h & ~y).sum()), int((~h & y).sum())
    tn = int(len(y) - tp - fp - fn)
    m.update({"pixels": int(len(y)), "accuracy": (tp + tn) / len(y), "errors": fp + fn,
              "error_rate": (fp + fn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = da.build_sets()
    model = da.load_model(DASearchTeacher, CKPT)
    pred = da.predict_t1(model, test)
    np.savez_compressed(OUT / "t1_search_ablation_predictions.npz", **pred)
    tgd = ((pred["gt"] > 0) & ~(pred["gc"] > 0)).ravel()
    rows = []
    for condition, key in (("T1", "p"), ("T1 w/o Search", "noda")):
        for region, mask in (("Overall", np.ones_like(tgd, dtype=bool)), ("TGD", tgd), ("Remaining", ~tgd)):
            rows.append({"method": condition, "region": region, "threshold": THRESHOLD,
                         **calc(pred[key].ravel(), pred["y"].ravel(), mask)})
    fields = ["method", "region", "threshold", "pixels", "accuracy", "errors", "error_rate", "auprc", "iou", "f1", "precision", "recall", "tp", "fp", "fn", "tn"]
    with (OUT / "teacher_search_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    t1 = {(r["region"]): r for r in rows if r["method"] == "T1"}
    off = {(r["region"]): r for r in rows if r["method"] == "T1 w/o Search"}
    delta = {region: {metric: t1[region][metric] - off[region][metric]
                      for metric in ("accuracy", "auprc", "iou", "f1", "precision", "recall", "error_rate")}
             for region in t1}
    report = {"experiment": "T1 Search utility: fixed-checkpoint inference intervention", "checkpoint": str(CKPT),
              "intervention": "use z_dual_noda rather than z_dual; bypasses retrieval and SearchWrite correction while retaining trained backbone, dual fusion, and decoder",
              "not_claimed": "This does not replace a separately trained matched no-search Teacher ablation.",
              "tgd_definition": "target geometry positive AND counter geometry negative", "delta_T1_minus_no_search": delta}
    (OUT / "teacher_search_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f'{r["method"]:16s} {r["region"]:9s} Acc={r["accuracy"]:.6f} F1={r["f1"]:.6f} IoU={r["iou"]:.6f} AUPRC={r["auprc"]:.6f}')
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
