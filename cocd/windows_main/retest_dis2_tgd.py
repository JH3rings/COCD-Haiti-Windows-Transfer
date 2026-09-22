#!/usr/bin/env python3
"""Inference-only TGD re-test for the retained DIS2-port checkpoint."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COCD = ROOT / "cocd"
sys.path.insert(0, str(COCD))
OUT = ROOT / "experiments" / "tgd_robustness_analysis"
# The v4 output path was cleaned after the monitored run.  The identical
# retained DIS2-port state is archived under the v1 protocol checkpoints.
CKPT = ROOT / "legacy_checkpoints" / "v1_protocol" / "DIS2_port_seed42.pt"
HISTORICAL = ROOT / "experiments" / "windows_main" / "cgsearch" / "test_monitored_results_seed42.csv"

spec = importlib.util.spec_from_file_location("ours_v2_runner", COCD / "scripts" / "23_train_ours_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# Keep the analysis log concise while rebuilding the pre-fit normalization cache.
runner._old.tqdm = lambda iterable, **_kwargs: iterable
THRESHOLD = 0.5


def measured(p, y, mask):
    p = p.ravel()[mask.ravel()]
    y = y.ravel()[mask.ravel()].astype(bool)
    h = p >= THRESHOLD
    tp, fp, fn = int((h & y).sum()), int((h & ~y).sum()), int((~h & y).sum())
    tn = int(len(y) - tp - fp - fn)
    d = float(len(y))
    result = runner.metrics(p, y, THRESHOLD)
    result.update({"pixels": int(d), "positive_pixels": int(y.sum()), "accuracy": (tp + tn) / d,
                   "errors": fp + fn, "error_rate": (fp + fn) / d, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return result


def main():
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    OUT.mkdir(parents=True, exist_ok=True)
    tid, vid, eid = runner.split()
    if vid:
        raise RuntimeError("Expected the frozen no-validation v4 split")
    train = runner.HaitiPairs(tid, True)
    test = runner.HaitiPairs(eid, False, train.stats)
    model = runner.OursV3Student("R1").to(runner.DEV)
    model.load_state_dict(torch.load(CKPT, map_location=runner.DEV, weights_only=True), strict=True)
    cache = OUT / "dis2_checkpoint_retest_predictions.npz"
    pred = dict(np.load(cache)) if cache.exists() else runner.predict_student(model, test)
    if not cache.exists():
        np.savez_compressed(cache, **pred)
    tgd = (pred["gt"] > 0) & ~(pred["gc"] > 0)
    rows = []
    for region, mask in (("Overall", np.ones_like(tgd, bool)), ("TGD", tgd), ("Remaining", ~tgd)):
        rows.append({"method": "DIS2-port", "region": region, "threshold": THRESHOLD, **measured(pred["p"], pred["y"], mask)})
    fields = ["method", "region", "threshold", "pixels", "positive_pixels", "accuracy", "errors", "error_rate", "iou", "f1", "precision", "recall", "auprc", "tp", "fp", "fn", "tn"]
    with (OUT / "dis2_checkpoint_retest_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    historical = list(csv.DictReader(HISTORICAL.open(encoding="utf-8-sig")))
    comparison = {}
    for label, legacy_region in (("Overall", "all"), ("TGD", "G10")):
        old = next(r for r in historical if r["method"] == "DIS2" and r["partition"] == "overall" and r["region"] == legacy_region)
        new = next(r for r in rows if r["region"] == label)
        comparison[label] = {m: float(new[m]) - float(old[m]) for m in ("iou", "f1", "precision", "recall", "auprc")}
    report = {"purpose": "inference-only re-test of retained DIS2-port checkpoint", "checkpoint": str(CKPT),
              "input": "target-orbit pre/post SAR plus orbit indicator only", "threshold": THRESHOLD,
              "tgd_definition": "existing G10: target geometry positive and counter geometry negative",
              "historical_csv_metric_difference_retest_minus_historical": comparison, "rows": rows}
    (OUT / "dis2_checkpoint_retest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f'{r["region"]:9s} Accuracy={r["accuracy"]:.6f} F1={r["f1"]:.6f} IoU={r["iou"]:.6f} '
              f'Precision={r["precision"]:.6f} Recall={r["recall"]:.6f} AUPRC={r["auprc"]:.6f} '
              f'Errors={r["errors"]}/{r["pixels"]}')
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
