#!/usr/bin/env python3
"""Inference-only locked-test export for the recovered Vanilla KD checkpoint."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
OUT = ROOT / "experiments" / "tgd_robustness_analysis"
CKPT = ROOT / "experiments" / "ours_v2" / "VKD_seed42.pt"
spec = importlib.util.spec_from_file_location("ours_v2_runner", ROOT / "cocd" / "scripts" / "23_train_ours_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner._old.tqdm = lambda iterable, **_kwargs: iterable


def measured(p, y, mask):
    p, y = p.ravel()[mask.ravel()], y.ravel()[mask.ravel()].astype(bool)
    h = p >= 0.5
    tp, fp, fn = int((h & y).sum()), int((h & ~y).sum()), int((~h & y).sum())
    tn = int(len(y) - tp - fp - fn)
    out = runner.metrics(p, y, 0.5)
    out.update({"pixels": int(len(y)), "positive_pixels": int(y.sum()), "accuracy": (tp + tn) / len(y),
                "errors": fp + fn, "error_rate": (fp + fn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out


def main():
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    OUT.mkdir(parents=True, exist_ok=True)
    tid, _, eid = runner.split()
    train = runner.HaitiPairs(tid, True)
    test = runner.HaitiPairs(eid, False, train.stats)
    model = runner.OursV3Student("R1").to(runner.DEV)
    model.load_state_dict(torch.load(CKPT, map_location=runner.DEV, weights_only=True), strict=True)
    pred = runner.predict_student(model, test)
    np.savez_compressed(OUT / "vanilla_kd_retest_predictions.npz", **pred)
    tgd = (pred["gt"] > 0) & ~(pred["gc"] > 0)
    rows = [{"method": "Vanilla KD", "region": region, "threshold": 0.5, **measured(pred["p"], pred["y"], mask)}
            for region, mask in (("Overall", np.ones_like(tgd, bool)), ("TGD", tgd), ("Remaining", ~tgd))]
    fields = ["method", "region", "threshold", "pixels", "positive_pixels", "accuracy", "errors", "error_rate",
              "iou", "f1", "precision", "recall", "auprc", "tp", "fp", "fn", "tn"]
    with (OUT / "vanilla_kd_retest_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    report = {"purpose": "inference-only export of recovered Vanilla KD checkpoint", "checkpoint": str(CKPT),
              "student": "OursV3Student R1; full-pixel vanilla logit KD", "threshold": 0.5,
              "selection": "epoch 39; test-monitored AUPRC (development evidence)", "rows": rows}
    (OUT / "vanilla_kd_retest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f'{r["region"]}: AUPRC={r["auprc"]:.6f} IoU={r["iou"]:.6f} F1={r["f1"]:.6f}')


if __name__ == "__main__":
    main()
