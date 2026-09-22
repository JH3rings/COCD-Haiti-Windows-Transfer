#!/usr/bin/env python3
"""Analysis-only re-test of external single-orbit baselines on the frozen Haiti test set.

Loads each completed v4 ``best.pt`` checkpoint, reconstructs the exact frozen
train normalization and test loader used at training time, performs inference
only, and writes regional TGD (formerly G10) metrics plus an integrity check
against the archived test predictions.  No model parameters are updated.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
COCD = ROOT / "cocd"
sys.path.insert(0, str(COCD))
OUT = ROOT / "experiments" / "tgd_robustness_analysis"
SOURCE = ROOT / "experiments" / "single_orbit_baselines_v3"

spec = importlib.util.spec_from_file_location("external_seats", COCD / "scripts" / "23_train_external_baselines.py")
seats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seats)
rapid = seats.rapid
# The shared dataset emits one progress bar per cache construction; suppress
# those cosmetic bars so this inference-only audit has concise terminal output.
rapid.tqdm = lambda iterable, **_kwargs: iterable

METHODS = (("boehm", "Boehm SAR U-Net++"), ("cdnette", "CDNetE Early Fusion"), ("mfewf", "MFEWF adapted"))
THRESHOLD = 0.5


def metrics(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    p, y = p[mask], y[mask].astype(bool)
    out = rapid.metric(p, y, t=THRESHOLD)
    h = p >= THRESHOLD
    out["pixels"] = int(mask.sum())
    out["errors"] = int((h != y).sum())
    out["error_rate"] = float(out["errors"] / max(1, out["pixels"]))
    return out


@torch.no_grad()
def infer(net, seat: str, kind: str, ds) -> dict:
    net.eval()
    out = {k: [] for k in ("p", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _real in seats.loader(ds, shuffle=False):
        x = torch.cat((a, d)).to(rapid.DEV)
        logits = seats.forward_seat(net, seat, x)
        p = seats.prob_seat(logits, kind).cpu().numpy()
        out["p"].append(p)
        out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy())
        out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.concatenate((np.zeros(len(a), dtype=np.int8), np.ones(len(a), dtype=np.int8))))
    return {k: np.concatenate(v) for k, v in out.items()}


def build(seat: str):
    if seat == "boehm":
        return seats.build_boehm(pretrained=False)
    if seat == "cdnette":
        return seats.build_cdnette(pretrained=False)
    if seat == "mfewf":
        return seats.build_mfewf(pretrained=False)
    raise ValueError(seat)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rapid.seed()
    train_ids, val_ids, test_ids = rapid.split()
    if val_ids:
        raise RuntimeError("Frozen v4 protocol is expected to have no validation locations")
    # This reconstructs the training-set normalization exactly; test-only data
    # are not used to fit normalization statistics.
    train = rapid.HaitiPairs(train_ids, True)
    test = rapid.HaitiPairs(test_ids, False, train.stats)
    rows, integrity = [], {}
    for seat, label in METHODS:
        net, kind = build(seat)
        ckpt = SOURCE / seat / "best.pt"
        net.load_state_dict(torch.load(ckpt, map_location=rapid.DEV, weights_only=True))
        net.to(rapid.DEV)
        pred = infer(net, seat, kind, test)
        old = np.load(SOURCE / seat / "test_predictions.npz")
        same_shape = pred["p"].shape == old["p"].shape
        max_abs = float(np.max(np.abs(pred["p"] - old["p"]))) if same_shape else None
        integrity[label] = {"checkpoint": str(ckpt), "archived_prediction_shape_match": same_shape,
                            "max_abs_probability_difference_vs_archived": max_abs}
        np.savez_compressed(OUT / f"{seat}_checkpoint_retest_predictions.npz", **pred)
        tgd = (pred["gt"] > 0) & ~(pred["gc"] > 0)
        masks = {"Overall": np.ones_like(tgd, dtype=bool), "TGD": tgd, "Remaining": ~tgd}
        for region, mask in masks.items():
            m = metrics(pred["p"].ravel(), pred["y"].ravel(), mask.ravel())
            rows.append({"method": label, "region": region, **m})
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = ["method", "region", "pixels", "errors", "error_rate", "auprc", "iou", "f1", "precision", "recall"]
    with (OUT / "external_checkpoint_retest_region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    report = {"purpose": "inference-only external checkpoint re-test", "threshold": THRESHOLD,
              "tgd_definition": "existing G10 mask: target geometry positive and counter geometry negative",
              "split": {"train_locations_for_normalization": len(train_ids), "test_locations": len(test_ids)},
              "integrity": integrity, "rows": rows}
    (OUT / "external_checkpoint_retest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f'{r["method"]:22s} {r["region"]:9s} AUPRC={r["auprc"]:.6f} IoU={r["iou"]:.6f} errors={r["errors"]}/{r["pixels"]} ({r["error_rate"]:.4%})')
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
