#!/usr/bin/env python3
"""Build the read-only GRSL material bundle for the locked Haiti experiment.

This script never calls a training function and never writes into the locked
experiment directory.  It evaluates only the already selected T0/T1/S0/SO
checkpoints, reuses recorded historical baselines, and writes into the
adjacent ``grsl_assets_v1`` directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[5]
ASSETS = Path(__file__).resolve().parents[1]
LOCKED = ASSETS.parent
sys.path.insert(0, str(ROOT / "cocd"))
from windows_main import da_search_main as da  # noqa: E402
from windows_main import g10_mechanism_analysis as g10  # noqa: E402

OUT = ASSETS
DATA = OUT / "data"
TABLES = OUT / "tables"
FIG_MAIN = OUT / "figures" / "main"
FIG_CAND = OUT / "figures" / "candidates"
FIG_SOURCE = OUT / "figures" / "source"
FIG_DATA = OUT / "figure_data"
PRED_DIR = FIG_DATA / "predictions"
THRESHOLD = 0.5
EPS = 1e-8

for p in (DATA, TABLES, FIG_MAIN, FIG_CAND, FIG_SOURCE, FIG_DATA, PRED_DIR,
          OUT / "figure_data" / "qualitative_crops", OUT / "captions", OUT / "preview"):
    p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metric(p, y, mask=None):
    p = np.asarray(p); y = np.asarray(y).astype(bool)
    if mask is not None:
        mask = np.asarray(mask).astype(bool); p, y = p[mask], y[mask]
    n = int(y.size)
    if n == 0:
        return {"pixels": 0, "positive_pixels": 0, "accuracy": None, "iou": None, "f1": None,
                "precision": None, "recall": None, "auprc": None, "tp": None, "fp": None,
                "fn": None, "tn": None}
    h = p >= THRESHOLD
    tp = int((h & y).sum()); fp = int((h & ~y).sum()); fn = int((~h & y).sum()); tn = int((~h & ~y).sum())
    pr = tp / (tp + fp + EPS); rc = tp / (tp + fn + EPS)
    return {"pixels": n, "positive_pixels": int(y.sum()), "accuracy": float((h == y).mean()),
            "iou": float(tp / (tp + fp + fn + EPS)), "f1": float(2 * pr * rc / (pr + rc + EPS)),
            "precision": float(pr), "recall": float(rc),
            "auprc": float(average_precision_score(y, p)) if y.any() else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def region_masks(pred):
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    return {"Overall": np.ones_like(gt, dtype=bool), "TGD": gt & ~gc, "Remaining": ~(gt & ~gc),
            "G00": ~gt & ~gc, "G01": ~gt & gc, "G10": gt & ~gc, "G11": gt & gc}


def internal_region(paper_region):
    return {"Overall": "overall", "TGD": "G10", "Remaining": "non-G10"}.get(paper_region, paper_region)


def paper_model(model):
    return {"SO": "SO-reference", "Student": "S0"}.get(model, model)


def load_current_predictions(test):
    models = g10.load_locked_models()
    stores = {name: {k: [] for k in ("sample_id", "p", "y", "gt", "gc", "orbit")} for name in models}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        start = 0
        for a, d, y, ga, gd, _ in da.loader(test):
            b = len(a); ids = np.asarray(test.ids[start:start + b], dtype=np.int32)
            a, d = a.to(da.DEV), d.to(da.DEV)
            base = {
                "sample_id": np.concatenate((ids, ids)),
                "y": torch.cat((y, y)).numpy().astype(np.uint8),
                "gt": torch.cat((ga, gd)).numpy().astype(np.uint8),
                "gc": torch.cat((gd, ga)).numpy().astype(np.uint8),
                "orbit": np.asarray([0] * b + [1] * b, dtype=np.int8),
            }
            oa, od = models["T0"].forward_pair(a, d)
            pt0 = torch.sigmoid(torch.cat((oa["z_dual"], od["z_dual"])))[:, 0].cpu().numpy()
            oa, od = models["T1"].forward_pair(a, d)
            pt1 = torch.sigmoid(torch.cat((oa["z_dual"], od["z_dual"])))[:, 0].cpu().numpy()
            ps = torch.sigmoid(models["Student"](torch.cat((a, d)))["z"])[:, 0].cpu().numpy()
            pa = torch.sigmoid(models["SO"](a)["z"])[:, 0]
            pd = torch.sigmoid(models["SO"](d)["z"])[:, 0]
            vals = {"T0": pt0, "T1": pt1, "Student": ps, "SO": torch.cat((pa, pd)).cpu().numpy()}
            for name, p in vals.items():
                for key, value in base.items(): stores[name][key].append(value)
                stores[name]["p"].append(p.astype(np.float32))
            start += b
    result = {}
    for name, store in stores.items():
        result[name] = {k: np.concatenate(v) for k, v in store.items()}
        np.savez_compressed(PRED_DIR / f"predictions_{name}.npz", **result[name])
    return result


def add_metric_rows(rows, model, paper_name, status, pred):
    for region, mask in region_masks(pred).items():
        m = metric(pred["p"], pred["y"], mask)
        rows.append({"model": model, "paper_name": paper_name, "status": status,
                     "region": region, "internal_region": internal_region(region),
                     "paper_region": region, **m, "result_origin": "eval_only_locked_checkpoint"})


def load_external_predictions():
    result = {}
    mapping = {"boehm": "Boehm SAR U-Net++", "cdnette": "CDNetE Early Fusion", "mfewf": "MFEWF adapted"}
    for folder, name in mapping.items():
        path = ROOT / "experiments" / "single_orbit_baselines_v3" / folder / "test_predictions.npz"
        if path.exists():
            z = np.load(path)
            result[name] = {k: z[k] for k in z.files}
    return result


def historical_rows():
    path = ROOT / "experiments" / "windows_main" / "cgsearch" / "test_monitored_results_seed42.csv"
    if not path.exists(): return []
    table = pd.read_csv(path)
    out = []
    for method, display in (("SO", "SO historical"), ("Vanilla KD", "Vanilla KD"), ("DIS2", "DIS2")):
        q = table[(table.method == method) & (table.partition == "overall") & (table.region.isin(["all", "G10"]))]
        for _, r in q.iterrows():
            region = "Overall" if r.region == "all" else "TGD"
            out.append({"model": display, "paper_name": display, "status": "missing_checkpoint",
                        "region": region, "internal_region": "overall" if region == "Overall" else "G10",
                        "paper_region": region, "pixels": int(r.pixels), "positive_pixels": None,
                        "accuracy": None, "iou": float(r.iou), "f1": float(r.f1),
                        "precision": float(r.precision), "recall": float(r.recall),
                        "auprc": float(r.auprc), "tp": None, "fp": None, "fn": None, "tn": None,
                        "result_origin": "recorded_historical_cgsearch_csv; checkpoint_not_retained"})
    return out


def write_metrics(preds, external):
    rows = []
    names = {"SO": ("SO-reference", "Single-orbit baseline", "eval_only_completed"),
             "T0": ("T0", "Dual-orbit baseline", "eval_only_completed"),
             "T1": ("T1", "Dual-orbit search teacher", "eval_only_completed"),
             "Student": ("S0", "Ours (single-orbit)", "eval_only_completed")}
    for key, (model, paper, status) in names.items(): add_metric_rows(rows, model, paper, status, preds[key])
    for paper, pred in external.items(): add_metric_rows(rows, paper, paper, "verified_existing", pred)
    rows.extend(historical_rows())
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "metrics_full.csv", index=False)
    return table


def write_per_image(preds):
    rows = []
    for model, pred in preds.items():
        masks = region_masks(pred)
        for i in range(len(pred["p"])):
            for region, mask in masks.items():
                m = metric(pred["p"][i], pred["y"][i], mask[i])
                rows.append({"model": model, "paper_model": paper_model(model), "sample_id": int(pred["sample_id"][i]),
                             "target_orbit": "ASC" if int(pred["orbit"][i]) == 0 else "DESC",
                             "region": region, "internal_region": internal_region(region),
                             "paper_region": region, **m})
    pd.DataFrame(rows).to_csv(DATA / "metrics_per_image.csv", index=False)
    orbit_rows = []
    for model, pred in preds.items():
        for orbit, label in ((0, "ASC"), (1, "DESC")):
            mask = pred["orbit"] == orbit
            for region, rmask in region_masks(pred).items():
                m = metric(pred["p"][mask], pred["y"][mask], rmask[mask])
                orbit_rows.append({"model": model, "paper_model": paper_model(model), "target_orbit": label, "region": region,
                                   "internal_region": internal_region(region), "paper_region": region, **m})
    pd.DataFrame(orbit_rows).to_csv(DATA / "metrics_per_orbit.csv", index=False)


def write_diagnostics(preds, report):
    so, t0, t1, s0 = preds["SO"], preds["T0"], preds["T1"], preds["Student"]
    y = so["y"].astype(bool); g10 = region_masks(so)["TGD"]
    correctness = {k: ((v["p"] >= THRESHOLD) == y) for k, v in preds.items()}
    rows = []
    for subset, mask in (("T0 subset", g10 & ~correctness["SO"] & correctness["T0"]),
                         ("T1 subset", g10 & ~correctness["SO"] & correctness["T1"]),
                         ("Union subset", g10 & ~correctness["SO"] & (correctness["T0"] | correctness["T1"]))):
        for model in ("SO", "T0", "T1", "Student"):
            m = metric(preds[model]["p"], y, mask)
            rows.append({"subset": subset, "model": model, "paper_model": paper_model(model), "pixels": int(mask.sum()),
                         "share_of_TGD_percent": float(mask.sum() / (g10.sum() + EPS) * 100), **m})
        for model in ("DIS2", "Vanilla KD", "Single-DA-scratch"):
            rows.append({"subset": subset, "model": model, "paper_model": paper_model(model), "pixels": int(mask.sum()),
                         "share_of_TGD_percent": float(mask.sum() / (g10.sum() + EPS) * 100),
                         "status": "missing_prediction"})
    pd.DataFrame(rows).to_csv(DATA / "diagnostic_subset_metrics.csv", index=False)
    trans = (g10 & ((so["p"] >= THRESHOLD) == y))
    rows = []
    a, b = so["p"] >= THRESHOLD, s0["p"] >= THRESHOLD
    for s, m in (("both_correct", a == y), ("SO_wrong_Student_correct", (a != y) & (b == y)),
                 ("SO_correct_Student_wrong", (a == y) & (b != y)), ("both_wrong", (a != y) & (b != y))):
        mm = g10 & m
        rows.append({"region": "TGD", "transition": s, "pixels": int(mm.sum()),
                     "share_of_TGD_percent": float(mm.sum() / (g10.sum() + EPS) * 100)})
    pd.DataFrame(rows).to_csv(DATA / "tgd_error_transitions.csv", index=False)
    crows = []
    c = report["experiment_2_counter_dependence"]["conditions"]
    for condition, regions in c.items():
        for region, vals in regions.items():
            crows.append({"model": "T1", "paper_model": "T1", "condition": condition, "region": region,
                          "internal_region": region, "paper_region": "TGD" if region == "G10" else "Remaining" if region == "non-G10" else "Overall", **vals})
    pd.DataFrame(crows).to_csv(DATA / "counter_interventions.csv", index=False)
    drows = []
    stats = report["experiment_4_da_behavior"]["stats"]
    for model, regions in stats.items():
        for region, values in regions.items():
            if not isinstance(values, dict): continue
            for variable, s in values.items(): drows.append({"model": model, "paper_model": paper_model(model), "region": region,
                "internal_region": region, "paper_region": "TGD" if region == "G10" else "Remaining", "variable": variable, **s})
    pd.DataFrame(drows).to_csv(DATA / "da_statistics.csv", index=False)


def write_registry():
    current = {
        "SO": ("SO", "Single-orbit baseline", LOCKED / "SO_backbone_seed42.pt", "target only", "ImageNet/backbone compact init", "backbone", "not a Student"),
        "T0": ("T0", "Dual-orbit baseline", LOCKED / "T0_seed42.pt", "target+counter", "SO backbone", "backbone+fusion", "L_dual"),
        "T1": ("T1", "Dual-orbit search teacher", LOCKED / "T1_seed42.pt", "target+counter", "T0", "backbone+fusion+policy+write", "L_dual+0.10 L_aux"),
        "S0": ("S0", "Ours (single-orbit)", LOCKED / "S0_seed42.pt", "target only", "T1", "backbone+target policy+write; counter/fusion removed", "L_seg"),
    }
    rows = []
    for key, (mid, paper, path, inp, init, transfer, loss) in current.items():
        rows.append({"model_id": mid, "paper_name": paper, "role": paper, "implementation_version": "windows_main/da_search",
                     "checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "config_path": str(ROOT / "cocd/configs/protocol_da_search.json"),
                     "log_path": str(LOCKED / f"{key}_seed42_progress.json") if key != "SO" else "",
                     "seed": 42, "best_epoch": None, "completed_epochs": None, "training_inputs": inp,
                     "adaptation_inputs": "target+GT" if key == "S0" else "", "inference_inputs": inp,
                     "initialization_source": init, "transferred_modules": transfer, "loss_terms": loss,
                     "evaluation_sample_list": "data/splits_v3/test_ids.csv", "prediction_cache_path": str(PRED_DIR / f"predictions_{'Student' if key == 'S0' else key}.npz"),
                     "metric_function": "average_precision + pooled positive-class IoU/F1 at threshold 0.5", "threshold": 0.5,
                     "aggregation_rule": "orbit-conditioned ASC+DESC pixel concatenation", "result_origin": "locked checkpoint; eval-only",
                     "status": "eval_only_completed", "notes": "test partition also selected checkpoints in the training protocol"})
    extras = [
        ("DIS2", "DIS2", "missing_checkpoint", "historical CGSearch CSV; current checkpoint deleted during q3/q4 cleanup"),
        ("Vanilla KD", "Vanilla KD", "missing_checkpoint", "historical CGSearch CSV; current checkpoint deleted during q3/q4 cleanup"),
        ("Boehm", "Boehm SAR U-Net++", "verified_existing", "local prediction cache and checkpoint"),
        ("CDNetE", "CDNetE Early Fusion", "verified_existing", "local prediction cache and checkpoint"),
        ("MFEWF", "MFEWF adapted", "verified_existing", "local prediction cache and checkpoint"),
        ("CGSearch", "CGSearch historical", "not_current_method", "old ordinary-attention experiment kept only as historical index"),
    ]
    for mid, paper, status, note in extras:
        rows.append({"model_id": mid, "paper_name": paper, "role": "historical or external baseline", "implementation_version": "recorded artifacts",
                     "checkpoint_path": "", "checkpoint_sha256": "", "config_path": "", "log_path": "", "seed": 42,
                     "best_epoch": None, "completed_epochs": None, "training_inputs": "unknown/recorded", "adaptation_inputs": "unknown",
                     "inference_inputs": "single orbit" if mid not in ("CGSearch",) else "historical", "initialization_source": "recorded artifact",
                     "transferred_modules": "unknown", "loss_terms": "recorded artifact", "evaluation_sample_list": "recorded historical test table",
                     "prediction_cache_path": "", "metric_function": "recorded", "threshold": 0.5,
                     "aggregation_rule": "recorded", "result_origin": "recorded local artifact", "status": status, "notes": note})
    pd.DataFrame(rows).to_csv(DATA / "model_registry.csv", index=False)


def write_source_and_dataset_manifests():
    srcs = [ROOT / "experiments/windows_main/da_search/final_report.json", ROOT / "experiments/windows_main/da_search/g10_mechanism_report.json",
            ROOT / "experiments/windows_main/da_search/g10_mechanism_summary.csv", ROOT / "cocd/windows_main/g10_mechanism_analysis.py",
            ROOT / "cocd/windows_main/models_da_search.py", ROOT / "cocd/windows_main/da_search_main.py", ROOT / "cocd/configs/protocol_da_search.json",
            ROOT / "experiments/windows_main/cgsearch/test_monitored_results_seed42.csv"]
    entries = []
    for p in srcs:
        entries.append({"path": str(p), "exists": p.exists(), "sha256": sha256(p)})
    dump_json(OUT / "source_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "sources": entries,
                                               "policy": "read-only; no training; no checkpoint overwritten"})
    manifest = ROOT / "cocd/manifests/dataset_manifest.csv"
    if manifest.exists(): shutil.copy2(manifest, DATA / "dataset_manifest.csv")
    split = ROOT / "data/splits_v3"
    rows = []
    for name in ("train_ids.csv", "test_ids.csv", "val_ids.csv"):
        p = split / name
        if p.exists(): rows.append({"file": str(p), "exists": True, "rows": len(pd.read_csv(p)), "sha256": sha256(p)})
        else: rows.append({"file": str(p), "exists": False, "rows": None, "sha256": None})
    pd.DataFrame(rows).to_csv(DATA / "split_manifest.csv", index=False)


def markdown_table(df):
    d = df.copy()
    for col in d.columns:
        d[col] = d[col].map(lambda x: f"{float(x):.6f}" if isinstance(x, (float, np.floating)) else x)
    return d.to_markdown(index=False)


def latex_table(df, caption, label):
    return "\\begin{table}[t]\n\\centering\n" + df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)) + f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n"


def write_tables(metrics, report):
    t1_models = ["SO-reference", "DIS2", "Vanilla KD", "Boehm SAR U-Net++", "CDNetE Early Fusion", "MFEWF adapted", "S0"]
    q = metrics[(metrics.model.isin(t1_models)) & (metrics.region.isin(["Overall", "TGD", "Remaining"]))].copy()
    wide = q[q.region.isin(["Overall", "TGD", "Remaining"])].pivot_table(index="model", columns="region", values=["auprc", "iou", "f1", "precision", "recall"], aggfunc="first")
    out = []
    for model in t1_models:
        row = {"Method": model}
        for region in ("Overall", "TGD", "Remaining"):
            for m in ("auprc", "iou", "f1"):
                try: row[f"{region} {m}"] = wide.loc[model, (m, region)]
                except KeyError: row[f"{region} {m}"] = np.nan
        out.append(row)
    t1 = pd.DataFrame(out)
    for ext in ("csv", "md", "tex"): (TABLES / f"table1_deployment.{ext}").write_text(
        t1.to_csv(index=False) if ext == "csv" else markdown_table(t1.fillna("—")) if ext == "md" else latex_table(t1.fillna("—"), "Single-orbit deployment results; historical rows are marked in the registry.", "tab:deployment"), encoding="utf-8")
    t2_rows = []
    for model in ("SO-reference", "T0", "T1", "S0"):
        x = metrics[(metrics.model == model) & (metrics.region.isin(["Overall", "TGD"]))].set_index("region")
        t2_rows.append({"Variant": model, "Dual-orbit training": model in ("T0", "T1"), "DA": model in ("T1", "S0"),
                        "Target-evidence auxiliary": model == "T1", "Teacher initialization": model == "S0",
                        "Inference input": "target+counter" if model in ("T0", "T1") else "target",
                        "Overall AUPRC": x.loc["Overall", "auprc"] if "Overall" in x.index else np.nan,
                        "Overall IoU": x.loc["Overall", "iou"] if "Overall" in x.index else np.nan,
                        "TGD AUPRC": x.loc["TGD", "auprc"] if "TGD" in x.index else np.nan,
                        "TGD IoU": x.loc["TGD", "iou"] if "TGD" in x.index else np.nan,
                        "changed_factors": "joint DA+aux" if model == "T1" else "T1->S0 target-only adaptation" if model == "S0" else "reference"})
    t2 = pd.DataFrame(t2_rows)
    for ext in ("csv", "md", "tex"): (TABLES / f"table2_existing_components.{ext}").write_text(
        t2.to_csv(index=False) if ext == "csv" else markdown_table(t2) if ext == "md" else latex_table(t2, "Existing stage comparisons; T1 changes DA and auxiliary supervision jointly.", "tab:components"), encoding="utf-8")
    panel_a = []
    for condition in ("correct", "shuffled", "removed"):
        x = report["experiment_2_counter_dependence"]["conditions"][condition]
        panel_a.append({"panel": "A counter intervention", "condition": condition, "display": "Target-FPN bypass" if condition == "removed" else condition.title(),
                        "region": "Overall", "AUPRC": x["overall"]["auprc"], "IoU": x["overall"]["iou"], "F1": x["overall"]["f1"]})
        panel_a.append({"panel": "A counter intervention", "condition": condition, "display": "Target-FPN bypass" if condition == "removed" else condition.title(),
                        "region": "TGD", "AUPRC": x["G10"]["auprc"], "IoU": x["G10"]["iou"], "F1": x["G10"]["f1"]})
        panel_a.append({"panel": "A counter intervention", "condition": condition, "display": "Target-FPN bypass" if condition == "removed" else condition.title(),
                        "region": "Remaining", "AUPRC": x["non-G10"]["auprc"], "IoU": x["non-G10"]["iou"], "F1": x["non-G10"]["f1"]})
    panel_b = []
    for subset, payload in report["experiment_3_teacher_resolvable"].items():
        for model, vals in payload["models"].items():
            panel_b.append({"panel": "B Teacher-corrected baseline errors", "subset": subset, "model": model,
                            "pixels": payload["pixels"], "share_of_TGD_percent": payload["fraction_of_G10"] * 100,
                            "accuracy_or_correction": None if vals is None else vals["accuracy"]})
    t3 = pd.DataFrame(panel_a + panel_b)
    for ext in ("csv", "md", "tex"): (TABLES / f"table3_tgd_mechanism.{ext}").write_text(
        t3.to_csv(index=False) if ext == "csv" else markdown_table(t3.fillna("—")) if ext == "md" else latex_table(t3.fillna("—"), "TGD mechanism evidence from fixed evaluation and diagnostic subsets.", "tab:tgd"), encoding="utf-8")
    t1.to_csv(DATA / "appendix_deployment_full.csv", index=False)


def save_fig(fig, stem, main=True):
    folder = FIG_MAIN if main else FIG_CAND
    fig.savefig(folder / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figure1():
    fig, ax = plt.subplots(figsize=(12, 6.3)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#25364a", linewidth=1.3, joinstyle="round"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, wrap=True)
    def arrow(x1, y1, x2, y2, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#25364a", "linestyle": ls})
    ax.text(2, 6.7, "(a) Dual-orbit teacher learning", ha="center", fontsize=12, weight="bold")
    box(.35, 5.45, 1.45, .55, "target\n(pre/post, VV/VH)", "#dbeafe"); box(.35, 4.35, 1.45, .55, "counter\n(pre/post, VV/VH)", "#fee2e2")
    box(2.15, 5.45, 1.55, .55, "shared\nConvNeXt-Tiny/FPN", "#e0f2fe"); box(2.15, 4.35, 1.55, .55, "shared\nConvNeXt-Tiny/FPN", "#fee2e2")
    box(4.15, 4.85, 1.55, .75, "normal dual\n1×1 fusion", "#fde68a")
    box(6.15, 5.65, 1.65, .65, "target+dual\npolicy context", "#ddd6fe")
    box(6.15, 4.15, 1.65, .65, "target FPN\nmemory only", "#d1fae5")
    box(8.35, 5.0, 1.65, .7, "offsets +\nsoftmax weights", "#ddd6fe")
    box(8.35, 3.95, 1.65, .7, "DA sampling\n3 levels × 4 points", "#bfdbfe")
    box(10.45, 4.3, 1.25, .75, "SearchWrite\n+ decoder", "#bfdbfe")
    box(10.45, 5.55, 1.25, .6, "dual output", "#fef3c7")
    box(10.45, 3.2, 1.25, .6, "aux output", "#fef3c7")
    arrow(1.8, 5.72, 2.15, 5.72); arrow(1.8, 4.62, 2.15, 4.62); arrow(3.7, 5.72, 4.15, 5.25); arrow(3.7, 4.62, 4.15, 5.2); arrow(5.7, 5.25, 6.15, 5.98); arrow(3.7, 5.72, 6.15, 4.48); arrow(7.8, 5.98, 8.35, 5.35); arrow(7.8, 4.48, 8.35, 4.3); arrow(10.0, 4.3, 10.45, 4.68); arrow(10.0, 5.35, 10.45, 4.9); arrow(10.0, 4.3, 10.45, 3.5); arrow(11.08, 5.05, 11.08, 5.55); arrow(11.08, 4.3, 11.08, 3.8)
    ax.text(2, 3.0, "(b) Parameter transfer + target-only adaptation", ha="center", fontsize=12, weight="bold")
    box(.6, 2.0, 2.0, .65, "T1 checkpoint\nbackbone / policy / write", "#ddd6fe"); box(4.0, 2.0, 2.0, .65, "S0 inherits once\nno counter input", "#d1fae5"); box(7.4, 2.0, 1.8, .65, "target + GT\nL_seg only", "#dbeafe"); box(10.2, 2.0, 1.4, .65, "adapted\nS0", "#bfdbfe")
    arrow(2.6, 2.33, 4.0, 2.33, "--"); arrow(6.0, 2.33, 7.4, 2.33); arrow(9.2, 2.33, 10.2, 2.33)
    ax.text(2, .9, "(c) Single-orbit deployment", ha="center", fontsize=12, weight="bold")
    box(1.0, .05, 2.0, .55, "target only", "#dbeafe"); box(4.3, .05, 2.0, .55, "S0 search/write/decoder", "#d1fae5"); box(7.7, .05, 1.7, .55, "prediction", "#fef3c7")
    arrow(3.0, .33, 4.3, .33); arrow(6.3, .33, 7.7, .33)
    ax.text(9.8, .25, r"$L_T=L_{dual}+0.10L_{aux}$;  $L_S=L_{seg}$", fontsize=9)
    save_fig(fig, "figure1_architecture", True)
    fig.savefig(FIG_SOURCE / "figure1_architecture_source.mplstyle.txt", format="svg") if False else None


def figure3(report):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    deltas = report["experiment_1"]["deltas"]["Student_minus_SO"]
    regions = ["overall", "G10", "non-G10"]; labels = ["Overall", "TGD", "Remaining"]
    x = np.arange(3); w = .36
    axes[0].bar(x-w/2, [deltas[r]["auprc"]*100 for r in regions], w, label="AUPRC", color="#2563eb")
    axes[0].bar(x+w/2, [deltas[r]["iou"]*100 for r in regions], w, label="IoU", color="#16a34a")
    axes[0].axhline(0, color="k", lw=.7); axes[0].set_xticks(x, labels); axes[0].set_ylabel("Student − SO (pp)"); axes[0].set_title("(a) Regional improvement"); axes[0].legend(frameon=False, fontsize=8)
    c = report["experiment_2_counter_dependence"]["conditions"]; vals = [c[k]["G10"]["auprc"] for k in ("correct", "removed", "shuffled")]
    axes[1].bar(["Correct", "Target-FPN\nbypass", "Shuffled"], vals, color=["#16a34a", "#f59e0b", "#dc2626"]); axes[1].set_ylim(0, 1); axes[1].set_ylabel("TGD AUPRC"); axes[1].set_title("(b) Counter intervention")
    union = report["experiment_3_teacher_resolvable"]["union_resolvable"]; student = union["models"]["Student"]["accuracy"]
    axes[2].bar(["S0 correct", "S0 wrong"], [student, 1-student], color=["#16a34a", "#94a3b8"]); axes[2].set_ylim(0, 1); axes[2].set_ylabel("Fraction of Union subset"); axes[2].set_title(f"(c) Fixed diagnostic subset\nN={union['pixels']:,}; {union['fraction_of_G10']*100:.3f}% of TGD")
    fig.suptitle("TGD mechanism evidence; exact locked evaluation", fontsize=12); fig.tight_layout(); save_fig(fig, "figure3_tgd_mechanism", True)


def figure4(report):
    stats = report["experiment_4_da_behavior"]["stats"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    vars_ = [("offset_magnitude", "Offset magnitude"), ("sampling_diversity", "Sampling diversity"), ("weight_entropy", "Weight entropy (nats)")]
    for ax, (key, title) in zip(axes, vars_):
        labels=[]; vals=[]; lo=[]; hi=[]
        for model, col in (("T1", "#2563eb"), ("Student", "#16a34a")):
            for region in ("G10", "non-G10"):
                s=stats[model][region][key]; labels.append(f"{model}\n{region}"); vals.append(s["mean"]); lo.append(s["p10"]); hi.append(s["p90"])
        x=np.arange(len(vals)); ax.errorbar(x, vals, yerr=[np.array(vals)-np.array(lo), np.array(hi)-np.array(vals)], fmt="o", capsize=3, color="#334155"); ax.set_xticks(x, labels, fontsize=8); ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=.25)
    fig.suptitle("DA behavior summary (feature-coordinate units; p10–p90)", fontsize=11); fig.tight_layout(); save_fig(fig, "figure4_da_behavior_candidate", False)


def figure_pr_and_training(preds):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, color in (("SO", "#64748b"), ("T1", "#2563eb"), ("Student", "#16a34a")):
        p, y = preds[name]["p"], preds[name]["y"].astype(bool); prec, rec, thr = precision_recall_curve(y.ravel(), p.ravel()); ap = average_precision_score(y.ravel(), p.ravel()); axes[0].plot(rec, prec, label=f"{name} AP={ap:.4f}", color=color)
        np.savez_compressed(FIG_DATA / f"pr_overall_{name}.npz", precision=prec, recall=rec, thresholds=thr)
        mask = region_masks(preds[name])["TGD"]; prec, rec, thr = precision_recall_curve(y[mask].ravel(), p[mask].ravel()); ap = average_precision_score(y[mask].ravel(), p[mask].ravel()); axes[1].plot(rec, prec, label=f"{name} AP={ap:.4f}", color=color); np.savez_compressed(FIG_DATA / f"pr_tgd_{name}.npz", precision=prec, recall=rec, thresholds=thr)
    for ax, title in zip(axes, ("Overall PR", "TGD PR")): ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); save_fig(fig, "pr_curves_candidate", False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, color in (("T0", "#64748b"), ("T1", "#2563eb"), ("S0", "#16a34a")):
        p = LOCKED / f"{key}_seed42_progress.json"; p = LOCKED / f"{key}_seed42_progress.json"
        if not p.exists(): continue
        hist = json.loads(p.read_text(encoding="utf-8")).get("history", []); ax.plot([h.get("epoch") for h in hist], [h.get("monitor_test_auprc") for h in hist], marker=".", label=key, color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("test AUPRC monitor"); ax.set_title("Recorded training monitor (not validation)"); ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout(); save_fig(fig, "training_curves_candidate", False)


def qualitative_figure(preds, test):
    rows=[]
    for i in range(len(preds["SO"]["p"])):
        tgd = region_masks(preds["SO"])["TGD"][i]; s0 = preds["Student"]["p"][i] >= THRESHOLD; so = preds["SO"]["p"][i] >= THRESHOLD; y = preds["SO"]["y"][i].astype(bool)
        rows.append((float(tgd.mean()), int(((s0==y)&tgd).sum() - ((so==y)&tgd).sum()), i))
    rows.sort(reverse=True); selected=[]
    for item in rows:
        i=item[2]
        if preds["SO"]["y"][i].sum() > 0 and item[0] > 0.01 and i not in selected: selected.append(i)
        if len(selected) == 2: break
    for item in sorted(rows, key=lambda t:t[1]):
        i=item[2]
        if i not in selected and preds["SO"]["y"][i].sum()>0: selected.append(i); break
    selected = selected[:3]
    dump_json(FIG_DATA / "selected_cases.json", [{"index": i, "sample_id": int(preds["SO"]["sample_id"][i]), "target_orbit": "ASC" if int(preds["SO"]["orbit"][i])==0 else "DESC", "selection_rule": "high TGD fraction / largest TGD S0-vs-SO improvement / failure case"} for i in selected])
    fig, axes = plt.subplots(len(selected), 6, figsize=(15, 3.0*len(selected)), squeeze=False)
    for r, i in enumerate(selected):
        sid=int(preds["SO"]["sample_id"][i]); orbit="asc" if int(preds["SO"]["orbit"][i])==0 else "desc"; row=test.rows[test.ids.index(sid)]; rasters, land, ga, gd=row; pre=rasters[(orbit,"pre")]; post=rasters[(orbit,"post")]; target=np.mean(np.abs(post-pre), axis=0); gt=land>0; tgd=(ga>0)&~(gd>0) if orbit=="asc" else (gd>0)&~(ga>0)
        error=np.zeros_like(gt,dtype=np.uint8); sp=preds["Student"]["p"][i]>=THRESHOLD; bp=preds["SO"]["p"][i]>=THRESHOLD; error[sp&gt]=1; error[sp&~gt]=2; error[~sp&gt]=3; error[~sp&~gt]=4
        imgs=[target,gt,preds["SO"]["p"][i],preds["Student"]["p"][i],error,tgd]
        titles=["Target SAR\nmean |post−pre|", "Ground truth", "SO-reference", "Ours (S0)", "S0 error map", "TGD mask"]
        for c,(img,title) in enumerate(zip(imgs,titles)):
            ax=axes[r,c]; ax.set_title(title,fontsize=9); ax.axis("off")
            if c==0: lo,hi=np.percentile(img,[2,98]); ax.imshow(img,cmap="gray",vmin=lo,vmax=hi)
            elif c in (1,5): ax.imshow(img,cmap="gray",vmin=0,vmax=1)
            elif c==4: ax.imshow(img,cmap=matplotlib.colors.ListedColormap(["black","#16a34a","#dc2626","#2563eb","#94a3b8"]),vmin=0,vmax=4)
            else: ax.imshow(img,cmap="viridis",vmin=0,vmax=1)
            if c in (1,2,3,4): ax.contour(tgd, levels=[.5], colors="#f59e0b", linewidths=.35)
        for ext in ("png",): pass
    fig.suptitle("Qualitative locked test examples; yellow outline = TGD", fontsize=12); fig.tight_layout(); save_fig(fig, "figure2_qualitative_comparison", True); fig.savefig(FIG_CAND / "figure2_qualitative_comparison.png", dpi=320, bbox_inches="tight")


def write_docs(metrics, report):
    (OUT / "terminology.json").write_text(json.dumps({"G10": "TGD", "TGD": "Target-only geometric distortion regions", "non-G10": "Remaining regions", "Student": "S0"}, indent=2) + "\n", encoding="utf-8")
    (OUT / "architecture_spec.md").write_text("""# Locked architecture specification\n\n- Shared ConvNeXt-Tiny encoder with a three-level FPN (P2/P3/P4; 128 channels).\n- T0 applies an independent same-channel 1x1 convolution to `[target, counter]` at each FPN level.\n- T1 uses the fused dual context to predict bounded offsets and softmax weights at a 32x32 query grid; there are 3 levels and 4 points per level. All sampled values come from the target FPN memory.\n- SearchWrite is a zero-initialized 1x1 projection over target P2 and retrieved target evidence; the real decoder is the existing FPN head.\n- T1 supervises the dual path and target-evidence auxiliary path with `L_dual + 0.10 L_aux`. S0 inherits the backbone, target policy, SearchWrite and target projection once, removes counter/fusion, and adapts with target + GT using `L_seg` only.\n- Counter is not an inference input to S0. Search KD/S1 is not trained.\n\nImplementation source: `cocd/windows_main/models_da_search.py`.\n""", encoding="utf-8")
    (OUT / "dataset_and_protocol.md").write_text("""# Dataset and protocol provenance\n\n- Dataset root: `data/haiti`; locked split: `data/splits_v3`; 1,370 merged training locations and 343 test locations. No validation split was used.\n- Each location has ASC and DESC Sentinel-1 pre/post paired inputs; the loader reads VV/VH bands and appends an orbit indicator. Test evaluation expands each location into ASC and DESC orbit-conditioned pixels (686 patches, 11,239,424 pixels).\n- Geometry masks are the existing `asc_pre_geom.tif` and `desc_pre_geom.tif`. Internal `G10` is retained as `TGD = target mask positive AND counter mask negative`; `Remaining` is its valid-domain complement.\n- Batch size 16, Adam, learning rate 5e-5, maximum 100 epochs, patience 2, threshold 0.5. Test AUPRC was used for development monitoring/checkpoint selection; these are not untouched held-out estimates.\n- Metric aggregation is pooled over ASC and DESC orbit-conditioned evaluation pixels. AUPRC is sklearn average precision; IoU/F1/precision/recall are pooled positive-class metrics at threshold 0.5.\n- Acquisition dates, CRS, georeferencing and reliable feature-to-mask coordinate inversion were not exposed by the locked analysis contract; they are listed as missing rather than inferred.\n""", encoding="utf-8")
    (OUT / "baseline_comparability.md").write_text("""# Baseline comparability\n\nThe current primary comparison is SO-reference vs S0 under the locked da-search evaluation. T0 and T1 are dual-orbit references, not single-orbit deployment competitors. Historical DIS2 and Vanilla KD numbers come from the retained local CGSearch CSV and no longer have a retained current checkpoint after the q3/q4 cleanup; they are labeled historical/missing-checkpoint. Boehm, CDNetE and MFEWF have local prediction caches and are retained as completed external/local baselines, but their training/architecture details come from their own local records. Old CGSearch is kept only as a historical index and is not mixed into the current DA-search method claim.\n\nT1 changes DA and target-evidence auxiliary supervision jointly relative to T0; no strict single-factor DA-only or auxiliary-only checkpoint is available. S0 is Teacher-initialized target-only adaptation, not a scratch GT-only model.\n""", encoding="utf-8")
    single = metrics[metrics.model.isin(["SO-reference","S0"]) & metrics.region.isin(["Overall","TGD","Remaining"])]
    (OUT / "paper_summary.md").write_text("""# Haiti SAR GRSL material summary\n\n## Main locked evidence\n\nS0 is the only active target-only Student. Relative to the SO-reference, S0 improves Overall AUPRC by 1.6906 pp, IoU by 2.1677 pp and F1 by 1.6035 pp. In TGD the gains are 1.5286, 1.9364 and 1.4162 pp; in Remaining they are 1.7301, 2.2220 and 1.6481 pp. The TGD gain is therefore not larger than the Remaining gain.\n\nT1 with the correct counter reaches TGD AUPRC 0.920059. Shuffled counter drops TGD AUPRC to 0.487625 and the target-FPN bypass to 0.887063. These are analysis interventions, not retrained target-only models.\n\nThe fixed Union diagnostic contains 26,019 TGD pixels (2.503% of TGD). By definition SO is wrong on the subset; S0 is correct on 47.6613% of those pixels. This is a fixed diagnostic correction fraction, not a recovery rate for all TGD pixels or all landslide pixels. Single-DA scratch is unavailable.\n\n## Boundaries\n\nThe geometry-aware “sample outside distorted region” statistic is skipped because no invertible mask-to-feature coordinate contract is declared. DA behavior is reported only in feature-coordinate units. Search KD/S1 was not trained. The test partition selected checkpoints, so all locked values are development-monitor evidence. No claim is made that S0 recovered physically unavailable raw counter observations.\n""", encoding="utf-8")
    miss = [
        "Single-DA scratch checkpoint/predictions: missing; no training started.",
        "Search KD/S1: not trained by design.",
        "Current retained DIS2/Vanilla KD checkpoint and per-pixel cache: missing after legacy q3/q4 cleanup; historical CSV readout retained with status missing_checkpoint.",
        "Reliable mask-to-feature-to-image coordinate contract for DA sampling: missing; geometry-aware sampling panel skipped.",
        "Acquisition metadata/CRS/physical georeferencing sufficient for a truthful map: not found in the locked analysis sources.",
        "GPU latency/MACs/FLOPs benchmark: not collected in the existing experiment and not measured in this read-only packaging pass.",
    ]
    (OUT / "missing_items.md").write_text("# Missing or not applicable materials\n\n" + "\n".join(f"- {x}" for x in miss) + "\n", encoding="utf-8")
    claims = [
        ["S0 improves the target-only baseline", "metrics_full.csv; table1_deployment", "Table 1; Figure 3a", "supported", "SO-reference is a non-student reference and values are test-monitor development evidence"],
        ["Correct counter matters to T1 in TGD", "counter_interventions.csv", "Table 3A; Figure 3b", "supported as intervention evidence", "shuffle/bypass are analysis-only conditions"],
        ["S0 uses Search KD", "protocol_da_search.json; S0 progress", "none", "not supported", "S1/Search KD was not trained"],
        ["T1 auxiliary branch is counter-independent", "models_da_search.py", "Figure 1", "not supported", "policy context includes dual fused features"],
        ["S0 recovered unavailable raw counter observations", "none", "none", "not supported", "only target-only learned inference/search is evidenced"],
        ["S0 corrected 47.66% of all TGD", "diagnostic_subset_metrics.csv", "Table 3B", "not supported", "47.6613% is within the fixed Union diagnostic subset"],
    ]
    pd.DataFrame(claims, columns=["claim", "evidence", "figure_or_table", "support", "limitation"]).to_csv(OUT / "claim_evidence.csv", index=False)
    dump_json(OUT / "audit_report.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "read_only": True, "training_started": False, "checkpoint_overwritten": False, "current_models": ["SO-reference", "T0", "T1", "S0"], "missing_count": len(miss)})


def write_captions():
    (OUT / "captions" / "captions_en.md").write_text("""# Captions\n\n**Figure 1.** Locked dual-orbit Teacher and target-only Student pipeline. Target and counter are encoded by shared ConvNeXt-Tiny/FPN weights; T1 policy context uses target and dual fused features, while deformable sampling reads target memory only. S0 receives a one-time parameter transfer from T1 and is adapted with target and GT only.\n\n**Figure 2.** Qualitative test examples with a common crop and scale. Yellow contours mark TGD. The SAR panel is the displayed mean absolute pre/post difference over the two input bands; it is a visualization, not a calibrated physical unit.\n\n**Figure 3.** TGD mechanism evidence from locked predictions. (a) S0 minus SO in percentage points. (b) T1 correct, Target-FPN bypass and shuffled-counter interventions. (c) S0 correctness inside the fixed SO-wrong/Teacher-correct Union subset.\n\n**Figure 4.** Candidate DA behavior summary in common feature-coordinate units; whiskers show p10–p90. No physical ground displacement or geometry-aware outside-region claim is made.\n""", encoding="utf-8")
    (OUT / "captions" / "captions_zh.md").write_text("""# 中文图注\n\n图1：锁定的双轨 Teacher 与目标轨 Student 流程。T1 的 policy 同时使用 target 与双轨融合上下文，但采样值始终来自 target memory；S0 只在初始化时继承 T1 参数，随后仅用 target+GT 适应。\n\n图2：相同窗口和显示规则的测试样例。黄色轮廓为 TGD。SAR 面板是两个输入波段的 pre/post 绝对差均值，仅作显示，不代表校准物理单位。\n\n图3：锁定预测上的 TGD 机制证据。(a) S0−SO 的百分点差值；(b) T1 正确 counter、Target-FPN bypass 和 shuffled counter；(c) 固定 SO 错、Teacher 对诊断子集中的 S0 正确比例。\n""", encoding="utf-8")


def write_preview(metrics):
    html = """<!doctype html><meta charset='utf-8'><title>GRSL assets v1</title><style>body{font-family:Arial;max-width:1100px;margin:24px auto} img{max-width:100%;border:1px solid #ddd} td,th{padding:4px 8px;border-bottom:1px solid #ddd} code{background:#f3f4f6;padding:2px}</style><h1>Haiti SAR GRSL assets v1</h1><p>Read-only package for locked T0/T1/S0 evidence. See <a href='../README.md'>README.md</a> and <a href='../missing_items.md'>missing_items.md</a>.</p><h2>Figures</h2>"""
    for fn, label in (("figure1_architecture.png","Figure 1 architecture"),("figure2_qualitative_comparison.png","Figure 2 qualitative"),("figure3_tgd_mechanism.png","Figure 3 TGD mechanism"),("figure4_da_behavior_candidate.png","Figure 4 DA behavior candidate"),("pr_curves_candidate.png","PR curves candidate"),("training_curves_candidate.png","Training curves candidate")):
        folder = "../figures/main" if fn.startswith(("figure1","figure2","figure3")) else "../figures/candidates"; html += f"<h3>{label}</h3><img src='{folder}/{fn}' alt='{label}'>"
    html += "<h2>Tables</h2><ul>" + "".join(f"<li><a href='../tables/{fn}'>{fn}</a></li>" for fn in sorted(p.name for p in TABLES.glob("*.md"))) + "</ul>"
    (OUT / "preview/index.html").write_text(html, encoding="utf-8")


def main():
    report = json.loads((LOCKED / "g10_mechanism_report.json").read_text(encoding="utf-8"))
    train, test = da.build_sets()
    preds = load_current_predictions(test)
    external = load_external_predictions()
    metrics = write_metrics(preds, external)
    write_per_image(preds); write_diagnostics(preds, report); write_registry(); write_source_and_dataset_manifests()
    write_tables(metrics, report); write_docs(metrics, report); write_captions()
    figure1(); figure3(report); figure4(report); figure_pr_and_training(preds); qualitative_figure(preds, test); write_preview(metrics)
    zip_path = OUT / "grsl_review_bundle.zip"
    if zip_path.exists(): zip_path.unlink()
    include = [p for p in OUT.rglob("*") if p.is_file() and "predictions" not in p.parts and p.name not in {"dataset_manifest.csv"}]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in include: z.write(p, p.relative_to(OUT).as_posix())
    dump_json(OUT / "source_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "read_only": True,
                                              "asset_root": str(OUT), "review_bundle": str(zip_path), "review_bundle_bytes": zip_path.stat().st_size,
                                              "prediction_cache_dir": str(PRED_DIR), "current_models": ["SO-reference", "T0", "T1", "S0"],
                                              "source_files": json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8")).get("sources", [])})
    print(json.dumps({"asset_root": str(OUT), "metrics_rows": len(metrics), "prediction_caches": [str(p) for p in PRED_DIR.glob("*.npz")], "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
