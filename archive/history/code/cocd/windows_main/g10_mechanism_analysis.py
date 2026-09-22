#!/usr/bin/env python3
"""G10 mechanism evidence for the locked Haiti DA-search experiment.

This is an analysis-only entry point.  It loads the existing SO/T0/T1/S0
checkpoints, does not train or overwrite any checkpoint, and writes the two
requested G10 evidence artifacts under ``experiments/windows_main/da_search``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

PACKAGE = Path(__file__).resolve().parents[2]
COCD = PACKAGE / "cocd"
sys.path.insert(0, str(COCD))

from windows_main import da_search_main as da  # noqa: E402
from windows_main.models_da_search import (  # noqa: E402
    DAStudent,
    DASearchTeacher,
    NormalDualTeacher,
    TargetOnlyBaseline,
    pyramid,
    pyramid_from_encoded,
)


OUT = da.OUT
REPORT_PATH = OUT / "g10_mechanism_report.json"
CSV_PATH = OUT / "g10_mechanism_summary.csv"
SEED = da.SEED
EPS = 1e-8
POLICY_LEVEL_SIZES = (32, 16, 8)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(model, path: Path):
    model = model.to(da.DEV)
    model.load_state_dict(torch.load(path, map_location=da.DEV, weights_only=True))
    return model.eval()


def load_locked_models() -> dict:
    so = TargetOnlyBaseline().to(da.DEV)
    da.load_so_backbone(so, str(OUT / "SO_backbone_seed42.pt"))
    so.eval()
    return {
        "SO": so,
        "T0": load_state(NormalDualTeacher(), OUT / "T0_seed42.pt"),
        "T1": load_state(DASearchTeacher(), OUT / "T1_seed42.pt"),
        "Student": load_state(DAStudent(), OUT / "S0_seed42.pt"),
    }


def region_masks(pred: dict) -> dict[str, np.ndarray]:
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    g10 = gt & ~gc
    return {"overall": np.ones_like(g10, dtype=bool), "G10": g10, "non-G10": ~g10}


def metric_mask(p, y, mask) -> dict:
    p = np.asarray(p)
    y = np.asarray(y).astype(bool)
    mask = np.asarray(mask).astype(bool)
    pf, yf = p[mask], y[mask]
    n = int(mask.sum())
    if n == 0:
        return {"pixels": 0, "accuracy": None, "iou": None, "f1": None,
                "precision": None, "recall": None, "auprc": None}
    hit = pf >= 0.5
    tp = int((hit & yf).sum()); fp = int((hit & ~yf).sum()); fn = int((~hit & yf).sum())
    tn = int((~hit & ~yf).sum())
    precision = tp / (tp + fp + EPS); recall = tp / (tp + fn + EPS)
    return {
        "pixels": n,
        "accuracy": float((hit == yf).mean()),
        "iou": float(tp / (tp + fp + fn + EPS)),
        "f1": float(2 * precision * recall / (precision + recall + EPS)),
        "precision": float(precision), "recall": float(recall),
        "auprc": float(average_precision_score(yf, pf)) if yf.any() else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def performance_table(preds: dict[str, dict]) -> dict:
    out = {}
    for name, pred in preds.items():
        out[name] = {}
        masks = region_masks(pred)
        for region, mask in masks.items():
            out[name][region] = metric_mask(pred["p"], pred["y"], mask)
    return out


def append_prediction(store: dict, p: torch.Tensor, y, gt, gc, orbit, policy=None) -> None:
    store["p"].append(torch.sigmoid(p)[:, 0].detach().cpu().numpy())
    store["y"].append(torch.cat((y, y)).cpu().numpy())
    store["gt"].append(torch.cat((gt, gc)).cpu().numpy())
    store["gc"].append(torch.cat((gc, gt)).cpu().numpy())
    store["orbit"].append(np.array(["asc"] * len(y) + ["desc"] * len(y)))
    if policy is not None:
        store["offsets"].append(torch.cat((policy[0]["offsets"], policy[1]["offsets"])).detach().cpu().numpy())
        store["weights"].append(torch.cat((policy[0]["weights"], policy[1]["weights"])).detach().cpu().numpy())


def finish_prediction(store: dict) -> dict:
    out = {}
    for key, vals in store.items():
        if vals:
            out[key] = np.concatenate(vals)
    return out


def removed_counter_output(model: DASearchTeacher, target: tuple[torch.Tensor, ...], output_size) -> dict:
    """Analysis-only counter-off bypass using the existing target path.

    This does not feed an artificial all-zero SAR image.  The counter-dependent
    fusion is bypassed and the target FPN is used as the dual feature, while the
    existing T1 policy/search/write operators are left unchanged.
    """
    dual = target
    policy = model.policy(model.context(target, dual))
    retrieved = model.search(target, policy)
    correction = model.write(target[0], retrieved)
    return {"z_dual": model.logits(dual[0] + correction, output_size), "policy": policy}


def shuffled_t1_pair(model: DASearchTeacher, a, d, generator):
    """T1 outputs with a batch-shuffled counter, preserving each pair's SAR time order."""
    b = a.shape[0]
    perm = torch.randperm(b, generator=generator).to(a.device)
    if b > 1 and torch.equal(perm.cpu(), torch.arange(b)):
        perm = torch.roll(perm, shifts=1, dims=0)
    features = model.backbone.encode(torch.cat((a, d, d[perm], a[perm]), dim=0))
    fa = tuple(x[:b] for x in features); fd = tuple(x[b:2 * b] for x in features)
    fca = tuple(x[2 * b:3 * b] for x in features); fcd = tuple(x[3 * b:] for x in features)
    ta = pyramid_from_encoded(model.backbone, fa); td = pyramid_from_encoded(model.backbone, fd)
    ca = pyramid_from_encoded(model.backbone, fca); cd = pyramid_from_encoded(model.backbone, fcd)
    return model.from_pyramids(ta, ca, tuple(a.shape[-2:])), model.from_pyramids(td, cd, tuple(d.shape[-2:]))


def collect_t1_counter_conditions(model, dataset) -> dict:
    """Return correct/shuffled/counter-off T1 predictions and correct policies."""
    model.eval()
    conditions = {}
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1701)
    for condition in ("correct", "shuffled", "removed"):
        store = {k: [] for k in ("p", "y", "gt", "gc", "orbit")}
        if condition == "correct":
            store.update({"offsets": [], "weights": []})
        with torch.no_grad():
            for a, d, y, ga, gd, _ in da.loader(dataset):
                a, d = a.to(da.DEV), d.to(da.DEV)
                if condition == "correct":
                    oa, od = model.forward_pair(a, d)
                elif condition == "shuffled":
                    oa, od = shuffled_t1_pair(model, a, d, generator)
                else:
                    features = model.backbone.encode(torch.cat((a, d), dim=0))
                    b = len(a)
                    fa = tuple(x[:b] for x in features); fd = tuple(x[b:] for x in features)
                    ta = pyramid_from_encoded(model.backbone, fa); td = pyramid_from_encoded(model.backbone, fd)
                    oa = removed_counter_output(model, ta, tuple(a.shape[-2:]))
                    od = removed_counter_output(model, td, tuple(d.shape[-2:]))
                policy = (oa, od) if condition == "correct" else None
                append_prediction(store, torch.cat((oa["z_dual"], od["z_dual"])), y, ga, gd, None,
                                  policy=policy)
        conditions[condition] = finish_prediction(store)
    return conditions


def collect_student_policy(model: DAStudent, dataset) -> dict:
    model.eval()
    store = {k: [] for k in ("p", "y", "gt", "gc", "orbit", "offsets", "weights")}
    with torch.no_grad():
        for a, d, y, ga, gd, _ in da.loader(dataset):
            out = model(torch.cat((a.to(da.DEV), d.to(da.DEV))))
            store["p"].append(torch.sigmoid(out["z"])[:, 0].cpu().numpy())
            store["y"].append(torch.cat((y, y)).numpy())
            store["gt"].append(torch.cat((ga, gd)).numpy())
            store["gc"].append(torch.cat((gd, ga)).numpy())
            store["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
            store["offsets"].append(out["offsets"].cpu().numpy())
            store["weights"].append(out["weights"].cpu().numpy())
    return finish_prediction(store)


def query_region_mask(pred: dict, region: str) -> np.ndarray:
    """Map the exact 128x128 geometry mask to the 32x32 policy query grid."""
    g10 = (pred["gt"] > 0) & ~(pred["gc"] > 0)
    # The policy query grid is 32x32 and the input mask is 128x128: this is an
    # exact 4x4 block reduction, not an interpolated geometry approximation.
    qg = g10.reshape(g10.shape[0], 32, 4, 32, 4).any(axis=(2, 4))
    return qg if region == "G10" else ~qg


def distribution_stats(values: np.ndarray) -> dict:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None,
                "min": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
            "min": float(values.min()), "max": float(values.max())}


def pairwise_mean_distance(points: np.ndarray, chunk: int = 4096) -> np.ndarray:
    flat = points.reshape(-1, points.shape[-2], 2)
    tri = np.triu_indices(points.shape[-2], 1)
    out = []
    for start in range(0, len(flat), chunk):
        q = flat[start:start + chunk]
        dist = np.sqrt(((q[:, :, None, :] - q[:, None, :, :]) ** 2).sum(axis=-1))
        out.append(dist[:, tri[0], tri[1]].mean(axis=1))
    return np.concatenate(out).reshape(points.shape[:-2])


def policy_behavior_stats(policy_pred: dict) -> dict:
    """Summarize bounded offsets, entropy, common-grid locations and diversity."""
    raw = policy_pred["offsets"]
    bounded = 2.0 * np.tanh(raw)
    weights = policy_pred["weights"]
    magnitude = np.sqrt((bounded ** 2).sum(axis=3)).mean(axis=(1, 2))  # N,H,W
    entropy = -(weights.clip(1e-8) * np.log(weights.clip(1e-8))).sum(axis=(1, 2))
    n, levels, points, _, h, w = bounded.shape
    coords = []
    yy_q = (np.arange(h, dtype=np.float32) + 0.5)
    xx_q = (np.arange(w, dtype=np.float32) + 0.5)
    base_y, base_x = np.meshgrid(yy_q, xx_q, indexing="ij")
    for level, (hm, wm) in enumerate(zip(POLICY_LEVEL_SIZES, POLICY_LEVEL_SIZES)):
        # Map every sampled feature coordinate into the common finest (32x32)
        # feature coordinate system, preserving the operator's align_corners=False grid.
        by = (np.arange(h, dtype=np.float32) + 0.5) / h * hm - 0.5
        bx = (np.arange(w, dtype=np.float32) + 0.5) / w * wm - 0.5
        gy, gx = np.meshgrid(by, bx, indexing="ij")
        sy = gy[None, None, :, :] + bounded[:, level, :, 0]
        sx = gx[None, None, :, :] + bounded[:, level, :, 1]
        common_y = (sy + 0.5) / hm * h - 0.5
        common_x = (sx + 0.5) / wm * w - 0.5
        coords.append(np.stack((common_y, common_x), axis=-1))
    points = np.concatenate(coords, axis=1)  # N, L*P, H, W, 2
    points = np.moveaxis(points, 1, 3)       # N, H, W, L*P, 2
    diversity = pairwise_mean_distance(points)
    qg = query_region_mask(policy_pred, "G10")
    out = {}
    for region, mask in (("G10", qg), ("non-G10", ~qg)):
        flat_mask = mask.reshape(-1)
        region_points = points.reshape(-1, points.shape[-2], 2)[flat_mask]
        values = {
            "offset_magnitude": magnitude.reshape(-1)[flat_mask],
            "weight_entropy": entropy.reshape(-1)[flat_mask],
            "sampling_diversity": diversity.reshape(-1)[flat_mask],
            "sampling_y_common32": region_points[..., 0],
            "sampling_x_common32": region_points[..., 1],
        }
        out[region] = {name: distribution_stats(value) for name, value in values.items()}
    out["mapping"] = "exact 128-to-32 4x4 query-cell reduction; sample locations are reported in common 32x32 feature coordinates"
    return out


def policy_behavior_comparison(t1_policy: dict, student_policy: dict) -> dict:
    result = {}
    for region in ("G10", "non-G10"):
        mask = query_region_mask(t1_policy, region).reshape(-1)
        to = 2.0 * np.tanh(t1_policy["offsets"])
        so = 2.0 * np.tanh(student_policy["offsets"])
        tw, sw = t1_policy["weights"], student_policy["weights"]
        result[region] = {
            "offset_abs_difference": float(np.abs(to - so).mean(axis=(1, 2, 3)).reshape(-1)[mask].mean()),
            "weight_abs_difference": float(np.abs(tw - sw).mean(axis=(1, 2)).reshape(-1)[mask].mean()),
        }
    return result


def counter_drops(condition_metrics: dict) -> dict:
    out = {}
    for region in ("overall", "G10", "non-G10"):
        correct = condition_metrics["correct"][region]
        out[region] = {}
        for condition, label in (("shuffled", "DropG10shuffle"), ("removed", "DropG10remove")):
            out[region][condition] = {metric: (correct[metric] - condition_metrics[condition][region][metric])
                                      if correct[metric] is not None and condition_metrics[condition][region][metric] is not None else None
                                      for metric in ("auprc", "iou", "f1")}
    return out


def teacher_resolvable(preds: dict) -> dict:
    so, t0, t1, student = preds["SO"], preds["T0"], preds["T1"], preds["Student"]
    y = so["y"].astype(bool)
    so_correct = (so["p"] >= 0.5) == y
    t0_correct = (t0["p"] >= 0.5) == y
    t1_correct = (t1["p"] >= 0.5) == y
    g10 = region_masks(so)["G10"]
    masks = {
        "T0_resolvable": g10 & ~so_correct & t0_correct,
        "T1_resolvable": g10 & ~so_correct & t1_correct,
        "union_resolvable": g10 & ~so_correct & (t0_correct | t1_correct),
    }
    out = {}
    for name, mask in masks.items():
        total = int(mask.sum()); g10_total = int(g10.sum())
        out[name] = {
            "pixels": total,
            "fraction_of_G10": float(total / (g10_total + EPS)),
            "definition": "G10 AND SO wrong AND selected dual Teacher correct",
            "models": {},
        }
        for model_name in ("SO", "Student"):
            out[name]["models"][model_name] = metric_mask(preds[model_name]["p"], y, mask)
            out[name]["models"][model_name]["correction_fraction_vs_SO"] = (
                out[name]["models"][model_name]["accuracy"] if model_name == "Student" else 0.0)
        out[name]["models"]["Single-DA-scratch"] = None
        out[name]["student_gain_vs_SO_accuracy"] = out[name]["models"]["Student"]["accuracy"]
        out[name]["student_gain_vs_single_DA_scratch_accuracy"] = None
    return out


def verdicts(perf, counter, resolvable, behavior, single_scratch_available=False) -> dict:
    so = perf["SO"]["G10"]; t0 = perf["T0"]["G10"]; t1 = perf["T1"]["G10"]
    q1_metrics = ("auprc", "iou", "f1")
    q1_t0 = all(t0[m] > so[m] for m in q1_metrics)
    q1_t1 = all(t1[m] > so[m] for m in q1_metrics)
    q1 = "YES" if q1_t0 and q1_t1 else ("PARTIAL" if q1_t0 or q1_t1 else "NO")
    drops = counter["drops"]["G10"]
    q2_shuffle = all(drops["shuffled"][m] > 0 for m in q1_metrics)
    q2_removed = all(drops["removed"][m] > 0 for m in q1_metrics)
    q2 = "YES" if q2_shuffle and q2_removed else ("PARTIAL" if q2_shuffle or q2_removed else "NO")
    union = resolvable["union_resolvable"]
    student_acc = union["models"]["Student"]["accuracy"]
    so_acc = union["models"]["SO"]["accuracy"]
    q3_student_better = student_acc is not None and so_acc is not None and student_acc > so_acc
    q3 = "YES" if q3_student_better and single_scratch_available else ("PARTIAL" if q3_student_better else "NO")
    # A scratch DA arm is not present in the locked artifacts, and the geometry
    # coordinate test is intentionally skipped.  This keeps Q4 auxiliary.
    q4 = "PARTIAL" if behavior.get("student_teacher_comparison") else "NO"
    return {
        "Q1": {"verdict": q1, "T0_G10_all_metrics_above_SO": q1_t0, "T1_G10_all_metrics_above_SO": q1_t1},
        "Q2": {"verdict": q2, "shuffle_all_metric_drops_positive": q2_shuffle,
               "removed_all_metric_drops_positive": q2_removed},
        "Q3": {"verdict": q3, "student_better_than_SO_on_union": q3_student_better,
               "single_DA_scratch_available": single_scratch_available},
        "Q4": {"verdict": q4, "teacher_student_behavior_comparison_available": bool(behavior.get("student_teacher_comparison")),
               "single_DA_scratch_available": single_scratch_available,
               "geometry_aware_sampling": "skipped"},
    }


def add_row(rows, section, model, condition, region, metric, value, pixels=None, denominator=None,
            delta_name=None, delta_value=None, notes=None):
    rows.append({"section": section, "model": model, "condition": condition, "region": region,
                 "metric": metric, "value": value, "pixels": pixels, "denominator": denominator,
                 "delta_name": delta_name, "delta_value": delta_value, "notes": notes})


def make_csv_rows(report: dict) -> list[dict]:
    rows = []
    for model, regions in report["experiment_1"]["performance_by_region"].items():
        for region, values in regions.items():
            for metric in ("auprc", "iou", "f1", "accuracy", "recall"):
                add_row(rows, "experiment_1_performance", model, "normal", region, metric, values.get(metric), values.get("pixels"))
    for delta_name, delta in report["experiment_1"]["deltas"].items():
        for region, values in delta.items():
            for metric, value in values.items():
                add_row(rows, "experiment_1_delta", delta_name, "normal", region, metric, value,
                        delta_name=delta_name, delta_value=value)
    for condition, regions in report["experiment_2_counter_dependence"]["conditions"].items():
        for region, values in regions.items():
            for metric in ("auprc", "iou", "f1", "accuracy", "recall"):
                add_row(rows, "experiment_2_counter", "T1", condition, region, metric, values.get(metric), values.get("pixels"))
    for condition, regions in report["experiment_2_counter_dependence"]["drops"].items():
        for region, values in regions.items():
            for metric, value in values.items():
                add_row(rows, "experiment_2_drop", "T1", condition, region, metric, value,
                        delta_name="correct-minus-perturbed", delta_value=value)
    for subset, payload in report["experiment_3_teacher_resolvable"].items():
        for model, values in payload["models"].items():
            if values is None:
                add_row(rows, "experiment_3_resolvable", model, subset, "Omega_TR", "available", None,
                        payload["pixels"], payload["fraction_of_G10"], notes="locked single-DA scratch result unavailable")
                continue
            for metric in ("accuracy", "recall", "f1", "iou", "auprc", "correction_fraction_vs_SO"):
                add_row(rows, "experiment_3_resolvable", model, subset, "Omega_TR", metric, values.get(metric),
                        payload["pixels"], notes="same fixed SO-wrong/Teacher-correct subset")
    for model, regions in report["experiment_4_da_behavior"]["stats"].items():
        for region, values in regions.items():
            if not isinstance(values, dict):
                continue
            for metric, stats in values.items():
                if not isinstance(stats, dict):
                    continue
                add_row(rows, "experiment_4_da_behavior", model, "normal", region, metric, stats.get("mean"),
                        stats.get("count"), notes=json.dumps(stats, ensure_ascii=False))
    for region, values in report["experiment_5_student_teacher_behavior"]["student_teacher_comparison"].items():
        for metric, value in values.items():
            add_row(rows, "experiment_5_behavior_comparison", "Student-vs-T1", "normal", region, metric, value)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = da.build_sets()
    models = load_locked_models()
    preds = {
        "SO": da.predict_baseline(models["SO"], test),
        "T0": da.predict_t0(models["T0"], test),
        "T1": da.predict_t1(models["T1"], test),
        "Student": da.predict_student(models["Student"], test),
    }
    performance = performance_table(preds)
    deltas = {
        "Student_minus_SO": {
            region: {metric: performance["Student"][region][metric] - performance["SO"][region][metric]
                     for metric in ("auprc", "iou", "f1")}
            for region in ("overall", "G10", "non-G10")
        },
        "Student_minus_Single_DA_scratch": {
            region: {metric: None for metric in ("auprc", "iou", "f1")}
            for region in ("overall", "G10", "non-G10")
        },
    }

    condition_preds = collect_t1_counter_conditions(models["T1"], test)
    condition_metrics = {name: performance_table({"T1": pred})["T1"] for name, pred in condition_preds.items()}
    drops = counter_drops(condition_metrics)

    t1_policy = condition_preds["correct"]
    student_policy = collect_student_policy(models["Student"], test)
    behavior = {
        "stats": {"T1": policy_behavior_stats(t1_policy), "Student": policy_behavior_stats(student_policy)},
        "student_teacher_comparison": policy_behavior_comparison(t1_policy, student_policy),
        "Single-DA-scratch": None,
        "geometry_aware_sampling": {
            "status": "skipped",
            "reason": "The project stores geometry masks at 128x128 while DA sampling spans multi-level feature maps; no declared invertible mask-to-feature coordinate contract exists, so no approximate outside-region claim is made.",
        },
    }

    resolvable = teacher_resolvable(preds)
    single_scratch = {"available": False, "checkpoint": None,
                      "reason": "No locked Single-DA-scratch checkpoint or result exists in the current experiment artifacts; no new training was started."}
    report = {
        "protocol": {
            "seed": SEED, "split": "data/splits_v3", "train_locations": len(train.ids),
            "test_locations": len(test.ids), "validation_used": False,
            "test_opened_for_monitoring": True, "threshold": 0.5,
            "g10_definition": "existing G10 = target geometry mask positive and counter geometry mask negative",
        },
        "locked_checkpoints": {
            "SO": str(OUT / "SO_backbone_seed42.pt"),
            "T0": str(OUT / "T0_seed42.pt"), "T1": str(OUT / "T1_seed42.pt"),
            "Student": str(OUT / "S0_seed42.pt"),
        },
        "single_DA_scratch": single_scratch,
        "experiment_1": {"performance_by_region": performance, "deltas": deltas},
        "experiment_2_counter_dependence": {
            "conditions": condition_metrics,
            "drops": drops,
            "shuffled_definition": "batch permutation of whole counter samples; each counter pre/post pair remains intact",
            "removed_definition": "analysis-only target-FPN bypass of T1 counter fusion; no artificial zero SAR input",
        },
        "experiment_3_teacher_resolvable": resolvable,
        "experiment_4_da_behavior": behavior,
        "experiment_5_student_teacher_behavior": {
            "student_teacher_comparison": behavior["student_teacher_comparison"],
            "single_DA_scratch_comparison": None,
            "interpretation": "behavior-level auxiliary evidence; no exact pointwise equality claim",
        },
    }
    report["verdicts"] = verdicts(performance, {"drops": drops}, resolvable, behavior,
                                  single_scratch_available=single_scratch["available"])
    report["artifacts"] = {"report": str(REPORT_PATH), "csv": str(CSV_PATH)}
    write_json(REPORT_PATH, report)
    pd.DataFrame(make_csv_rows(report)).to_csv(CSV_PATH, index=False)
    print(json.dumps({"report": str(REPORT_PATH), "csv": str(CSV_PATH), "verdicts": report["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
