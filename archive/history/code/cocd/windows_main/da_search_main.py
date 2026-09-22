#!/usr/bin/env python3
"""Train T0/T1/S0 for normal dual fusion and target-memory DA search."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PACKAGE = Path(__file__).resolve().parents[2]
os.environ.setdefault("COCD_SPLIT_DIR", str(PACKAGE / "data" / "splits_v3"))
os.environ.setdefault("COCD_OUT_ROOT", str(PACKAGE / "experiments"))
COCD = PACKAGE / "cocd"
sys.path.insert(0, str(COCD))
from paths import OUT_ROOT, DEV, describe  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_da_search import (  # noqa: E402
    DAStudent,
    DASearchTeacher,
    NormalDualTeacher,
    TargetOnlyBaseline,
    init_student_from_teacher,
    init_t1_from_t0,
    load_so_backbone,
    normalized_feature_l1,
    parameter_counts,
)

OUT = OUT_ROOT / "windows_main" / "da_search"
SO_INIT = OUT / "SO_backbone_seed42.pt"
SEED = 42
BATCH = 16
MAX_EPOCHS = 100
PATIENCE = 2
LAMBDA_AUX = 0.10
LAMBDA_DA = 0.10
EPS = 1e-8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed(value: int = SEED) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def seg_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(z[:, 0], y)


def metric(p, y, threshold=0.5):
    p, y = np.asarray(p).ravel(), np.asarray(y).astype(bool).ravel()
    h = p >= threshold
    tp, fp, fn = int((h & y).sum()), int((h & ~y).sum()), int((~h & y).sum())
    pr, rc = tp / (tp + fp + EPS), tp / (tp + fn + EPS)
    return {"iou": float(tp / (tp + fp + fn + EPS)),
            "f1": float(2 * pr * rc / (pr + rc + EPS)),
            "precision": float(pr), "recall": float(rc),
            "auprc": float(average_precision_score(y, p)) if y.any() else float("nan")}


def build_sets():
    train_ids, val_ids, test_ids = split()
    if val_ids:
        raise RuntimeError("DA search requires the pinned merged train split with no validation set")
    train = HaitiPairs(train_ids, True)
    test = HaitiPairs(test_ids, False, train.stats)
    return train, test


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def predict_t0(model, dataset):
    model.eval(); out = {k: [] for k in ("p", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _ in loader(dataset):
        oa, od = model.forward_pair(a.to(DEV), d.to(DEV))
        p = torch.sigmoid(torch.cat((oa["z_dual"], od["z_dual"])))[:, 0]
        out["p"].append(p.cpu().numpy()); out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy()); out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def predict_t1(model, dataset):
    model.eval(); out = {k: [] for k in ("p", "search", "noda", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _ in loader(dataset):
        oa, od = model.forward_pair(a.to(DEV), d.to(DEV))
        for key, value in (("p", torch.cat((oa["z_dual"], od["z_dual"]))),
                           ("search", torch.cat((oa["z_search"], od["z_search"]))),
                           ("noda", torch.cat((oa["z_dual_noda"], od["z_dual_noda"])) )):
            out[key].append(torch.sigmoid(value)[:, 0].cpu().numpy())
        out["y"].append(torch.cat((y, y)).numpy()); out["gt"].append(torch.cat((ga, gd)).numpy())
        out["gc"].append(torch.cat((gd, ga)).numpy()); out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def predict_student(model, dataset):
    model.eval(); out = {k: [] for k in ("p", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _ in loader(dataset):
        p = torch.sigmoid(model(torch.cat((a.to(DEV), d.to(DEV))))["z"])[:, 0]
        out["p"].append(p.cpu().numpy()); out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy()); out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def predict_baseline(model, dataset):
    """Evaluate the ordinary target-only reference, not a Student model."""
    model.eval(); out = {k: [] for k in ("p", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _ in loader(dataset):
        pa = torch.sigmoid(model(a.to(DEV))["z"])[:, 0]
        pd = torch.sigmoid(model(d.to(DEV))["z"])[:, 0]
        out["p"].append(torch.cat((pa, pd)).cpu().numpy())
        out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy())
        out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


def region_mask(pred, region):
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    return {"G00": ~gt & ~gc, "G01": ~gt & gc,
            "G10": gt & ~gc, "G11": gt & gc}.get(region, np.ones_like(gt, bool))


def rows(name, pred, field="p"):
    ans = []; p, y = pred[field], pred["y"]
    for part, pm in (("overall", np.ones(len(pred["orbit"]), bool)),
                     ("asc", pred["orbit"] == "asc"), ("desc", pred["orbit"] == "desc")):
        where = np.broadcast_to(pm[:, None, None], p.shape)
        for region in ("all", "G00", "G01", "G10", "G11"):
            mask = where & (np.ones_like(y, bool) if region == "all" else region_mask(pred, region))
            if mask.any():
                ans.append({"method": name, "partition": part, "region": region,
                            "threshold": 0.5, "pixels": int(mask.sum()), **metric(p[mask], y[mask])})
    return ans


def overall(pred, field="p"):
    return metric(pred[field], pred["y"])


def train_t0(train, test, epochs=MAX_EPOCHS):
    final = OUT / f"T0_seed{SEED}.pt"; latest = OUT / f"T0_seed{SEED}_latest.pt"
    progress = OUT / f"T0_seed{SEED}_progress.json"
    if final.exists(): return final
    seed(); model = NormalDualTeacher().to(DEV)
    load_so_backbone(model, str(SO_INIT))
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    best, best_epoch, bad, best_state, hist = -float("inf"), 0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train(); terms = []
        for a, d, y, *_ in tqdm(loader(train, True), desc=f"T0 {epoch:03d}/{epochs}", unit="batch"):
            oa, od = model.forward_pair(a.to(DEV), d.to(DEV))
            loss = 0.5 * (seg_loss(oa["z_dual"], y.to(DEV)) + seg_loss(od["z_dual"], y.to(DEV)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); terms.append(float(loss.detach()))
        score = overall(predict_t0(model, test)); improved = score["auprc"] > best
        if improved: best, best_epoch, bad, best_state = score["auprc"], epoch, 0, clone_state(model)
        else: bad += 1
        rec = {"epoch": epoch, "train_loss": float(np.mean(terms)), "monitor_test_auprc": score["auprc"],
               "best": best, "best_epoch": best_epoch, "bad": bad, "improved": improved}
        hist.append(rec); log(f"T0 epoch={epoch} loss={rec['train_loss']:.5f} test={best:.5f}@{best_epoch}")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                    "best": best, "best_epoch": best_epoch, "best_state": best_state,
                    "history": hist, "complete": False}, latest)
        write_json(progress, {"method": "T0", "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                              "best": best, "best_epoch": best_epoch, "bad": bad, "history": hist,
                              "monitor": "test_auprc", "monitor_partition": "test", "validation_used": False,
                              "test_opened": True, "monitor_risk": "test selects stopping/checkpoint"})
        if bad >= PATIENCE: break
    if best_state is None: best_state = clone_state(model)
    model.load_state_dict(best_state); torch.save(model.state_dict(), final)
    write_json(progress, {"method": "T0", "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                          "best": best, "best_epoch": best_epoch, "history": hist, "complete": True,
                          "stopped_by": "early_stop_or_ceiling", "checkpoint": str(final),
                          "monitor_risk": "test selects stopping/checkpoint"})
    return final


def train_t1(train, test, t0_path, epochs=MAX_EPOCHS):
    final = OUT / f"T1_seed{SEED}.pt"; latest = OUT / f"T1_seed{SEED}_latest.pt"; progress = OUT / f"T1_seed{SEED}_progress.json"
    if final.exists(): return final
    seed(); t0 = NormalDualTeacher().to(DEV); t0.load_state_dict(torch.load(t0_path, map_location=DEV, weights_only=True)); t0.eval()
    model = DASearchTeacher().to(DEV); init_t1_from_t0(model, t0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    best, best_epoch, bad, best_state, hist = -float("inf"), 0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train(); vals = []
        for a, d, y, *_ in tqdm(loader(train, True), desc=f"T1 {epoch:03d}/{epochs}", unit="batch"):
            oa, od = model.forward_pair(a.to(DEV), d.to(DEV)); yy = y.to(DEV)
            dual = 0.5 * (seg_loss(oa["z_dual"], yy) + seg_loss(od["z_dual"], yy))
            aux = 0.5 * (seg_loss(oa["z_search"], yy) + seg_loss(od["z_search"], yy))
            loss = dual + LAMBDA_AUX * aux
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            vals.append((float(loss.detach()), float(dual.detach()), float(aux.detach())))
        pred = predict_t1(model, test); score = overall(pred); improved = score["auprc"] > best
        if improved: best, best_epoch, bad, best_state = score["auprc"], epoch, 0, clone_state(model)
        else: bad += 1
        m = np.mean(vals, axis=0); rec = {"epoch": epoch, "train_loss": float(m[0]), "L_dual": float(m[1]),
             "L_aux": float(m[2]), "lambda_aux_L_aux": LAMBDA_AUX * float(m[2]),
             "aux_to_dual": LAMBDA_AUX * float(m[2]) / (float(m[1]) + EPS),
             "monitor_test_auprc": score["auprc"], "best": best, "best_epoch": best_epoch, "bad": bad, "improved": improved}
        hist.append(rec); log(f"T1 epoch={epoch} dual={m[1]:.5f} aux={m[2]:.5f} ratio={rec['aux_to_dual']:.3f} test={best:.5f}@{best_epoch}")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch, "best_state": best_state, "history": hist, "complete": False}, latest)
        write_json(progress, {"method": "T1", "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                              "best": best, "best_epoch": best_epoch, "bad": bad, "history": hist,
                              "lambda_aux": LAMBDA_AUX, "monitor": "test_auprc", "monitor_partition": "test",
                              "validation_used": False, "test_opened": True, "monitor_risk": "test selects stopping/checkpoint"})
        if bad >= PATIENCE: break
    if best_state is None: best_state = clone_state(model)
    model.load_state_dict(best_state); torch.save(model.state_dict(), final)
    write_json(progress, {"method": "T1", "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                          "best": best, "best_epoch": best_epoch, "history": hist, "complete": True,
                          "stopped_by": "early_stop_or_ceiling", "checkpoint": str(final), "lambda_aux": LAMBDA_AUX,
                          "monitor_risk": "test selects stopping/checkpoint"})
    return final


def train_student(train, test, t1_path, epochs=MAX_EPOCHS):
    variant = "S0"
    final = OUT / f"{variant}_seed{SEED}.pt"; latest = OUT / f"{variant}_seed{SEED}_latest.pt"; progress = OUT / f"{variant}_seed{SEED}_progress.json"
    if final.exists(): return final
    seed(); teacher = DASearchTeacher().to(DEV); teacher.load_state_dict(torch.load(t1_path, map_location=DEV, weights_only=True)); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    model = DAStudent().to(DEV); init_student_from_teacher(model, teacher)
    lam = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    best, best_epoch, bad, best_state, hist = -float("inf"), 0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train(); vals = []
        for a, d, y, _ga, _gd, real in tqdm(loader(train, True), desc=f"{variant} {epoch:03d}/{epochs}", unit="batch"):
            a, d, yy = a.to(DEV), d.to(DEV), torch.cat((y, y)).to(DEV)
            x = torch.cat((a, d)); out = model(x); land = seg_loss(out["z"], yy); action = land * 0.0
            if lam > 0 and bool(real.any()):
                sel = real.to(DEV).bool(); cm = torch.cat((sel, sel))
                with torch.no_grad():
                    ta, td = teacher.forward_pair(a[sel], d[sel])
                    pol = {k: torch.cat((ta["policy"][k], td["policy"][k])) for k in ("offsets", "weights")}
                oracle = model.forward_with_policy(x[cm], pol)
                action = normalized_feature_l1(out["R"][cm], oracle["R"])
            loss = land + lam * action
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            vals.append((float(loss.detach()), float(land.detach()), float(action.detach())))
        pred = predict_student(model, test); score = overall(pred); improved = score["auprc"] > best
        if improved: best, best_epoch, bad, best_state = score["auprc"], epoch, 0, clone_state(model)
        else: bad += 1
        m = np.mean(vals, axis=0); rec = {"epoch": epoch, "train_loss": float(m[0]), "L_land": float(m[1]),
             "L_DA": float(m[2]), "lambda_DA_L_DA": lam * float(m[2]),
             "DA_to_land": lam * float(m[2]) / (float(m[1]) + EPS), "lambda_DA": lam,
             "monitor_test_auprc": score["auprc"], "best": best, "best_epoch": best_epoch, "bad": bad, "improved": improved}
        hist.append(rec); log(f"{variant} epoch={epoch} land={m[1]:.5f} L_DA={m[2]:.5f} ratio={rec['DA_to_land']:.3f} test={best:.5f}@{best_epoch}")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch, "best_state": best_state, "history": hist, "complete": False}, latest)
        write_json(progress, {"method": variant, "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                              "best": best, "best_epoch": best_epoch, "bad": bad, "history": hist,
                              "lambda_DA": lam, "monitor": "test_auprc", "monitor_partition": "test",
                              "validation_used": False, "test_opened": True, "monitor_risk": "test selects stopping/checkpoint"})
        if bad >= PATIENCE: break
    if best_state is None: best_state = clone_state(model)
    model.load_state_dict(best_state); torch.save(model.state_dict(), final)
    write_json(progress, {"method": variant, "seed": SEED, "epoch": epoch, "max_epochs": epochs,
                          "best": best, "best_epoch": best_epoch, "history": hist, "complete": True,
                          "stopped_by": "early_stop_or_ceiling", "checkpoint": str(final), "lambda_DA": lam,
                          "monitor_risk": "test selects stopping/checkpoint"})
    return final


@torch.no_grad()
def oracle_probe(teacher, student, dataset):
    teacher.eval(); student.eval(); out = {k: [] for k in ("normal", "oracle", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _ in loader(dataset):
        a, d = a.to(DEV), d.to(DEV); x = torch.cat((a, d)); normal = student(x)
        ta, td = teacher.forward_pair(a, d)
        pol = {k: torch.cat((ta["policy"][k], td["policy"][k])) for k in ("offsets", "weights")}
        oracle = student.forward_with_policy(x, pol)
        out["normal"].append(torch.sigmoid(normal["z"])[:, 0].cpu().numpy()); out["oracle"].append(torch.sigmoid(oracle["z"])[:, 0].cpu().numpy())
        out["y"].append(torch.cat((y, y)).numpy()); out["gt"].append(torch.cat((ga, gd)).numpy()); out["gc"].append(torch.cat((gd, ga)).numpy()); out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def mechanism_diagnostics(teacher, student, dataset):
    teacher.eval(); student.eval(); sums = {k: [] for k in ("offset_abs", "weight_entropy", "policy_offset_error", "policy_weight_error")}
    region_vals = {g: [] for g in ("G10", "non-G10")}; shuffled = {k: [] for k in ("offset", "weight")}
    for a, d, y, ga, gd, _ in loader(dataset):
        a, d = a.to(DEV), d.to(DEV); ta, td = teacher.forward_pair(a, d)
        pol = {k: torch.cat((ta["policy"][k], td["policy"][k])) for k in ("offsets", "weights")}
        sp = student(torch.cat((a, d)))["policy"]
        off = pol["offsets"]; wt = pol["weights"]
        sums["offset_abs"].append(off.abs().mean().item())
        ent = -(wt.clamp_min(1e-8) * wt.clamp_min(1e-8).log()).sum((1, 2)).mean().item()
        sums["weight_entropy"].append(ent)
        sums["policy_offset_error"].append((sp["offsets"] - off).abs().mean().item())
        sums["policy_weight_error"].append((sp["weights"] - wt).abs().mean().item())
        # A shuffled counter changes only the policy context; values remain target-only.
        if len(a) > 1:
            shuffled_counter = torch.roll(d, shifts=1, dims=0)
            sa, sd = teacher.forward_pair(a, shuffled_counter)
            po = {k: torch.cat((sa["policy"][k], sd["policy"][k])) for k in ("offsets", "weights")}
            shuffled["offset"].append((off - po["offsets"]).abs().mean().item())
            shuffled["weight"].append((wt - po["weights"]).abs().mean().item())
        g10 = torch.cat(((ga > 0) & ~(gd > 0), (gd > 0) & ~(ga > 0))).to(DEV)
        mag = off.abs().mean((1, 2, 3)); qg = F.adaptive_avg_pool2d(g10.float()[:, None], mag.shape[-2:])[:, 0] > 0
        region_vals["G10"].extend(mag[qg].cpu().tolist()); region_vals["non-G10"].extend(mag[~qg].cpu().tolist())
    return {"mean": {k: float(np.mean(v)) if v else None for k, v in sums.items()},
            "G10_vs_nonG10_offset_abs": {k: float(np.mean(v)) if v else None for k, v in region_vals.items()},
            "shuffled_counter_policy_change": {k: float(np.mean(v)) if v else None for k, v in shuffled.items()}}


def load_model(cls, path):
    model = cls().to(DEV); model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)); return model.eval()


def evaluate(train, test, t0, t1, s0):
    t0p, t1p = predict_t0(t0, test), predict_t1(t1, test)
    s0p = predict_student(s0, test) if s0 else None
    baseline = load_model(TargetOnlyBaseline, SO_INIT) if SO_INIT.exists() else None
    baselinep = predict_baseline(baseline, test) if baseline else None
    all_rows = rows("SO-reference", baselinep) if baselinep else []
    all_rows += rows("T0-Normal-Dual", t0p) + rows("T1-Dual+DA", t1p) + rows("T1-target-search-aux", t1p, "search")
    if s0p: all_rows += rows("S0-GT-only", s0p)
    oracle = oracle_probe(t1, s0, test) if s0 else None
    diag = mechanism_diagnostics(t1, s0, test) if s0 else None
    write_json(OUT / "oracle_probe.json", {"normal": overall(oracle, "normal") if oracle else None,
        "oracle": overall(oracle, "oracle") if oracle else None,
        "G10_normal": metric(oracle["normal"][_region_mask(oracle, "G10")], oracle["y"][_region_mask(oracle, "G10")]) if oracle else None,
        "G10_oracle": metric(oracle["oracle"][_region_mask(oracle, "G10")], oracle["y"][_region_mask(oracle, "G10")]) if oracle else None})
    write_json(OUT / "mechanism_diagnostics.json", diag or {})
    pd.DataFrame(all_rows).to_csv(OUT / "test_monitored_results_seed42.csv", index=False)
    def brief(pred, field="p"):
        if pred is None:
            return None
        m = {"overall": overall(pred, field)}
        for region in ("G00", "G01", "G10", "G11"):
            mask = _region_mask(pred, region)
            m[region] = metric(pred[field][mask], pred["y"][mask])
        return m
    oracle_gate = None
    gate_path = OUT / "oracle_gate.json"
    if gate_path.exists():
        oracle_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    counts = parameter_counts(t0, t1, s0)
    report = {
        "decision": "PARTIALLY_SUPPORTED",
        "decision_reason": "T0/T1 and S0 were evaluated; the Teacher-policy oracle is retained as a diagnostic and no second Student arm is active.",
        "protocol": {"seed": SEED, "batch": BATCH, "max_epochs": MAX_EPOCHS, "validation_used": False,
                     "test_opened_for_monitoring": True, "lambda_aux": LAMBDA_AUX, "lambda_DA": LAMBDA_DA},
        "architecture": {"backbone": "ConvNeXt-Tiny + three-level FPN", "channels": 128,
                          "levels": 3, "points_per_level": 4, "policy_grid": [32, 32],
                          "offset_shape": ["B", 3, 4, 2, 32, 32], "weight_shape": ["B", 3, 4, 32, 32],
                          "memory": "target FPN only", "teacher_fusion": "per-level 1x1 Conv([T,C])",
                          "student_inputs": "target only", "student_teacher_policy": "detached offsets and weights"},
        "parameter_counts": counts,
        "metrics": {"SO_reference": brief(baselinep) if baselinep else None,
                    "T0": brief(t0p), "T1_dual": brief(t1p), "T1_aux": brief(t1p, "search"),
                    "S0": brief(s0p),
                    "oracle_normal": brief(oracle, "normal") if oracle else None,
                    "oracle_teacher_policy": brief(oracle, "oracle") if oracle else None},
        "student_definition": "S0 is the only active target-only Student; SO-reference is a non-student baseline.",
        "gates": {"teacher": json.loads((OUT / "teacher_gates.json").read_text(encoding="utf-8")) if (OUT / "teacher_gates.json").exists() else None,
                  "oracle": oracle_gate},
        "diagnostics": diag or {},
        "artifacts": {"test_table": str(OUT / "test_monitored_results_seed42.csv"),
                      "oracle_probe": str(OUT / "oracle_probe.json"),
                      "mechanism_diagnostics": str(OUT / "mechanism_diagnostics.json")}
    }
    write_json(OUT / "final_report.json", report)
    return {"SO-reference": baselinep, "T0": t0p, "T1": t1p, "S0": s0p, "oracle": oracle}


def _region_mask(pred, region):
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    return {"G10": gt & ~gc, "G00": ~gt & ~gc, "G01": ~gt & gc, "G11": gt & gc}[region]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("T0", "T1", "S0", "validate", "all"))
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    print(describe()); print(f"[DA protocol] split={os.environ['COCD_SPLIT_DIR']} batch={BATCH} epochs={args.epochs} "
                              f"lambda_aux={LAMBDA_AUX} lambda_DA={LAMBDA_DA} monitor=test_auprc test_opened=True", flush=True)
    train, test = build_sets()
    t0_path = OUT / f"T0_seed{SEED}.pt"
    if args.stage in ("T0", "all"): t0_path = train_t0(train, test, args.epochs)
    if args.stage == "T0": return
    if not t0_path.exists(): raise SystemExit("missing T0 checkpoint")
    t0 = load_model(NormalDualTeacher, t0_path)
    if args.stage in ("T1", "all"): t1_path = train_t1(train, test, t0_path, args.epochs)
    else: t1_path = OUT / f"T1_seed{SEED}.pt"
    if args.stage == "T1": return
    if not t1_path.exists(): raise SystemExit("missing T1 checkpoint")
    t1 = load_model(DASearchTeacher, t1_path)
    if args.stage == "all":
        # Gate 1: normal dual versus the current ordinary target-only reference.
        # Gate 2: T1 must not lose T0 overall AUPRC; the auxiliary branch is
        # reported.  The reference is not a Student and is never used to
        # initialise S0.
        t0p, t1p = predict_t0(t0, test), predict_t1(t1, test)
        if not SO_INIT.exists(): raise SystemExit(f"missing target-only reference backbone: {SO_INIT}")
        so = load_model(TargetOnlyBaseline, SO_INIT); sop = predict_baseline(so, test)
        gate1 = overall(t0p)["auprc"] > overall(sop)["auprc"]
        gate2 = overall(t1p)["auprc"] >= overall(t0p)["auprc"]
        write_json(OUT / "teacher_gates.json", {"gate1_T0_dual_value": gate1, "gate2_T1_not_degraded": gate2,
            "T0": overall(t0p), "T1_dual": overall(t1p), "T1_aux": overall(t1p, "search"),
            "SO_reference": overall(sop),
            "SO_role": "non-student target-only reference; no q3/q4 correction network"})
        if not (gate1 and gate2):
            log("T0/T1 gate failed; stopping before Student Search KD")
            evaluate(train, test, t0, t1, None); return
        s0_path = train_student(train, test, t1_path, args.epochs)
        s0 = load_model(DAStudent, s0_path)
        oracle = oracle_probe(t1, s0, test)
        oracle_gate = overall(oracle, "oracle")["auprc"] > overall(oracle, "normal")["auprc"] and metric(oracle["oracle"][_region_mask(oracle, "G10")], oracle["y"][_region_mask(oracle, "G10")])["auprc"] > metric(oracle["normal"][_region_mask(oracle, "G10")], oracle["y"][_region_mask(oracle, "G10")])["auprc"]
        write_json(OUT / "oracle_gate.json", {"supported": oracle_gate, "overall": {"normal": overall(oracle, "normal"), "oracle": overall(oracle, "oracle")}})
        if not oracle_gate:
            log("Oracle Search diagnostic did not pass; retaining S0 as the only Student")
            evaluate(train, test, t0, t1, s0); return
        evaluate(train, test, t0, t1, s0)
    elif args.stage == "S0":
        s_path = train_student(train, test, t1_path, args.epochs)
        evaluate(train, test, t0, t1, load_model(DAStudent, s_path))
    elif args.stage == "validate":
        s0 = load_model(DAStudent, OUT / f"S0_seed{SEED}.pt") if (OUT / f"S0_seed{SEED}.pt").exists() else None
        evaluate(train, test, t0, t1, s0)


if __name__ == "__main__":
    main()
