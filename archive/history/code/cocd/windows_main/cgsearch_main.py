#!/usr/bin/env python3
"""Train the single CGSearch Teacher/Student experiment under the frozen v4 rule.

The runner intentionally follows the existing Windows path: the pinned v3
train IDs are used as the merged 1370-location training set, the 343-location
test partition is monitored every epoch with patience two, and threshold 0.5
is never searched.  This file adds only the CGSearch stages and does not alter
the dataset API or any legacy model.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PACKAGE = Path(__file__).resolve().parents[2]
os.environ.setdefault("COCD_SPLIT_DIR", str(PACKAGE / "data" / "splits_v3"))
os.environ.setdefault("COCD_OUT_ROOT", str(PACKAGE / "experiments"))
COCD = PACKAGE / "cocd"
sys.path.insert(0, str(COCD))

# The legacy loader reads the UTF-8 protocol without specifying an encoding.
_spec = importlib.util.spec_from_file_location("cg_v2", COCD / "scripts" / "23_train_ours_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_read_text = Path.read_text
Path.read_text = lambda self, encoding=None, errors=None: _read_text(
    self, encoding="utf-8" if encoding is None else encoding, errors=errors)
try:
    assert _spec.loader is not None
    _spec.loader.exec_module(v2)
finally:
    Path.read_text = _read_text

from paths import OUT_ROOT, DEV, describe  # noqa: E402
from models.landslide_cocd_v2 import OursV3Student  # noqa: E402
from windows_main.models_cgsearch import (  # noqa: E402
    CGSearchStudent,
    CGSearchTeacher,
    action_distance,
    copy_teacher_to_student,
    load_so_backbone,
    parameter_counts,
)

OUT = OUT_ROOT / "windows_main" / "cgsearch"
SEED = 42
BATCH = 16
MAX_EPOCHS = 100
PATIENCE = 2
EPS = 1e-8


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def seg_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits[:, 0], y)


def metric(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    p = np.asarray(p).ravel()
    y = np.asarray(y).astype(bool).ravel()
    h = p >= threshold
    tp = int((h & y).sum())
    fp = int((h & ~y).sum())
    fn = int((~h & y).sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return {
        "iou": float(tp / (tp + fp + fn + EPS)),
        "f1": float(2 * precision * recall / (precision + recall + EPS)),
        "precision": float(precision),
        "recall": float(recall),
        "auprc": float(average_precision_score(y, p)) if y.any() else float("nan"),
    }


def build_sets():
    train_ids, val_ids, test_ids = v2.split()
    if val_ids:
        raise RuntimeError("CGSearch must use the pinned merged train set; validation is not allowed")
    train_set = v2.HaitiPairs(train_ids, True)
    test_set = v2.HaitiPairs(test_ids, False, train_set.stats)
    return train_set, test_set


def clone_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def predict_teacher(model: CGSearchTeacher, dataset) -> dict:
    model.eval()
    out = {k: [] for k in ("p", "self", "dual", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _real in loader(dataset):
        a, d = a.to(DEV), d.to(DEV)
        oa, od = model.forward_pair(a, d)
        self_p = torch.sigmoid(torch.cat((oa["z_self"], od["z_self"])))[:, 0]
        dual_p = torch.sigmoid(torch.cat((oa["z_dual"], od["z_dual"])))[:, 0]
        out["p"].append(dual_p.cpu().numpy())
        out["self"].append(self_p.cpu().numpy())
        out["dual"].append(dual_p.cpu().numpy())
        out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy())
        out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def predict_student(model: CGSearchStudent, dataset) -> dict:
    model.eval()
    out = {k: [] for k in ("p", "zero", "self", "y", "gt", "gc", "orbit")}
    for a, d, y, ga, gd, _real in loader(dataset):
        x = torch.cat((a.to(DEV), d.to(DEV)))
        normal = model(x)
        zero = model.forward_zero_guidance(x)
        p = torch.sigmoid(normal["z"])[:, 0]
        p0 = torch.sigmoid(zero["z"])[:, 0]
        self_p = torch.sigmoid(normal["z_self"])[:, 0]
        out["p"].append(p.cpu().numpy())
        out["zero"].append(p0.cpu().numpy())
        out["self"].append(self_p.cpu().numpy())
        out["y"].append(torch.cat((y, y)).numpy())
        out["gt"].append(torch.cat((ga, gd)).numpy())
        out["gc"].append(torch.cat((gd, ga)).numpy())
        out["orbit"].append(np.array(["asc"] * len(a) + ["desc"] * len(a)))
    return {k: np.concatenate(v) for k, v in out.items()}


def rows(name: str, pred: dict, field: str = "p") -> list[dict]:
    p, y = pred[field], pred["y"]
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    ans = []
    for partition, part_mask in (
        ("overall", np.ones(len(pred["orbit"]), dtype=bool)),
        ("asc", pred["orbit"] == "asc"),
        ("desc", pred["orbit"] == "desc"),
    ):
        where = np.broadcast_to(part_mask[:, None, None], p.shape)
        for region, region_mask in (
            ("all", np.ones_like(gt, dtype=bool)),
            ("G00", ~gt & ~gc), ("G01", ~gt & gc),
            ("G10", gt & ~gc), ("G11", gt & gc),
        ):
            mask = where & region_mask
            if not mask.any():
                continue
            ans.append({"method": name, "partition": partition, "region": region,
                        "threshold": 0.5, "pixels": int(mask.sum()),
                        "positive_pixels": int(y[mask].sum()),
                        **metric(p[mask], y[mask])})
    return ans


def overall_auprc(pred: dict, field: str = "p") -> float:
    return metric(pred[field], pred["y"])["auprc"]


def train_teacher(train_set, test_set, epochs: int = MAX_EPOCHS) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / f"CGSearch-Teacher_seed{SEED}.pt"
    latest = OUT / f"CGSearch-Teacher_seed{SEED}_latest.pt"
    progress_path = OUT / f"CGSearch-Teacher_seed{SEED}_progress.json"
    if final.exists():
        log(f"Teacher checkpoint already exists: {final}")
        return final
    set_seed(SEED)
    model = CGSearchTeacher().to(DEV)
    source = PACKAGE / "experiments" / "ours_v2" / "SO_wm_seed42.pt"
    load_info = load_so_backbone(model, str(source))
    log(f"Teacher initialized from SO checkpoint: {source}")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    best, best_epoch, bad, best_state = -float("inf"), 0, 0, None
    start, history = 1, []
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get("complete", False):
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start = int(state["epoch"]) + 1
            history = state.get("history", [])
            best, best_epoch, bad = state["best"], state["best_epoch"], state["bad"]
            best_state = state.get("best_state")
            log(f"resuming CGSearch Teacher at epoch {start}")
    for epoch in range(start, epochs + 1):
        model.train()
        losses = []
        bar = tqdm(loader(train_set, True), desc=f"CGSearch-Teacher {epoch:03d}/{epochs}", unit="batch")
        for a, d, y, *_ in bar:
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            oa, od = model.forward_pair(a, d)
            loss = 0.25 * (seg_loss(oa["z_self"], y) + seg_loss(oa["z_dual"], y) +
                           seg_loss(od["z_self"], y) + seg_loss(od["z_dual"], y))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")
        pred = predict_teacher(model, test_set)
        score = overall_auprc(pred, "p")
        improved = score > best
        if improved:
            best, best_epoch, bad, best_state = score, epoch, 0, clone_state(model)
            flag = "NEW BEST"
        else:
            bad += 1
            flag = f"no gain ({bad}/{PATIENCE})"
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "monitor_test_auprc_dual": score, "monitor_partition": "test",
               "best": best, "best_epoch": best_epoch, "improved": improved, "bad": bad}
        history.append(rec)
        log(f"CGSearch-Teacher epoch={epoch} loss={rec['train_loss']:.5f} "
            f"test_auprc={score:.5f} best={best:.5f}@{best_epoch} [{flag}]")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad,
                   "best_state": best_state, "history": history, "complete": False}
        torch.save(payload, latest)
        write_json(progress_path, {
            "method": "CGSearch-Teacher", "seed": SEED, "epoch": epoch,
            "max_epochs": epochs, "selection": "best monitored test AUPRC",
            "monitor": "test_auprc", "monitor_partition": "test", "patience": PATIENCE,
            "best": best, "best_epoch": best_epoch, "history": history,
            "validation_used": False, "test_opened": True, "load_info": load_info,
            "monitor_risk": "test AUPRC selects stopping/checkpoint; not independent test evidence",
        })
        if bad >= PATIENCE:
            log(f"CGSearch Teacher early stop at epoch {epoch}")
            break
    if best_state is None:
        best_state, best_epoch = clone_state(model), epoch
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final)
    torch.save({"model": model.state_dict(), "epoch": epoch, "best": best,
                "best_epoch": best_epoch, "history": history, "complete": True}, latest)
    write_json(progress_path, {"method": "CGSearch-Teacher", "seed": SEED,
                               "epoch": epoch, "max_epochs": epochs,
                               "selection": "best monitored test AUPRC", "monitor": "test_auprc",
                               "monitor_partition": "test", "patience": PATIENCE,
                               "best": best, "best_epoch": best_epoch, "history": history,
                               "checkpoint": str(final), "validation_used": False,
                               "test_opened": True, "load_info": load_info,
                               "monitor_risk": "test AUPRC selects stopping/checkpoint; not independent test evidence"})
    log(f"wrote {final}; selected epoch={best_epoch}; monitored test AUPRC={best:.5f}")
    return final


def load_teacher(path: Path) -> CGSearchTeacher:
    model = CGSearchTeacher().to(DEV)
    model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True), strict=True)
    return model.eval()


def search_action_loss(student_out: dict, teacher_out: dict,
                       y: torch.Tensor, g10: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The sole KD term: relative search-action distance with detached Teacher."""
    rho_s = student_out["rho"]
    rho_t = teacher_out["rho"].detach()
    d_action = action_distance(rho_s, rho_t).reshape(rho_s.shape[0], 1,
                                                       rho_s.shape[-2] if rho_s.ndim == 4 else 1,
                                                       rho_s.shape[-1] if rho_s.ndim == 4 else -1)
    # rho is [B,Nq,Nk]; recover P3's square grid from the Teacher/Student taps.
    h3, w3 = student_out["P3"].shape[-2:]
    d_action = action_distance(rho_s, rho_t).reshape(rho_s.shape[0], 1, h3, w3)
    with torch.no_grad():
        e_self = F.binary_cross_entropy_with_logits(student_out["teacher_z_self"][:, 0], y,
                                                    reduction="none")
        e_dual = F.binary_cross_entropy_with_logits(student_out["teacher_z_dual"][:, 0], y,
                                                    reduction="none")
        gain = ((e_self - e_dual) / (e_self + EPS)).clamp(0, 1)
        gain3 = F.adaptive_avg_pool2d(gain[:, None], (h3, w3))
        g10frac = F.adaptive_avg_pool2d(g10[:, None].float(), (h3, w3))
        weight = gain3 * (1.0 + g10frac)
    denom = weight.sum()
    value = (weight * d_action).sum() / (denom + EPS) if float(denom) > 0 else rho_s.sum() * 0.0
    return value, {"L_action": float(value.detach()), "weight_mean": float(weight.mean().detach()),
                   "gain_mean": float(gain.mean().detach()), "g10_fraction": float(g10.float().mean().detach())}


def student_batch_loss(student: CGSearchStudent, teacher: CGSearchTeacher,
                       a, d, y, ga, gd, real, lam: float):
    x, y2 = torch.cat((a, d)), torch.cat((y, y))
    out = student(x)
    land = seg_loss(out["z"], y2)
    stats = {"L_land": float(land.detach()), "L_action": 0.0,
             "lambda_L_action": 0.0, "aux_to_land": 0.0,
             "gain_mean": 0.0, "weight_mean": 0.0, "g10_fraction": 0.0}
    real_mask = torch.cat((real, real)).bool()
    if lam <= 0 or not real_mask.any():
        return land, stats
    # Keep the Teacher completely outside the Student graph and use only real
    # pre/post pairs for the search-action term.
    selected = real.bool()
    with torch.no_grad():
        ta, td = teacher.forward_pair(a[selected], d[selected])
        tz_self = torch.cat((ta["z_self"], td["z_self"]))
        tz_dual = torch.cat((ta["z_dual"], td["z_dual"]))
        rho_t = torch.cat((ta["rho"], td["rho"]))
        g10 = torch.cat(((ga[selected] > 0) & ~(gd[selected] > 0),
                         (gd[selected] > 0) & ~(ga[selected] > 0)))
    student_selected = {k: v[real_mask] for k, v in out.items() if torch.is_tensor(v)}
    student_selected.update({"teacher_z_self": tz_self, "teacher_z_dual": tz_dual})
    action, extra = search_action_loss(student_selected, {"rho": rho_t},
                                       torch.cat((y[selected], y[selected])), g10)
    loss = land + lam * action
    stats.update({"L_action": float(action.detach()), "lambda_L_action": float((lam * action).detach()),
                  "aux_to_land": float((lam * action / (land.detach() + EPS))), **extra})
    return loss, stats


def train_student(variant: str, train_set, test_set, teacher: CGSearchTeacher,
                  epochs: int = MAX_EPOCHS) -> Path:
    final = OUT / f"{variant}_seed{SEED}.pt"
    latest = OUT / f"{variant}_seed{SEED}_latest.pt"
    progress_path = OUT / f"{variant}_seed{SEED}_progress.json"
    if final.exists():
        log(f"{variant} checkpoint already exists: {final}")
        return final
    set_seed(SEED)
    model = CGSearchStudent().to(DEV)
    copy_teacher_to_student(model, teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    lam = 0.0 if variant.endswith("GT") else 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    best, best_epoch, bad, best_state = -float("inf"), 0, 0, None
    start, history = 1, []
    if latest.exists():
        state = torch.load(latest, map_location=DEV, weights_only=False)
        if not state.get("complete", False):
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start = int(state["epoch"]) + 1
            history = state.get("history", [])
            best, best_epoch, bad, best_state = state["best"], state["best_epoch"], state["bad"], state.get("best_state")
            log(f"resuming {variant} at epoch {start}")
    for epoch in range(start, epochs + 1):
        model.train()
        terms = []
        bar = tqdm(loader(train_set, True), desc=f"{variant} {epoch:03d}/{epochs}", unit="batch")
        for a, d, y, ga, gd, real in bar:
            a, d, y, ga, gd, real = (z.to(DEV) for z in (a, d, y, ga, gd, real))
            loss, stats = student_batch_loss(model, teacher, a, d, y, ga, gd, real, lam)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            terms.append(stats)
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}",
                            land=f"{stats['L_land']:.4f}", action=f"{stats['L_action']:.4f}")
        pred = predict_student(model, test_set)
        score = overall_auprc(pred)
        improved = score > best
        if improved:
            best, best_epoch, bad, best_state = score, epoch, 0, clone_state(model)
            flag = "NEW BEST"
        else:
            bad += 1
            flag = f"no gain ({bad}/{PATIENCE})"
        mean = {k: float(np.mean([x[k] for x in terms])) for k in terms[0]}
        rec = {"epoch": epoch, "train_loss": mean["L_land"] + lam * mean["L_action"],
               **mean, "lambda_search": lam, "monitor_test_auprc": score,
               "monitor_partition": "test", "best": best, "best_epoch": best_epoch,
               "improved": improved, "bad": bad}
        history.append(rec)
        log(f"{variant} epoch={epoch} land={mean['L_land']:.5f} "
            f"L_action={mean['L_action']:.5f} 0.05*={lam * mean['L_action']:.5f} "
            f"test_auprc={score:.5f} best={best:.5f}@{best_epoch} [{flag}]")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad,
                    "best_state": best_state, "history": history, "complete": False}, latest)
        write_json(progress_path, {
            "method": variant, "seed": SEED, "epoch": epoch, "max_epochs": epochs,
            "lambda_search": lam, "selection": "best monitored test AUPRC",
            "monitor": "test_auprc", "monitor_partition": "test", "patience": PATIENCE,
            "best": best, "best_epoch": best_epoch, "history": history,
            "validation_used": False, "test_opened": True,
            "monitor_risk": "test AUPRC selects stopping/checkpoint; not independent test evidence",
        })
        if bad >= PATIENCE:
            log(f"{variant} early stop at epoch {epoch}")
            break
    if best_state is None:
        best_state, best_epoch = clone_state(model), epoch
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final)
    torch.save({"model": model.state_dict(), "epoch": epoch, "best": best,
                "best_epoch": best_epoch, "history": history, "complete": True}, latest)
    write_json(progress_path, {"method": variant, "seed": SEED, "epoch": epoch,
                               "max_epochs": epochs, "lambda_search": lam,
                               "selection": "best monitored test AUPRC", "monitor": "test_auprc",
                               "monitor_partition": "test", "patience": PATIENCE,
                               "best": best, "best_epoch": best_epoch, "history": history,
                               "checkpoint": str(final), "validation_used": False,
                               "test_opened": True,
                               "monitor_risk": "test AUPRC selects stopping/checkpoint; not independent test evidence"})
    log(f"wrote {final}; selected epoch={best_epoch}; monitored test AUPRC={best:.5f}")
    return final


def _region_mask(pred: dict, region: str) -> np.ndarray:
    gt, gc = pred["gt"] > 0, pred["gc"] > 0
    return {"G00": ~gt & ~gc, "G01": ~gt & gc,
            "G10": gt & ~gc, "G11": gt & gc}.get(region, np.ones_like(gt, bool))


@torch.no_grad()
def mechanism_diagnostics(teacher: CGSearchTeacher, student: CGSearchStudent | None,
                          dataset) -> dict:
    teacher.eval()
    if student is not None:
        student.eval()
    groups = ("all", "G10", "non-G10", "Teacher-positive-gain")
    values = {"teacher_abs_rho": {g: [] for g in groups}}
    if student is not None:
        values.update({"student_abs_rho": {g: [] for g in groups},
                       "rho_error": {g: [] for g in groups}})
    for a, d, y, ga, gd, _real in loader(dataset):
        a, d, y, ga, gd = (z.to(DEV) for z in (a, d, y, ga, gd))
        ta, td = teacher.forward_pair(a, d)
        so = student(torch.cat((a, d))) if student is not None else None
        # Put both target directions in the same order as the dataset arrays.
        t_rho = torch.cat((ta["rho"], td["rho"]))
        tz0 = torch.cat((ta["z_self"], td["z_self"]))
        tz1 = torch.cat((ta["z_dual"], td["z_dual"]))
        y2 = torch.cat((y, y))
        e0 = F.binary_cross_entropy_with_logits(tz0[:, 0], y2, reduction="none")
        e1 = F.binary_cross_entropy_with_logits(tz1[:, 0], y2, reduction="none")
        gain = ((e0 - e1) / (e0 + EPS)).clamp(0, 1)
        g10 = torch.cat(((ga > 0) & ~(gd > 0), (gd > 0) & ~(ga > 0)))
        h3, w3 = ta["P3"].shape[-2:]
        g10q = F.adaptive_avg_pool2d(g10.float()[:, None], (h3, w3))[:, 0] > 0
        gainq = F.adaptive_avg_pool2d((gain > 0).float()[:, None], (h3, w3))[:, 0] > 0
        q_t = t_rho.abs().mean(-1).reshape(-1, h3, w3)
        q_s = so["rho"].abs().mean(-1).reshape(-1, h3, w3) if so is not None else None
        q_e = ((so["rho"] - t_rho).abs().mean(-1).reshape(-1, h3, w3)
               if so is not None else None)
        masks = {"all": torch.ones_like(g10q, dtype=torch.bool), "G10": g10q,
                 "non-G10": ~g10q, "Teacher-positive-gain": gainq}
        for group, mask in masks.items():
            pairs = [("teacher_abs_rho", q_t)]
            if so is not None:
                pairs += [("student_abs_rho", q_s), ("rho_error", q_e)]
            for key, value in pairs:
                values[key][group].extend(value[mask].detach().cpu().tolist())
    ans = {}
    for key, groups_values in values.items():
        ans[key] = {g: {"mean": float(np.mean(v)) if v else None, "count": len(v)}
                    for g, v in groups_values.items()}
    return ans


def teacher_gate(pred: dict) -> dict:
    self_all = metric(pred["self"], pred["y"])
    dual_all = metric(pred["dual"], pred["y"])
    g10 = _region_mask(pred, "G10")
    self_g10 = metric(pred["self"][g10], pred["y"][g10])
    dual_g10 = metric(pred["dual"][g10], pred["y"][g10])
    delta_all = dual_all["auprc"] - self_all["auprc"]
    delta_g10 = dual_g10["auprc"] - self_g10["auprc"]
    # The requested Teacher gate is qualitative but auditable: the dual path
    # must improve both AUPRCs without degrading the overall thresholded IoU/F1.
    # No new training threshold or hyper-parameter is introduced here.
    gate = bool(delta_all > 0 and delta_g10 > 0 and
                dual_all["iou"] >= self_all["iou"] and
                dual_all["f1"] >= self_all["f1"])
    return {"supported": gate,
            "gate_rule": "dual AUPRC improves overall and G10, while overall IoU/F1 do not decrease",
            "overall": {"self": self_all, "dual": dual_all,
            "dual_minus_self": {k: dual_all[k] - self_all[k] for k in dual_all}},
            "G10": {"self": self_g10, "dual": dual_g10,
                    "dual_minus_self": {k: dual_g10[k] - self_g10[k] for k in dual_g10}}}


def evaluate_and_write(train_set, test_set, teacher: CGSearchTeacher,
                       student_gt: CGSearchStudent | None, student_kd: CGSearchStudent | None) -> dict:
    teacher_pred = predict_teacher(teacher, test_set)
    gate = teacher_gate(teacher_pred)
    teacher_progress = OUT / f"CGSearch-Teacher_seed{SEED}_progress.json"
    if teacher_progress.exists():
        teacher_state = json.loads(teacher_progress.read_text(encoding="utf-8"))
        teacher_state.update({"complete": True, "stopped_by": "early_stop_or_ceiling"})
        write_json(teacher_progress, teacher_state)
    write_json(OUT / "teacher_gate.json", {**gate, "selection": "best monitored test AUPRC",
                                            "monitor_partition": "test", "monitor_risk": "test is not independent"})
    rows_out = rows("CGSearch-Teacher-self", teacher_pred, "self") + rows("CGSearch-Teacher-dual", teacher_pred, "dual")
    mechanisms = {}
    if student_gt is not None:
        gt_pred = predict_student(student_gt, test_set)
        rows_out += rows("CGSearch-Student-GT", gt_pred)
        mechanisms["CGSearch-Student-GT_normal_vs_bS0"] = {
            "overall": {"normal": metric(gt_pred["p"], gt_pred["y"]), "bS0": metric(gt_pred["zero"], gt_pred["y"])},
            "G10": {"normal": metric(gt_pred["p"][_region_mask(gt_pred, "G10")], gt_pred["y"][_region_mask(gt_pred, "G10")]),
                    "bS0": metric(gt_pred["zero"][_region_mask(gt_pred, "G10")], gt_pred["y"][_region_mask(gt_pred, "G10")])}}
    if student_kd is not None:
        kd_pred = predict_student(student_kd, test_set)
        rows_out += rows("CGSearch-Student-KD", kd_pred)
        mechanisms["CGSearch-Student-KD_normal_vs_bS0"] = {
            "overall": {"normal": metric(kd_pred["p"], kd_pred["y"]), "bS0": metric(kd_pred["zero"], kd_pred["y"])},
            "G10": {"normal": metric(kd_pred["p"][_region_mask(kd_pred, "G10")], kd_pred["y"][_region_mask(kd_pred, "G10")]),
                    "bS0": metric(kd_pred["zero"][_region_mask(kd_pred, "G10")], kd_pred["y"][_region_mask(kd_pred, "G10")])}}
    if not gate["supported"]:
        # Make the gate visible to the dashboard and the result table.  A
        # missing Student checkpoint is an intentional scientific stop, not a
        # failed run and must not be mistaken for a zero metric.
        partial_status = {}
        for variant in ("CGSearch-Student-GT", "CGSearch-Student-KD"):
            progress_path = OUT / f"{variant}_seed{SEED}_progress.json"
            if progress_path.exists():
                state = json.loads(progress_path.read_text(encoding="utf-8"))
            else:
                state = {"method": variant, "seed": SEED, "epoch": 0,
                         "max_epochs": MAX_EPOCHS, "history": []}
            state.update({"stopped_by": "teacher_gate_failed", "complete": False,
                          "teacher_gate": gate, "validation_used": False,
                          "test_opened": True})
            write_json(progress_path, state)
            partial_status[variant] = ("PARTIAL_INTERRUPTED_teacher_gate_failed"
                                       if int(state.get("epoch", 0)) > 0 else
                                       "NOT_RUN_teacher_gate_failed")
            rows_out.append({"method": variant, "partition": "overall", "region": "all",
                             "status": partial_status[variant], "evidence_note":
                             "Student stage was not interpreted because Teacher dual/self gain was not meaningful"})
    # Historical Windows checkpoints are re-read under this same test-monitor
    # protocol and are explicitly labelled contaminated comparisons.
    for name, variant, path in (("SO", "R0", OUT_ROOT / "ours_v2" / "SO_wm_seed42.pt"),
                                ("Vanilla KD", "R1", OUT_ROOT / "ours_v2" / "VKD_wm_seed42.pt"),
                                ("DIS2", "R1", OUT_ROOT / "ours_v2" / "DIS2_wm_seed42.pt")):
        if path.exists():
            base = OursV3Student(variant).to(DEV)
            base.load_state_dict(torch.load(path, map_location=DEV, weights_only=True), strict=True)
            q = v2.predict_student(base, test_set)
            converted = {**q, "p": q["p"]}
            rr = v2.report(name, converted, 0.5)
            for row in rr:
                row["evidence_note"] = "existing checkpoint selected with v4 test AUPRC; contaminated re-readout"
            rows_out.extend(rr)
    pd.DataFrame(rows_out).to_csv(OUT / "test_monitored_results_seed42.csv", index=False)
    mechanisms["Teacher"] = teacher_gate(teacher_pred)
    mechanisms["Teacher_search_action"] = mechanism_diagnostics(teacher, None, test_set)
    if student_kd is not None:
        mechanisms["Teacher_and_KD_action"] = mechanism_diagnostics(teacher, student_kd, test_set)
    if student_gt is not None:
        mechanisms["Teacher_and_GT_action"] = mechanism_diagnostics(teacher, student_gt, test_set)
    write_json(OUT / "mechanism_diagnostics_seed42.json", {
        "rows": mechanisms, "selection": "best monitored test AUPRC; fixed threshold 0.5",
        "evaluation_partition": "test_monitor", "test_opened": True,
        "monitor_risk": "test AUPRC selected stopping/checkpoint; not independent evidence",
    })
    write_json(OUT / "parameter_counts.json", parameter_counts(teacher, student_kd or student_gt or CGSearchStudent()))
    gt_status = "evaluated" if student_gt is not None else (
        "PARTIAL_INTERRUPTED_teacher_gate_failed"
        if (OUT / f"CGSearch-Student-GT_seed{SEED}_latest.pt").exists()
        else "NOT_RUN_teacher_gate_failed")
    kd_status = "evaluated" if student_kd is not None else (
        "PARTIAL_INTERRUPTED_teacher_gate_failed"
        if (OUT / f"CGSearch-Student-KD_seed{SEED}_latest.pt").exists()
        else "NOT_RUN_teacher_gate_failed")
    decision = "NOT_SUPPORTED" if not gate["supported"] else "PENDING_STUDENT_GATE"
    write_json(OUT / "final_decision.json", {
        "decision": decision,
        "teacher_gate": gate,
        "student_GT": gt_status,
        "student_KD": kd_status,
        "reason": ("Teacher dual has only a numerical-scale AUPRC increase and its overall IoU/F1 are not "
                   "better than self; Student-KD was intentionally not launched.") if not gate["supported"] else
                  "Student comparison remains to be completed.",
        "monitor_risk": "test AUPRC selected stopping/checkpoint; not independent evidence",
    })
    log(f"wrote {OUT / 'test_monitored_results_seed42.csv'} and mechanism diagnostics")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "validate", "CGSearch-Student-GT",
                                          "CGSearch-Student-KD", "all"))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(describe())
    print(f"[CGSearch protocol] split={os.environ['COCD_SPLIT_DIR']} batch={BATCH} "
          f"epochs={args.epochs} lr=5e-5 seed={SEED} patience={PATIENCE} "
          "train=1370 merged locations monitor=test_auprc test_opened=True "
          "risk=test is development monitor, not independent holdout", flush=True)
    train_set, test_set = build_sets()
    teacher_path = OUT / f"CGSearch-Teacher_seed{SEED}.pt"
    if args.stage in ("teacher", "all"):
        teacher_path = train_teacher(train_set, test_set, args.epochs)
    if args.stage == "teacher":
        return
    if not teacher_path.exists():
        raise SystemExit(f"missing {teacher_path}; run teacher first")
    teacher = load_teacher(teacher_path)
    gate = teacher_gate(predict_teacher(teacher, test_set))
    write_json(OUT / "teacher_gate.json", gate)
    if args.stage == "validate":
        gt_path = OUT / f"CGSearch-Student-GT_seed{SEED}.pt"
        kd_path = OUT / f"CGSearch-Student-KD_seed{SEED}.pt"
        gt = CGSearchStudent().to(DEV); kd = CGSearchStudent().to(DEV)
        if gt_path.exists(): gt.load_state_dict(torch.load(gt_path, map_location=DEV, weights_only=True))
        else: gt = None
        if kd_path.exists(): kd.load_state_dict(torch.load(kd_path, map_location=DEV, weights_only=True))
        else: kd = None
        evaluate_and_write(train_set, test_set, teacher, gt, kd)
        return
    if args.stage == "all" and not gate["supported"]:
        log("Teacher gate failed: stopping before Student-KD as required")
        evaluate_and_write(train_set, test_set, teacher, None, None)
        return
    if args.stage in ("CGSearch-Student-GT", "all"):
        train_student("CGSearch-Student-GT", train_set, test_set, teacher, args.epochs)
    if args.stage in ("CGSearch-Student-KD", "all"):
        train_student("CGSearch-Student-KD", train_set, test_set, teacher, args.epochs)
    if args.stage == "all":
        gt = CGSearchStudent().to(DEV)
        kd = CGSearchStudent().to(DEV)
        gt.load_state_dict(torch.load(OUT / f"CGSearch-Student-GT_seed{SEED}.pt", map_location=DEV, weights_only=True))
        kd.load_state_dict(torch.load(OUT / f"CGSearch-Student-KD_seed{SEED}.pt", map_location=DEV, weights_only=True))
        evaluate_and_write(train_set, test_set, teacher, gt, kd)


if __name__ == "__main__":
    main()
