"""Equal mean per-batch-r contribution comparison for S2 / RKD / DIS2-RS.

This is an isolated extension of the v4 raw-training benchmark. It preserves
the three existing KD definitions and changes only their external scalar
weighting: a smooth EMA controller updates lambda every few training steps to
track a 10% weighted KD contribution. Existing S0-Raw and S2-Raw artifacts are
left untouched and are recorded as references only.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cocd"))

from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_teacher_target_search_v2 import (  # noqa: E402
    TargetSearchTeacherV2, load_so_backbone)
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402
from windows_main.train_sgri_powc_masked_v4 import (  # noqa: E402
    BATCH, EPS, LR, MAX_EPOCHS, PATIENCE, SEED, WD, set_seed, write_json)
from windows_main.train_sgri_powc_masked_v4 import loader as standard_loader  # noqa: E402
from windows_main.train_raw_cver_v1 import s2_terms  # noqa: E402

OUT = Path(__file__).resolve().parent
TEACHER_CKPT = ROOT / "experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
INIT_CKPT = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
NATIVE_DIR = ROOT / "experiments/windows_main/raw_cver_v1_seed42"
OLD_KD_DIR = ROOT / "experiments/windows_main/multitoone_kd_benchmark_20260927"

TARGET_SHARE = 0.10
EMA_BETA = 0.75
LAMBDA_LOG_GAIN = 1.00
LAMBDA_MAX_FACTOR_PER_UPDATE = 1.25
LAMBDA_UPDATE_EVERY_STEPS = 2
TEMPERATURE = 1.0
ARMS = ("S2-10%", "RKD-10%", "DIS2-RS-10%")


def masked_metrics(p, y, g, c):
    regions = {
        "Overall": np.ones_like(g, dtype=bool),
        "BothValid": (g == 0) & (c == 0),
        "TargetValid": g == 0,
        "OverallValid": (g == 0) | (c == 0),
        "G10": (g > 0) & (c == 0),
        "G01": (g == 0) & (c > 0),
    }
    out = {}
    for name, mask in regions.items():
        m = mask.reshape(-1)
        yy, pp = y.reshape(-1)[m].astype(bool), p.reshape(-1)[m]
        pred = pp >= 0.5
        tp, fp, fn = int((pred & yy).sum()), int((pred & ~yy).sum()), int((~pred & yy).sum())
        out[name] = {
            "AUPRC": float(average_precision_score(yy, pp)),
            "IoU": float(tp / (tp + fp + fn + EPS)),
            "F1": float(2 * tp / (2 * tp + fp + fn + EPS)),
            "pixels": int(m.sum()),
            "positive_pixels": int(yy.sum()),
        }
    return out


def evaluate(student, test):
    student.eval()
    ps, ys, gas, gds = [], [], [], []
    with torch.inference_mode():
        for a, d, y, ga, gd, _ in standard_loader(test):
            oa, od = student(a.to(DEV)), student(d.to(DEV))
            ps.extend((torch.sigmoid(oa["z"][:, 0]).cpu().numpy(),
                       torch.sigmoid(od["z"][:, 0]).cpu().numpy()))
            ys.extend((y.numpy(), y.numpy()))
            gas.extend((ga.numpy(), gd.numpy()))
            gds.extend((gd.numpy(), ga.numpy()))
    p, y, g, c = tuple(np.concatenate(q, axis=0) for q in (ps, ys, gas, gds))
    return masked_metrics(p, y, g, c)


def train_loader_for_epoch(train, epoch):
    # Identical sample order across arms, as in the original matched KD runner.
    generator = torch.Generator(device="cpu").manual_seed(SEED + epoch)
    return DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0,
                      generator=generator)


def bernoulli_kl(student_logits, teacher_logits):
    """Canonical RKD per-pixel KL(Bernoulli(teacher) || Bernoulli(student))."""
    zt, zs = teacher_logits / TEMPERATURE, student_logits / TEMPERATURE
    pt = torch.sigmoid(zt).detach()
    log_pt, log_qt = F.logsigmoid(zt).detach(), F.logsigmoid(-zt).detach()
    log_ps, log_qs = F.logsigmoid(zs), F.logsigmoid(-zs)
    return (pt * (log_pt - log_ps) + (1 - pt) * (log_qt - log_qs)).mean()


class MultiResolutionAlign(nn.Module):
    """Existing DIS2-RS training-only identity-initialized 1x1 adapters."""
    def __init__(self, channels=128, levels=3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])
        for layer in self.proj:
            nn.init.dirac_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        return tuple(layer(f) for layer, f in zip(self.proj, features))


def classwise_multires_loss(student_features, teacher_features, y, align):
    """Existing DIS2-RS GT-class attention-weighted cosine feature loss."""
    s_features = align(student_features)
    level_losses, class_values = [], {"fg": [], "bg": []}
    for fs, ft in zip(s_features, teacher_features):
        ft = ft.detach()
        target = F.interpolate(y[:, None], size=ft.shape[-2:], mode="area")
        fg_mask = F.adaptive_max_pool2d(y[:, None], output_size=ft.shape[-2:])[:, 0] > 0
        bg_mask = F.adaptive_max_pool2d((1 - y)[:, None], output_size=ft.shape[-2:])[:, 0] > 0
        fg_mask = fg_mask & (target[:, 0] >= 0.5)
        bg_mask = bg_mask & ~fg_mask
        distance = 1 - F.cosine_similarity(fs, ft, dim=1, eps=1e-6)
        teacher_attention = ft.detach().abs().mean(1)
        terms = []
        for name, mask in (("fg", fg_mask), ("bg", bg_mask)):
            raw = teacher_attention * mask.float()
            norm = raw.sum((-2, -1), keepdim=True)
            count = mask.float().sum((-2, -1), keepdim=True)
            att = raw * count / norm.clamp_min(EPS)
            per_image = (distance * att).sum((-2, -1)) / count[:, 0, 0].clamp_min(1)
            present = count[:, 0, 0] > 0
            if bool(present.any()):
                term = per_image[present].mean()
                terms.append(term)
                class_values[name].append(term.detach())
        if terms:
            level_losses.append(torch.stack(terms).mean())
    if not level_losses:
        return student_features[0].sum() * 0, {k: None for k in class_values}
    perclass = {k: (float(torch.stack(v).mean()) if v else None) for k, v in class_values.items()}
    return torch.stack(level_losses).mean(), perclass


def prepare_teacher():
    teacher = TargetSearchTeacherV2().to(DEV)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEV, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def mean_batch_share(lam, seg_values, kd_values):
    seg = np.asarray(seg_values, dtype=np.float64)
    kd = np.asarray(kd_values, dtype=np.float64)
    shares = lam * kd / np.maximum(seg + lam * kd, EPS)
    return float(shares.mean())


def calibrate_lambda_for_mean_share(seg_values, kd_values):
    """Solve lambda so mean per-batch r_i over initial train pass is 10%."""
    low, high = 0.0, 1.0
    while mean_batch_share(high, seg_values, kd_values) < TARGET_SHARE and high < 1e12:
        high *= 2.0
    if mean_batch_share(high, seg_values, kd_values) < TARGET_SHARE:
        raise RuntimeError("initial training batches cannot attain target mean r_KD")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if mean_batch_share(mid, seg_values, kd_values) < TARGET_SHARE:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def update_controller(lam, r_ema):
    r_ema = min(max(r_ema, EPS), 1.0 - EPS)
    target_logit = math.log(TARGET_SHARE / (1.0 - TARGET_SHARE))
    ema_logit = math.log(r_ema / (1.0 - r_ema))
    log_delta = LAMBDA_LOG_GAIN * (target_logit - ema_logit)
    max_delta = math.log(LAMBDA_MAX_FACTOR_PER_UPDATE)
    log_delta = min(max(log_delta, -max_delta), max_delta)
    return lam * math.exp(log_delta)


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def calibrate_initial_lambda(name, student, align, teacher, train):
    """Calibrate from a no-update pass over the full initial train epoch.

    This avoids making lambda depend on the unusually variable first minibatch.
    It reads only train IDs and performs no optimizer steps or checkpoint choice.
    The S2 component normalization remains its original first-batch definition.
    """
    student.eval()
    if align is not None:
        align.eval()
    seg_values, kd_values = [], []
    s2_components = None
    with torch.inference_mode():
        for a, d, y, ga, gd, _real in train_loader_for_epoch(train, 1):
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            if name == "S2-10%":
                lp, le, outs = s2_terms(student, teacher, a, d, ga.to(DEV), gd.to(DEV))
                if s2_components is None:
                    s2_components = {"policy_init": max(float(lp), EPS),
                                     "evidence_init": max(float(le), EPS),
                                     "note": "canonical S2 component normalization from first batch; retained unchanged"}
                seg = 0.5 * (
                    F.binary_cross_entropy_with_logits(outs[0]["z"][:, 0], y) +
                    F.binary_cross_entropy_with_logits(outs[1]["z"][:, 0], y))
                kd_raw = 0.5 * lp / s2_components["policy_init"] + 0.5 * le / s2_components["evidence_init"]
            else:
                with torch.no_grad():
                    teacher_a, teacher_d = teacher.forward_pair(a, d, "counter")
                    zta, ztd = teacher_a["z"], teacher_d["z"]
                    tfa, tfd = teacher_a["T"], teacher_d["T"]
                student_a, student_d = student(a), student(d)
                seg = 0.5 * (
                    F.binary_cross_entropy_with_logits(student_a["z"][:, 0], y) +
                    F.binary_cross_entropy_with_logits(student_d["z"][:, 0], y))
                kd_logit = 0.5 * (
                    bernoulli_kl(student_a["z"][:, 0], zta[:, 0]) +
                    bernoulli_kl(student_d["z"][:, 0], ztd[:, 0]))
                if name == "RKD-10%":
                    kd_raw = kd_logit
                else:
                    feat_a, _ = classwise_multires_loss(student_a["T"], tfa, y, align)
                    feat_d, _ = classwise_multires_loss(student_d["T"], tfd, y, align)
                    kd_raw = 0.5 * (feat_a + feat_d) + kd_logit
            seg_values.append(float(seg))
            kd_values.append(float(kd_raw))
    if not seg_values or not kd_values or not np.isfinite(kd_values).all():
        raise RuntimeError(f"invalid full-train initial loss calibration for {name}")
    seg_mean, kd_mean = float(np.mean(seg_values)), float(np.mean(kd_values))
    if kd_mean <= EPS:
        raise RuntimeError(f"initial mean KD is zero for {name}; cannot calibrate a nonzero KD share")
    lam = calibrate_lambda_for_mean_share(seg_values, kd_values)
    student.train()
    if align is not None:
        align.train()
    record = {**(s2_components or {}), "initial_train_seg_mean": seg_mean,
              "initial_train_kd_raw_mean": kd_mean, "initial_mean_batch_r": mean_batch_share(lam, seg_values, kd_values),
              "initial_lambda": lam,
              "initial_epoch_batches": len(seg_values), "target_share": TARGET_SHARE,
              "rule": "no-update full-train initialization calibration solves mean per-batch r; lambda subsequently follows EMA of observed r"}
    print(f"{name} full-train calibration {json.dumps(record)}", flush=True)
    return lam, seg_mean, kd_mean, record


def train_arm(name, train, test, teacher):
    ckpt_path = OUT / f"{name}_seed42_best.pt"
    progress_path = OUT / f"{name}_progress.json"
    summary_path = OUT / f"{name}_summary.json"
    if progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8")).get("complete"):
        print(f"SKIP complete {name}", flush=True)
        return

    set_seed(SEED)
    student = StudentTargetSearchV2().to(DEV)
    load_so_backbone(student, str(INIT_CKPT))
    align = MultiResolutionAlign().to(DEV) if name == "DIS2-RS-10%" else None
    params = list(student.parameters()) + (list(align.parameters()) if align is not None else [])
    optimizer = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)

    best, bad_epochs, best_epoch, best_state, best_align = -1.0, 0, None, None, None
    lam, seg_ema, kd_ema, calibration = calibrate_initial_lambda(name, student, align, teacher, train)
    r_ema = TARGET_SHARE
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        student.train()
        if align is not None:
            align.train()
        batch_logs, lambda_updates, class_align_logs = [], [], []
        lambda_start = lam
        batches = train_loader_for_epoch(train, epoch)
        total_batches = len(batches)
        for step, (a, d, y, ga, gd, real) in enumerate(batches, start=1):
            a, d, y = a.to(DEV), d.to(DEV), y.to(DEV)
            optimizer.zero_grad(set_to_none=True)

            if name == "S2-10%":
                lp, le, outs = s2_terms(student, teacher, a, d, ga.to(DEV), gd.to(DEV))
                seg = 0.5 * (
                    F.binary_cross_entropy_with_logits(outs[0]["z"][:, 0], y) +
                    F.binary_cross_entropy_with_logits(outs[1]["z"][:, 0], y))
                kd_raw = 0.5 * lp / calibration["policy_init"] + 0.5 * le / calibration["evidence_init"]
                kd_components = {"policy_js": float(lp.detach()), "evidence_distance": float(le.detach())}
                feature_loss = torch.zeros((), device=DEV)
            else:
                with torch.no_grad():
                    teacher_a, teacher_d = teacher.forward_pair(a, d, "counter")
                    zta, ztd = teacher_a["z"], teacher_d["z"]
                    tfa, tfd = teacher_a["T"], teacher_d["T"]
                student_a, student_d = student(a), student(d)
                seg = 0.5 * (
                    F.binary_cross_entropy_with_logits(student_a["z"][:, 0], y) +
                    F.binary_cross_entropy_with_logits(student_d["z"][:, 0], y))
                kd_logit = 0.5 * (
                    bernoulli_kl(student_a["z"][:, 0], zta[:, 0]) +
                    bernoulli_kl(student_d["z"][:, 0], ztd[:, 0]))
                feature_loss = torch.zeros((), device=DEV)
                kd_components = {"logit_kl": float(kd_logit.detach())}
                if name == "RKD-10%":
                    kd_raw = kd_logit
                else:
                    feat_a, fg_a = classwise_multires_loss(student_a["T"], tfa, y, align)
                    feat_d, fg_d = classwise_multires_loss(student_d["T"], tfd, y, align)
                    feature_loss = 0.5 * (feat_a + feat_d)
                    kd_raw = feature_loss + kd_logit
                    kd_components["feature_cosine"] = float(feature_loss.detach())
                    fg_align = {k: float(np.mean([v for v in (fg_a[k], fg_d[k]) if v is not None]))
                                if any(v is not None for v in (fg_a[k], fg_d[k])) else None
                                for k in ("fg", "bg")}
                    class_align_logs.append(fg_align)

            seg_value, kd_value = float(seg.detach()), float(kd_raw.detach())
            if not math.isfinite(kd_value):
                raise RuntimeError(f"invalid KD magnitude for {name}: {kd_value}")
            weighted_kd = lam * kd_raw
            loss = seg + weighted_kd
            actual_r = float((weighted_kd.detach() / loss.detach().clamp_min(EPS)).cpu())
            loss.backward()
            optimizer.step()
            batch_logs.append({"seg": seg_value, "kd_raw": kd_value, "lambda": lam,
                               "weighted_kd": float(weighted_kd.detach()),
                               "actual_share": actual_r, **kd_components})
            # Track raw loss scales and the actual per-batch ratio. Lambda only
            # changes after observing the current batch, for use on later steps.
            seg_ema = EMA_BETA * seg_ema + (1.0 - EMA_BETA) * seg_value
            kd_ema = EMA_BETA * kd_ema + (1.0 - EMA_BETA) * kd_value
            r_ema = EMA_BETA * r_ema + (1.0 - EMA_BETA) * actual_r
            if step % LAMBDA_UPDATE_EVERY_STEPS == 0 or step == total_batches:
                old_lam = lam
                lam = update_controller(lam, r_ema)
                lambda_updates.append({"step": step, "lambda_before": old_lam,
                                       "lambda_after": lam, "r_ema": r_ema,
                                       "seg_ema": seg_ema, "kd_raw_ema": kd_ema})

        metrics = evaluate(student, test)
        monitor = metrics["Overall"]["AUPRC"]
        seg_sum = sum(r["seg"] for r in batch_logs)
        weighted_kd_sum = sum(r["weighted_kd"] for r in batch_logs)
        epoch_share = weighted_kd_sum / (seg_sum + weighted_kd_sum + EPS)
        shares = [r["actual_share"] for r in batch_logs]
        seg_epoch, kd_epoch = _mean(batch_logs, "seg"), _mean(batch_logs, "kd_raw")
        lambda_next = lam
        row = {"epoch": epoch, "monitor_test_auprc_raw_overall": monitor,
               "metrics": metrics,
               "train_loss_seg_mean": seg_epoch,
               "train_loss_kd_raw_mean": kd_epoch,
               "train_weighted_kd_mean": _mean(batch_logs, "weighted_kd"),
               "train_weighted_kd_share_aggregate": epoch_share,
               "train_weighted_kd_share_batch_mean": float(np.mean(shares)),
               "train_weighted_kd_share_batch_min": float(np.min(shares)),
               "train_weighted_kd_share_batch_max": float(np.max(shares)),
               "lambda_start": lambda_start, "lambda_end": lambda_next, "lambda_next": lambda_next,
               "seg_loss_ema_end": seg_ema, "kd_raw_loss_ema_end": kd_ema, "r_ema_end": r_ema,
               "lambda_updates": lambda_updates}
        if name == "S2-10%":
            row.update({"train_policy_js_mean": _mean(batch_logs, "policy_js"),
                        "train_evidence_distance_mean": _mean(batch_logs, "evidence_distance")})
        elif name == "DIS2-RS-10%":
            row["train_logit_kl_mean"] = _mean(batch_logs, "logit_kl")
            row["train_feature_cosine_mean"] = _mean(batch_logs, "feature_cosine")
            row["classwise_feature_alignment"] = {
                k: (float(np.nanmean([r[k] for r in class_align_logs if r[k] is not None]))
                    if any(r[k] is not None for r in class_align_logs) else None)
                for k in ("fg", "bg")
            }
        else:
            row["train_logit_kl_mean"] = _mean(batch_logs, "logit_kl")
        history.append(row)
        lam = lambda_next

        if monitor > best:
            best, bad_epochs, best_epoch = monitor, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            best_align = ({k: v.detach().cpu().clone() for k, v in align.state_dict().items()}
                          if align is not None else None)
        else:
            bad_epochs += 1
        payload = {
            "arm": name, "seed": SEED, "epoch": epoch, "best_epoch": best_epoch,
            "best_test_auprc": best, "patience": PATIENCE, "max_epochs": MAX_EPOCHS,
            "monitor_risk": "canonical test AUPRC monitor/checkpoint selection; optimistic test estimate",
            "train_ids": len(train.rows), "test_ids": len(test.rows), "validation_ids": 0,
            "batch_size": BATCH,
            "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WD},
            "initialization": str(INIT_CKPT), "teacher": str(TEACHER_CKPT),
            "distillation": ("canonical S2 policy JS + same-target-memory evidence cosine, including original query weighting"
                             if name == "S2-10%" else
                             "Bernoulli KL, T=1, full raw extent" if name == "RKD-10%" else
                             "existing P2/P3/P4 GT-class attention-weighted cosine + Bernoulli logit KL; full raw extent"),
        "kd_budget_controller": {"target_share": TARGET_SHARE, "ema_beta": EMA_BETA,
                                     "update_every_steps": LAMBDA_UPDATE_EVERY_STEPS,
                                     "log_gain": LAMBDA_LOG_GAIN,
                                     "max_lambda_factor_per_update": LAMBDA_MAX_FACTOR_PER_UPDATE,
                                     "definition": "actual weighted KD / (seg + actual weighted KD)"},
            "calibration": calibration, "history": history, "complete": False}
        write_json(progress_path, payload)
        print(f"{name} epoch={epoch} AP={monitor:.6f} best={best:.6f}@{best_epoch} "
              f"kd_share_batch_mean={float(np.mean(shares)):.4f} pooled={epoch_share:.4f} "
              f"batch_range=[{min(shares):.4f},{max(shares):.4f}] "
              f"lambda={row['lambda_start']:.5g} next={row['lambda_next']:.5g}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"no selected checkpoint for {name}")
    student.load_state_dict(best_state)
    checkpoint = {"student": best_state, "training_only_alignment": best_align,
                  "arm": name, "best_epoch": best_epoch, "lambda_final": lam,
                  "calibration": calibration, "kd_budget_controller": {
                      "target_share": TARGET_SHARE, "ema_beta": EMA_BETA,
                      "update_every_steps": LAMBDA_UPDATE_EVERY_STEPS,
                      "log_gain": LAMBDA_LOG_GAIN,
                      "max_lambda_factor_per_update": LAMBDA_MAX_FACTOR_PER_UPDATE}}
    torch.save(checkpoint, ckpt_path)
    best_metrics = evaluate(student, test)
    write_json(summary_path, {"arm": name, "best_epoch": best_epoch,
                              "monitor_test_auprc": best, "metrics": best_metrics,
                              "mechanism": history[best_epoch - 1],
                              "checkpoint": str(ckpt_path), "calibration": calibration})
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload.update({"complete": True, "checkpoint": str(ckpt_path)})
    write_json(progress_path, payload)
    print(f"COMPLETE {name} best={best:.6f}@{best_epoch}", flush=True)


def reference_metrics():
    refs = {}
    for key, path in (("S0-Raw", NATIVE_DIR / "S0-Raw_summary.json"),
                      ("S2-Raw-native", NATIVE_DIR / "S2-Raw_summary.json"),
                      ("RKD-old", OLD_KD_DIR / "RKD_summary.json"),
                      ("DIS2-RS-old", OLD_KD_DIR / "DIS2-RS_summary.json")):
        refs[key] = json.loads(path.read_text(encoding="utf-8"))
    return refs


def build_report():
    refs = reference_metrics()
    rows = {}
    for arm in ARMS:
        summary = json.loads((OUT / f"{arm}_summary.json").read_text(encoding="utf-8"))
        rows[arm] = summary
    reference_map = {
        "S0-Raw": refs["S0-Raw"]["best_metrics"],
        "S2-Raw-native": refs["S2-Raw-native"]["best_metrics"],
        "RKD-old": refs["RKD-old"]["metrics"],
        "DIS2-RS-old": refs["DIS2-RS-old"]["metrics"],
    }
    report = ["# Matched 10% actual KD contribution: Raw S2 / RKD / DIS2-RS",
              "",
              "独立目录诊断实验。三个 KD 的逐项定义、数据划分、Raw 输入、网络、优化器与测试 AUPRC 监控/早停规则沿用原 runner；唯一方法变化是外层 λ 由 EMA 控制，使实际加权 KD contribution 跟踪 10%。S0-Raw 与 S2-Raw-native 只作为既有参考，不与新训练组混算。",
              "",
              "| Method | Best epoch | Overall AP / IoU / F1 | BothValid AP / IoU / F1 | TargetValid AP / IoU / F1 | OverallValid AP / IoU / F1 | G10 AP / IoU / F1 | G01 AP / IoU / F1 |",
              "|---|---:|---|---|---|---|---|---|"]

    def fmt(m, region):
        x = m[region]
        return f"{x['AUPRC']:.4f} / {x['IoU']:.4f} / {x['F1']:.4f}"

    for arm in ARMS:
        s = rows[arm]
        m = s["metrics"]
        report.append(f"| {arm} | {s['best_epoch']} | {fmt(m, 'Overall')} | {fmt(m, 'BothValid')} | {fmt(m, 'TargetValid')} | {fmt(m, 'OverallValid')} | {fmt(m, 'G10')} | {fmt(m, 'G01')} |")
    for name in ("S0-Raw", "S2-Raw-native"):
        s = refs["S0-Raw"] if name == "S0-Raw" else refs["S2-Raw-native"]
        m = s["best_metrics"]
        report.append(f"| {name} (historical ref) | {s['best_epoch']} | {fmt(m, 'Overall')} | {fmt(m, 'BothValid')} | {fmt(m, 'TargetValid')} | — | {fmt(m, 'G10')} | {fmt(m, 'G01')} |")

    s2_native_progress = json.loads((NATIVE_DIR / "S2-Raw_progress.json").read_text(encoding="utf-8"))
    s2_native_share = s2_native_progress["history"][s2_native_progress["best_epoch"] - 1]["mechanism"]["kd_contribution"]
    report.extend(["", "## Historical fixed-lambda checkpoints (context only; not matched-budget rows)", "",
                   "| Method | Best epoch | Overall AP | Best-epoch KD contribution |", "|---|---:|---:|---:|"])
    report.append(f"| S2-Raw-native | {refs['S2-Raw-native']['best_epoch']} | {refs['S2-Raw-native']['best_metrics']['Overall']['AUPRC']:.4f} | {s2_native_share:.2%} |")
    for key, label in (("RKD-old", "RKD prior fixed-λ"), ("DIS2-RS-old", "DIS2-RS prior fixed-λ")):
        old = refs[key]
        share = old["mechanism"]["train_weighted_kd_share_mean"]
        report.append(f"| {label} | {old['best_epoch']} | {old['metrics']['Overall']['AUPRC']:.4f} | {share:.2%} |")

    report.extend(["", "## Epoch budget check", "",
                   "The exact per-batch ratio is `r_i = λ_i·L_KD,i / (L_seg,i + λ_i·L_KD,i)`. Two epoch summaries are shown because averaging these ratios is not the same as pooling the loss contributions: `pooled` is `Σ(λ·L_KD)/Σ(L_seg+λ·L_KD)`, while `batch mean` is the arithmetic mean of `r_i` across optimizer steps. Both are included to make the budget interpretation explicit.",
                   "", "| Method | Pooled epoch r: min / max / mean; epochs in 8–12% | Mean of batch r: min / max / mean; epochs in 8–12% |", "|---|---|---|"])
    budget_summary = {}
    for arm in ARMS:
        progress = json.loads((OUT / f"{arm}_progress.json").read_text(encoding="utf-8"))
        pooled = [h["train_weighted_kd_share_aggregate"] for h in progress["history"]]
        batch_mean = [h["train_weighted_kd_share_batch_mean"] for h in progress["history"]]
        budget_summary[arm] = {
            "epochs": len(pooled),
            "pooled_min": float(min(pooled)), "pooled_max": float(max(pooled)),
            "pooled_mean": float(np.mean(pooled)),
            "pooled_in_band": int(sum(0.08 <= x <= 0.12 for x in pooled)),
            "batch_mean_min": float(min(batch_mean)), "batch_mean_max": float(max(batch_mean)),
            "batch_mean_mean": float(np.mean(batch_mean)),
            "batch_mean_in_band": int(sum(0.08 <= x <= 0.12 for x in batch_mean)),
        }
        b = budget_summary[arm]
        report.append(f"| {arm} | {b['pooled_min']:.2%} / {b['pooled_max']:.2%} / {b['pooled_mean']:.2%}; {b['pooled_in_band']}/{b['epochs']} | {b['batch_mean_min']:.2%} / {b['batch_mean_max']:.2%} / {b['batch_mean_mean']:.2%}; {b['batch_mean_in_band']}/{b['epochs']} |")

    report.extend(["", "## Actual weighted KD contribution by epoch", "",
                   "The target is the arithmetic mean of the exact training-step ratios `r_i = λ_i·L_KD,i / (L_seg,i + λ_i·L_KD,i)`. Initial λ is solved from a no-update full train pass so mean `r_i` is 10%; during training, the observed `r_i` EMA (beta 0.75) updates λ every two steps, capped at a 1.25× change. Segmentation and raw KD loss EMAs are also recorded. Pooled epoch loss ratio is shown separately because it differs from mean `r_i`.",
                   "", "| Method | Epoch | λ used → next | Seg mean | KD raw mean | KD share aggregate | Batch share mean [min,max] |", "|---|---:|---:|---:|---:|---:|---|"])
    for arm in ARMS:
        progress = json.loads((OUT / f"{arm}_progress.json").read_text(encoding="utf-8"))
        for h in progress["history"]:
            report.append(f"| {arm} | {h['epoch']} | {h['lambda_start']:.5g} → {h['lambda_next']:.5g} | {h['train_loss_seg_mean']:.5g} | {h['train_loss_kd_raw_mean']:.5g} | {h['train_weighted_kd_share_aggregate']:.2%} | {h['train_weighted_kd_share_batch_mean']:.2%} [{h['train_weighted_kd_share_batch_min']:.2%}, {h['train_weighted_kd_share_batch_max']:.2%}] |")

    report.extend(["", "## Protocol and interpretation", "",
                   "- Seed 42; same student initialization checkpoint; Adam, lr 5e-5, batch size 16, weight decay 0; max 100 epochs; patience 2; test Overall AUPRC selects best checkpoint under the locked project protocol (optimistic test selection risk applies equally to all new arms).",
                   "- Each arm receives the same epoch-specific shuffled sample order. Teacher checkpoint and model initialization are fixed. No validation split, hyperparameter sweep, or test-driven λ selection is introduced.",
                   "- S2 contribution uses the existing S2 normalized composite (`0.5·policy_JS/policy_init + 0.5·evidence_distance/evidence_init`); its internal first-batch component normalization is preserved, while the outer scalar λ is dynamically adjusted.",
                   "- RKD and DIS2-RS retain their previous raw loss definitions. S0-Raw and S2-Raw-native are historical references and are not part of the matched-10% three-method comparison.",
                   "- DIS2-RS is the same simplified classwise multi-resolution adaptation used by the previous runner, not a claim of exact official DIS2 reproduction.",
                   ""])
    (OUT / "equal_kd10_report.md").write_text("\n".join(report), encoding="utf-8")

    import csv
    with (OUT / "kd_budget_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "epoch", "lambda_used", "lambda_next", "seg_mean", "kd_raw_mean",
                         "weighted_kd_mean", "kd_share_aggregate", "kd_share_batch_mean",
                         "kd_share_batch_min", "kd_share_batch_max"])
        for arm in ARMS:
            progress = json.loads((OUT / f"{arm}_progress.json").read_text(encoding="utf-8"))
            for h in progress["history"]:
                writer.writerow([arm, h["epoch"], h["lambda_start"], h["lambda_next"],
                                 h["train_loss_seg_mean"], h["train_loss_kd_raw_mean"],
                                 h["train_weighted_kd_mean"], h["train_weighted_kd_share_aggregate"],
                                 h["train_weighted_kd_share_batch_mean"], h["train_weighted_kd_share_batch_min"],
                                 h["train_weighted_kd_share_batch_max"]])
    with (OUT / "kd_budget_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "epochs", "pooled_min", "pooled_max", "pooled_mean",
                                               "pooled_epochs_in_8_12", "batch_mean_min", "batch_mean_max",
                                               "batch_mean_mean", "batch_mean_epochs_in_8_12"])
        writer.writeheader()
        for arm in ARMS:
            b = budget_summary[arm]
            writer.writerow({"method": arm, "epochs": b["epochs"], "pooled_min": b["pooled_min"],
                             "pooled_max": b["pooled_max"], "pooled_mean": b["pooled_mean"],
                             "pooled_epochs_in_8_12": b["pooled_in_band"],
                             "batch_mean_min": b["batch_mean_min"], "batch_mean_max": b["batch_mean_max"],
                             "batch_mean_mean": b["batch_mean_mean"],
                             "batch_mean_epochs_in_8_12": b["batch_mean_in_band"]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, test_ids = split()
    if val_ids:
        raise RuntimeError(f"locked protocol requires zero validation IDs, got {len(val_ids)}")
    train = HaitiPairs(train_ids, True)
    test = HaitiPairs(test_ids, False, train.stats)
    if len(train_ids) != 1370 or len(test_ids) != 343:
        raise RuntimeError(f"split changed: train={len(train_ids)}, test={len(test_ids)}")
    teacher = prepare_teacher()
    protocol = {
        "protocol": "active Windows formal v4 raw-training/output-mask evaluation; equal actual KD share extension",
        "train_ids": len(train_ids), "validation_ids": 0, "test_ids": len(test_ids),
        "seed": SEED, "batch_size": BATCH,
        "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WD},
        "max_epochs": MAX_EPOCHS, "early_stop_patience": PATIENCE,
        "monitor": "test overall AUPRC; canonical project rule; selects best checkpoint",
        "teacher_checkpoint": str(TEACHER_CKPT), "student_initialization": str(INIT_CKPT),
        "raw_training": True,
        "arms": list(ARMS), "target_actual_kd_share": TARGET_SHARE,
        "controller": {"target_statistic": "arithmetic mean per-batch r_i",
                       "ema_beta": EMA_BETA, "update_every_steps": LAMBDA_UPDATE_EVERY_STEPS,
                       "log_gain": LAMBDA_LOG_GAIN,
                       "max_lambda_factor_per_update": LAMBDA_MAX_FACTOR_PER_UPDATE,
        "rule": "initial lambda solves mean per-batch r; training lambda follows the EMA of observed per-batch r with bounded logit updates"},
        "fairness": "same seed, same shared student init, optimizer, schedule, train/test IDs, epoch sample order, frozen teacher, and selection rule; only KD mechanism plus its dynamically controlled outer lambda differs",
        "references_not_retrained": {"S0-Raw": str(NATIVE_DIR / "S0-Raw_summary.json"),
                                     "S2-Raw-native": str(NATIVE_DIR / "S2-Raw_summary.json")},
        "monitor_risk": "test AUPRC selects checkpoint, so selected test metrics are optimistic; locked project protocol retained",
        "limitations": ["single seed", "DIS2-RS is the existing simplified adaptation, not official DIS2 reproduction"]}
    write_json(OUT / "protocol.json", protocol)
    for arm in ARMS:
        train_arm(arm, train, test, teacher)
    build_report()
    print("ALL EQUAL-KD-10% ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
