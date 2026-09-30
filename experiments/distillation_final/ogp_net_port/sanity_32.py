from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import DEV, SEED, make_loaders, seed_everything, write_json  # noqa: E402
from ogp_losses import dist_response, segmentation_loss, semi_hard_pixel_contrastive  # noqa: E402
from ogp_model import OGPNetPort, load_backbone_state  # noqa: E402
from train_ogp import evaluate, horizontal_flip_pair, load_single_teacher, teacher_features, train_loader  # noqa: E402


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    values = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def main():
    seed_everything(SEED)
    train, _test, split_info = make_loaders()
    subset = Subset(train, list(range(32)))
    teacher, teacher_path = load_single_teacher()
    model = OGPNetPort().to(DEV)
    init = teacher_path
    load_backbone_state(model.branch_asc, str(init))
    load_backbone_state(model.branch_desc, str(init))
    _, initial_metrics, _ = evaluate(model, subset)
    model.train()
    batch = next(iter(train_loader(subset)))
    asc, desc, y = (item.to(DEV) for item in batch[:3])
    out = model.forward_pair(asc, desc)
    with torch.no_grad():
        fa_t, da_t, za_t = teacher_features(teacher, asc)
        fd_t, dd_t, zd_t = teacher_features(teacher, desc)
    dur_a = dist_response(out["z_asc"], za_t[:, 0:1])
    dur_d = dist_response(out["z_desc"], zd_t[:, 0:1])
    dur = 0.5 * sum(dur_a) + 0.5 * sum(dur_d)
    feature = 0.5 * (F.mse_loss(out["features_asc"][2], fa_t[2]) +
                     F.mse_loss(out["features_asc"][3], fa_t[3]) +
                     F.mse_loss(out["d_asc"], da_t) +
                     F.mse_loss(out["features_desc"][2], fd_t[2]) +
                     F.mse_loss(out["features_desc"][3], fd_t[3]) +
                     F.mse_loss(out["d_desc"], dd_t))
    contrast = semi_hard_pixel_contrastive(out["contrast_features"], y, 0.2)
    params = list(model.parameters())
    kd_checks = {name: {"finite": bool(torch.isfinite(value)), "value": float(value.detach().item()),
                         "gradient_norm": grad_norm(value, params)}
                 for name, value in (("DIST", dur), ("encoder_decoder_MSE", feature),
                                     ("pixel_contrastive", contrast))}

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0)
    train_losses = []
    steps = 100
    failure_step = None
    failure_reason = None
    max_abs_logit = 0.0
    it = iter(train_loader(subset))
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader(subset))
            batch = next(it)
        asc, desc, y = (item.to(DEV) for item in batch[:3])
        asc, desc, y = horizontal_flip_pair(asc, desc, y)
        out = model.forward_pair(asc, desc)
        with torch.no_grad():
            fa_t, da_t, za_t = teacher_features(teacher, asc)
            fd_t, dd_t, zd_t = teacher_features(teacher, desc)
        seg = segmentation_loss(out["z_fused"], y)
        yd = F.interpolate(y[:, None].float(), size=out["aux_asc"].shape[-2:], mode="nearest")[:, 0].long()
        aux = 0.5 * (segmentation_loss(out["aux_asc"], yd) + segmentation_loss(out["aux_desc"], yd))
        da = dist_response(out["z_asc"], za_t[:, 0:1]); dd = dist_response(out["z_desc"], zd_t[:, 0:1])
        dur = 0.5 * (sum(da) + sum(dd))
        feature = 0.5 * (F.mse_loss(out["features_asc"][2], fa_t[2]) +
                         F.mse_loss(out["features_asc"][3], fa_t[3]) + F.mse_loss(out["d_asc"], da_t) +
                         F.mse_loss(out["features_desc"][2], fd_t[2]) +
                         F.mse_loss(out["features_desc"][3], fd_t[3]) + F.mse_loss(out["d_desc"], dd_t))
        contrast = semi_hard_pixel_contrastive(out["contrast_features"], y, 0.2)
        loss = seg + aux + dur + feature + contrast
        if not torch.isfinite(loss):
            failure_step = step + 1
            failure_reason = "nonfinite_loss"
            break
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grads_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                           for p in model.parameters())
        if not grads_finite:
            failure_step = step + 1
            failure_reason = "nonfinite_gradient"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().item()))
        # Detect forward overflow immediately, including the final update.
        with torch.no_grad():
            test_logits = model.forward_single(asc, "asc")
            finite_logits = bool(torch.isfinite(test_logits).all())
            if finite_logits:
                max_abs_logit = max(max_abs_logit, float(test_logits.abs().max().item()))
        if not finite_logits:
            failure_step = step + 1
            failure_reason = "nonfinite_deploy_logits_after_update"
            break

    has_nonfinite_params = any(not torch.isfinite(p).all() for p in model.parameters())
    before = None
    if not has_nonfinite_params and failure_reason is None:
        _, before, _ = evaluate(model, subset)
    # Use the branch-only inference interface for the required single-orbit deploy check.
    before_iou_f1 = ({k: before["Overall"][k] for k in ("IoU", "F1")}
                     if before is not None else {"IoU": None, "F1": None})
    # The overfit score is measured after updates; the initial branch-only score is captured from a fresh init below.
    final_f = bool(torch.isfinite(torch.tensor(train_losses)).all().item() and not has_nonfinite_params)
    first_loss = train_losses[0] if train_losses else None
    last_mean = sum(train_losses[-10:]) / min(10, len(train_losses)) if train_losses else None
    report = {"method": "OGP-Net-Port", "sanity_kind": "32 training rows; no formal checkpoint selection",
        "seed": SEED, "training_rows": 32, "locked_split": split_info,
        "stage1_frozen_teacher": str(teacher_path), "steps": steps,
        "optimizer": {"name": "SGD", "lr": 5e-3, "paper_initial_lr": 5e-3,
                      "momentum": 0.9, "weight_decay": 0.0,
                      "global_grad_norm_clip": 1.0,
                      "adaptation": "fixed grad norm clip required after unbounded Eq.12 overflowed"},
        "failure_step": failure_step, "nonfinite_parameters": bool(has_nonfinite_params),
        "failure_reason": failure_reason, "max_abs_deploy_logit_during_updates": max_abs_logit,
        "loss_first": first_loss, "loss_last_10_mean": last_mean,
        "all_training_losses_finite": bool(final_f), "single_orbit_output": True,
        "kd_terms": kd_checks,
        "initial_deploy_metrics": {k: initial_metrics["Overall"][k] for k in ("IoU", "F1")},
        "final_deploy_metrics": before_iou_f1,
        "passed": bool(final_f and train_losses and train_losses[-1] < train_losses[0] and
                        before_iou_f1["IoU"] is not None and before_iou_f1["IoU"] > initial_metrics["Overall"]["IoU"] and
                        before_iou_f1["F1"] > initial_metrics["Overall"]["F1"] and
                        all(v["finite"] and v["gradient_norm"] > 0 for v in kd_checks.values()))}
    write_json(HERE.parent / "sanity_32_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
