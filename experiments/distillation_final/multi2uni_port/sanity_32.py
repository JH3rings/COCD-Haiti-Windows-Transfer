from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import DEV, EPS, LR, SEED, bce, make_loaders, seed_everything, write_json  # noqa: E402
from multi2uni_losses import cad_loss, fad_loss  # noqa: E402
from multi2uni_model import Multi2UniPort  # noqa: E402
from train_multi2uni import calibration_terms, evaluate  # noqa: E402


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def main():
    seed_everything(SEED)
    train, _test, split_info = make_loaders()
    subset = Subset(train, list(range(32)))
    loader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)
    init = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
    model = Multi2UniPort(str(init)).to(DEV)
    initial_metrics, _ = evaluate(model, subset)

    calibration_batch = next(iter(loader))
    lambda_fad, lambda_cad, seg0, fad0, cad0 = calibration_terms(model, calibration_batch)
    asc, desc, y = (v.to(DEV) for v in calibration_batch[:3])
    model.train()
    out = model.forward_pair(asc, desc)
    fad = 0.5 * (fad_loss(out["teacher_fused"], out["student_asc"]) +
                 fad_loss(out["teacher_fused"], out["student_desc"]))
    cad = cad_loss(out["teacher_asc"][3], out["teacher_desc"][3],
                   out["student_asc"][3], out["student_desc"][3])
    student_params = list(model.student.parameters())
    kd_checks = {"FAD": {"raw": float(fad.detach().item()), "gradient_norm_to_student": grad_norm(fad, student_params)},
                 "CAD": {"raw": float(cad.detach().item()), "gradient_norm_to_student": grad_norm(cad, student_params)}}

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    finite_losses = []
    steps = 500
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        asc, desc, y = (v.to(DEV) for v in batch[:3])
        out = model.forward_pair(asc, desc)
        seg = (bce(out["teacher_logits"], y) +
               0.5 * (bce(out["student_logits_asc"], y) + bce(out["student_logits_desc"], y)))
        fad = 0.5 * (fad_loss(out["teacher_fused"], out["student_asc"]) +
                     fad_loss(out["teacher_fused"], out["student_desc"]))
        cad = cad_loss(out["teacher_asc"][3], out["teacher_desc"][3],
                       out["student_asc"][3], out["student_desc"][3])
        loss = seg + lambda_fad * fad + lambda_cad * cad
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        finite_losses.append(float(loss.detach().item()))

    final_metrics, _ = evaluate(model, subset)
    initial_iou, initial_f1 = initial_metrics["Overall"]["IoU"], initial_metrics["Overall"]["F1"]
    final_iou, final_f1 = final_metrics["Overall"]["IoU"], final_metrics["Overall"]["F1"]
    report = {"method": "Multi2Uni-Port", "sanity_kind": "32 training rows; no formal checkpoint selection",
        "seed": SEED, "training_rows": 32, "locked_split": split_info, "steps": steps,
        "initial_loss_magnitude_calibration": {"seg": seg0, "fad": fad0, "cad": cad0,
            "lambda_fad": lambda_fad, "lambda_cad": lambda_cad,
            "target_each_auxiliary_contribution_fraction": 0.10, "calibration_batches": 1},
        "loss_first": finite_losses[0], "loss_last_10_mean": sum(finite_losses[-10:]) / 10,
        "all_training_losses_finite": bool(torch.isfinite(torch.tensor(finite_losses)).all()),
        "kd_terms": kd_checks, "single_orbit_student_output": True,
        "initial_deploy_metrics": {"IoU": initial_iou, "F1": initial_f1},
        "final_deploy_metrics": {"IoU": final_iou, "F1": final_f1},
        "passed": bool(torch.isfinite(torch.tensor(finite_losses)).all() and
            finite_losses[-1] < finite_losses[0] and final_iou > initial_iou and final_f1 > initial_f1 and
            all(v["raw"] > 0 and v["gradient_norm_to_student"] > 0 for v in kd_checks.values()))}
    write_json(HERE.parent / "sanity_32_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

