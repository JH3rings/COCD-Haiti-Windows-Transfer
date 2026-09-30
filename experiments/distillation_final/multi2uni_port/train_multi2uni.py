from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import (  # noqa: E402
    DEV, EPS, MAX_EPOCHS, PATIENCE, SEED, bce, make_loaders, metric_rows,
    overall_ap, seed_everything, write_json,
)
from multi2uni_losses import cad_loss, fad_loss  # noqa: E402
from multi2uni_model import Multi2UniPort  # noqa: E402

OUT = HERE.parent
BATCH = 16
LR = 5e-5
WEIGHT_DECAY = 0.0


def evaluate(model, dataset):
    model.eval()
    pred, labels, gtarget, gcounter = [], [], [], []
    test_loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    with torch.no_grad():
        for asc, desc, y, ga, gd, _ in test_loader:
            za = model.predict_single(asc.to(DEV))[:, 0]
            zd = model.predict_single(desc.to(DEV))[:, 0]
            pred.extend((torch.sigmoid(za).cpu().numpy(), torch.sigmoid(zd).cpu().numpy()))
            labels.extend((y.numpy(), y.numpy()))
            gtarget.extend((ga.numpy(), gd.numpy()))
            gcounter.extend((gd.numpy(), ga.numpy()))
    arrays = tuple(np.concatenate(parts, axis=0) for parts in (pred, labels, gtarget, gcounter))
    return metric_rows(arrays), overall_ap(arrays)


def calibration_terms(model, batch):
    asc, desc, y = (value.to(DEV) for value in batch[:3])
    with torch.no_grad():
        # We only need the initial scalar magnitudes; run a graph-free model pass.
        out = model.forward_pair(asc, desc)
        seg = (bce(out["teacher_logits"], y) +
               0.5 * (bce(out["student_logits_asc"], y) + bce(out["student_logits_desc"], y)))
        fad = 0.5 * (fad_loss(out["teacher_fused"], out["student_asc"]) +
                     fad_loss(out["teacher_fused"], out["student_desc"]))
        cad = cad_loss(out["teacher_asc"][3], out["teacher_desc"][3],
                       out["student_asc"][3], out["student_desc"][3])
    lambda_fad = 0.10 * float(seg.item()) / max(float(fad.item()), EPS)
    lambda_cad = 0.10 * float(seg.item()) / max(float(cad.item()), EPS)
    return lambda_fad, lambda_cad, float(seg.item()), float(fad.item()), float(cad.item())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True
    train, test, split_info = make_loaders()
    loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0)
    model = Multi2UniPort(str(ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt")).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=WEIGHT_DECAY)
    lambda_fad = lambda_cad = None
    initial_calibration = None
    history, best_state = [], None
    best_score, best_epoch, bad_epochs = float("-inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        model.train()
        sums = {k: 0.0 for k in ("seg_teacher", "seg_student", "seg_total", "fad_raw",
                                  "cad_raw", "fad_weighted", "cad_weighted", "total")}
        n_batches = 0
        for batch in loader:
            asc, desc, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
            out = model.forward_pair(asc, desc)
            seg_teacher = bce(out["teacher_logits"], y)
            seg_student = 0.5 * (bce(out["student_logits_asc"], y) +
                                 bce(out["student_logits_desc"], y))
            seg_total = seg_teacher + seg_student
            fad_raw = 0.5 * (fad_loss(out["teacher_fused"], out["student_asc"]) +
                             fad_loss(out["teacher_fused"], out["student_desc"]))
            cad_raw = cad_loss(out["teacher_asc"][3], out["teacher_desc"][3],
                               out["student_asc"][3], out["student_desc"][3])
            if lambda_fad is None:
                lambda_fad = 0.10 * float(seg_total.detach().item()) / max(float(fad_raw.detach().item()), EPS)
                lambda_cad = 0.10 * float(seg_total.detach().item()) / max(float(cad_raw.detach().item()), EPS)
                initial_calibration = {"seg_total": float(seg_total.detach().item()),
                    "fad_raw": float(fad_raw.detach().item()), "cad_raw": float(cad_raw.detach().item()),
                    "lambda_fad": lambda_fad, "lambda_cad": lambda_cad,
                    "target_each_auxiliary_contribution_fraction": 0.10,
                    "calibration_batches": 1, "sweep": False}
            fad_weighted, cad_weighted = lambda_fad * fad_raw, lambda_cad * cad_raw
            loss = seg_total + fad_weighted + cad_weighted
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in (("seg_teacher", seg_teacher), ("seg_student", seg_student),
                               ("seg_total", seg_total), ("fad_raw", fad_raw), ("cad_raw", cad_raw),
                               ("fad_weighted", fad_weighted), ("cad_weighted", cad_weighted), ("total", loss)):
                sums[key] += float(value.detach().item())
            n_batches += 1

        means = {k: v / max(1, n_batches) for k, v in sums.items()}
        metrics, score = evaluate(model, test)
        row = {"epoch": epoch, **means,
               "fad_contribution_fraction_of_seg": means["fad_weighted"] / max(means["seg_total"], EPS),
               "cad_contribution_fraction_of_seg": means["cad_weighted"] / max(means["seg_total"], EPS),
               "test_overall_auprc": score, "test_metrics": metrics,
               "epoch_seconds": time.time() - started}
        history.append(row)
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT / "Multi2Uni-Port_seed42_best.pt")
            torch.save({k.removeprefix("student."): v.detach().cpu().clone()
                        for k, v in best_state.items() if k.startswith("student.")},
                       OUT / "Multi2Uni-Port_deploy_student_seed42_best.pt")
        else:
            bad_epochs += 1

        protocol = {
            "arm": "Multi2Uni-Port", "seed": SEED, "split": split_info,
            "batch_size": BATCH, "optimizer": {"name": "Adam", "lr": LR,
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WEIGHT_DECAY},
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "selection_monitor": "test Overall AUPRC (locked project protocol)",
            "geometry_use": "post-hoc evaluation only; no geometry input/loss/KD",
            "symmetric_views": ["ASC target with DESC auxiliary", "DESC target with ASC auxiliary"],
            "teacher": "two independent ConvNeXt encoders + four bidirectional gated MEF blocks + shared FPN decoder",
            "student": "one ConvNeXtTinyFPN; only student retained for inference",
            "FAD": "one-level orthonormal Haar DWT; detail bands LH/HL/HH; encoder c2/c3; MSE",
            "CAD": "full global c5 cross-view affinity versus target-only self-affinity; L1; no local crop",
            "initial_loss_magnitude_calibration": initial_calibration,
            "published_details_unresolved": ["exact MEF equations/configuration", "FAD levels/wavelet and exact loss form",
                "CAD exact layer/formula", "loss coefficients", "optimizer/schedule"],
            "claim_scope": "mechanism-faithful port, not official numerical reproduction",
        }
        write_json(OUT / "progress.json", {"protocol": protocol, "history": history,
                    "best_epoch": best_epoch, "best_test_overall_auprc": best_score})
        print(f"Multi2Uni-Port epoch={epoch} testAP={score:.6f} "
              f"FAD%={row['fad_contribution_fraction_of_seg']:.3f} "
              f"CAD%={row['cad_contribution_fraction_of_seg']:.3f} best={best_epoch}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Multi2Uni-Port did not produce a best checkpoint")
    model.load_state_dict(best_state)
    metrics, score = evaluate(model, test)
    summary = {"method": "Multi2Uni-Port", "protocol": protocol,
               "best_epoch": best_epoch, "best_test_overall_auprc": best_score,
               "metrics": metrics, "history": history,
               "single_orbit_inference_parameters": sum(p.numel() for p in model.student.parameters()),
               "training_parameters": sum(p.numel() for p in model.parameters()),
               "initial_loss_magnitude_calibration": initial_calibration,
               "student_checkpoint": str(OUT / "Multi2Uni-Port_deploy_student_seed42_best.pt")}
    write_json(OUT / "summary.json", summary)
    print(f"Multi2Uni-Port COMPLETE best_epoch={best_epoch} bestAP={score:.6f}", flush=True)


if __name__ == "__main__":
    main()

