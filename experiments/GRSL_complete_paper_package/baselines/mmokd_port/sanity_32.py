from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
OUT = HERE.parent
ROOT = HERE.parents[4]
COCD = ROOT / "cocd"
if str(COCD) not in sys.path:
    sys.path.insert(0, str(COCD))
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
sys.path.insert(0, str(OUT))

from paths import DEV
from windows_main.data import HaitiPairs, split
from common import SO_INIT, metric_rows
from mmokd_model import MMOKDPort
from mmokd_losses import mmokd_terms


SEED = 42
SANITY_EPOCHS = 300
SANITY_BATCH = 8


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def evaluate_train(model, dataset):
    model.eval()
    pa, pd, labels, geom_asc, geom_desc = [], [], [], [], []
    losses = []
    with torch.no_grad():
        for asc, desc, y, g_a, g_d, _ in DataLoader(dataset, batch_size=SANITY_BATCH,
                                                    shuffle=False, num_workers=0):
            out = model.forward_pair(asc.to(DEV), desc.to(DEV))
            pa.append(torch.softmax(out["student_asc"]["logits"], 1)[:, 1].cpu().numpy())
            pd.append(torch.softmax(out["student_desc"]["logits"], 1)[:, 1].cpu().numpy())
            labels.append(y.numpy()); geom_asc.append(g_a.numpy()); geom_desc.append(g_d.numpy())
            losses.append(float(mmokd_terms(out, y.to(DEV))["seg"].item()))
    p_a, p_d = np.concatenate(pa), np.concatenate(pd)
    y = np.concatenate(labels, axis=0)
    g_a, g_d = np.concatenate(geom_asc, axis=0), np.concatenate(geom_desc, axis=0)
    combined = metric_rows((np.concatenate((p_a, p_d)), np.concatenate((y, y)),
                            np.concatenate((g_a, g_d)), np.concatenate((g_d, g_a))))
    asc_metrics = metric_rows((p_a, y, g_a, g_d))
    desc_metrics = metric_rows((p_d, y, g_d, g_a))
    return float(np.mean(losses)), {"combined": combined, "ASC": asc_metrics, "DESC": desc_metrics}


def main():
    seed_all()
    train_ids, val_ids, _ = split()
    if len(val_ids) != 0:
        raise RuntimeError(f"Expected the locked no-validation split, got {len(val_ids)} val IDs")
    all_train = HaitiPairs(train_ids, train_modes=True)
    positive_ids = [sample_id for sample_id, row in zip(train_ids, all_train.rows)
                    if float(row[1].sum()) > 0]
    rng = random.Random(SEED)
    rng.shuffle(positive_ids)
    if len(positive_ids) < 32:
        raise RuntimeError(f"Only {len(positive_ids)} positive training locations available")
    sanity_ids = positive_ids[:32]
    sanity = HaitiPairs(sanity_ids, train_modes=False, stats=all_train.stats)

    model = MMOKDPort(SO_INIT).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    initial_seg, initial_metrics = evaluate_train(model, sanity)
    trace = []
    gradient_check = None
    train_loader = DataLoader(sanity, batch_size=SANITY_BATCH, shuffle=True, num_workers=0)

    for epoch in range(1, SANITY_EPOCHS + 1):
        model.train(); epoch_rows = []
        for batch in train_loader:
            asc, desc, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_pair(asc, desc)
            terms = mmokd_terms(outputs, y)
            if gradient_check is None:
                checks = {}
                groups = {
                    "feature_kd": tuple(model.student_asc.sfg.parameters()),
                    "relation_kd": tuple(model.student_asc.decoder.parameters()),
                    "response_kd": tuple(model.student_asc.decoder.classifier.parameters()),
                }
                for name, params in groups.items():
                    grads = torch.autograd.grad(terms[name], params, retain_graph=True,
                                                allow_unused=True)
                    norm_sq = sum(float(g.detach().pow(2).sum()) for g in grads if g is not None)
                    checks[name] = {"raw_loss": float(terms[name].detach()),
                                    "student_gradient_norm": float(norm_sq ** 0.5),
                                    "nonzero": bool(norm_sq > 0)}
                grad_t = torch.autograd.grad(terms["seg_teacher"],
                                             tuple(model.teacher.mff.parameters()),
                                             retain_graph=True, allow_unused=True)
                teacher_norm_sq = sum(float(g.detach().pow(2).sum()) for g in grad_t if g is not None)
                checks["teacher_seg_to_mff_gradient_norm"] = float(teacher_norm_sq ** 0.5)
                gradient_check = checks
            terms["total"].backward()
            optimizer.step()
            epoch_rows.append({k: float(v.detach()) for k, v in terms.items()})
        trace.append({"epoch": epoch, **{k: float(np.mean([r[k] for r in epoch_rows]))
                                        for k in epoch_rows[0]}})
        if epoch in (1, SANITY_EPOCHS):
            seg, metrics = evaluate_train(model, sanity)
            trace[-1]["train_seg_eval"] = seg
            for branch in ("combined", "ASC", "DESC"):
                trace[-1][f"train_iou_{branch.lower()}"] = metrics[branch]["Overall"]["IoU"]
                trace[-1][f"train_f1_{branch.lower()}"] = metrics[branch]["Overall"]["F1"]

    final_seg, final_metrics = evaluate_train(model, sanity)
    model.eval()
    with torch.no_grad():
        probe = next(iter(DataLoader(sanity, batch_size=2, shuffle=False, num_workers=0)))
        asc_logits = model.predict_asc(probe[0].to(DEV))
        desc_logits = model.predict_desc(probe[1].to(DEV))
    deployment = {
        "asc_branch_alone_output_shape": list(asc_logits.shape),
        "desc_branch_alone_output_shape": list(desc_logits.shape),
        "asc_branch_alone_finite": bool(torch.isfinite(asc_logits).all()),
        "desc_branch_alone_finite": bool(torch.isfinite(desc_logits).all()),
    }
    checks_ok = (all(v.get("nonzero", False) and v.get("raw_loss", 0.0) > 0
                     for v in gradient_check.values() if isinstance(v, dict)) and
                 gradient_check.get("teacher_seg_to_mff_gradient_norm", 0.0) > 0)
    passed = (np.isfinite(final_seg) and final_seg < initial_seg and
              all(final_metrics[branch]["Overall"]["IoU"] > initial_metrics[branch]["Overall"]["IoU"]
                  and final_metrics[branch]["Overall"]["F1"] > initial_metrics[branch]["Overall"]["F1"]
                  for branch in ("ASC", "DESC")) and
              checks_ok and deployment["asc_branch_alone_finite"] and
              deployment["desc_branch_alone_finite"])
    report = {
        "method": "MMOKD-Port", "sanity_gate": "32 positive training locations, paired ASC/DESC; no test/validation used",
        "seed": SEED, "location_count": len(sanity_ids), "training_locations": sanity_ids,
        "epochs": SANITY_EPOCHS, "batch_size": SANITY_BATCH,
        "initial_seg_loss": initial_seg, "final_seg_loss": final_seg,
        "initial_train_metrics": initial_metrics, "final_train_metrics": final_metrics,
        "kd_gradient_checks": gradient_check, "deployment_single_branch_probe": deployment,
        "history": trace, "passed": bool(passed),
        "failure_reasons": [],
    }
    if not (np.isfinite(final_seg) and final_seg < initial_seg): report["failure_reasons"].append("segmentation loss did not fall")
    for branch in ("ASC", "DESC"):
        if final_metrics[branch]["Overall"]["IoU"] <= initial_metrics[branch]["Overall"]["IoU"]:
            report["failure_reasons"].append(f"{branch} target-only training IoU did not rise")
        if final_metrics[branch]["Overall"]["F1"] <= initial_metrics[branch]["Overall"]["F1"]:
            report["failure_reasons"].append(f"{branch} target-only training F1 did not rise")
    if not checks_ok: report["failure_reasons"].append("one or more KD terms or fusion-Teacher supervision did not produce a nonzero gradient")
    if not (deployment["asc_branch_alone_finite"] and deployment["desc_branch_alone_finite"]): report["failure_reasons"].append("single-orbit branch inference failed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity_32_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "sanity_32_curve.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader(); writer.writerows(trace)
    print(json.dumps({"passed": report["passed"], "initial_seg": initial_seg,
        "final_seg": final_seg, "initial_iou": initial_metrics["combined"]["Overall"]["IoU"],
        "initial_asc_iou": initial_metrics["ASC"]["Overall"]["IoU"],
        "final_asc_iou": final_metrics["ASC"]["Overall"]["IoU"],
        "initial_asc_f1": initial_metrics["ASC"]["Overall"]["F1"],
        "final_asc_f1": final_metrics["ASC"]["Overall"]["F1"],
        "initial_desc_iou": initial_metrics["DESC"]["Overall"]["IoU"],
        "final_desc_iou": final_metrics["DESC"]["Overall"]["IoU"],
        "initial_desc_f1": initial_metrics["DESC"]["Overall"]["F1"],
        "final_desc_f1": final_metrics["DESC"]["Overall"]["F1"],
        "failure_reasons": report["failure_reasons"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
