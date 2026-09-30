from __future__ import annotations

import csv
import json
import random
import sys
import time
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
from common import (BATCH, LR, MAX_EPOCHS, PATIENCE, SEED, SO_INIT,
                    make_loaders, metric_rows, overall_ap, write_json)
from mmokd_model import MMOKDPort, SingleOrbitBranch, load_haiti_initialization
from mmokd_losses import mmokd_terms
from models.landslide_cocd import ConvNeXtTinyFPN


WEIGHT_DECAY = 0.0
MICRO_BATCH = 4  # gradient accumulation preserves effective batch 16 on 16-GB cards
ALPHA_DISTILL = 1.0
BETA_SUPERVISION = 1.0
EPS = 1e-8


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def pair_prediction(model, dataset, batch_size=BATCH):
    model.eval()
    pred_a, pred_d, labels, geom_asc, geom_desc = [], [], [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for asc, desc, y, ga, gd, _ in loader:
            out_a = model.student_asc(asc.to(DEV))
            out_d = model.student_desc(desc.to(DEV))
            pred_a.append(torch.softmax(out_a["logits"], dim=1)[:, 1].cpu().numpy())
            pred_d.append(torch.softmax(out_d["logits"], dim=1)[:, 1].cpu().numpy())
            labels.append(y.numpy()); geom_asc.append(ga.numpy()); geom_desc.append(gd.numpy())
    p_a, p_d = np.concatenate(pred_a, axis=0), np.concatenate(pred_d, axis=0)
    y = np.concatenate(labels, axis=0)
    ga, gd = np.concatenate(geom_asc, axis=0), np.concatenate(geom_desc, axis=0)
    arrays = (np.concatenate((p_a, p_d), axis=0), np.concatenate((y, y), axis=0),
              np.concatenate((ga, gd), axis=0), np.concatenate((gd, ga), axis=0))
    return {"combined": metric_rows(arrays),
            "ASC": metric_rows((p_a, y, ga, gd)),
            "DESC": metric_rows((p_d, y, gd, ga))}, overall_ap(arrays)


def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def profile_one_orbit(branch, initialization):
    params = sum(p.numel() for p in branch.parameters())
    result = {"parameters_per_deployable_orbit": int(params),
              "supported_forward_gflops_per_orbit": None,
              "unsupported_operator_count": None,
              "profile_note": "fvcore supported-op trace at 1x5x128x128; unsupported ops omitted"}
    try:
        from fvcore.nn import FlopCountAnalysis

        class LogitsOnly(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__(); self.wrapped = wrapped
            def forward(self, x):
                return self.wrapped(x)["logits"]

        base = ConvNeXtTinyFPN()
        load_haiti_initialization(base, Path(initialization))
        template = SingleOrbitBranch(base)
        template.load_state_dict(branch.state_dict(), strict=True)
        wrapper = LogitsOnly(template.eval()).to(DEV)
        with torch.no_grad():
            analysis = FlopCountAnalysis(wrapper, torch.zeros(1, 5, 128, 128, device=DEV))
            result["supported_forward_gflops_per_orbit"] = float(analysis.total()) / 1e9
            result["unsupported_operator_count"] = len(analysis.unsupported_ops())
    except Exception as exc:  # profiler support varies by installed fvcore build
        result["profile_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sanity_file = OUT / "sanity_32_report.json"
    if not sanity_file.exists():
        raise FileNotFoundError("Required sanity gate is missing; run sanity_32.py first")
    sanity = json.loads(sanity_file.read_text(encoding="utf-8"))
    if not sanity.get("passed", False):
        raise RuntimeError("32-sample sanity gate failed; formal training is blocked")

    seed_all()
    torch.backends.cudnn.benchmark = True
    train, test, split_info = make_loaders()
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0)
    model = MMOKDPort(SO_INIT).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=WEIGHT_DECAY)
    best_score, best_epoch, bad_epochs = float("-inf"), None, 0
    best_state, history = None, []
    full_path = OUT / "best_checkpoint.pt"
    asc_path = OUT / "student_asc_best.pt"
    desc_path = OUT / "student_desc_best.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time(); model.train()
        names = ("total", "seg", "seg_teacher", "seg_student_asc", "seg_student_desc",
                 "feature_kd", "relation_kd", "response_kd", "distill")
        sums = {name: 0.0 for name in names}; batches = 0
        for batch in train_loader:
            asc, desc, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
            optimizer.zero_grad(set_to_none=True)
            batch_terms = {name: 0.0 for name in names}
            for start in range(0, len(asc), MICRO_BATCH):
                stop = min(start + MICRO_BATCH, len(asc))
                n = stop - start
                outputs = model.forward_pair(asc[start:stop], desc[start:stop])
                terms = mmokd_terms(outputs, y[start:stop])
                loss = (BETA_SUPERVISION * terms["seg"] +
                        ALPHA_DISTILL * terms["distill"])
                terms["total"] = loss
                weight = n / len(asc)
                (loss * weight).backward()
                for key in names:
                    batch_terms[key] += float(terms[key].detach()) * weight
            optimizer.step()
            for key in names:
                sums[key] += batch_terms[key]
            batches += 1

        means = {key: value / max(batches, 1) for key, value in sums.items()}
        evaluation, score = pair_prediction(model, test)
        row = {"epoch": epoch, **means,
               "test_overall_auprc": score,
               "test_overall_iou": evaluation["combined"]["Overall"]["IoU"],
               "test_overall_f1": evaluation["combined"]["Overall"]["F1"],
               "test_targetvalid_auprc": evaluation["combined"]["TargetValid"]["AUPRC"],
               "test_bothvalid_auprc": evaluation["combined"]["BothValid"]["AUPRC"],
               "kd_alpha": ALPHA_DISTILL, "seg_beta": BETA_SUPERVISION,
               "epoch_seconds": time.time() - started}
        history.append(row)
        improved = score > best_score
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save(best_state, full_path)
            torch.save({key.removeprefix("student_asc."): value.detach().cpu().clone()
                        for key, value in best_state.items() if key.startswith("student_asc.")}, asc_path)
            torch.save({key.removeprefix("student_desc."): value.detach().cpu().clone()
                        for key, value in best_state.items() if key.startswith("student_desc.")}, desc_path)
        else:
            bad_epochs += 1
        write_csv(OUT / "training_curve.csv", history)
        loss_history = [{k: r[k] for k in ("epoch", "seg", "seg_teacher", "seg_student_asc",
                                           "seg_student_desc", "feature_kd", "relation_kd",
                                           "response_kd", "distill", "total", "kd_alpha", "seg_beta")}
                        for r in history]
        write_csv(OUT / "loss_terms.csv", loss_history)
        payload = {
            "method": "MMOKD-Port", "seed": SEED, "split": split_info,
            "effective_batch_size": BATCH, "micro_batch_size": MICRO_BATCH,
            "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999],
                          "eps": 1e-8, "weight_decay": WEIGHT_DECAY},
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "selection_monitor": "test Overall AUPRC; inherited locked project rule",
            "geometry_use": "post-hoc only; geometry is absent from all model inputs and losses",
            "branches": "online ASC/DESC fusion Teacher plus separate ASC-only and DESC-only Students",
            "loss": {"alpha_distill": ALPHA_DISTILL, "beta_supervision": BETA_SUPERVISION,
                     "distill": "mean(feature MSE, decoder relation MSE, T=4 response KL)",
                     "supervision": "mean of branch CE+soft-Dice losses"},
            "history": history, "best_epoch": best_epoch,
            "best_test_overall_auprc": best_score, "complete": False,
            "monitor_risk": "test AUPRC selects checkpoint under existing protocol",
        }
        write_json(OUT / "progress.json", payload)
        print(f"MMOKD-Port epoch={epoch} seg={means['seg']:.5f} distill={means['distill']:.5f} "
              f"AP={score:.6f} best={best_score:.6f}@{best_epoch} bad={bad_epochs}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("MMOKD-Port produced no selected checkpoint")
    model.load_state_dict(best_state)
    final_metrics, final_score = pair_prediction(model, test)
    region_rows = []
    for orbit, metrics in final_metrics.items():
        for region, values in metrics.items():
            region_rows.append({"Method": "MMOKD-Port", "Orbit": orbit, "Region": region,
                                **values, "BestEpoch": best_epoch})
    write_csv(OUT / "region_metrics.csv", region_rows)
    compute = {
        "method": "MMOKD-Port", "best_epoch": best_epoch,
        "training_parameters": int(sum(p.numel() for p in model.parameters())),
        "student_asc_inference": profile_one_orbit(model.student_asc, SO_INIT),
        "student_desc_inference": profile_one_orbit(model.student_desc, SO_INIT),
        "training_only_branches": ["paired ASC/DESC encoders", "symmetric MFF", "fusion FPN decoder"],
        "inference_only_inputs": "one 5-channel target-orbit stack per branch; no counter image or fusion Teacher",
        "supported_op_flops_caveat": "fvcore forward trace omits unsupported operators; null if unavailable",
    }
    write_json(OUT / "compute_cost.json", compute)
    summary = {
        "method": "MMOKD-Port", "status": "COMPLETE_MECHANISM_FAITHFUL_PORT",
        "protocol": {
            "seed": SEED, "split": split_info, "optimizer": {"name": "Adam", "lr": LR,
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WEIGHT_DECAY},
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "selection_monitor": "test Overall AUPRC (locked project protocol)",
            "geometry": "post-hoc only; no geometry in model or training losses",
            "train_modes": "existing three canonical training modes; 4,110 paired rows/epoch",
        },
        "loss": {"alpha_distill": ALPHA_DISTILL, "beta_supervision": BETA_SUPERVISION,
                 "components": ["low/high encoder MSE", "full decoder pixel-cosine relation MSE",
                                "two-class response KL T=4", "CE+soft Dice on all three branches"]},
        "best_epoch": best_epoch, "best_test_overall_auprc": best_score,
        "metrics": final_metrics["combined"], "ASC_metrics": final_metrics["ASC"],
        "DESC_metrics": final_metrics["DESC"], "history": history,
        "training_parameters": compute["training_parameters"],
        "inference_parameters_per_orbit": compute["student_asc_inference"]["parameters_per_deployable_orbit"],
        "compute_cost": compute,
        "checkpoints": {"full": str(full_path), "student_asc": str(asc_path),
                        "student_desc": str(desc_path)},
        "sanity_gate": str(sanity_file), "source_fidelity": "mechanism-faithful port; paper coefficients and exact MFF/SFG operators unresolved",
        "monitor_risk": "test Overall AUPRC is used for epoch selection under the locked project protocol; results are test-monitored development evidence",
    }
    write_json(OUT / "summary.json", summary)
    payload.update({"complete": True, "summary": str(OUT / "summary.json"),
                    "checkpoint": str(full_path)})
    write_json(OUT / "progress.json", payload)
    print(f"MMOKD-Port COMPLETE best_epoch={best_epoch} AP={final_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
