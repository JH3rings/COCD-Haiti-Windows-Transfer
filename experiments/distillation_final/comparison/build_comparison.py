from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments/windows_main"
FINAL = ROOT / "experiments/distillation_final"
RAW = EXP / "raw_cver_v1_seed42"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_metric(summary, old=False):
    metrics = summary["best_metrics"] if old else summary["metrics"]
    out = {}
    for region in ("Overall", "BothValid", "TargetValid", "G10", "G01"):
        row = metrics[region]
        for key in ("AUPRC", "IoU", "F1"):
            out[f"{region}_{key}"] = row[key]
    return out


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    s0_summary = read_json(RAW / "S0-Raw_summary.json")
    s2_summary = read_json(RAW / "S2-Raw_summary.json")
    s0_progress = read_json(RAW / "S0-Raw_progress.json")
    s2_progress = read_json(RAW / "S2-Raw_progress.json")
    hinton = read_json(FINAL / "hinton_kd/summary.json")
    dis2 = read_json(FINAL / "dis2_port/summary.json")
    s2stage = read_json(FINAL / "ours_s2_2stage/summary.json")
    hinton_history = hinton["history"]
    dis2_history = dis2["history"]
    s2stage_history = s2stage["history"]
    profile = read_json(OUT / "flop_profile.json")

    params_student = profile["params"]["Student"]
    params_teacher = profile["params"]["Teacher"]
    params_dis2_full = profile["params"]["DIS2_full"]
    params_dis2_paths = profile["params"]["DIS2_deployment_paths_both_orbits"]
    student_infer_gflops = profile["Student_S0_Hinton_S2_inference"]["supported_forward_gflops"]
    teacher_pair_gflops = profile["CG0_Teacher_two_orbit_pair_forward"]["supported_forward_gflops"]
    s2_kd_pair_gflops = profile["S2_SearchKD_training_forward_pair"]["supported_forward_gflops"]
    dis2_infer_gflops = profile["DIS2_ASC_target_inference"]["supported_forward_gflops"]
    dis2_train_direction_gflops = profile["DIS2_training_forward_one_direction_epoch6"]["supported_forward_gflops"]

    metric_sources = [
        ("S0-Raw", "COMPLETE", flatten_metric(s0_summary, old=True), s0_summary["best_epoch"],
         params_student, student_infer_gflops, "StudentTargetSearchV2, target orbit only"),
        ("Hinton-KD-Port", "COMPLETE", flatten_metric(hinton), hinton["best_epoch"],
         params_student, student_infer_gflops, "S0 deploy path, target orbit only"),
        ("DIS2-Port", "COMPLETE", flatten_metric(dis2), dis2["best_epoch"],
         params_dis2_paths, dis2_infer_gflops, "target orbit in one of two 5-channel slots; other slot zero"),
        ("OGP-Net-Port", "PAUSED_SOURCE_PARAMETER_UNRESOLVED", {}, None, None, None,
         "not trained; numeric contrastive temperature and BN-gamma channel-exchange cutoff unresolved"),
        ("Multi2Uni-Port", "PAUSED_SOURCE_DETAILS_UNRESOLVED", {}, None, None, None,
         "not trained; full-text/supplement/code parameters unavailable"),
        ("S2-Raw-native", "HISTORICAL_REFERENCE_GEOMETRY_WEIGHTED_KD", flatten_metric(s2_summary, old=True),
         s2_summary["best_epoch"], params_student, student_infer_gflops,
         "historical S2 uses CG10 and geometry-derived G10/G01 KD query weighting"),
        ("S2-2Stage", "COMPLETE", flatten_metric(s2stage), s2stage["best_epoch"],
         params_student, student_infer_gflops, "current uniform-query Search-KD, target orbit only"),
    ]
    table = []
    for name, status, values, best_epoch, params, flops, deployment in metric_sources:
        row = {"Method": name, "Status": status, "BestEpoch": best_epoch,
               "InferenceParams": params, "InferenceSupportedGFLOPsPerOrbit": flops,
               "DeploymentInput": deployment}
        row.update(values)
        table.append(row)
    table_fields = ["Method", "Status", "Overall_AUPRC", "Overall_IoU", "Overall_F1",
                    "BothValid_AUPRC", "BothValid_IoU", "BothValid_F1",
                    "TargetValid_AUPRC", "TargetValid_IoU", "TargetValid_F1",
                    "G10_AUPRC", "G10_IoU", "G10_F1", "G01_AUPRC", "G01_IoU", "G01_F1",
                    "BestEpoch", "InferenceParams", "InferenceSupportedGFLOPsPerOrbit", "DeploymentInput"]
    write_csv(OUT / "final_table.csv", table, table_fields)

    curves = []
    for arm, progress in (("S0-Raw", s0_progress), ("S2-Raw-native", s2_progress)):
        for h in progress["history"]:
            me = h.get("mechanism", {})
            curves.append({"Method": arm, "epoch": h["epoch"], "seg_raw_or_train_total": h.get("train_loss"),
                           "policy_raw": me.get("policy_js"), "evidence_raw": me.get("evidence_cos"),
                           "weighted_policy": None, "weighted_evidence": None,
                           "kd_scalar_share": me.get("kd_contribution", 0.0),
                           "total_loss": h.get("train_loss"), "test_overall_auprc": h.get("monitor_test_auprc_raw_overall"),
                           "seg_to_shared_backbone": None, "policy_to_shared_backbone": None,
                           "evidence_to_shared_backbone": None})
    for h in hinton_history:
        curves.append({"Method": "Hinton-KD-Port", "epoch": h["epoch"],
                       "seg_raw_or_train_total": h["seg_raw"], "policy_raw": None, "evidence_raw": h["soft_kd_raw"],
                       "weighted_policy": None, "weighted_evidence": h["kd_weighted"],
                       "kd_scalar_share": h["kd_scalar_share"], "total_loss": h["train_total"],
                       "test_overall_auprc": h["test_overall_auprc"],
                       "seg_to_shared_backbone": None, "policy_to_shared_backbone": None,
                       "evidence_to_shared_backbone": None})
    for h in dis2_history:
        curves.append({"Method": "DIS2-Port", "epoch": h["epoch"],
                       "seg_raw_or_train_total": h["segmentation"], "policy_raw": None, "evidence_raw": h["kl_raw"],
                       "weighted_policy": None, "weighted_evidence": h["kl_weighted"],
                       "kd_scalar_share": h["aux_scalar_share"], "total_loss": h["total"],
                       "test_overall_auprc": h["test_overall_auprc"],
                       "seg_to_shared_backbone": None, "policy_to_shared_backbone": None,
                       "evidence_to_shared_backbone": None})
    for h in s2stage_history:
        gn = h.get("gradient_norms_first_batch") or {}
        curves.append({"Method": "S2-2Stage", "epoch": h["epoch"],
                       "seg_raw_or_train_total": h["seg"], "policy_raw": h["policy"],
                       "evidence_raw": h["evidence"], "weighted_policy": h["weighted_policy"],
                       "weighted_evidence": h["weighted_evidence"], "kd_scalar_share": h["kd_scalar_share"],
                       "total_loss": h["total"], "test_overall_auprc": h["test_overall_auprc"],
                       "seg_to_shared_backbone": gn.get("seg_to_shared_backbone"),
                       "policy_to_shared_backbone": gn.get("policy_to_shared_backbone"),
                       "evidence_to_shared_backbone": gn.get("evidence_to_shared_backbone")})
    write_csv(OUT / "loss_curves.csv", curves)
    kd_rows = [{"Method": r["Method"], "epoch": r["epoch"], "kd_scalar_share": r["kd_scalar_share"],
                "test_overall_auprc": r["test_overall_auprc"]} for r in curves]
    write_csv(OUT / "kd_contribution_curves.csv", kd_rows)

    train_rows = 4110
    n_steps = math.ceil(train_rows / 16)
    s0_pair = 2 * student_infer_gflops
    hinton_pair = s0_pair + teacher_pair_gflops
    dis2_pair = 2 * dis2_train_direction_gflops
    s2_pair = s2_kd_pair_gflops
    epochs = {"S0-Raw": 19, "Hinton-KD-Port": len(hinton_history), "DIS2-Port": len(dis2_history),
              "S2-Raw-native": 44, "S2-2Stage": len(s2stage_history)}
    train_pair = {"S0-Raw": s0_pair, "Hinton-KD-Port": hinton_pair,
                  "DIS2-Port": dis2_pair, "S2-Raw-native": s2_pair, "S2-2Stage": s2_pair}
    trainable = {"S0-Raw": params_student, "Hinton-KD-Port": params_student,
                 "DIS2-Port": params_dis2_full, "S2-Raw-native": params_student,
                 "S2-2Stage": params_student}
    frozen_teacher = {"Hinton-KD-Port": params_teacher, "S2-Raw-native": params_teacher,
                      "S2-2Stage": params_teacher}
    compute_rows = []
    for method in ("S0-Raw", "Hinton-KD-Port", "DIS2-Port", "S2-Raw-native", "S2-2Stage"):
        ep = epochs[method]
        total_gf = train_pair[method] * train_rows * ep
        if method == "S2-2Stage":
            total_gf = train_rows * (4 * s0_pair + (ep - 4) * s2_pair)
        compute_rows.append({"Method": method, "TrainRowsPerEpoch": train_rows,
            "StepsPerEpoch_BS16": n_steps, "ObservedEpochs": ep,
            "TrainableParams": trainable[method], "FrozenTeacherParams": frozen_teacher.get(method, 0),
            "InferenceParams": params_dis2_paths if method == "DIS2-Port" else params_student,
            "InferenceSupportedGFLOPsPerOrbit": dis2_infer_gflops if method == "DIS2-Port" else student_infer_gflops,
            "TrainSupportedForwardGFLOPsPerPairedRow": train_pair[method],
            "TrainSupportedForwardTFLOPs_ObservedEpochs": total_gf * 1e-3,
            "TrainingProtocol": "seed42 Adam lr=5e-5 batch16; max100 patience2; test Overall AP monitor"})
    compute_rows += [
        {"Method": "OGP-Net-Port", "TrainingProtocol": "paused; numeric primary-source parameters unresolved"},
        {"Method": "Multi2Uni-Port", "TrainingProtocol": "paused; key architecture/loss parameters unavailable"},
    ]
    write_csv(OUT / "compute_cost.csv", compute_rows)

    def fmt(v, digits=5):
        return "—" if v is None else f"{v:.{digits}f}"
    md_rows = []
    md_cols = ["Method", "Status", "Overall_AUPRC", "Overall_IoU", "Overall_F1",
               "BothValid_AUPRC", "BothValid_IoU", "BothValid_F1",
               "TargetValid_AUPRC", "TargetValid_IoU", "TargetValid_F1", "BestEpoch"]
    for r in table:
        cells=[]
        for col in md_cols:
            value=r.get(col)
            if isinstance(value, int):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append(fmt(value))
            else:
                cells.append("—" if value is None else str(value))
        md_rows.append("| " + " | ".join(cells) + " |")

    hinton_best = next(x for x in hinton_history if x["epoch"] == hinton["best_epoch"])
    dis2_best = next(x for x in dis2_history if x["epoch"] == dis2["best_epoch"])
    s2_best = next(x for x in s2stage_history if x["epoch"] == s2stage["best_epoch"])
    native_best = next(x for x in s2_progress["history"] if x["epoch"] == s2_progress["best_epoch"])
    s0_ap = s0_summary["best_metrics"]["Overall"]["AUPRC"]
    stage_ap = s2stage["metrics"]["Overall"]["AUPRC"]
    native_ap = s2_summary["best_metrics"]["Overall"]["AUPRC"]
    stage_gain = stage_ap - s0_ap
    native_gain = native_ap - s0_ap
    stop_epochs = {"S0-Raw": s0_progress["epoch"], "S2-Raw-native": s2_progress["epoch"],
                   "Hinton-KD-Port": len(hinton_history), "DIS2-Port": len(dis2_history),
                   "S2-2Stage": len(s2stage_history)}
    report = f"""# Final knowledge-distillation benchmark

## Locked protocol and evidence boundary

- Fixed project split: **1,370 train / 0 validation / 343 test locations**, 4,110 train rows after the project's three training modes. No split was redrawn.
- Seed 42, batch 16, Adam lr `5e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0`, max 100 epochs, patience 2. Every completed new arm selects best checkpoint by the inherited **test Overall AUPRC** monitor; this is test-monitored development evidence, not an independent unbiased holdout estimate.
- Geometry codes are post-hoc only in Hinton-KD-Port, DIS2-Port, and S2-2Stage. BothValid means target=0 and counter=0; TargetValid means target=0; G10/G01 are diagnostic regions. No geometry enters these three methods' training loss, model input, search, or KD.
- All primary results are single-seed first-pass. OGP-Net-Port and Multi2Uni-Port are **paused**, not scored, because critical method-defining values could not be resolved from accessible primary sources.

## Results

| Method | Status | Overall AP | IoU | F1 | BothValid AP | BothValid IoU | BothValid F1 | TargetValid AP | TargetValid IoU | TargetValid F1 | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(md_rows)}

Full precision, G10/G01 metrics, best epochs, parameter counts, and supported-operator FLOPs are in `final_table.csv` and `compute_cost.csv`.

## Main findings

1. Among completed methods, S2-2Stage has the highest test-monitored Overall AP (**{stage_ap:.6f}**), followed by historical S2-Raw-native (**{native_ap:.6f}**), Hinton-KD-Port (**{hinton['metrics']['Overall']['AUPRC']:.6f}**), S0-Raw (**{s0_ap:.6f}**), and DIS2-Port (**{dis2['metrics']['Overall']['AUPRC']:.6f}**). The ranking is conditional on the one seed and test-based checkpoint selection; OGP/Multi2Uni are unknown.
2. In the completed run, S2-2Stage exceeds historical S2-Raw-native by **{stage_ap-native_ap:+.6f} AP** and S0-Raw by **{stage_gain:+.6f} AP**. This is not evidence of a statistically stable difference.
3. S2-2Stage best epoch {s2stage['best_epoch']} has weighted KD scalar share **{s2_best['kd_scalar_share']*100:.2f}%** (policy {s2_best['weighted_policy']:.6g}; evidence {s2_best['weighted_evidence']:.6g}); the configured initial target was 20%, while epoch 6 actually measured {s2stage_history[5]['kd_scalar_share']*100:.2f}% after epoch-level losses changed. At best epoch 44, the historical S2-Raw-native KD scalar share was **{native_best['mechanism']['kd_contribution']*100:.2f}%**. The two-stage schedule therefore avoided the native run's high KD scalar share while retaining its selected AP in this run.
4. S2-2Stage remains a single-orbit Student at inference. Its model has {params_student/1e6:.2f}M inference parameters; DIS2's two target-orbit deploy paths together have {params_dis2_paths/1e6:.2f}M. Thus Search-KD is not smaller by parameter count than the DIS2 deploy paths, although its supported-operator fvcore estimate is similar ({student_infer_gflops:.3f} vs {dis2_infer_gflops:.3f} GFLOPs per orbit). The FLOP counts omit listed unsupported ops and are partial estimates.

## Loss and method-faithfulness notes

- **Hinton-KD-Port:** `L_soft + 0.5 L_seg`, two-class logits `[-z,z]`, T=2, per-pixel mean `T² KL`. Best epoch {hinton['best_epoch']}, KD scalar share {hinton_best['kd_scalar_share']*100:.2f}%.
- **DIS2-Port:** author DLKD_ver4 distinct/supplementary branches, hierarchical fusion, classwise decoder, normalized multi-level/penultimate feature KD, T=2 logit KD, orthogonality, full/missing/unimodal and multi-scale segmentation; symmetric ASC/DESC target directions; KL weight 0 for epochs 1–5, 5 from epoch 6. Best epoch {dis2['best_epoch']}, auxiliary share {dis2_best['aux_scalar_share']*100:.2f}%. It is a Haiti mechanism port, not numerical reproduction of the original Potsdam result. The port fixes only an upstream classwise map append duplication in three lines; see `unresolved_hyperparameters.md`.
- **S2-Raw-native:** retained historical canonical checkpoint and scores, but code inspection shows its old KD query weight uses geometry-derived G10/G01 and its Teacher is CG10 (geometry-TGD weighted training). It is shown as historical context, not compliant with the current geometry-post-hoc-only rule.
- **S2-2Stage:** uses geometry-free CG0 as frozen Teacher; epochs 1–5 segmentation only; epoch-level loss normalization after epoch 5; equal policy/evidence weight and one cosine decay from 1 to 0.25. No additional module or loss was introduced. The KD inputs stop-gradient before the shared backbone by the canonical Search-KD definition, so best-epoch policy/evidence-to-backbone gradient norms are 0; segmentation-to-backbone norm was {s2_best['gradient_norms_first_batch']['seg_to_shared_backbone']:.6g} on the recorded first batch.
- **OGP-Net-Port paused:** accessible AAAI paper identifies DMC/DUR, DIST, gated unit, and semi-hard pixel contrastive construction, but does not resolve a numeric contrastive temperature or near-zero BN-gamma exchange cutoff; no author implementation was found. The cited PCD temperature belongs to a different objective and was not substituted.
- **Multi2Uni-Port paused:** publisher preview confirms MEF + DWT-FAD + CAD, but the accessible sources do not specify FAD levels/wavelet, CAD level, temperatures/weights, teacher order, or MEF configuration. Retired CO-FAD/CO-CAD settings were not reused.

## Compute accounting

`flop_profile.json` and `profile_cost.py` record a batch-one JIT trace at 128×128. The table reports only fvcore-supported forward operations. It excludes unsupported elementwise, sampling, softmax/attention, padding, and backward work; training totals multiply the supported forward trace by the exact observed train rows and epochs and are therefore lower-bound forward-only accounting, not total training FLOPs. DIS2's method-native architecture prevents a shared ConvNeXt initialization; it uses seed-42 random init as a documented architecture exception. All ConvNeXt Student arms load the same SO initialization.

## References

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- Kieu et al., [DIS2 paper](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html), [author source](https://github.com/nhikieu/DIS2).
- Xu et al., [OGP-Net, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32743).
- Sun et al., [Multi-to-Uni Modality Knowledge Distillation, Information Fusion 2026](https://doi.org/10.1016/j.inffus.2026.104696).

## Cleanup

`distillation_deletion_manifest.txt` records the obsolete KD experiment trees and dedicated runners deleted after results were written. Canonical S0-Raw/S2-Raw evidence, current Teacher checkpoints, fixed split/data audit, shared source modules, third-party DIS2 source, and the paper package were retained. `retest_dis2_tgd.py` was retained because existing baseline and audit documents still depend on it.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
