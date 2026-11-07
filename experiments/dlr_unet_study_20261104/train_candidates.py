"""Train independent N1 and N2 candidates against the frozen paired D0 runs."""
import json
from pathlib import Path
import traceback

import torch

from probe_candidate_kd import candidate_losses
from replicate_output_kd import OUT as CONTROLS, seed
from train_distillation import (ROOT, REFERENCE, Features, Inputs, OpticalInputs,
                                common, evaluate_frozen, fit, kd, segmentation)

OUT = Path(__file__).resolve().parent / "candidate_runs_20261105"
QUEUE = [(arm, number) for number in (43, 44) for arm in ("N1", "N2")]


def target_forward(teacher, batch):
    with torch.no_grad():
        return teacher(batch["optical"])


def calibrate(arm, model, teacher, probe):
    model.eval()
    s = model(probe["x"])
    t = target_forward(teacher, probe)
    terms, relation_cells, boundary_cells = candidate_losses(model, teacher, s, t, probe, arm)
    params = tuple(model.parameters())
    def norm(loss):
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return sum(g.square().sum() for g in gradients if g is not None).sqrt().item()
    base, extra = norm(segmentation(s, probe)), norm(terms[arm])
    assert base > 0 and extra > 0 and torch.isfinite(torch.tensor([base, extra])).all()
    return {"lambda": min(1000., .05 * base / extra), "seg_grad_norm": base,
            "aux_grad_norm": extra, "samples": probe["sample_id"],
            "relation_cells": relation_cells, "boundary_cells": boundary_cells}


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((CONTROLS / "status.json").read_text())["state"] == "COMPLETE"
    common.setup()
    frames = common.frames()
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    weight_out = json.loads((ROOT / "D0/calibration.json").read_text())["weights"]["out"]
    train = OpticalInputs(frames["train"], stats, optical, augment=True)
    val = Inputs(frames["val"], stats)
    probe = common.cuda(next(iter(common.loader(OpticalInputs(frames["train"].iloc[:2], stats, optical)))))
    teacher = Features(12).cuda().to(memory_format=torch.channels_last)
    teacher_path = ROOT / "Teacher/best.pt"
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=True))
    teacher.eval().requires_grad_(False)
    common.write(OUT / "config.json", {
        "queue": QUEUE, "teacher_sha256": common.sha(teacher_path),
        "temperature": 10, "lambda_out": weight_out, "aux_initial_gradient_ratio": .05,
        "N1": "GT class prototype margin at decoder stride 2, teacher-correct support",
        "N2": "mean LH/HL/HH normalized local probability energy on valid teacher-correct GT boundary cells",
        "test_after_all_freeze": True, "combination": "NOT_QUEUED",
    })
    for arm, number in QUEUE:
        name = f"{arm}_seed{number}"
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        if (folder / "selection.json").exists():
            continue
        common.write(OUT / "status.json", {"state": "TRAINING", "arm": name})
        seed(number)
        model = Features(4).cuda().to(memory_format=torch.channels_last)
        record = calibrate(arm, model, teacher, probe)
        common.write(folder / "calibration.json", record)
        seed(number)
        gen = torch.Generator().manual_seed(number + 81)
        def objective(student, batch):
            t = target_forward(teacher, batch)
            s = student(batch["x"])
            terms, _, _ = candidate_losses(student, teacher, s, t, batch, arm)
            return (segmentation(s, batch) + weight_out * kd.output_kd(s, t, batch["valid"])
                    + record["lambda"] * terms[arm])
        fit(model, common.loader(train, True, gen), val, gen, folder, objective)
        del model
        torch.cuda.empty_cache()
    common.write(OUT / "status.json", {"state": "EVALUATING"})
    test = Inputs(frames["test"], stats)
    comparison = []
    for arm, number in QUEUE:
        folder = OUT / f"{arm}_seed{number}"
        model = Features(4).cuda().to(memory_format=torch.channels_last)
        evaluate_frozen(model, test, folder)
        baseline = CONTROLS / f"D0_seed{number}"
        candidate_score, d0_score = [json.loads((x / "test_metrics.json").read_text()) for x in (folder, baseline)]
        candidate_val, d0_val = [json.loads((x / "selection.json").read_text())["validation_IoU"]
                                 for x in (folder, baseline)]
        comparison.append({
            "arm": arm, "seed": number, "validation_gain_pp": 100*(candidate_val-d0_val),
            "all120_gain_pp": 100*(candidate_score["IoU"]-d0_score["IoU"]),
            "ind44_gain_pp": 100*(candidate_score["per_official_subset"]["testind"]["IoU"]-
                                  d0_score["per_official_subset"]["testind"]["IoU"]),
        })
        del model
    common.write(OUT / "comparison.json", comparison)
    common.write(OUT / "status.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(OUT / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
