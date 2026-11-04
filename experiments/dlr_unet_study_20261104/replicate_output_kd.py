"""Run paired B1/D0 replications and two output-KD controls."""
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from train_distillation import (BASE, ROOT, REFERENCE, Features, Inputs, OpticalInputs,
                                common, evaluate_frozen, fit, kd, segmentation)

OUT = Path(__file__).resolve().parent / "mechanism_runs_20261105"


def seed(value):
    common.seed()
    import random
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def objective_for(arm, teacher, sar_teacher, weight):
    def objective(student, batch):
        z = student(batch["x"])
        loss = segmentation(z, batch)
        if arm == "B1":
            return loss
        with torch.no_grad():
            target = sar_teacher(batch["x"]) if arm == "SAR_self_KD" else teacher(batch["optical"])
        valid = batch["valid"].float()
        if arm == "optical_confidence_KD":
            # Confidence is computed at T=1. Normalize so the average KD
            # weight over valid pixels stays the same as in ordinary D0.
            confidence = (2 * target.sigmoid() - 1).abs() * valid
            valid = confidence * (valid.sum() / confidence.sum().clamp_min(1e-6))
        return loss + weight * kd.output_kd(z, target, valid)
    return objective


def main():
    OUT.mkdir(exist_ok=True)
    common.setup()
    frames = common.frames()
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    weights = json.loads((ROOT / "D0/calibration.json").read_text())["weights"]
    teacher = Features(12).cuda().to(memory_format=torch.channels_last)
    teacher.load_state_dict(torch.load(ROOT / "Teacher/best.pt", map_location="cpu", weights_only=True))
    teacher.eval().requires_grad_(False)
    sar_teacher = common.PaperUNet(4).cuda().to(memory_format=torch.channels_last)
    sar_teacher.load_state_dict(torch.load(BASE / "best.pt", map_location="cpu", weights_only=True))
    sar_teacher.eval().requires_grad_(False)
    queue = [(43, "B1"), (43, "D0"), (44, "B1"), (44, "D0"),
             (42, "SAR_self_KD"), (42, "optical_confidence_KD")]
    common.write(OUT / "config.json", {"queue": queue, "lambda_out": weights["out"],
                 "temperature": 10, "test_after_all_freeze": True,
                 "source_teacher_sha256": common.sha(ROOT / "Teacher/best.pt"),
                 "source_sar_teacher_sha256": common.sha(BASE / "best.pt"),
                 "split_sha256": common.sha(Path(common.HERE) / "split_manifest.csv")})
    val = Inputs(frames["val"], stats)
    for number, arm in queue:
        folder = OUT / f"{arm}_seed{number}"
        folder.mkdir(exist_ok=True)
        if (folder / "selection.json").exists():
            continue
        common.write(OUT / "status.json", {"state": "TRAINING", "arm": folder.name})
        seed(number)
        model = common.PaperUNet(4).cuda().to(memory_format=torch.channels_last)
        if number == 42:
            model.load_state_dict(torch.load(BASE / "initial.pt", map_location="cpu", weights_only=True))
        seed(number)
        use_optical = arm in ("D0", "optical_confidence_KD")
        train = (OpticalInputs(frames["train"], stats, optical, augment=True) if use_optical
                 else Inputs(frames["train"], stats, augment=True))
        gen = torch.Generator().manual_seed(number + 81)
        objective = objective_for(arm, teacher, sar_teacher, weights["out"])
        fit(model, common.loader(train, True, gen), val, gen, folder, objective)
        del model, train
        teacher.enc.clear()
        teacher.dec.clear()
        torch.cuda.empty_cache()
    common.write(OUT / "status.json", {"state": "EVALUATING"})
    test = Inputs(frames["test"], stats)
    for number, arm in queue:
        folder = OUT / f"{arm}_seed{number}"
        model = common.PaperUNet(4).cuda().to(memory_format=torch.channels_last)
        evaluate_frozen(model, test, folder)
        del model
    common.write(OUT / "status.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(OUT / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
