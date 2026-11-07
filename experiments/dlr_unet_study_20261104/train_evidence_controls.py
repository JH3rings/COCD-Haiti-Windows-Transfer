"""Queue two separate controller pilots after the N1/N2 experiment freezes."""
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F

from sar_evidence_control import EvidenceUNet
from search_spectrum import JointEvidenceUNet, SearchEvidenceUNet, amplitude, covariance
from train_candidates import OUT as CANDIDATES
from replicate_output_kd import seed
from train_distillation import (BASE, ROOT, REFERENCE, Inputs, OpticalInputs, common,
                                evaluate_frozen, fit, kd, segmentation)

OUT = Path(__file__).resolve().parent / "evidence_control_runs_20261105"
FAMILIES = {"modulation": EvidenceUNet, "spectrum": SearchEvidenceUNet}
if "--joint" in sys.argv:
    OUT = OUT.with_name("joint_evidence_runs_20261105")
    FAMILIES = {"joint": JointEvidenceUNet}
TEACHER_MARGIN = .002


class ControllerModel(nn.Module):
    def __init__(self, family, optical, backbone):
        super().__init__()
        self.net = FAMILIES[family](backbone, optical_teacher=optical)
        self.optical = optical
        backbone.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.net.backbone.eval()
        return self

    def details(self, x):
        return self.net(x[:, :4], x[:, 4:] if self.optical else None)

    def forward(self, x):
        return self.details(x)["z"]


def optical_view(dataset, shuffled=False):
    class View(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data, self.frame = data, data.frame
            self.indices = list(range(len(data)))
            if shuffled:
                events = self.frame.event_id.tolist()
                assert len(set(events)) > 1
                self.indices = [next((i+j) % len(events) for j in range(1, len(events))
                                     if events[(i+j) % len(events)] != event)
                                for i, event in enumerate(events)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            row = self.data[i]
            optical = self.data[self.indices[i]]["optical"] if shuffled else row["optical"]
            return {**row, "x": torch.cat((row["x"], optical), 0)}
    return View(dataset)


def auxiliary(kind, s, t, valid):
    mask = F.avg_pool2d(valid.float(), 4).eq(1)
    if kind == "joint_modulation":
        # Compare what both gains do to the same teacher-searched SAR tensor.
        reference = t["searched"]
        loss = F.smooth_l1_loss(s["gain"] * reference, t["gain"] * reference,
                                reduction="none").mean(1, keepdim=True)
    elif kind == "modulation":
        loss = F.smooth_l1_loss(s["delta"], t["delta"], reduction="none").mean(1, keepdim=True)
    else:
        # The spectral expression is for interior samples, away from invalid pixels.
        mask = F.avg_pool2d(mask.float(), 5, 1, 2).eq(1)
        transform = amplitude if kind == "spectrum" else covariance
        loss = (transform(s) - transform(t)).square().mean(-1).unsqueeze(1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def objective(teacher=None, out_weight=0., extra_weights=None):
    def loss(model, batch):
        s = model.details(batch["x"])
        total = segmentation(s["z"], batch)
        if teacher is None:
            return total
        with torch.no_grad():
            t = teacher.details(torch.cat((batch["x"], batch["optical"]), 1))
        total = total + out_weight * kd.output_kd(s["z"], t["z"], batch["valid"])
        for kind, weight in (extra_weights or {}).items():
            total = total + weight * auxiliary(kind, s, t, batch["valid"])
        return total
    return loss


def build(family, optical):
    seed(42)
    backbone = common.PaperUNet(4)
    backbone.load_state_dict(torch.load(BASE / "best.pt", map_location="cpu", weights_only=True))
    return ControllerModel(family, optical, backbone).cuda().to(memory_format=torch.channels_last)


def load_selected(model, folder):
    selected = json.loads((folder / "selection.json").read_text())
    path = folder / "best.pt"
    assert selected["epochs_completed"] == 100 and common.sha(path) == selected["checkpoint_sha256"]
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return selected


def run(family, arm, model, train, val, loss):
    folder = OUT / family / arm
    folder.mkdir(parents=True, exist_ok=True)
    common.write(OUT / "status.json", {"state": "TRAINING", "family": family, "arm": arm})
    if not (folder / "selection.json").exists():
        seed(42)
        generator = torch.Generator().manual_seed(123)
        fit(model, common.loader(train, True, generator), val, generator, folder, loss)
    return folder, load_selected(model, folder)


def calibrate(student, teacher, kinds, batch):
    student.eval()
    with torch.no_grad():
        t = teacher.details(torch.cat((batch["x"], batch["optical"]), 1))
    s = student.details(batch["x"])
    losses = {"seg": segmentation(s["z"], batch),
              "out": kd.output_kd(s["z"], t["z"], batch["valid"])}
    for kind in kinds:
        losses[kind] = auxiliary(kind, s, t, batch["valid"])
    params = tuple(p for p in student.parameters() if p.requires_grad)
    norms = {}
    for k, value in losses.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        norm = sum(g.square().sum() for g in gradients if g is not None).sqrt().item()
        assert torch.isfinite(torch.tensor(norm)) and norm > 0, (k, norm)
        norms[k] = norm
    return {k: min(1000., .05 * norms["seg"] / norms[k]) for k in norms if k != "seg"}, norms


def main():
    OUT.mkdir(exist_ok=True)
    while True:
        state = json.loads((CANDIDATES / "status.json").read_text())["state"]
        if state == "COMPLETE":
            break
        assert state != "FAILED", "N1/N2 failed; inspect candidate queue"
        common.write(OUT / "status.json", {"state": "WAITING_FOR_N1_N2"})
        time.sleep(20)
    common.setup()
    frames = common.frames()
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    training = OpticalInputs(frames["train"], stats, optical, augment=True)
    val_opt = OpticalInputs(frames["val"], stats, optical)
    val = Inputs(frames["val"], stats)
    probe = common.cuda(next(iter(common.loader(OpticalInputs(frames["train"].iloc[:4], stats, optical)))))
    common.write(OUT / "config.json", {"seed": 42, "epochs": 100, "val_every": 5,
                 "teacher_margin_validation_IoU": TEACHER_MARGIN, "base_sha256": common.sha(BASE / "best.pt"),
                 "backbone": "frozen B1, including BatchNorm statistics; only controller trains",
                 "scope": "single-seed pilot; frozen models tested once", "families": list(FAMILIES)})
    for family in FAMILIES:
        control = build(family, False)
        sar_dir, sar_val = run(family, "SAR_control", control, training, val, objective())
        del control
        teacher = build(family, True)
        teacher_dir, opt_val = run(family, "optical_teacher", teacher, optical_view(training),
                                   optical_view(val_opt), objective())
        teacher.eval().requires_grad_(False)
        shuffled = common.evaluate(teacher, optical_view(val_opt, shuffled=True))
        common.write(teacher_dir / "shuffled_validation.json", shuffled)
        real_iou = opt_val["validation_IoU"]
        gate = real_iou - max(sar_val["validation_IoU"], shuffled["IoU"]) >= TEACHER_MARGIN
        common.write(OUT / family / "teacher_gate.json", {
            "pass": gate, "optical": real_iou, "sar": sar_val["validation_IoU"],
            "shuffled_optical": shuffled["IoU"], "margin": TEACHER_MARGIN,
            "shuffle": "cross-event reassignment on validation only, inference intervention"})
        if not gate:
            del teacher
            continue
        student_arms = [("output_KD", ())]
        if family == "joint":
            student_arms.append(("output_plus_spectrum_and_modulation", ("spectrum", "joint_modulation")))
        else:
            student_arms.append(("output_plus_" + family, (family,)))
        if family == "spectrum":
            student_arms.append(("output_plus_covariance", ("covariance",)))
        output_weight = None
        for arm, kinds in student_arms:
            student = build(family, False)
            weights, norms = calibrate(student, teacher, kinds, probe)
            if output_weight is None:
                output_weight = weights["out"]
            extra = {kind: weights[kind] for kind in kinds}
            folder = OUT / family / arm
            folder.mkdir(exist_ok=True)
            common.write(folder / "calibration.json", {"output": output_weight,
                         "extra": extra, "norms": norms})
            run(family, arm, student, training, val,
                objective(teacher, output_weight, extra))
            del student
        del teacher
    common.write(OUT / "status.json", {"state": "EVALUATING"})
    test = Inputs(frames["test"], stats)
    test_opt = optical_view(OpticalInputs(frames["test"], stats, optical))
    for family in FAMILIES:
        for folder in (OUT / family).iterdir():
            if not (folder / "selection.json").exists():
                continue
            optical_mode = folder.name == "optical_teacher"
            model = build(family, optical_mode)
            evaluate_frozen(model, test_opt if optical_mode else test, folder)
            del model
    common.write(OUT / "status.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(OUT / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
