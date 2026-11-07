"""Train-only feasibility check for independent relation and frequency losses."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from train_distillation import Features, OpticalInputs, ROOT, REFERENCE, common, segmentation


def relation(x, foreground, background):
    x = F.normalize(x.float(), dim=1)
    prototypes = [F.normalize((x * region).sum((2, 3), keepdim=True), dim=1)
                  for region in (foreground, background)]
    return (x * (prototypes[0] - prototypes[1])).sum(1, keepdim=True)


def candidate_losses(student, teacher, s, t, batch, mode=None):
    valid, label = batch["valid"].float(), batch["mask"]
    feature = None
    if mode != "N2":
        feature, relation_count = feature_loss(student, teacher, t, valid, label)
        if mode == "N1":
            return {"N1": feature}, relation_count, 0
    subbands, boundary_count = frequency_losses(s, t, valid, label)
    if mode == "N2":
        return {"N2": sum(subbands[k] for k in ("LH", "HL", "HH")) / 3}, 0, boundary_count
    return {"N1": feature, **subbands}, relation_count, boundary_count


def feature_loss(student, teacher, t, valid, label):
    size = student.dec[2].shape[-2:]
    mask = F.interpolate(valid, size=size, mode="area").eq(1)
    y = F.interpolate(label, size=size, mode="area").ge(.5)
    correct = F.interpolate(t.detach(), size=size, mode="area").ge(0).eq(y)
    foreground, background = mask & y, mask & ~y
    usable = foreground.any((2, 3), keepdim=True) & background.any((2, 3), keepdim=True)
    w = mask & correct & usable
    sr = relation(student.dec[2], foreground, background)
    tr = relation(teacher.dec[2].detach(), foreground, background)
    feature = (F.smooth_l1_loss(sr, tr, reduction="none") * w).sum() / w.sum().clamp_min(1)
    return feature, int(w.sum())


def frequency_losses(s, t, valid, label):
    def haar(z):
        z = z.float().sigmoid()
        a, b, c, d = z[..., ::2, ::2], z[..., ::2, 1::2], z[..., 1::2, ::2], z[..., 1::2, 1::2]
        return ((a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2)
    border = F.max_pool2d(label, 4, 2, 1) > -F.max_pool2d(-label, 4, 2, 1)
    window = border & F.avg_pool2d(valid * t.detach().ge(0).eq(label.bool()), 2).eq(1)
    def scale_energy(z):
        e = z.abs()
        mean = (e * window).sum((2, 3), keepdim=True) / window.sum((2, 3), keepdim=True).clamp_min(1)
        return e / (mean.detach() + .01)
    subbands = {}
    for name, a, b in zip(("LL", "LH", "HL", "HH"), haar(s), haar(t.detach())):
        if name != "LL":
            a, b = scale_energy(a), scale_energy(b)
        subbands[name] = (F.smooth_l1_loss(a, b, reduction="none") * window).sum() / window.sum().clamp_min(1)
    return subbands, int(window.sum())


def main():
    torch.set_num_threads(4)
    frames = common.frames()
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    teacher, student = Features(12).eval().requires_grad_(False), Features(4).eval()
    teacher.load_state_dict(torch.load(ROOT / "Teacher/best.pt", map_location="cpu", weights_only=True))
    student.load_state_dict(torch.load(ROOT / "D0/best.pt", map_location="cpu", weights_only=True))
    params = [*student.unet.enc[0].parameters(), *student.unet.dec[2].parameters()]
    report = []
    for start in (0, 60, 120, 180):
        ds = OpticalInputs(frames["train"].iloc[start:start+2], stats, optical)
        batch = next(iter(common.loader(ds)))
        with torch.no_grad():
            t = teacher(batch["optical"])
        s = student(batch["x"])
        losses, relation_pixels, boundary_cells = candidate_losses(student, teacher, s, t, batch)
        def gradient(loss):
            gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                              for p, g in zip(params, gs)])
        base = gradient(segmentation(s, batch))
        scores = {}
        for name, loss in losses.items():
            g = gradient(loss)
            assert torch.isfinite(g).all()
            scores[name] = {"loss": float(loss.detach()), "grad_norm": float(g.norm()),
                            "cosine_seg": float(F.cosine_similarity(g, base, dim=0))}
        report.append({"samples": batch["sample_id"], "relation_cells": relation_pixels,
                       "boundary_cells": boundary_cells, "metrics": scores})
    common.write(Path(__file__).resolve().parent / "mechanism_audit/candidate_probe.json", report)


if __name__ == "__main__":
    main()
