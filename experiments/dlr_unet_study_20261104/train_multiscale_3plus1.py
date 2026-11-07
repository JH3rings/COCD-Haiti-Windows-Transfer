"""Official DLR 3+1 benchmark with three-scale SAR evidence control."""
import json
from pathlib import Path
import traceback

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import Inputs
from replicate_output_kd import seed
from search_spectrum import SearchEvidenceUNet, amplitude
from train_baseline import common, fit, segmentation
from train_distillation import (ROOT, REFERENCE, OpticalInputs, evaluate_frozen, kd)
from train_evidence_controls import optical_view

OUT = Path(__file__).resolve().parent / "multiscale_3plus1_results"
SAR_FIELDS = tuple(f"{t}_g{p}" for t in ("PRE1", "PRE2", "PRE3", "POST1") for p in ("VV", "VH"))
STRIDES = (8, 4, 2)
DEC_CHANNELS = (512, 256, 128)
ARMS = {
    "SAR_control": (), "optical_teacher": (), "output_KD": ("out",),
    "output_spectrum": ("out", "spectrum"), "output_modulation": ("out", "modulation"),
    "full": ("out", "spectrum", "modulation"), "MFSAD": ("out", "bk", "freq", "jscad"),
}


class SARInputs(Inputs):
    sar_fields = SAR_FIELDS


class PrivilegedInputs(OpticalInputs):
    sar_fields = SAR_FIELDS


class TerrainInputs(SARInputs):
    def __init__(self, frame, stats, augment=False):
        super().__init__(frame, stats, augment)
        mean = np.asarray(stats["terrain_mean"], np.float32)[:, None, None]
        std = np.asarray(stats["terrain_std"], np.float32)[:, None, None]
        for row, sample in zip(self.frame.itertuples(), self.cache):
            with h5py.File(row.h5_path, "r") as f:
                fields = np.stack([f[key][int(row.h5_index), 0] for key in ("None_DEM", "None_SLOPE")])
            assert np.isfinite(fields).all()
            terrain = torch.from_numpy(np.clip((fields - mean) / std, -8, 8).astype(np.float32))
            sample["x"] = torch.cat((sample["x"], terrain), 0)


class ScaleControl(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.controller = self.head(channels, 11)
        self.modulator = self.head(channels, channels)
        self.register_buffer("anchors", torch.tensor([[0., 0.], [.75, 0.], [0., .75], [-.75, -.75]]))

    @staticmethod
    def head(channels, output):
        model = nn.Sequential(nn.Conv2d(channels + 16, 32, 1), nn.GELU(), nn.Conv2d(32, output, 1))
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
        return model

    def forward(self, evidence, context):
        searched, search = SearchEvidenceUNet.intervene(self, evidence, context)
        gain = .5 * self.modulator(torch.cat((searched, context), 1)).tanh()
        return searched * (1 + gain), {**search, "searched": searched, "gain": gain}


class MultiScaleUNet(nn.Module):
    def __init__(self, backbone, optical=False, adapters=False):
        super().__init__()
        self.backbone, self.optical = backbone, optical
        self.context = nn.Sequential(nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.GELU())
        self.controls = nn.ModuleList(ScaleControl(c) for c in DEC_CHANNELS)
        if adapters:
            def adapter(c):
                return nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
            self.enc_adapters = nn.ModuleList(adapter(c) for c in (256, 512, 1024))
            self.dec_adapters = nn.ModuleList(adapter(c) for c in DEC_CHANNELS)
        for module in self.frozen_modules():
            module.requires_grad_(False).eval()

    def frozen_modules(self):
        return self.backbone.enc[0], self.backbone.enc[1], self.backbone.up[0], self.backbone.dec[0]

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen_modules():
            module.eval()
        return self

    def details(self, x):
        assert x.shape[1] == (20 if self.optical else 8)
        sar = x[:, :8].contiguous(memory_format=torch.channels_last)
        context = self.context(x[:, 8:] if self.optical else torch.cat((sar, torch.zeros_like(sar[:, :4])), 1))
        encoded, decoded, controls = [], [], []
        x = sar
        for i, block in enumerate(self.backbone.enc):
            x = block(F.max_pool2d(x, 2) if i else x)
            encoded.append(x)
        for i, (up, block, skip) in enumerate(zip(self.backbone.up, self.backbone.dec, encoded[-2::-1])):
            x = block(torch.cat((skip, up(x)), 1))
            if i < len(self.controls):
                local = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x, ctl = self.controls[i](x, local)
                controls.append(ctl)
                decoded.append(x)
        return {"z": self.backbone.head(x), "controls": controls, "encoded": encoded, "decoded": decoded}

    def forward(self, x):
        return self.details(x)["z"]


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def raw_losses(model, s, t, batch, keys):
    valid = batch["valid"].float()
    out = {"out": kd.output_kd(s["z"], t["z"], valid)}
    groups = {k: [] for k in keys if k != "out"}
    for i, stride in enumerate(STRIDES):
        left, right = s["controls"][i], t["controls"][i]
        mask = F.avg_pool2d(valid, stride).eq(1)
        if "spectrum" in groups:
            interior = F.avg_pool2d(mask.float(), 5, 1, 2).eq(1)
            # Cycles per input pixel multiplied by feature stride.
            frequencies = tuple(r * stride for r in (.025, .04, .06))
            difference = amplitude(left, frequencies) - amplitude(right, frequencies)
            groups["spectrum"].append(masked_mean(difference.square().mean(-1).unsqueeze(1), interior))
        if "modulation" in groups:
            reference = right["searched"]
            difference = F.smooth_l1_loss(left["gain"] * reference, right["gain"] * reference, reduction="none")
            groups["modulation"].append(masked_mean(difference.mean(1, keepdim=True), mask))
        if "bk" in groups:
            a, b = model.enc_adapters[i](s["encoded"][i+2]), t["encoded"][i+2]
            mask_bk = F.interpolate(valid, size=a.shape[-2:], mode="nearest")
            groups["bk"].append(masked_mean((a-b).square().mean(1, keepdim=True), mask_bk))
            freq, joint = kd.mfsad_neck(model.dec_adapters[i](s["decoded"][i]), t["decoded"][i], mask.float())
            groups["freq"].append(freq)
            groups["jscad"].append(joint)
    out.update({k: torch.stack(v).mean() for k, v in groups.items()})
    return out


def calibrate(model, teacher, batch, keys):
    model.eval()
    with torch.no_grad():
        t = teacher.details(torch.cat((batch["x"], batch["optical"]), 1))
    s = model.details(batch["x"])
    losses = raw_losses(model, s, t, batch, keys)
    losses["seg"] = segmentation(s["z"], batch)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    norms = {}
    for key, value in losses.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        norm = sum(g.square().sum() for g in gradients if g is not None).sqrt().item()
        assert np.isfinite(norm) and norm > 0, (key, norm)
        norms[key] = norm
    return norms, {k: min(1000., .05 * norms["seg"] / norms[k]) for k in keys}


def plain(channels):
    seed(42)
    return common.PaperUNet(channels).cuda().to(memory_format=torch.channels_last)


def build(arm):
    backbone = plain(8)
    backbone.load_state_dict(torch.load(OUT / "B1_3plus1/best.pt", map_location="cpu", weights_only=True))
    return MultiScaleUNet(backbone, arm == "optical_teacher", arm == "MFSAD").cuda().to(memory_format=torch.channels_last)


def run(arm, model, train, val, objective):
    dest = OUT / arm
    dest.mkdir(exist_ok=True)
    common.write(OUT / "status.json", {"state": "TRAINING", "arm": arm})
    if not (dest / "selection.json").exists():
        seed(42)
        generator = torch.Generator().manual_seed(123)
        fit(model, common.loader(train, True, generator), val, generator, dest, objective)
    selected = json.loads((dest / "selection.json").read_text())
    assert selected["epochs_completed"] == 100 and common.sha(dest / "best.pt") == selected["checkpoint_sha256"]
    model.load_state_dict(torch.load(dest / "best.pt", map_location="cpu", weights_only=True))
    return selected


def main():
    OUT.mkdir(exist_ok=True)
    common.setup()
    frames = common.frames()
    assert {k: len(v) for k, v in frames.items()} == {"train": 355, "val": 88, "test": 120}
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    common.write(OUT / "config.json", {"fields": SAR_FIELDS, "epochs": 100, "val_every": 5,
                 "strides": STRIDES, "seed": 42, "threshold": .5,
                 "reference_split_sha256": common.sha(Path(common.HERE) / "split_manifest.csv"),
                 "protocol": "355/88/120 official split, joint 8-SAR validity, test after all freeze",
                 "arms": ["B1_3plus1", *ARMS, "terrain_3plus1"],
                 "frozen": "first two encoder blocks and first decoder block; deeper encoder and subsequent decoder train",
                 "baseline_normalization": "locked train-only per-polarization normalization"})
    train = SARInputs(frames["train"], stats, augment=True)
    val = SARInputs(frames["val"], stats)
    base = plain(8)
    run("B1_3plus1", base, train, val, lambda model, batch: segmentation(model(batch["x"]), batch))
    del base
    train_opt = PrivilegedInputs(frames["train"], stats, optical, augment=True)
    val_opt = PrivilegedInputs(frames["val"], stats, optical)
    probe = common.cuda(next(iter(common.loader(PrivilegedInputs(frames["train"].iloc[:4], stats, optical)))))
    sar = build("SAR_control")
    sar_selection = run("SAR_control", sar, train, val, lambda m, batch: segmentation(m(batch["x"]), batch))
    del sar
    teacher = build("optical_teacher")
    opt_selection = run("optical_teacher", teacher, optical_view(train_opt), optical_view(val_opt),
                        lambda m, batch: segmentation(m(batch["x"]), batch))
    teacher.eval().requires_grad_(False)
    shuffled = common.evaluate(teacher, optical_view(val_opt, shuffled=True))
    common.write(OUT / "teacher_diagnostic.json", {"optical": opt_selection["validation_IoU"],
                 "sar": sar_selection["validation_IoU"], "shuffled_optical": shuffled["IoU"]})
    shared_weights = {}
    for arm, keys in list(ARMS.items())[2:]:
        student = build(arm)
        norms, weights = calibrate(student, teacher, probe, keys)
        for key in ("out", "spectrum", "modulation"):
            if key in weights:
                weights[key] = shared_weights.setdefault(key, weights[key])
        folder = OUT / arm
        folder.mkdir(exist_ok=True)
        common.write(folder / "calibration.json", {"weights": weights, "gradient_norms": norms})
        def objective(model, batch):
            with torch.no_grad():
                t = teacher.details(torch.cat((batch["x"], batch["optical"]), 1))
            s = model.details(batch["x"])
            raw = raw_losses(model, s, t, batch, keys)
            return segmentation(s["z"], batch) + sum(weights[k] * value for k, value in raw.items())
        run(arm, student, train_opt, val, objective)
        del student
        torch.cuda.empty_cache()
    del teacher, train, train_opt, val_opt, probe
    torch.cuda.empty_cache()
    terrain = plain(10)
    run("terrain_3plus1", terrain, TerrainInputs(frames["train"], stats, augment=True),
        TerrainInputs(frames["val"], stats), lambda m, batch: segmentation(m(batch["x"]), batch))
    del terrain
    common.write(OUT / "status.json", {"state": "EVALUATING"})
    test = SARInputs(frames["test"], stats)
    comparison = {}
    for arm in ("B1_3plus1", *ARMS, "terrain_3plus1"):
        if arm == "terrain_3plus1":
            model, data = plain(10), TerrainInputs(frames["test"], stats)
        elif arm == "B1_3plus1":
            model, data = plain(8), test
        elif arm == "optical_teacher":
            model, data = build(arm), optical_view(PrivilegedInputs(frames["test"], stats, optical))
        else:
            model, data = build(arm), test
        evaluate_frozen(model, data, OUT / arm)
        metrics = json.loads((OUT / arm / "test_metrics.json").read_text())
        comparison[arm] = {"all120_IoU": metrics["IoU"], "ind44_IoU": metrics["per_official_subset"]["testind"]["IoU"]}
        del model, data
    common.write(OUT / "comparison.json", comparison)
    common.write(OUT / "status.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(OUT / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
