"""Sequential optical Teacher and U-Net D0–D3 with a completed B1 gate."""
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import distillation_losses as kd
from train_baseline import Inputs, OUT as BASE, REFERENCE, common, fit, segmentation

ROOT = Path(__file__).resolve().parent / "results"
OPTICAL = tuple(f"{time}_{band}" for time in ("PRE1", "POST1")
                for band in ("B02", "B03", "B04", "B08", "B11", "B12"))


def read_optical(row):
    with h5py.File(row.h5_path, "r") as f:
        return np.stack([f[k][int(row.h5_index), 0] for k in OPTICAL]).astype(np.float32) / 10000


def fit_optical(frame):
    sums, squares, count = np.zeros((3, 12))
    for row in frame.itertuples():
        x = read_optical(row).reshape(12, -1).astype(np.float64)
        valid = np.isfinite(x) & (x > 0)
        x = np.where(valid, x, 0)
        sums += x.sum(1)
        squares += (x*x).sum(1)
        count += valid.sum(1)
    assert (count > 0).all()
    mean = sums/count
    return {"mean": mean.tolist(), "std": np.sqrt(np.maximum(squares/count-mean**2, 1e-6)).tolist()}


class OpticalInputs(Inputs):
    def __init__(self, frame, stats, optical_stats, teacher=False, augment=False):
        super().__init__(frame, stats, augment)
        mean = np.asarray(optical_stats["mean"], np.float32)[:, None, None]
        std = np.asarray(optical_stats["std"], np.float32)[:, None, None]
        for row, item in zip(self.frame.itertuples(), self.cache):
            x = read_optical(row)
            assert np.isfinite(x).mean() > .99
            x = np.clip((np.where(np.isfinite(x), x, 0)-mean)/std, -8, 8)
            item["x" if teacher else "optical"] = torch.from_numpy(x)


class Features(nn.Module):
    def __init__(self, channels, adapters=False):
        super().__init__()
        self.unet = common.PaperUNet(channels)
        self.enc, self.dec = {}, {}
        for i in (2, 3, 4):
            self.unet.enc[i].register_forward_hook(self.capture(self.enc, i))
        for i in (0, 1, 2):
            self.unet.dec[i].register_forward_hook(self.capture(self.dec, i))
        if adapters:
            self.enc_adapter = nn.ModuleList(self.adapter(c) for c in (256, 512, 1024))
            self.dec_adapter = nn.ModuleList(self.adapter(c) for c in (512, 256, 128))

    @staticmethod
    def capture(storage, i):
        def record(_module, _inputs, output):
            storage[i] = output
        return record

    @staticmethod
    def adapter(channels):
        return nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                             nn.BatchNorm2d(channels), nn.SiLU())

    def forward(self, x):
        self.enc.clear()
        self.dec.clear()
        return self.unet(x.contiguous(memory_format=torch.channels_last))


def raw_terms(arm, student, teacher, logits, target, valid):
    terms = {"out": kd.output_kd(logits, target, valid)}
    if arm == "D0":
        return terms
    att, wave, bk, freq, joint = [], [], [], [], []
    for i in (0, 1, 2):
        s, t = student.dec[i], teacher.dec[i].detach()
        v = F.interpolate(valid.float(), size=s.shape[-2:], mode="nearest")
        if arm == "D3":
            a, b = student.enc_adapter[i](student.enc[i+2]), teacher.enc[i+2].detach()
            m = F.interpolate(valid.float(), size=a.shape[-2:], mode="nearest")
            bk.append(((a-b).square()*m).sum() / (m.sum().clamp_min(1)*a.shape[1]))
            f, j = kd.mfsad_neck(student.dec_adapter[i](s), t, v)
            freq.append(f)
            joint.append(j)
        else:
            att.append(kd.attention(s, t, v))
            if arm == "D2":
                wave.append(kd.wavelet(s, t, v))
    for key, values in (("att", att), ("wave", wave), ("bk", bk), ("freq", freq), ("jscad", joint)):
        if values:
            terms[key] = torch.stack(values).mean()
    return terms


def evaluate_frozen(model, dataset, folder):
    selection = json.loads((folder / "selection.json").read_text())
    checkpoint = folder / "best.pt"
    assert selection["epochs_completed"] == 100 and common.sha(checkpoint) == selection["checkpoint_sha256"]
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    if not (folder / "test_metrics.json").exists():
        marker = folder / "test_started.json"
        assert not marker.exists(), "Check interrupted test before retry"
        common.write(marker, selection)
        score = common.evaluate(model, dataset, folder / "test_predictions.npz")
        common.write(folder / "test_metrics.json", score)


def main():
    status = ROOT / "distillation_status.json"
    while not (BASE / "test_metrics.json").exists():
        assert not (BASE / "error.json").exists(), "B1 reported a failure"
        common.write(status, {"state": "WAITING_FOR_B1"})
        time.sleep(15)
    selection = json.loads((BASE / "selection.json").read_text())
    history = json.loads((BASE / "history.json").read_text())
    assert len(history) == selection["epochs_completed"] == 100
    assert common.sha(BASE / "best.pt") == selection["checkpoint_sha256"]
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert selection["validation_IoU"] >= .70, "B1 validation gate: inspect baseline before KD"
    common.setup()
    frames = common.frames()
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    opt_stats = fit_optical(frames["train"])
    common.write(ROOT / "optical_normalization.json", opt_stats)
    common.write(status, {"state": "TRAINING", "arm": "Teacher"})
    tdir = ROOT / "Teacher"
    tdir.mkdir(exist_ok=True)
    common.seed()
    teacher = Features(12).cuda().to(memory_format=torch.channels_last)
    if not (tdir / "selection.json").exists():
        train = OpticalInputs(frames["train"], stats, opt_stats, teacher=True, augment=True)
        val = OpticalInputs(frames["val"], stats, opt_stats, teacher=True)
        gen = torch.Generator().manual_seed(123)
        fit(teacher, common.loader(train, True, gen), val, gen, tdir,
            lambda model, batch: segmentation(model(batch["x"]), batch))
        del train, val
    evaluate_frozen(teacher, OpticalInputs(frames["test"], stats, opt_stats, teacher=True), tdir)
    teacher.eval().requires_grad_(False)
    for storage in (teacher.enc, teacher.dec):
        storage.clear()
    train = OpticalInputs(frames["train"], stats, opt_stats, augment=True)
    val = Inputs(frames["val"], stats)
    probe = common.cuda(next(iter(common.loader(
        OpticalInputs(frames["train"].iloc[:2], stats, opt_stats)))))
    shared_weights = {}
    for arm in ("D0", "D1", "D2", "D3"):
        common.write(status, {"state": "TRAINING", "arm": arm})
        folder = ROOT / arm
        folder.mkdir(exist_ok=True)
        common.seed()
        model = Features(4, adapters=arm == "D3").cuda().to(memory_format=torch.channels_last)
        initial = torch.load(BASE / "initial.pt", weights_only=True, map_location="cpu")
        model.unet.load_state_dict(initial)
        with torch.no_grad():
            target = teacher(probe["optical"])
        logits = model(probe["x"])
        raw = raw_terms(arm, model, teacher, logits, target, probe["valid"])
        seg = segmentation(logits, probe)
        norms = {}
        for key, value in {"seg": seg, **raw}.items():
            gradients = torch.autograd.grad(value, tuple(model.unet.parameters()),
                                            allow_unused=True, retain_graph=True)
            norm = sum(g.square().sum() for g in gradients if g is not None).sqrt().item()
            assert np.isfinite(norm) and norm > 0, key
            norms[key] = norm
        weights = {}
        for key in raw:
            weight = min(1000., .05 * norms["seg"] / norms[key])
            if key in ("out", "att"):
                weight = shared_weights.setdefault(key, weight)
            weights[key] = weight
        common.write(folder / "calibration.json", {"weights": weights, "gradient_norms": norms,
                     "samples": probe["sample_id"], "temperature": 10})
        del raw, logits, seg, gradients
        model.unet.load_state_dict(initial)  # restore BatchNorm after calibration
        common.seed()
        def objective(student, batch):
            with torch.no_grad():
                t = teacher(batch["optical"])
            s = student(batch["x"])
            losses = raw_terms(arm, student, teacher, s, t, batch["valid"])
            return segmentation(s, batch) + sum(weights[k]*v for k, v in losses.items())
        if not (folder / "selection.json").exists():
            gen = torch.Generator().manual_seed(123)
            fit(model, common.loader(train, True, gen), val, gen, folder, objective)
        evaluate_frozen(model, Inputs(frames["test"], stats), folder)
        del model
        torch.cuda.empty_cache()
    common.write(status, {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(ROOT / "distillation_status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
