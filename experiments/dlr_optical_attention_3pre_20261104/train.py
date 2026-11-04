"""Isolated 3+1 experiment using the verified DLR training lifecycle."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "dlr_threelevel_frequency_v2_20261103"
OUT = HERE / "results"
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("frequency_runtime", OLD / "run.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
b = runtime.b
from losses import cosine, output_kd, wcsd
from network import ResidualTeacher, StudentSystem
from model import Adapter
import model_kd
import paper_mfsad

b.HERE = HERE
b.OUT = OUT
runtime.OUT = OUT
b.ARMS = {arm: {"teacher": None if arm == "base" else "residual"}
          for arm in ("base", "MFSAD", "E1", "E3")}
b.SHARES = {"base": {}, "MFSAD": dict.fromkeys(("bk", "freq", "jscad", "kd"), .1),
            "E1": {"att": .05, "out": .05}, "E3": {"att": .05, "out": .05, "wave": .05}}
BaseInputs = b.DLRInputs


class Inputs(BaseInputs):
    def __init__(self, frame, stats, include_post=False, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.cache = []
        mean = np.asarray(stats["sar_mean"], np.float32)[None, :, None, None]
        std = np.asarray(stats["sar_std"], np.float32)[None, :, None, None]
        om = np.asarray(stats["optical_mean"], np.float32)[:, None, None]
        os = np.asarray(stats["optical_std"], np.float32)[:, None, None]
        for row in self.frame.itertuples():
            assert row.orbit in ("ASC", "DES")
            with h5py.File(row.h5_path, "r") as src:
                i = int(row.h5_index)
                sar = np.stack([[src[f"{t}_{p}"][i, 0] for p in ("gVV", "gVH")]
                                for t in ("PRE1", "PRE2", "PRE3", "POST1")])
                valid = (np.isfinite(sar) & (sar > 0)).all(axis=1)
                label = src["None_MASK"][i]
                assert np.isin(label, (0, 1)).all()
                normalized = np.clip((b.sar_db(sar) - mean) / std, -8, 8)
                normalized *= valid[:, None]
                orbit = np.full_like(label, float(row.orbit == "ASC"))
                tensor = np.concatenate((normalized.reshape(8, *label.shape[-2:]), orbit, valid), axis=0)
                item = {"sar": torch.from_numpy(tensor.astype(np.float32)),
                        "mask": torch.from_numpy(label),
                        "valid": torch.from_numpy(valid.all(axis=0, keepdims=True)),
                        "sample_id": row.sample_id, "event_id": row.event_id}
                if include_post:
                    for time, key in (("PRE1", "opt_pre"), ("POST1", "opt_post")):
                        optical = np.stack([src[f"{time}_{band}"][i, 0] for band in
                                            ("B02", "B03", "B04", "B08", "B11", "B12")])
                        assert np.isfinite(optical).mean() > .99 and (optical > 0).any()
                        item[key] = torch.from_numpy(np.clip(
                            (np.nan_to_num(optical, posinf=0, neginf=0) / 10000 - om) / os, -8, 8))
                self.cache.append(item)


def temporal_features(encoder, sar):
    orbit = sar[:, 8:9]
    features = [encoder(torch.cat((sar[:, 2*i:2*i+2], orbit), dim=1)) for i in range(4)]
    reference = []
    for level, post in enumerate(features[3]):
        weights = F.interpolate(sar[:, 9:12], size=post.shape[-2:], mode="area").unsqueeze(2)
        pre = torch.stack([features[i][level] for i in range(3)], dim=1)
        reference.append((weights * pre).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6))
    return reference, features[3]


class Student(model_kd.SARStudent):
    def forward(self, sar):
        pre, post = temporal_features(self.backbone, sar)
        features = self._fpn(pre, post)
        logits = F.interpolate(self.decoder(features[0]), size=sar.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"z": logits, "change": features, "feat_pre": pre, "feat_post": post}


class Teacher(ResidualTeacher):
    def forward(self, optical_pre, optical_post, sar):
        pre, post = temporal_features(self.sar_backbone, sar)
        op, oq = self.backbone(optical_pre), self.backbone(optical_post)
        changes = []
        for i, (a, c, d, e) in enumerate(zip(pre, post, op, oq)):
            ds = self.sar_change[i](torch.cat((a, c, c-a, (c-a).abs()), 1))
            do = self.change[i](torch.cat((d, e, e-d, (e-d).abs()), 1))
            context = torch.cat((ds, do), 1)
            changes.append(ds + self.gates[i](context).sigmoid() * self.residual[i](context))
        pyramid = [None] * 4
        for i in reversed(range(4)):
            lateral = self.lateral[i](changes[i])
            if i < 3:
                lateral = lateral + F.interpolate(pyramid[i+1], size=lateral.shape[-2:], mode="nearest")
            pyramid[i] = self.smooth[i](lateral)
        logits = F.interpolate(self.decoder(pyramid[0]), size=sar.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"z": logits, "E": changes, "change": pyramid, "feat_pre": pre, "feat_post": post}


def fresh_teacher():
    previous = model_kd.PicoEncoder
    model_kd.PicoEncoder = runtime.prior.OriginalEncoder
    try:
        return Teacher(b.base_initial())
    finally:
        model_kd.PicoEncoder = previous


def build_system(arm):
    b.seed()
    student = Student(None)
    student.load_state_dict(torch.load(OUT / "student_initial.pt", weights_only=True))
    model = StudentSystem(student, arm)
    if arm != "MFSAD":
        model.adapters.load_state_dict(torch.load(OUT / "adapter_initial.pt", weights_only=True))
    return model.cuda()


def attention(x, mask):
    x = x.float().abs() * mask
    return x.mean(dim=1).flatten(1), x.sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1)


def terms(arm, system, student, teacher, batch, weights, probe_wave=False):
    if arm == "MFSAD":
        return paper_mfsad.losses("mfsad", system, student, teacher, batch, weights)
    att, wave = [], []
    valid = batch["valid"]
    for adapter, source, target in zip(system.adapters, student["D"], teacher["E"]):
        source, target = adapter(source), target.detach()
        mask = F.interpolate(valid.float(), size=source.shape[-2:], mode="nearest")
        pairs = zip(attention(source, mask), attention(target, mask))
        use = mask.flatten(1).any(dim=1)
        att.append(sum(((1 - F.cosine_similarity(s, t)) * use).sum() /
                       use.sum().clamp_min(1) for s, t in pairs))
        if arm == "E3" or probe_wave:
            wave.append(wcsd(source, target, valid))
    raw = {"att": torch.stack(att).mean(),
           "out": output_kd(student["z"], teacher["z"], valid)}
    if wave:
        raw["wave"] = torch.stack(wave).mean()
    return raw, sum(weights.get(key, 0.) * loss for key, loss in raw.items())


def calibrate(arm, model, teacher, kind, batches):
    path = OUT / "shared_calibration.json"
    if arm == "E3":
        report = json.loads(path.read_text())
    else:
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        with runtime.preserve_rng():
            model.train()
            norms = {}
            for batch in batches:
                student = model(batch["sar"])
                with torch.no_grad():
                    target = b.teacher_forward(kind, teacher, batch)
                raw, _ = terms(arm, model, student, target, batch, {}, probe_wave=arm == "E1")
                raw["seg"] = b.seg_loss(student["z"], batch["mask"], batch["valid"])
                for name, loss in raw.items():
                    gradients = torch.autograd.grad(loss, tuple(model.student.parameters()),
                                                    retain_graph=True, allow_unused=True)
                    norm = sum(g.square().sum() for g in gradients if g is not None).sqrt().item()
                    assert np.isfinite(norm) and norm > 0, name
                    norms[name] = norms.get(name, 0.) + norm / len(batches)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        share = .1 if arm == "MFSAD" else .05
        report = {"gradient_norms": norms, "lambda_all": {
            k: min(1000., share * norms["seg"] / v) for k, v in norms.items() if k != "seg"}}
        if arm == "E1":
            b.write(path, report)
    return {**report, "lambda": {k: report["lambda_all"][k] for k in b.SHARES[arm]}}


b.DLRInputs = Inputs
b.build_system = build_system
b.aux_terms = terms
b.calibrate = calibrate
b.evaluate = runtime.original_evaluate
runtime.fresh_teacher = fresh_teacher
runtime.build_system = build_system


def prepare():
    reg = runtime.resolve(runtime.ROOT)
    for key, dest in [("split", HERE / "split_manifest.csv"),
                      ("normalization", OUT / "normalization.json"),
                      ("initial", OUT / "student_initial.pt")]:
        source = runtime.ROOT / reg["registry"]["locked_artifacts"][key]["path"]
        if not dest.exists():
            shutil.copyfile(source, dest)
    if not (OUT / "teacher_initial.pt").exists():
        b.seed()
        torch.save(fresh_teacher().state_dict(), OUT / "teacher_initial.pt")
    if not (OUT / "adapter_initial.pt").exists():
        b.seed(43)
        torch.save(nn.ModuleList([Adapter(128, 128) for _ in range(4)]).state_dict(),
                   OUT / "adapter_initial.pt")
    assert {key: len(value) for key, value in b.frames().items()} == {"train": 355, "val": 88, "test": 120}
    b.write(OUT / "run_inputs.json", {
        "policy": json.loads((HERE / "protocol.json").read_text()),
        "normalization": "frozen training-only per-polarization statistics shared by all four dates",
        "evaluation_mask": "joint numerical validity of all four SAR dates",
        "hashes": {str(p.relative_to(HERE)): b.sha(p) for p in
                   [HERE / "split_manifest.csv", OUT / "normalization.json", *OUT.glob("*initial.pt")]}
    })


def smoke():
    stats = json.loads((OUT / "normalization.json").read_text())
    data = Inputs(b.frames()["train"].iloc[:2], stats, include_post=True)
    batch = b.batch_cuda(next(iter(b.loader(data))))
    teacher = runtime.build_teacher("residual")
    prediction = b.teacher_forward("residual", teacher, batch)
    b.seg_loss(prediction["z"], batch["mask"], batch["valid"]).backward()
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
               for enc in (teacher.backbone, teacher.sar_backbone))
    teacher.zero_grad(set_to_none=True)
    teacher.eval().requires_grad_(False)
    with torch.no_grad():
        target = b.teacher_forward("residual", teacher, batch)
    for arm in b.ARMS:
        model = build_system(arm)
        pred = model(batch["sar"])
        assert pred["z"].shape == batch["mask"].shape
        seg = b.seg_loss(pred["z"], batch["mask"], batch["valid"])
        aux = 0
        if arm != "base":
            raw, aux = terms(arm, model, pred, target, batch, dict.fromkeys(b.SHARES[arm], .01))
            assert all(torch.isfinite(value) for value in raw.values())
            for value in raw.values():
                gradients = torch.autograd.grad(value, tuple(model.student.parameters()),
                                                allow_unused=True, retain_graph=True)
                assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        (seg + aux).backward()
        assert all(p.grad is None for p in teacher.parameters())
        del model, pred, seg, aux
    b.write(OUT / "checks.json", {"state": "PASS", "train_samples": 2, "test_loaded": False})


def main():
    OUT.mkdir(exist_ok=True)
    b.setup()
    prepare()
    b.write(OUT / "controller_status.json", {"state": "CHECKING"})
    smoke()
    for arm in ("T_residual", *b.ARMS):
        b.write(OUT / "controller_status.json", {"state": "TRAINING", "arm": arm})
        if arm == "T_residual":
            b.train_teacher("residual")
        else:
            b.train_arm(arm)
    b.write(OUT / "controller_status.json", {"state": "EVALUATING"})
    # Reuse the frozen evaluation gate, without the earlier study's report generator.
    import inspect
    finish = inspect.getsource(runtime.finish).split(" import deliver")[0]
    exec(compile(finish, "frozen_evaluation", "exec"), runtime.__dict__)
    runtime.finish()
    b.write(OUT / "complete.json", {"state": "COMPLETE"})
    b.write(OUT / "controller_status.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        b.write(OUT / "controller_status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
