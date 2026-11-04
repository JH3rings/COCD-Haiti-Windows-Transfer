"""Post-hoc prediction diagnostics and train-only checkpoint gradient probes."""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
import torch

from train_distillation import (Features, OpticalInputs, ROOT, fit_optical,
                                raw_terms, segmentation, common, REFERENCE)

OUT = ROOT.parent / "mechanism_audit"


def prediction_audit(frames):
    frame = frames["test"]
    labels = []
    change = []
    for row in frame.itertuples():
        with h5py.File(row.h5_path, "r") as f:
            i = int(row.h5_index)
            labels.append(f["None_MASK"][i, 0] > .5)
            pre, post = [np.stack([f[f"{t}_g{p}"][i, 0] for p in ("VV", "VH")])
                         for t in ("PRE1", "POST1")]
            change.append(np.abs(10*np.log10(np.maximum(post, 1e-8)) -
                                 10*np.log10(np.maximum(pre, 1e-8))).mean(0))
    y = np.stack(labels)
    change = np.stack(change)
    boundary = np.stack([ndimage.binary_dilation(m, iterations=2) ^
                         ndimage.binary_erosion(m, iterations=2) for m in y])
    pred = {}
    valid = None
    for arm in ("B1", "D0", "D1", "D2", "D3", "Teacher"):
        with np.load(ROOT / arm / "test_predictions.npz") as f:
            assert list(f["sample_ids"]) == frame.sample_id.tolist()
            if valid is None:
                valid = f["valid"].astype(bool)
            assert np.array_equal(valid, f["valid"])
            pred[arm] = f["probability"]
    independent = (frame.official_split == "testind").to_numpy()
    weak_threshold = float(np.quantile(change[y & valid], .25))
    results = {"weak_definition": "retrospective lowest quartile of valid GT-positive mean absolute VV/VH dB change; diagnostic only",
               "weak_threshold_db": weak_threshold, "scopes": {}}
    for scope, use in (("all120", np.ones(len(y), bool)), ("ind44", independent)):
        v = valid & use[:, None, None]
        weak = (change <= weak_threshold) & y & v
        small = []
        for j in np.flatnonzero(use):
            cc, n = ndimage.label(y[j] & v[j])
            for k in range(1, n+1):
                region = cc == k
                if region.sum() <= 100:
                    small.append((j, region))
        entries = {}
        for arm, prob in pred.items():
            entries[arm] = {**common.metrics(y[v], prob[v]),
                            "brier": float(np.mean((prob[v]-y[v])**2)),
                            "boundary_band": common.metrics(y[v & boundary], prob[v & boundary]),
                            "weak_positive_pixels": int(weak.sum()),
                            "weak_recall": float((prob[weak] >= .5).mean()),
                            "small_components": len(small),
                            "small_component_hit_rate": float(np.mean([
                                (prob[j][r] >= .5).mean() >= .1 for j, r in small])),
                            "event_iou": {e: common.metrics(y[v & (frame.event_id == e).to_numpy()[:, None, None]],
                                         prob[v & (frame.event_id == e).to_numpy()[:, None, None]])["IoU"]
                                         for e in frame.event_id[use].unique()}}
        before, after, teacher = (pred[k] >= .5 for k in ("B1", "D0", "Teacher"))
        recovered = v & y & ~before & after
        lost = v & y & before & ~after
        results["scopes"][scope] = {"models": entries, "d0_recovered_positive_pixels": int(recovered.sum()),
                                     "d0_lost_positive_pixels": int(lost.sum()),
                                     "teacher_agrees_on_recovered": float(teacher[recovered].mean())}
    common.write(OUT / "predictions.json", results)


def gradients(frames):
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    optical = json.loads((ROOT / "optical_normalization.json").read_text())
    teacher = Features(12).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(ROOT / "Teacher/best.pt", map_location="cpu", weights_only=True))
    report = {}
    for arm in ("D0", "D1", "D2", "D3"):
        student = Features(4, adapters=arm == "D3").cuda().eval()
        student.load_state_dict(torch.load(ROOT / arm / "best.pt", map_location="cpu", weights_only=True))
        params = list(student.unet.named_parameters())
        group = {"shallow": "enc.0.", "middle": "enc.2.", "deep": "enc.4.", "decoder": "dec."}
        weights = json.loads((ROOT / arm / "calibration.json").read_text())["weights"]
        report[arm] = []
        for offset in (0, 60, 120, 180):
            ds = OpticalInputs(frames["train"].iloc[offset:offset+2], stats, optical)
            batch = common.cuda(next(iter(common.loader(ds))))
            with torch.no_grad():
                t = teacher(batch["optical"])
            s = student(batch["x"])
            values = raw_terms(arm, student, teacher, s, t, batch["valid"])
            values["seg"] = segmentation(s, batch)
            gs = {k: torch.autograd.grad(v, [p for _, p in params], retain_graph=True,
                                         allow_unused=True) for k, v in values.items()}
            record = {"samples": batch["sample_id"], "groups": {}}
            for stage, prefix in group.items():
                ids = [i for i, (name, _) in enumerate(params) if name.startswith(prefix)]
                vec = {k: torch.cat([(g[i] if g[i] is not None else torch.zeros_like(params[i][1])).flatten()
                                     for i in ids]) for k, g in gs.items()}
                record["groups"][stage] = {k: {"weighted_norm_ratio": float(weights[k]*v.norm()/vec["seg"].norm().clamp_min(1e-12)),
                                               "cosine_seg": float(torch.nn.functional.cosine_similarity(v, vec["seg"], dim=0)),
                                               "cosine_output": float(torch.nn.functional.cosine_similarity(v, vec["out"], dim=0))}
                                            for k, v in vec.items() if k != "seg"}
            report[arm].append(record)
            del gs, values, s
    common.write(OUT / "gradients.json", report)


if __name__ == "__main__":
    common.setup()
    frames = common.frames()
    prediction_audit(frames)
    gradients(frames)
