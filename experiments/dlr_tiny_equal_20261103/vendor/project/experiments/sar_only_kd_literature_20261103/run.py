"""SAR-only dual-temporal landslide segmentation under optical-change KD.

Baseline + Hinton logit KD + GOLD-style multi-level distillation + M-FSAD-KD
adaptation, all sharing one student graph, input tensor, initial state, split
and training recipe. Test is touched once per frozen model.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_erosion, label
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "dlr_change_guided_cocd_v2"))
sys.path.insert(0, str(ROOT / "cocd"))

from model_kd import (  # noqa: E402
    StudentSystem, attention_transfer, cosine_feature_loss, difference_attention,
    freq_loss, jscad_loss, logit_kd, make_teacher, masked_mse, seg_loss,
)
from data import DLRInputs, numerical_valid, raw, sar_db  # noqa: E402
from windows_main.pico_optimization_v2 import schedule_before_step  # noqa: E402


def make_optimizer(module, prefix=""):
    """Same grouping/ratios as the project recipe, tolerant of a name prefix.

    Encoder stem/stages -> 2e-5, every new layer -> 1e-4, weight decay 0.01 with
    norm/bias/GRN exempt. Reproduced here because ``StudentSystem`` nests the
    deployed network under ``student.``.
    """
    exempt = set()
    for m in module.modules():
        kind = type(m).__name__.lower()
        if any(tag in kind for tag in ("norm", "grn", "globalresponse")):
            exempt.update(id(p) for p in m.parameters(recurse=False))
    groups, seen = {}, set()
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        assert id(param) not in seen
        seen.add(id(param))
        bare = name[len(prefix):] if prefix and name.startswith(prefix) else name
        encoder = bare.startswith(("backbone.stem.", "backbone.stages."))
        no_decay = param.ndim <= 1 or name.endswith(".bias") or id(param) in exempt
        key = ("encoder" if encoder else "new") + ("_no_decay" if no_decay else "_decay")
        if key not in groups:
            lr = 2e-5 if encoder else 1e-4
            groups[key] = dict(params=[], names=[], group_name=key, lr=lr, peak_lr=lr,
                               weight_decay=0.0 if no_decay else 0.01, schedule_step=0)
        groups[key]["params"].append(param)
        groups[key]["names"].append(name)
    assert seen == {id(p) for p in module.parameters() if p.requires_grad}
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8)

BATCH = 16
EPOCHS = 100
VAL_EVERY = 5

ARMS = {
    "base":          {"teacher": None,         "desc": "SAR-only dual-temporal segmentation, no distillation"},
    "logit":         {"teacher": "optical",    "desc": "Hinton response KD only (T=2)"},
    "gold":          {"teacher": "optical",    "desc": "GOLD-style feature+attention+output KD, dynamic weights"},
    "mfsad":         {"teacher": "multimodal", "desc": "M-FSAD-KD adaptation: Bk+Freq+JSCAD+KD, multimodal teacher"},
    "mfsad_oo":      {"teacher": "optical",    "desc": "M-FSAD-KD adaptation with the optical-only teacher"},
    "mfsad_nofreq":  {"teacher": "multimodal", "desc": "M-FSAD-KD without Freq (speckle-band control)"},
}
TEACHERS = {"optical": "OpticalTeacher", "multimodal": "MultimodalTeacher"}

# Pre-declared gradient-share targets, applied to the epoch-1 measured gradient
# norms. The papers publish raw coefficients tied to their own loss scales; a
# gradient-share target is the scale-free equivalent of the same calibration.
SHARES = {
    "base": {},
    "logit": {"kd": 0.050},
    "gold": {"feat": 0.100, "att": 0.050, "kd": 0.040},
    "mfsad": {"bk": 0.100, "freq": 0.080, "jscad": 0.030, "kd": 0.040},
    "mfsad_oo": {"bk": 0.100, "freq": 0.080, "jscad": 0.030, "kd": 0.040},
    "mfsad_nofreq": {"bk": 0.100, "jscad": 0.030, "kd": 0.040},
}


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(50):
        try:
            temp.replace(p)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def seed(n=42):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available()


def frames():
    d = pd.read_csv(HERE / "split_manifest.csv")
    return {s: d[d.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def loader(ds, shuffle=False, generator=None):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True, generator=generator)


def batch_cuda(b):
    return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


def counts(y, p):
    return {"TP": int((y & p).sum()), "FP": int((~y & p).sum()), "FN": int((y & ~p).sum()), "TN": int((~y & ~p).sum())}


def metrics(y, prob):
    y = np.asarray(y, dtype=bool)
    prob = np.asarray(prob)
    c = counts(y, prob >= 0.5)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    return {**c,
            "IoU": tp / max(tp + fp + fn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "Precision": tp / max(tp + fp, 1),
            "Recall": tp / max(tp + fn, 1),
            "AUPRC": float(average_precision_score(y, prob)) if y.any() and (~y).any() else None,
            "foreground_fraction": float(y.mean()),
            "predicted_positive_fraction": float((prob >= 0.5).mean()),
            "background_FP_rate": fp / max(fp + tn, 1),
            "pixels": len(y)}


def boundary_band_f1(y, prob):
    """F1 restricted to a 2-pixel band around the ground-truth boundary."""
    band = binary_dilation(y, iterations=2) & ~binary_erosion(y, iterations=2)
    if band.sum() == 0:
        return None
    return float(metrics(y[band], prob[band])["F1"])


def roughness(y, prob):
    """Mean absolute prediction gradient over true-negative pixels.

    A proxy for speckle-driven high-frequency structure in the output.
    """
    gx = np.abs(np.diff(prob, axis=1))
    gy = np.abs(np.diff(prob, axis=0))
    tot = gx[:-1, :] + gy[:, :-1]
    sel = (y[:-1, :-1] == 0) & (y[1:, :-1] == 0) & (y[:-1, 1:] == 0)
    return float(tot[sel].mean()) if sel.sum() > 0 else None


@torch.no_grad()
def evaluate(arm, model, ds, teacher=None, tkind=None, save=None):
    model.eval()
    probs, ys, events, subsets = [], [], {}, {}
    records, samples, arrays, valids, labels = [], [], [], [], []
    sizes = {k: {"positive_pixels": 0, "detected_pixels": 0, "components": 0, "hit_components": 0}
             for k in ["<=100px", "101..1000px", ">1000px"]}
    band_f1, rough = [], []
    subset_by_id = ds.frame.set_index("sample_id").official_split.to_dict()
    for b in loader(ds):
        b = batch_cuda(b)
        o = model(b["sar"])
        p = o["z"].sigmoid().cpu().numpy()[:, 0]
        y = b["mask"].cpu().numpy()[:, 0] > 0.5
        v = b["valid"].cpu().numpy()[:, 0].astype(bool)
        for j in range(len(y)):
            a, pp, vv = y[j][v[j]], p[j][v[j]], v[j]
            ys.append(a)
            probs.append(pp)
            eid = b["event_id"][j]
            events.setdefault(eid, []).append((a, pp))
            subsets.setdefault(subset_by_id[b["sample_id"][j]], []).append((a, pp))
            m = metrics(a, pp)
            m["boundary_band_F1"] = boundary_band_f1(y[j], p[j])
            m["prediction_roughness_TN"] = roughness(y[j], p[j])
            records.append({"sample_id": b["sample_id"][j], "event_id": eid,
                            "official_subset": subset_by_id[b["sample_id"][j]], **m})
            if m["boundary_band_F1"] is not None:
                band_f1.append(m["boundary_band_F1"])
            if m["prediction_roughness_TN"] is not None:
                rough.append(m["prediction_roughness_TN"])
            if save is not None:
                samples.append(b["sample_id"][j])
                arrays.append(p[j])
                valids.append(v[j])
                labels.append(y[j])
                cc, n = label(y[j] & v[j])
                for k in range(1, n + 1):
                    mask = cc == k
                    area = int(mask.sum())
                    hit = int(((p[j] >= 0.5) & mask).sum())
                    key = "<=100px" if area <= 100 else "101..1000px" if area <= 1000 else ">1000px"
                    sizes[key]["positive_pixels"] += area
                    sizes[key]["detected_pixels"] += hit
                    sizes[key]["components"] += 1
                    sizes[key]["hit_components"] += int(hit >= 0.1 * area)
    result = metrics(np.concatenate(ys), np.concatenate(probs))
    result["per_event"] = {e: metrics(np.concatenate([x[0] for x in a]), np.concatenate([x[1] for x in a]))
                           for e, a in events.items()}
    result["per_official_subset"] = {k: {**metrics(np.concatenate([x[0] for x in a]), np.concatenate([x[1] for x in a])),
                                         "patches": len(a)} for k, a in subsets.items()}
    result["macro_event_IoU"] = float(np.mean([r["IoU"] for r in result["per_event"].values()]))
    result["boundary_band_F1_mean"] = float(np.mean(band_f1)) if band_f1 else None
    result["prediction_roughness_TN_mean"] = float(np.mean(rough)) if rough else None
    result["patches"] = len(ds)
    if save is not None:
        np.savez_compressed(save, sample_ids=np.array(samples), probability=np.stack(arrays),
                            valid=np.stack(valids), label=np.stack(labels))
        result["component_sizes"] = sizes
        pd.DataFrame(records).to_csv(Path(save).with_suffix(".samples.csv"), index=False)
    return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng(r):
    random.setstate(r["python"])
    np.random.set_state(r["numpy"])
    torch.set_rng_state(r["torch"])
    torch.cuda.set_rng_state_all(r["cuda"])


def base_initial():
    p = ROOT / "experiments/pico_v6_100ep_20261101/results/student_initial.pt"
    a = json.loads((ROOT / "configs/active_model.json").read_text())
    assert sha(p) == a["student_initial_sha256"]
    return torch.load(p, weights_only=True, map_location="cpu")


def teacher_forward(kind, teacher, b):
    if kind == "optical":
        return teacher(b["opt_pre"], b["opt_post"])
    return teacher(b["opt_pre"], b["opt_post"], b["sar"])


def student_forward(system, b):
    return system(b["sar"])  # optical tensors are never passed to the student


# --------------------------------------------------------------------------- #
# Per-sample distillation losses (dynamic weight allocation)
# --------------------------------------------------------------------------- #
def logit_kd_ps(zs, zt, valid, t=2.0):
    m = valid.float()[:, 0]
    ls = F.log_softmax(torch.cat([torch.zeros_like(zs), zs], 1) / t, 1)
    pt = F.softmax(torch.cat([torch.zeros_like(zt), zt], 1) / t, 1)
    d = F.kl_div(ls, pt, reduction="none").sum(1) * t ** 2
    return (d * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)


def att_transfer_ps(a_s, a_t, valid):
    m = F.interpolate(valid.float(), size=a_s.shape[-2:], mode="nearest")
    w = a_t * m
    w = w / (w.flatten(1).mean(1)[:, None, None, None] + 1e-6)
    num = ((a_s - a_t) ** 2 * w).flatten(1).sum(1)
    den = (w * m).flatten(1).sum(1).clamp_min(1)
    return num / den


def cosine_ps(a, b, valid):
    m = F.interpolate(valid.float(), size=a.shape[-2:], mode="nearest")
    cos = (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1, keepdim=True)
    return ((1 - cos) * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)


def dynamic_weights(to, valid):
    """GOLD dynamic allocation: emphasise samples where the teacher sees change."""
    a = (to["z"] > 0).float() * valid.float()
    conf = ((2 * to["z"].sigmoid() - 1).abs() * valid.float()).flatten(1).sum(1) / valid.float().flatten(1).sum(1).clamp_min(1)
    w = 1.0 + conf / (conf.mean() + 1e-6)
    return w / w.mean()


# --------------------------------------------------------------------------- #
# Auxiliary terms per arm
# --------------------------------------------------------------------------- #
def aux_terms(arm, system, so, to, b, lam):
    """Return {term: scaled loss} and the raw per-term values for logging."""
    valid = b["valid"]
    raw_terms, total = {}, 0.0
    if arm == "base":
        return raw_terms, total
    if arm == "logit":
        v = logit_kd(so["z"], to["z"], valid)
        raw_terms["kd"] = v
        return raw_terms, lam["kd"] * v
    if arm == "gold":
        w = dynamic_weights(to, valid)
        feats, atts = [], []
        for i in range(3):
            feats.append(cosine_ps(system.proj[i](so["change"][i]), to["change"][i], valid))
            atts.append(att_transfer_ps(difference_attention(so["change"][i]),
                                        difference_attention(to["change"][i]), valid))
        f = (torch.stack(feats).mean(0) * w).mean()
        a = (torch.stack(atts).mean(0) * w).mean()
        k = (logit_kd_ps(so["z"], to["z"], valid) * w).mean()
        raw_terms = {"feat": f, "att": a, "kd": k}
        return raw_terms, lam["feat"] * f + lam["att"] * a + lam["kd"] * k
    # M-FSAD family
    bk = system.bk_loss(so, to, valid)
    freq = 0.0
    if arm == "mfsad":
        freq = torch.stack([freq_loss(system.neck[i](so["change"][i]), to["change"][i], valid) for i in range(3)]).mean()
    jscad = torch.stack([jscad_loss(system.neck[i](so["change"][i]), to["change"][i], valid) for i in range(3)]).mean()
    kdv = logit_kd(so["z"], to["z"], valid)
    raw_terms = {"bk": bk, "jscad": jscad, "kd": kdv}
    total = lam["bk"] * bk + lam["jscad"] * jscad + lam["kd"] * kdv
    if arm == "mfsad":
        raw_terms["freq"] = freq
        total = total + lam["freq"] * freq
    return raw_terms, total


def measure_gradients(arm, system, teacher, tkind, batches, lam=None):
    """Gradient norm of each auxiliary term and of the segmentation term."""
    system.eval()
    norms = {}
    total = 0.0
    n = 0
    for b in batches:
        b = batch_cuda(b)
        so = student_forward(system, b)
        with torch.no_grad():
            to = teacher_forward(tkind, teacher, b) if teacher is not None else None
        term = seg_loss(so["z"], b["mask"], b["valid"])
        system.zero_grad(set_to_none=True)
        term.backward()
        total += float(torch.sqrt(sum((p.grad ** 2).sum() for p in system.parameters()
                                       if p.grad is not None and p.requires_grad)))
        n += 1
    norms["seg"] = total / n
    if arm == "base":
        return norms
    names = list(SHARES[arm].keys())
    for name in names:
        acc = 0.0
        for b in batches:
            b = batch_cuda(b)
            so = student_forward(system, b)
            with torch.no_grad():
                to = teacher_forward(tkind, teacher, b)
            valid = b["valid"]
            if name == "kd":
                val = logit_kd(so["z"], to["z"], valid)
            elif name == "feat":
                val = torch.stack([cosine_feature_loss(system.proj[i](so["change"][i]), to["change"][i], valid)
                                   for i in range(3)]).mean()
            elif name == "att":
                val = torch.stack([attention_transfer(difference_attention(so["change"][i]),
                                                      difference_attention(to["change"][i]), valid)
                                   for i in range(3)]).mean()
            elif name == "bk":
                val = system.bk_loss(so, to, valid)
            elif name == "freq":
                val = torch.stack([freq_loss(system.neck[i](so["change"][i]), to["change"][i], valid)
                                   for i in range(3)]).mean()
            elif name == "jscad":
                val = torch.stack([jscad_loss(system.neck[i](so["change"][i]), to["change"][i], valid)
                                   for i in range(3)]).mean()
            system.zero_grad(set_to_none=True)
            val.backward()
            acc += float(torch.sqrt(sum((p.grad ** 2).sum() for p in system.parameters()
                                        if p.grad is not None and p.requires_grad)))
        norms[name] = acc / max(len(batches), 1)
    return norms


def calibrate(arm, system, teacher, tkind, batches):
    norms = measure_gradients(arm, system, teacher, tkind, batches)
    lam = {}
    for name, share in SHARES[arm].items():
        g = norms.get(name, 0.0)
        lam[name] = float(np.clip(share * norms["seg"] / max(g, 1e-12), 1e-6, 1e3))
    return {"gradient_norms": norms, "shares": SHARES[arm], "lambda": lam}


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def build_teacher(kind):
    seed()
    m = make_teacher(kind, base_initial())
    path = OUT / f"teacher_{kind}_initial.pt"
    if path.exists():
        m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=False)
    return m.cuda()


def build_system(arm):
    seed()
    system = StudentSystem(base_initial(), arm)
    system.student.load_state_dict(torch.load(OUT / "student_initial.pt", weights_only=True, map_location="cpu"))
    return system.cuda()


def freeze_teacher(kind):
    m = build_teacher(kind)
    m.load_state_dict(torch.load(OUT / f"T_{kind}" / "best.pt", weights_only=True, map_location="cpu"))
    m.eval()
    m.requires_grad_(False)
    return m


def prepare():
    setup()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "config.json").exists(), "results/config.json already exists; refusing to overwrite"
    seed()
    s = StudentSystem(base_initial(), "base").student
    torch.save(s.state_dict(), OUT / "student_initial.pt")
    for kind in TEACHERS:
        seed()
        torch.save(make_teacher(kind, base_initial()).state_dict(), OUT / f"teacher_{kind}_initial.pt")
    cfg = {
        "study": "SAR-only dual-temporal landslide segmentation with optical-change distillation",
        "arms": ARMS,
        "teachers": {"optical": "PRE/POST 6-band optical, shared-weight encoder, change head, decoder",
                     "multimodal": "SAR PRE/POST + optical PRE/POST, per-level fusion, change head, decoder"},
        "student": "ConvNeXtV2-Pico shared-weight PRE/POST SAR (VV,VH,OrbitID), 4-level change head, FPN, decoder",
        "student_input": "one orbit, PRE/POST VV+VH + OrbitID; no optical, no counter-orbit, no DEM",
        "epochs": EPOCHS, "validation_every": VAL_EVERY, "early_stopping": False, "batch": BATCH,
        "loss": "masked BCE + 0.5 Dice; KD terms on epoch>=2 with epoch-1 gradient-share calibration",
        "shares": SHARES,
        "schedule": "AdamW pretrained2e-5/new1e-4, wd0.01 excluding norm/bias, first50 constant then cosine to 1%",
        "selection": "best pooled validation IoU at 0.5, evaluated every 5 epochs",
        "test_gate": "all arms frozen, then each tested once; no test-based selection",
        "split_sha256": sha(HERE / "split_manifest.csv"),
        "normalization_sha256": sha(OUT / "normalization.json"),
        "counts": {str(k): int(v) for k, v in
                   pd.read_csv(HERE / "split_manifest.csv").split.value_counts().items()},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          [HERE / "run.py", HERE / "model_kd.py", HERE / "audit.py",
                           ROOT / "experiments/dlr_change_guided_cocd_v2/data.py",
                           ROOT / "cocd/windows_main/pico_optimization_v2.py"]},
        "initial_sha256": {p.name: sha(p) for p in OUT.glob("*_initial.pt")},
        "parameters": {
            "student_deployed": int(sum(p.numel() for p in s.parameters())),
            "optical_teacher": int(sum(p.numel() for p in make_teacher("optical", base_initial()).parameters())),
            "multimodal_teacher": int(sum(p.numel() for p in make_teacher("multimodal", base_initial()).parameters())),
        },
    }
    write(OUT / "config.json", cfg)
    print("PREPARED", cfg["parameters"], flush=True)


def verify():
    c = json.loads((OUT / "config.json").read_text())
    for p, h in c["source_sha256"].items():
        assert sha(ROOT / p) == h, p
    assert sha(HERE / "split_manifest.csv") == c["split_sha256"]
    assert sha(OUT / "normalization.json") == c["normalization_sha256"]
    for name, h in c["initial_sha256"].items():
        assert sha(OUT / name) == h, name


def smoke():
    """Gradient routing, isolation and fit checks before any full run."""
    setup()
    verify()
    stats = json.loads((OUT / "normalization.json").read_text())
    f = frames()
    ds = DLRInputs(f["train"].iloc[:4], stats, include_post=True)
    b = batch_cuda(next(iter(loader(ds))))
    assert "opt_post" in b and "opt_pre" in b

    report = {"state": "PASS", "student_forward_args": ["sar"], "optical_in_student": "absent"}
    # 1. Student interface cannot receive optical data.
    import inspect
    sig = inspect.signature(StudentSystem.forward)
    report["student_signature"] = str(sig)
    assert list(sig.parameters) == ["self", "sar"]

    # 2. Teacher is frozen and untouched by distillation gradients.
    teacher = build_teacher("optical")
    teacher.eval()
    teacher.requires_grad_(False)
    system = build_system("gold")
    so = student_forward(system, b)
    with torch.no_grad():
        to = teacher_forward("optical", teacher, b)
    so["change"][0].retain_grad()
    so["feat_post"][2].retain_grad()
    so["feat_pre"][2].retain_grad()
    _, loss = aux_terms("gold", system, so, to, b, {"feat": 1.0, "att": 1.0, "kd": 1.0})
    system.zero_grad(set_to_none=True)
    loss.backward()
    grads = {"change0": float(so["change"][0].grad.norm()),
             "feat_pre": float(so["feat_pre"][2].grad.norm()),
             "feat_post": float(so["feat_post"][2].grad.norm())}
    assert min(grads.values()) > 0, grads
    report["gold_grad_to_sar_change"] = grads
    report["teacher_grad_none"] = all(p.grad is None for p in teacher.parameters())
    assert report["teacher_grad_none"]

    # 3. M-FSAD gradients reach the SAR encoder (not only the adapters).
    del system, so, to
    gc.collect()
    torch.cuda.empty_cache()
    system = build_system("mfsad")
    so = student_forward(system, b)
    with torch.no_grad():
        mt = build_teacher("multimodal").eval()
        mt.requires_grad_(False)
        to = teacher_forward("multimodal", mt, b)
    so["change"][2].retain_grad()
    _, loss = aux_terms("mfsad", system, so, to, b, {"bk": 1.0, "freq": 1.0, "jscad": 1.0, "kd": 1.0})
    system.zero_grad(set_to_none=True)
    loss.backward()
    enc = [p.grad for n, p in system.student.named_parameters() if n.startswith("backbone.stages.")]
    report["mfsad_grad_to_change_map"] = float(so["change"][2].grad.norm())
    report["mfsad_encoder_params_with_grad"] = int(sum(1 for g in enc if g is not None and torch.count_nonzero(g) > 0))
    report["mfsad_encoder_params_total"] = len(enc)
    assert report["mfsad_grad_to_change_map"] > 0 and report["mfsad_encoder_params_with_grad"] > 0.5 * len(enc)

    # 4. Adapters are training-only: the deployment graph is unaffected.
    sd = system.state_dict()
    student_keys = [k for k in sd if k.startswith("student.")]
    report["deployment_state_keys"] = len(student_keys)
    report["adapter_keys_excluded"] = [k for k in sd if not k.startswith("student.")][:5]

    # 5. Serialization round-trip.
    import io
    buf = io.BytesIO()
    torch.save(system.student.state_dict(), buf)
    buf.seek(0)
    system.student.load_state_dict(torch.load(buf, weights_only=True))
    report["serialization"] = "PASS"

    # 6. Train-only fit: the SAR student must be able to reduce its own loss.
    del system, so, to
    gc.collect()
    torch.cuda.empty_cache()
    fits = {}
    for arm in ("base",):
        seed()
        system = build_system(arm)
        opt = make_optimizer(system, prefix="student.")
        system.eval()
        with torch.no_grad():
            before = float(seg_loss(student_forward(system, b)["z"], b["mask"], b["valid"]))
        system.train()
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            l = seg_loss(student_forward(system, b)["z"], b["mask"], b["valid"])
            l.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        system.eval()
        with torch.no_grad():
            after = float(seg_loss(student_forward(system, b)["z"], b["mask"], b["valid"]))
        assert after < 0.9 * before, (before, after)
        fits[arm] = {"initial_loss": before, "final_loss": after, "updates": 60}
    report["train_only_fit"] = fits
    report["test_loaded"] = False
    write(OUT / "smoke_test.json", report)
    print("SMOKE PASS", json.dumps({k: report[k] for k in
                                    ["gold_grad_to_sar_change", "mfsad_grad_to_change_map",
                                     "mfsad_encoder_params_with_grad", "train_only_fit"]}, indent=2), flush=True)


def train_teacher(kind):
    dest = OUT / f"T_{kind}"
    dest.mkdir(exist_ok=True)
    if (dest / "selection.json").exists():
        return
    m = build_teacher(kind)
    seed()
    f = frames()
    stats = json.loads((OUT / "normalization.json").read_text())
    g = torch.Generator().manual_seed(123)
    dl = loader(DLRInputs(f["train"], stats, include_post=True, augment=True), True, g)
    vd = DLRInputs(f["val"], stats, include_post=True)
    opt = make_optimizer(m)
    history, best, selected, start = [], -1, None, 1
    if (dest / "latest.pt").exists():
        ck = torch.load(dest / "latest.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, best, selected, start = ck["history"], ck["best"], ck["selected"], ck["epoch"] + 1
        set_rng(ck["rng"])
        g.set_state(ck["generator"])
    for epoch in range(start, EPOCHS + 1):
        m.train()
        begin, total, count = time.time(), 0.0, 0
        for j, b in enumerate(dl):
            b = batch_cuda(b)
            schedule_before_step(opt, len(dl))
            opt.zero_grad(set_to_none=True)
            o = teacher_forward(kind, m, b)
            loss = seg_loss(o["z"], b["mask"], b["valid"])
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(b["sar"])
            count += len(b["sar"])
        score = None
        if epoch % VAL_EVERY == 0:
            # evaluate() expects a sar-only callable; teachers need their inputs.
            m.eval()

            @torch.no_grad()
            def _ev():
                probs, ys = [], []
                for vb in loader(vd):
                    vb = batch_cuda(vb)
                    p = teacher_forward(kind, m, vb)["z"].sigmoid().cpu().numpy()[:, 0]
                    y = vb["mask"].cpu().numpy()[:, 0] > 0.5
                    v = vb["valid"].cpu().numpy()[:, 0].astype(bool)
                    for k in range(len(y)):
                        ys.append(y[k][v[k]])
                        probs.append(p[k][v[k]])
                return metrics(np.concatenate(ys), np.concatenate(probs))
            score = _ev()
        if score is not None and score["IoU"] > best:
            best, selected = score["IoU"], epoch
            torch.save(m.state_dict(), dest / "best.pt")
        history.append({"epoch": epoch, "train_loss": total / count,
                        "validation": score, "seconds": time.time() - begin,
                        "lr": [x["lr"] for x in opt.param_groups]})
        write(dest / "history.json", history)
        torch.save({"epoch": epoch, "model": m.state_dict(), "optimizer": opt.state_dict(),
                    "history": history, "best": best, "selected": selected,
                    "rng": rng_state(), "generator": g.get_state()}, dest / "latest.tmp")
        (dest / "latest.tmp").replace(dest / "latest.pt")
        print("T", kind, epoch, "loss", round(total / count, 5),
              "val", None if score is None else round(score["IoU"], 5), flush=True)
    write(dest / "selection.json", {"state": "FROZEN", "epochs_completed": EPOCHS,
                                    "selected_epoch": selected, "validation_IoU": best,
                                    "checkpoint_sha256": sha(dest / "best.pt")})
    del m, dl
    gc.collect()
    torch.cuda.empty_cache()


def train_arm(arm):
    dest = OUT / arm
    dest.mkdir(exist_ok=True)
    if (dest / "selection.json").exists():
        return
    spec = ARMS[arm]
    system = build_system(arm)
    teacher = freeze_teacher(spec["teacher"]) if spec["teacher"] else None
    seed()
    f = frames()
    stats = json.loads((OUT / "normalization.json").read_text())
    g = torch.Generator().manual_seed(123)
    include_post = spec["teacher"] is not None
    dl = loader(DLRInputs(f["train"], stats, include_post=include_post, augment=True), True, g)
    vd = DLRInputs(f["val"], stats, include_post=False)
    opt = make_optimizer(system, prefix="student.")
    history, best, selected, start = [], -1, None, 1
    calib, lam = None, {k: 0.0 for k in SHARES[arm]}
    if (dest / "latest.pt").exists():
        ck = torch.load(dest / "latest.pt", map_location="cpu", weights_only=False)
        system.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history, best, selected, start = ck["history"], ck["best"], ck["selected"], ck["epoch"] + 1
        calib, lam = ck["calibration"], ck["lambda"]
        set_rng(ck["rng"])
        g.set_state(ck["generator"])
    for epoch in range(start, EPOCHS + 1):
        if arm != "base" and calib is None and epoch >= 2:
            calib = calibrate(arm, system, teacher, spec["teacher"],
                              [batch_cuda(x) for x in loader(DLRInputs(f["train"].iloc[:64], stats,
                                                                       include_post=include_post))])
            lam = calib["lambda"]
            write(dest / "calibration.json", calib)
            print(arm, "CALIBRATED", calib["gradient_norms"], lam, flush=True)
        system.train()
        begin, sums, count = time.time(), {}, 0
        for j, b in enumerate(dl):
            b = batch_cuda(b)
            schedule_before_step(opt, len(dl))
            opt.zero_grad(set_to_none=True)
            so = student_forward(system, b)
            seg = seg_loss(so["z"], b["mask"], b["valid"])
            terms = {"seg": seg}
            total = seg
            if teacher is not None and epoch >= 2:
                with torch.no_grad():
                    to = teacher_forward(spec["teacher"], teacher, b)
                raw_terms, aux = aux_terms(arm, system, so, to, b, lam)
                terms.update(raw_terms)
                total = total + aux
            assert torch.isfinite(total)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            n = len(b["sar"])
            count += n
            terms["total"] = total
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * n
            write(OUT / "progress.json", {"state": "TRAINING", "arm": arm, "epoch": epoch,
                                          "epochs": EPOCHS, "batch": j + 1, "batches": len(dl),
                                          "loss_terms": {k: float(v.detach()) for k, v in terms.items()},
                                          "best_val_IoU": best, "test_loaded": False})
        score = evaluate(arm, system, vd) if epoch % VAL_EVERY == 0 else None
        if score is not None and score["IoU"] > best:
            best, selected = score["IoU"], epoch
            torch.save(system.state_dict(), dest / "best.pt")
        history.append({"epoch": epoch, "train": {k: v / count for k, v in sums.items()},
                        "validation": score, "seconds": time.time() - begin,
                        "lambda": lam, "last_preclip_gradient_norm": float(grad),
                        "lr_groups": [{"group": x["group_name"], "lr": x["lr"]} for x in opt.param_groups]})
        write(dest / "history.json", history)
        torch.save({"epoch": epoch, "model": system.state_dict(), "optimizer": opt.state_dict(),
                    "history": history, "best": best, "selected": selected,
                    "calibration": calib, "lambda": lam,
                    "rng": rng_state(), "generator": g.get_state()}, dest / "latest.tmp")
        (dest / "latest.tmp").replace(dest / "latest.pt")
        print(arm, epoch, "seg", round(sums["seg"] / count, 5),
              "val", None if score is None else round(score["IoU"], 5), flush=True)
    write(dest / "selection.json", {"state": "FROZEN", "epochs_completed": EPOCHS,
                                    "selected_epoch": selected, "validation_IoU": best,
                                    "checkpoint_sha256": sha(dest / "best.pt"),
                                    "lambda": lam})
    del system, teacher, dl
    gc.collect()
    torch.cuda.empty_cache()


def weak_change_strata():
    """Pre-declared stratification by SAR change contrast (label-derived only)."""
    f = frames()
    rows = []
    for row in f["test"].itertuples():
        sar, _pre, _post, y = raw(row, False)
        y = np.squeeze(y)
        v = numerical_valid(sar)
        db = sar_db(sar)
        d = np.abs(db[2:4] - db[0:2]).mean(0)
        m = (y > 0.5) & v
        if m.sum() < 50 or (~m).sum() < 50:
            rows.append({"sample_id": row.sample_id, "contrast_db": np.nan, "event_id": row.event_id})
            continue
        rows.append({"sample_id": row.sample_id, "contrast_db": float(d[m].mean() - d[~m].mean()),
                     "event_id": row.event_id})
    return pd.DataFrame(rows)


def final_tests():
    f = frames()
    stats = json.loads((OUT / "normalization.json").read_text())
    for a in ARMS:
        s = json.loads((OUT / a / "selection.json").read_text())
        assert s["epochs_completed"] == EPOCHS and sha(OUT / a / "best.pt") == s["checkpoint_sha256"]
    for a in ARMS:
        if (OUT / a / "test_metrics.json").exists():
            continue
        assert not (OUT / a / "test_started.json").exists(), f"interrupted test for {a}"
        write(OUT / a / "test_started.json", {"all_frozen": True, "checkpoint_sha256": sha(OUT / a / "best.pt")})
        system = build_system(a)
        system.load_state_dict(torch.load(OUT / a / "best.pt", weights_only=True, map_location="cpu"))
        score = evaluate(a, system, DLRInputs(f["test"], stats, include_post=False), save=OUT / a / "test_predictions.npz")
        write(OUT / a / "test_metrics.json", score)
        del system
        torch.cuda.empty_cache()

    # Teacher test scores are reference-only (privileged, not deployable).
    for kind in TEACHERS:
        d = OUT / f"T_{kind}" / "test_metrics.json"
        if d.exists():
            continue
        m = freeze_teacher(kind)
        vd = DLRInputs(f["test"], stats, include_post=True)
        probs, ys = [], []
        for vb in loader(vd):
            vb = batch_cuda(vb)
            p = teacher_forward(kind, m, vb)["z"].sigmoid().cpu().numpy()[:, 0]
            y = vb["mask"].cpu().numpy()[:, 0] > 0.5
            v = vb["valid"].cpu().numpy()[:, 0].astype(bool)
            for k in range(len(y)):
                ys.append(y[k][v[k]])
                probs.append(p[k][v[k]])
        write(d, metrics(np.concatenate(ys), np.concatenate(probs)))
        del m
        torch.cuda.empty_cache()

    rows = []
    for a in ARMS:
        m = json.loads((OUT / a / "test_metrics.json").read_text())
        rows.append({"arm": a, "teacher": ARMS[a]["teacher"] or "-",
                     **{k: m[k] for k in ["IoU", "F1", "AUPRC", "Precision", "Recall"]},
                     "boundary_band_F1": m["boundary_band_F1_mean"],
                     "roughness_TN": m["prediction_roughness_TN_mean"],
                     "small_component_hit": m["component_sizes"]["<=100px"]["hit_components"] / max(m["component_sizes"]["<=100px"]["components"], 1),
                     "small_pixel_recall": m["component_sizes"]["<=100px"]["detected_pixels"] / max(m["component_sizes"]["<=100px"]["positive_pixels"], 1),
                     "mid_pixel_recall": m["component_sizes"]["101..1000px"]["detected_pixels"] / max(m["component_sizes"]["101..1000px"]["positive_pixels"], 1),
                     "large_pixel_recall": m["component_sizes"][">1000px"]["detected_pixels"] / max(m["component_sizes"][">1000px"]["positive_pixels"], 1),
                     "testind_IoU": m["per_official_subset"]["testind"]["IoU"],
                     "testspt_IoU": m["per_official_subset"]["testspt"]["IoU"]})
    for kind in TEACHERS:
        d = json.loads((OUT / f"T_{kind}" / "test_metrics.json").read_text())
        rows.append({"arm": f"T_{kind}", "teacher": "owner",
                     **{k: d[k] for k in ["IoU", "F1", "AUPRC", "Precision", "Recall"]},
                     "boundary_band_F1": None, "roughness_TN": None,
                     "small_component_hit": None, "small_pixel_recall": None,
                     "mid_pixel_recall": None, "large_pixel_recall": None,
                     "testind_IoU": d["per_official_subset"]["testind"]["IoU"],
                     "testspt_IoU": d["per_official_subset"]["testspt"]["IoU"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparison.csv", index=False)

    # Pre-declared weak-change stratification.
    strata = weak_change_strata()
    strata.to_csv(OUT / "test_strata.csv", index=False)
    valid = strata.dropna(subset=["contrast_db"]).sort_values("contrast_db")
    terc = np.array_split(np.arange(len(valid)), 3)
    labels = {}
    for name, idx in zip(["weak", "mid", "strong"], terc):
        for sid in valid.iloc[idx].sample_id:
            labels[sid] = name
    srows = []
    for a in ARMS:
        z = np.load(OUT / a / "test_predictions.npz", allow_pickle=True)
        ids, prob, valid_m, lab = z["sample_ids"], z["probability"], z["valid"], z["label"]
        agg = {}
        for i, sid in enumerate(ids):
            g = labels.get(str(sid))
            if g is None:
                continue
            m = valid_m[i].astype(bool)
            agg.setdefault(g, []).append((lab[i][m] > 0.5, prob[i][m]))
        row = {"arm": a}
        for g, items in agg.items():
            mm = metrics(np.concatenate([x[0] for x in items]), np.concatenate([x[1] for x in items]))
            row[f"{g}_IoU"] = mm["IoU"]
            row[f"{g}_Recall"] = mm["Recall"]
            row[f"{g}_AP"] = mm["AUPRC"]
        srows.append(row)
    pd.DataFrame(srows).to_csv(OUT / "comparison_strata.csv", index=False)
    write(OUT / "progress.json", {"state": "ALL_TRAINING_AND_FINAL_TESTS_COMPLETE",
                                  "arms": list(ARMS), "teachers": list(TEACHERS)})
    write(OUT / "comparison.json", {"table": json.loads(df.to_json(orient="records"))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "smoke", "teachers", "run", "test", "report"])
    args = ap.parse_args()
    try:
        if args.action == "prepare":
            prepare()
        elif args.action == "smoke":
            smoke()
        elif args.action == "teachers":
            setup()
            verify()
            for kind in TEACHERS:
                train_teacher(kind)
        elif args.action == "run":
            setup()
            verify()
            assert json.loads((OUT / "smoke_test.json").read_text())["state"] == "PASS"
            for a in ARMS:
                train_arm(a)
        elif args.action == "test":
            setup()
            final_tests()
        elif args.action == "report":
            final_tests()
    except Exception as e:
        write(OUT / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
        raise
