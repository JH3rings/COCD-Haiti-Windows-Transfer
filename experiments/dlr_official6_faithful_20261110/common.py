"""Shared recipe for the 2026-11-10 DLR official-input faithful comparison.

Input (user-fixed 2026-11-10): official DLR Table-1 setting, PRE1/POST1 gVV/gVH
+ DEM + slope (6 channels).  Teachers additionally receive Sentinel-2 PRE1/POST1
B02/B03/B04/B08/B11/B12 (12 channels).  Official 355/88/120 split unchanged.

Recipe (user-fixed): random dihedral augmentation, 200 epochs, validation
every 20 epochs, seed 42 for every arm.  Fixes applied relative to the
2026-11-04 recipe: deterministic kernels, drop_last (no 3-sample BN batch).
Everything else follows docs/DLR_CURRENT_TRAINING_PROTOCOL.md: BCE + per-image
foreground soft Dice 1:1, AdamW 1e-4, wd 0.01 (no decay on norm/bias), grad
clip 1, first half constant then cosine to 1%, threshold 0.5, validation
pooled IoU selects, test once after every arm is frozen.
"""
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = HERE / "results"
SPLIT = EXP / "dlr_unet_study_20261104" / "split_manifest.csv"
SAR_STATS = EXP / "dlr_bce_dice_equal_20261103" / "results" / "unet" / "normalization.json"
OPT_STATS = EXP / "dlr_unet_study_20261104" / "results" / "optical_normalization.json"

SEED = 42
EPOCHS = 200
VAL_EVERY = 20
BATCH = 16
LR = 1e-4
SAR_FIELDS = ("PRE1_gVV", "PRE1_gVH", "POST1_gVV", "POST1_gVH")
TERRAIN_FIELDS = ("None_DEM", "None_SLOPE")
OPTICAL_FIELDS = tuple(f"{t}_{b}" for t in ("PRE1", "POST1")
                       for b in ("B02", "B03", "B04", "B08", "B11", "B12"))
NDVI_FIELDS = ("PRE1_NDVI", "POST1_NDVI")


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def seed_all(value=SEED):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def setup():
    """Deterministic kernels: same seed -> same run (warn-only for the few CUDA
    ops without a deterministic backward, e.g. bilinear upsampling)."""
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def frames():
    d = pd.read_csv(SPLIT)
    out = {k: d[d.split == k].reset_index(drop=True) for k in ("train", "val", "test")}
    assert {k: len(v) for k, v in out.items()} == {"train": 355, "val": 88, "test": 120}
    return out


class Inputs(Dataset):
    """x: 6-channel student input; optical: 12-channel privileged input (teachers
    and training-only objectives); ndvi: PRE1/POST1 NDVI (training-only)."""

    def __init__(self, frame, augment=False, optical=True):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        s, o = read(SAR_STATS), read(OPT_STATS)
        sar_mean = np.asarray(s["sar_mean"] * 2, np.float32)[:, None, None]
        sar_std = np.asarray(s["sar_std"] * 2, np.float32)[:, None, None]
        ter_mean = np.asarray(s["terrain_mean"], np.float32)[:, None, None]
        ter_std = np.asarray(s["terrain_std"], np.float32)[:, None, None]
        opt_mean = np.asarray(o["mean"], np.float32)[:, None, None]
        opt_std = np.asarray(o["std"], np.float32)[:, None, None]
        self.cache = []
        for row in self.frame.itertuples():
            with h5py.File(row.h5_path, "r") as f:
                i = int(row.h5_index)
                ids = f.attrs["IDs_order"]
                sid = ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i])
                assert sid == row.sample_id
                sar = np.stack([f[k][i, 0] for k in SAR_FIELDS]).astype(np.float32)
                ter = np.stack([f[k][i, 0] for k in TERRAIN_FIELDS]).astype(np.float32)
                label = f["None_MASK"][i].astype(np.float32)
                opt = np.stack([f[k][i, 0] for k in OPTICAL_FIELDS]).astype(np.float32) / 10000 if optical else None
                ndvi = np.stack([f[k][i, 0] for k in NDVI_FIELDS]).astype(np.float32) if optical else None
            assert np.isin(label, (0, 1)).all() and np.isfinite(ter).all()
            valid = (np.isfinite(sar) & (sar > 0)).all(0)
            sar = (10 * np.log10(np.maximum(np.where(np.isfinite(sar), sar, 0), 1e-8)) - sar_mean) / sar_std
            sar[:, ~valid] = 0
            ter = (ter - ter_mean) / ter_std
            x = np.clip(np.concatenate((sar, ter)), -8, 8)
            item = {"x": torch.from_numpy(x), "mask": torch.from_numpy(label),
                    "valid": torch.from_numpy(valid[None]), "sample_id": sid,
                    "event_id": row.event_id, "official_split": row.official_split}
            if optical:
                ok = np.isfinite(opt).all(0)
                opt = np.clip((np.where(np.isfinite(opt), opt, 0) - opt_mean) / opt_std, -8, 8)
                ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
                item["optical"] = torch.from_numpy(opt.astype(np.float32))
                item["ndvi"] = torch.from_numpy(np.clip(ndvi, -1, 1).astype(np.float32))
                item["optical_ok"] = torch.from_numpy(ok[None])
            self.cache.append(item)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        b = dict(self.cache[i])
        if self.augment:
            turns = int(torch.randint(0, 4, ()).item())
            dims = []
            if torch.rand(()) < .5:
                dims.append(-1)
            if torch.rand(()) < .5:
                dims.append(-2)
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    if turns:
                        v = torch.rot90(v, turns, (-2, -1))
                    if dims:
                        v = v.flip(dims)
                    b[k] = v
        return b


def loader(ds, train=False, generator=None):
    return DataLoader(ds, batch_size=BATCH, shuffle=train, drop_last=train,
                      generator=generator, num_workers=0, pin_memory=True)


def cuda(b):
    return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


# --------------------------------------------------------------------------- #
# model: DLR paper U-Net (Figure 7), unchanged body
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """64/128/256/512/1024, four pools, four transpose-conv skip decoders.
    ``forward`` returns logits; the last call's features are kept in
    ``self.enc`` (5 maps, 128->8 px) and ``self.dec`` (4 maps, 16->128 px)."""

    widths = (64, 128, 256, 512, 1024)

    def __init__(self, channels, out=1):
        super().__init__()
        w = self.widths
        self.enc_blocks = nn.ModuleList(DoubleConv(a, b) for a, b in zip((channels,) + w[:-1], w))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, 2) for a, b in zip(w[:0:-1], w[-2::-1]))
        self.dec_blocks = nn.ModuleList(DoubleConv(2 * b, b) for b in w[-2::-1])
        self.head = nn.Conv2d(64, out, 1)
        self.enc, self.dec = [], []

    def features(self, x):
        enc = []
        for i, block in enumerate(self.enc_blocks):
            x = block(F.max_pool2d(x, 2) if i else x)
            enc.append(x)
        dec = []
        for up, block, skip in zip(self.up, self.dec_blocks, enc[-2::-1]):
            x = block(torch.cat((skip, up(x)), 1))
            dec.append(x)
        self.enc, self.dec = enc, dec
        return x

    def forward(self, x):
        return self.head(self.features(x))


def student(out=1):
    return UNet(6, out)


# --------------------------------------------------------------------------- #
# loss, optimiser, schedule
# --------------------------------------------------------------------------- #
def segmentation(logits, batch):
    mask, label = batch["valid"].float(), batch["mask"]
    bce = (F.binary_cross_entropy_with_logits(logits, label, reduction="none") * mask).sum()
    p = logits.sigmoid() * mask
    dims = (1, 2, 3)
    dice = (2 * (p * label).sum(dims) + 1e-6) / (p.sum(dims) + (mask * label).sum(dims) + 1e-6)
    return bce / mask.sum().clamp_min(1) + (1 - dice).mean()


def optimizer(params_named):
    groups = {True: [], False: []}
    for name, p in params_named:
        if p.requires_grad:
            groups[p.ndim <= 1 or name.endswith(".bias")].append(p)
    return torch.optim.AdamW([{"params": groups[False], "weight_decay": .01},
                              {"params": groups[True], "weight_decay": 0.}], lr=LR)


def lr_factor(step, total):
    half = total // 2
    if not 0 <= step < total:
        raise ValueError(step)
    return 1. if step < half else .01 + .99 * (1 + math.cos(math.pi * (step - half) / (total - 1 - half))) / 2


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def metrics(y, p):
    y = np.asarray(y, bool)
    pred = np.asarray(p) >= .5
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "IoU": tp / max(tp + fp + fn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "Precision": tp / max(tp + fp, 1),
            "Recall": tp / max(tp + fn, 1),
            "AUPRC": float(average_precision_score(y, p)) if y.any() and (~y).any() else None,
            "pixels": int(len(y))}


@torch.no_grad()
def evaluate(predict, ds, save=None, full=False):
    """``predict(batch) -> probability [B,1,H,W]``.  Pooled over SAR-valid pixels."""
    ys, ps, groups, saved = [], [], {}, {"ids": [], "p": [], "v": []}
    for b in loader(ds):
        b = cuda(b)
        p = predict(b).float().cpu().numpy()[:, 0]
        y = b["mask"].cpu().numpy()[:, 0] > .5
        v = b["valid"].cpu().numpy()[:, 0].astype(bool)
        for i in range(len(y)):
            yy, pp = y[i][v[i]], p[i][v[i]]
            ys.append(yy)
            ps.append(pp)
            for key in (("event", b["event_id"][i]), ("subset", b["official_split"][i])):
                groups.setdefault(key, []).append((yy, pp))
            if save is not None:
                saved["ids"].append(b["sample_id"][i])
                saved["p"].append(p[i].astype(np.float16))
                saved["v"].append(v[i])
    out = metrics(np.concatenate(ys), np.concatenate(ps))
    out["patches"] = len(ds)
    if full:
        for name, kind in (("per_event", "event"), ("per_official_subset", "subset")):
            out[name] = {k[1]: {**metrics(np.concatenate([a for a, _ in g]), np.concatenate([c for _, c in g])),
                                "patches": len(g)} for k, g in groups.items() if k[0] == kind}
        out["macro_event_IoU"] = float(np.mean([v["IoU"] for v in out["per_event"].values()]))
    if save is not None:
        np.savez_compressed(save, sample_ids=np.array(saved["ids"]), probability=np.stack(saved["p"]),
                            valid=np.stack(saved["v"]))
    return out


# --------------------------------------------------------------------------- #
# training loop shared by every arm
# --------------------------------------------------------------------------- #
def fit(name, model, train, val, objective, predict, extra_params=(), epochs=EPOCHS,
        val_every=VAL_EVERY, on_epoch_start=None):
    """Train ``model`` (plus training-only ``extra_params`` modules) with the
    shared recipe.  ``objective(batch, epoch) -> (loss, logs)``.
    Resumable from ``latest.pt``; writes history/selection/progress."""
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        return read(out / "selection.json")
    modules = nn.ModuleDict({"model": model, **{f"aux{i}": m for i, m in enumerate(extra_params)}})
    opt = optimizer(modules.named_parameters())
    gen = torch.Generator().manual_seed(SEED + 81)
    dl = loader(train, True, gen)
    total = epochs * len(dl)
    history, best, selected, start, step = [], -1., None, 1, 0
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        modules.load_state_dict(state["modules"])
        opt.load_state_dict(state["optimizer"])
        history, best, selected = state["history"], state["best"], state["selected"]
        start, step = state["epoch"] + 1, state["step"]
        restore_rng(state["rng"])
        gen.set_state(state["generator"])
    for epoch in range(start, epochs + 1):
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        modules.train()
        running, logs_sum, t0 = 0., {}, time.time()
        for index, batch in enumerate(dl, 1):
            batch = cuda(batch)
            for g in opt.param_groups:
                g["lr"] = LR * lr_factor(step, total)
            opt.zero_grad(set_to_none=True)
            loss, logs = objective(batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} epoch {epoch} step {step}: {logs}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modules.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            step += 1
            running += float(loss.detach())
            for k, v in logs.items():
                logs_sum[k] = logs_sum.get(k, 0.) + float(v)
        score = None
        if epoch % val_every == 0:
            modules.eval()
            score = evaluate(predict, val)
            if score["IoU"] > best:
                best, selected = score["IoU"], epoch
                torch.save(model.state_dict(), out / "best.pt")
        history.append({"epoch": epoch, "loss": running / len(dl), "seconds": round(time.time() - t0, 2),
                        "terms": {k: v / len(dl) for k, v in logs_sum.items()},
                        "validation": None if score is None else {k: score[k] for k in ("IoU", "F1", "AUPRC")}})
        write(out / "history.json", history)
        tmp = latest.with_suffix(".tmp")
        torch.save({"modules": modules.state_dict(), "optimizer": opt.state_dict(), "history": history,
                    "epoch": epoch, "step": step, "best": best, "selected": selected,
                    "rng": rng_state(), "generator": gen.get_state()}, tmp)
        tmp.replace(latest)
        write(RESULTS / "progress.json", {"arm": name, "epoch": epoch, "epochs": epochs,
                                          "best_validation_IoU": best, "loss": history[-1]["loss"]})
        print(name, epoch, f"loss {history[-1]['loss']:.4f}",
              "" if score is None else f"val IoU {score['IoU']:.4f}", flush=True)
    assert step == total and selected is not None
    torch.save(model.state_dict(), out / "final.pt")
    selection = {"state": "FROZEN", "epochs_completed": epochs, "steps": step, "selected_epoch": selected,
                 "validation_IoU": best, "checkpoint_sha256": sha(out / "best.pt")}
    write(out / "selection.json", selection)
    return selection
