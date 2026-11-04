"""Verify the frozen B0 reference and train the matched four-channel U-Net B1."""
import json
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import evaluate as common
from dataset import FIELDS, Inputs

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REFERENCE = EXPERIMENTS / "dlr_bce_dice_equal_20261103/results/unet"
OUT = HERE / "results/B1"


def segmentation(logits, batch):
    mask, label = batch["valid"].float(), batch["mask"]
    bce = (F.binary_cross_entropy_with_logits(logits, label, reduction="none") * mask).sum()
    probability = logits.sigmoid() * mask
    dims = (1, 2, 3)
    dice = (2 * (probability * label).sum(dims) + 1e-6) / (
        probability.sum(dims) + (mask * label).sum(dims) + 1e-6)
    return bce / mask.sum().clamp_min(1) + (1 - dice).mean()


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def verify_reference():
    selected = json.loads((REFERENCE / "selection.json").read_text())
    history = json.loads((REFERENCE / "history.json").read_text())
    assert len(history) == selected["epochs_completed"] == 100
    assert common.sha(REFERENCE / "best.pt") == selected["checkpoint_sha256"]
    validation = [row for row in history if row.get("validation")]
    assert [row["epoch"] for row in validation] == list(range(5, 101, 5))
    assert max(validation, key=lambda row: row["validation"]["IoU"])["epoch"] == selected["selected_epoch"]
    score = json.loads((REFERENCE / "test_metrics.json").read_text())
    assert score["patches"] == 120 and score["per_official_subset"]["testind"]["patches"] == 44
    for name in ("selection.json", "test_metrics.json"):
        dest = HERE / "results/B0" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REFERENCE / name, dest)


def build():
    model = common.PaperUNet(4)
    original = torch.load(REFERENCE / "initial.pt", map_location="cpu", weights_only=True)
    first = "enc.0.net.0.weight"
    # Match the initial weights of every unchanged layer to B0. The four-channel
    # first convolution uses PyTorch's native initialization for its new fan-in.
    original[first] = model.state_dict()[first]
    original["enc.0.net.0.bias"] = model.state_dict()["enc.0.net.0.bias"]
    model.load_state_dict(original, strict=True)
    return model.cuda().to(memory_format=torch.channels_last)


def baseline_objective(model, batch):
    return segmentation(model(batch["x"]), batch)


def fit(model, dl, val, generator, out, objective):
    opt = common.optimizer(model)
    history, best, selected, start, step = [], -1., None, 1, 0
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history, best, selected, start, step = (state["history"], state["best"],
                                                state["selected"], state["epoch"] + 1, state["step"])
        restore_rng(state["rng"])
        generator.set_state(state["generator"])
    for epoch in range(start, 101):
        model.train()
        running = 0.
        for index, batch in enumerate(dl, 1):
            batch = common.cuda(batch)
            for group in opt.param_groups:
                group["lr"] = 1e-4 * common.lr_factor(step, len(dl))
            opt.zero_grad(set_to_none=True)
            loss = objective(model, batch)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            step += 1
            running += loss.detach().item() * len(batch["x"])
            common.write(out / "progress.json", {"state": "TRAINING", "epoch": epoch,
                         "batch": index, "batches": len(dl), "best_validation_IoU": best})
        score = common.evaluate(model, val) if epoch % 5 == 0 else None
        if score is not None and score["IoU"] > best:
            best, selected = score["IoU"], epoch
            torch.save(model.state_dict(), out / "best.pt")
        history.append({"epoch": epoch, "train_loss": running / len(dl.dataset), "validation": score})
        common.write(out / "history.json", history)
        temporary = latest.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "history": history, "epoch": epoch, "step": step, "best": best,
                    "selected": selected, "rng": common.rng(), "generator": generator.get_state()}, temporary)
        temporary.replace(latest)
        print(out.name, epoch, "val IoU", None if score is None else score["IoU"], flush=True)
    assert step == 2300 and selected is not None
    common.write(out / "selection.json", {"state": "FROZEN", "epochs_completed": 100,
                 "selected_epoch": selected, "validation_IoU": best,
                 "checkpoint_sha256": common.sha(out / "best.pt")})
    return selected, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    common.setup()
    verify_reference()
    manifest = HERE / "split_manifest.csv"
    shutil.copyfile(REFERENCE / "split_manifest.csv", manifest)
    common.HERE = HERE
    frames = common.frames()
    assert {key: len(value) for key, value in frames.items()} == {"train": 355, "val": 88, "test": 120}
    stats = json.loads((REFERENCE / "normalization.json").read_text())
    common.seed()
    model = build()
    initial = OUT / "initial.pt"
    if not initial.exists():
        torch.save(model.state_dict(), initial)
    model.load_state_dict(torch.load(initial, weights_only=True, map_location="cpu"))
    common.write(OUT / "config.json", {
        "fields": FIELDS, "epochs": 100, "validation_every": 5, "batch_size": 16,
        "loss": "masked BCE + per-image foreground soft Dice 1:1",
        "selection": "validation pooled IoU at 0.5",
        "split_sha256": common.sha(manifest),
        "normalization_sha256": common.sha(REFERENCE / "normalization.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_sha256": common.sha(initial),
    })
    generator = torch.Generator().manual_seed(123)
    dl = common.loader(Inputs(frames["train"], stats, augment=True), True, generator)
    val = Inputs(frames["val"], stats)
    selected, best = fit(model, dl, val, generator, OUT, baseline_objective)
    if not (OUT / "test_metrics.json").exists():
        marker = OUT / "test_started.json"
        assert not marker.exists(), "Inspect interrupted final test before repeating"
        common.write(marker, {"selected_epoch": selected})
        model.load_state_dict(torch.load(OUT / "best.pt", weights_only=True, map_location="cpu"))
        score = common.evaluate(model, Inputs(frames["test"], stats), OUT / "test_predictions.npz")
        common.write(OUT / "test_metrics.json", score)
    common.write(OUT / "progress.json", {"state": "COMPLETE", "epochs": 100,
                 "selected_epoch": selected, "best_validation_IoU": best})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.write(OUT / "error.json", {"error": traceback.format_exc()})
        raise
