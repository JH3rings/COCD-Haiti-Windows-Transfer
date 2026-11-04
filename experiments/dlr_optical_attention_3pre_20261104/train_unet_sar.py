"""DLR paper U-Net adapted to eight SAR channels and the shared 3+1 protocol."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

import torch
from torch import nn

import train as shared

HERE = Path(__file__).resolve().parent
OUT = HERE / "unet_sar_results"
b = shared.b
source = HERE.parent / "dlr_unet_paper_reproduction" / "baseline.py"
spec = importlib.util.spec_from_file_location("paper_unet", source)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


class SARUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = paper.PaperUNet(channels=8)

    def forward(self, sar):
        # The first eight channels are four VV/VH pairs. Orbit and masks are
        # excluded from the network, and validity still masks loss and scoring.
        return {"z": self.unet(sar[:, :8].contiguous(memory_format=torch.channels_last))}


def build(_arm):
    b.seed()
    return SARUNet().cuda().to(memory_format=torch.channels_last)


def main():
    OUT.mkdir(exist_ok=True)
    b.OUT = OUT
    b.write = shared.runtime.original_write
    b.build_system = build
    b.setup()
    norm = OUT / "normalization.json"
    shutil.copyfile(HERE / "results/normalization.json", norm)
    stats = json.loads(norm.read_text())
    frames = b.frames()
    assert {key: len(value) for key, value in frames.items()} == {"train": 355, "val": 88, "test": 120}
    model = build("base")
    batch = b.batch_cuda(next(iter(b.loader(shared.Inputs(frames["train"].iloc[:2], stats)))))
    loss = b.seg_loss(model(batch["sar"])["z"], batch["mask"], batch["valid"])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    b.write(OUT / "config.json", {
        "input": "PRE1/2/3 and POST1 gVV/gVH, eight channels",
        "architecture": "DLR paper U-Net, random init, all parameters LR 1e-4",
        "protocol": "same 3+1 split, normalization, valid mask, augmentation and BCE+Dice",
        "epochs": 100, "validation_every": 5,
        "parameters": sum(p.numel() for p in model.parameters()),
        "split_sha256": b.sha(HERE / "split_manifest.csv"),
        "architecture_source_sha256": b.sha(source),
    })
    del model, batch, loss
    torch.cuda.empty_cache()
    b.train_arm("base")
    folder = OUT / "base"
    selection = json.loads((folder / "selection.json").read_text())
    checkpoint = folder / "best.pt"
    assert selection["epochs_completed"] == 100 and b.sha(checkpoint) == selection["checkpoint_sha256"]
    if not (folder / "test_metrics.json").exists():
        model = build("base")
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        result = b.evaluate("base", model, shared.Inputs(frames["test"], stats),
                            save=folder / "test_predictions.npz")
        b.write(folder / "test_metrics.json", result)
    b.write(OUT / "complete.json", {"state": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        shared.runtime.original_write(OUT / "error.json", {"error": traceback.format_exc()})
        raise
