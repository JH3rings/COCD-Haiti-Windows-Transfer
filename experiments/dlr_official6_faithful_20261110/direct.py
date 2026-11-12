"""Direct segmentation baselines (no teacher, no optical), 2026-11-12.

Same 6-channel student input (PRE1/POST1 gVV/gVH + DEM + slope), official
split, recipe, seed 42, validation-IoU selection and one test as the U-Net B0
in run.py.  Only the network changes:
  UNETPP  UNet++ with an ImageNet ResNet-34 encoder (as in Boehm et al. SAR
          adaptation; segmentation_models_pytorch 0.5.0)
  DLV3    DeepLabv3 with an ImageNet ResNet-34 encoder
ImageNet weights: the locally cached torchvision resnet34-b627a593.pth; the
first convolution is extended to 6 channels by cyclic repetition scaled by
3/6 (the smp rule for in_channels > 3).
    python direct.py --arms UNETPP DLV3
    python direct.py --test UNETPP DLV3
"""
import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

import common as C
import run as R

C.RESULTS = C.HERE / "results_direct"
WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet34-b627a593.pth"
NETS = {"UNETPP": smp.UnetPlusPlus, "DLV3": smp.DeepLabV3}


def make(arm):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(C.SEED)
        m = NETS[arm](encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=1)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    w = state["conv1.weight"]
    state["conv1.weight"] = torch.cat([w[:, i % 3:i % 3 + 1] for i in range(6)], 1) * 3 / 6
    r = torch.nn.Module.load_state_dict(m.encoder, state, strict=False)
    assert not r.unexpected_keys and all(k.endswith("num_batches_tracked") for k in r.missing_keys), r
    return m


def frozen(arm):
    m = make(arm)
    sel = C.read(C.RESULTS / arm / "selection.json")
    assert C.sha(C.RESULTS / arm / "best.pt") == sel["checkpoint_sha256"]
    m.load_state_dict(torch.load(C.RESULTS / arm / "best.pt", map_location="cpu", weights_only=True))
    return m.cuda().eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--test", nargs="*")
    a = ap.parse_args()
    C.setup()
    if a.test is not None:
        data = C.Inputs(C.frames()["test"], optical=False)
        for arm in a.test:
            out = C.RESULTS / arm
            if not (out / "test_metrics.json").exists():
                m = frozen(arm)
                C.write(out / "test_started.json", {"arm": arm})
                C.write(out / "test_metrics.json", C.evaluate(lambda b: m(b["x"]).sigmoid(), data,
                                                              save=out / "test_predictions.npz", full=True))
            r = C.read(out / "test_metrics.json")
            print(arm, {k: round(100 * r[k], 2) for k in ("IoU", "F1", "AUPRC")},
                  {k: round(100 * v["IoU"], 2) for k, v in r["per_official_subset"].items()}, flush=True)
        return
    f = C.frames()
    train, val = C.Inputs(f["train"], augment=True, optical=False), C.Inputs(f["val"], optical=False)
    for arm in a.arms:
        m = make(arm).cuda()
        C.write(C.RESULTS / arm / "config.json", {"arm": arm, "net": NETS[arm].__name__, "encoder": "resnet34",
                                                  "weights": str(WEIGHTS), "weights_sha256": C.sha(WEIGHTS),
                                                  "params": sum(p.numel() for p in m.parameters()),
                                                  "seed": C.SEED, "doc": __doc__})
        C.fit(arm, m, train, val, R.seg_only(m), R.student_predict(m))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
