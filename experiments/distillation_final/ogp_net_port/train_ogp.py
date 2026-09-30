from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import (  # noqa: E402
    DEV, MAX_EPOCHS, PATIENCE, RAW_DIR, SEED, make_loaders, metric_rows,
    overall_ap, seed_everything, write_json,
)
from ogp_losses import dist_response, segmentation_loss, semi_hard_pixel_contrastive  # noqa: E402
from ogp_model import OGPNetPort, load_backbone_state  # noqa: E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402

OUT = HERE.parent
OGP_BATCH = 8
TAU = 0.2
GAMMA_THRESHOLD = 1e-3
# Keep the published 5e-3 initial rate. A fixed global-norm clip is used because
# the raw, unweighted Eq.12 feature term otherwise overflows this Haiti port.
LR = 5e-3
MOMENTUM = 0.9
MAX_GRAD_NORM = 1.0


def train_loader(dataset):
    return DataLoader(dataset, batch_size=OGP_BATCH, shuffle=True, num_workers=0)


def load_single_teacher():
    path = RAW_DIR / "S0-Raw_seed42_best.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model = StudentTargetSearchV2().to(DEV)
    model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, path


def teacher_features(teacher, x):
    with torch.no_grad():
        feats = teacher.backbone.encode(x)
        c2, c3, c4, c5 = feats
        p5 = teacher.backbone.l5(c5)
        p4 = teacher.backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = teacher.backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = teacher.backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        decoder = teacher.backbone.head[:-1](p2)
        z = teacher.backbone.head[-1](decoder)
    return feats, decoder, z


def horizontal_flip_pair(asc, desc, y):
    flip = torch.rand(asc.shape[0], device=asc.device) < 0.5
    if flip.any():
        asc = asc.clone(); desc = desc.clone(); y = y.clone()
        asc[flip] = asc[flip].flip(-1)
        desc[flip] = desc[flip].flip(-1)
        y[flip] = y[flip].flip(-1)
    return asc, desc, y


def evaluate(model, dataset):
    model.eval()
    pred, labels, gtarget, gcounter = [], [], [], []
    eval_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for asc, desc, y, ga, gd, _ in eval_loader:
            asc, desc = asc.to(DEV), desc.to(DEV)
            za = model.forward_single(asc, "asc")
            zd = model.forward_single(desc, "desc")
            pred.extend((torch.softmax(za, 1)[:, 1].cpu().numpy(),
                         torch.softmax(zd, 1)[:, 1].cpu().numpy()))
            labels.extend((y.numpy(), y.numpy()))
            gtarget.extend((ga.numpy(), gd.numpy()))
            gcounter.extend((gd.numpy(), ga.numpy()))
    arrays = tuple(np.concatenate(items, axis=0) for items in (pred, labels, gtarget, gcounter))
    return arrays, metric_rows(arrays), overall_ap(arrays)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True
    train, test, split_info = make_loaders()
    teacher, teacher_path = load_single_teacher()
    model = OGPNetPort(gamma_threshold=GAMMA_THRESHOLD).to(DEV)
    # Stage-1's frozen target-only segmentation Teacher is also the shared
    # pretrained segmentation initialization for both Stage-2 orbit branches.
    load_backbone_state(model.branch_asc, str(teacher_path))
    load_backbone_state(model.branch_desc, str(teacher_path))
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.0)

    history, best_state = [], None
    best_score, best_epoch, bad_epochs = float("-inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        model.train()
        sums = {k: 0.0 for k in ("seg_main", "seg_aux", "dist_inter", "dist_intra",
                                  "feature_mse", "pixel_contrastive", "total")}
        n_batches = 0
        for asc, desc, y, _ga, _gd, _real in train_loader(train):
            asc, desc, y = asc.to(DEV), desc.to(DEV), y.to(DEV)
            asc, desc, y = horizontal_flip_pair(asc, desc, y)
            out = model.forward_pair(asc, desc)
            with torch.no_grad():
                fa_t, da_t, za_t = teacher_features(teacher, asc)
                fd_t, dd_t, zd_t = teacher_features(teacher, desc)

            # Eq. 12 supervises the GSU fused output plus the auxiliary head;
            # modality logits receive the paper's DUR response distillation.
            seg_main = segmentation_loss(out["z_fused"], y)
            yd = F.interpolate(y[:, None].float(), size=out["aux_asc"].shape[-2:], mode="nearest")[:, 0].long()
            seg_aux = 0.5 * (segmentation_loss(out["aux_asc"], yd) +
                             segmentation_loss(out["aux_desc"], yd))

            di_a, da_a = dist_response(out["z_asc"], za_t[:, 0:1])
            di_d, da_d = dist_response(out["z_desc"], zd_t[:, 0:1])
            dist_inter = 0.5 * (di_a + di_d)
            dist_intra = 0.5 * (da_a + da_d)
            feature_mse = 0.5 * (
                F.mse_loss(out["features_asc"][2], fa_t[2]) +
                F.mse_loss(out["features_asc"][3], fa_t[3]) +
                F.mse_loss(out["d_asc"], da_t) +
                F.mse_loss(out["features_desc"][2], fd_t[2]) +
                F.mse_loss(out["features_desc"][3], fd_t[3]) +
                F.mse_loss(out["d_desc"], dd_t))
            pixel_contrastive = semi_hard_pixel_contrastive(
                out["contrast_features"], y, temperature=TAU)
            loss = seg_main + seg_aux + dist_inter + dist_intra + feature_mse + pixel_contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            for key, value in (("seg_main", seg_main), ("seg_aux", seg_aux),
                               ("dist_inter", dist_inter), ("dist_intra", dist_intra),
                               ("feature_mse", feature_mse), ("pixel_contrastive", pixel_contrastive),
                               ("total", loss)):
                sums[key] += float(value.detach().item())
            n_batches += 1

        means = {k: v / max(n_batches, 1) for k, v in sums.items()}
        _arrays, metrics, score = evaluate(model, test)
        row = {"epoch": epoch, **means, "test_overall_auprc": score,
               "test_metrics": metrics, "channel_exchange_counts_last_batch": model.channel_exchange_counts,
               "epoch_seconds": time.time() - started}
        history.append(row)
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT / "OGP-Net-Port_seed42_best.pt")
        else:
            bad_epochs += 1

        protocol = {
            "arm": "OGP-Net-Port", "seed": SEED, "split": split_info,
            "train_batch_size": OGP_BATCH, "optimizer": {"name": "SGD", "lr": LR,
                "momentum": MOMENTUM, "weight_decay": 0.0,
                "global_grad_norm_clip": MAX_GRAD_NORM,
                "haiti_stability_adaptation": "fixed global norm clip after unbounded Eq.12 update overflowed"}, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "selection_monitor": "test Overall AUPRC (locked project protocol)",
            "stage1_pretrained_teacher": str(teacher_path), "stage1_teacher_frozen": True,
            "stage2_branch_initialization": "same frozen Stage-1 target-only Teacher backbone and decoder, both orbit branches",
            "geometry_use": "post-hoc evaluation only; no geometry input/loss/KD",
            "single_orbit_deployment": True, "DMC": ["shared convolution", "orbit-specific LayerNorm",
                "gamma-guided channel exchange", "odd-column spatial exchange at last two stages",
                "c2-c5 pixel contrastive"],
            "DUR": ["DIST two-class inter/intra Pearson response", "c4/c5/decoder MSE"],
            "GSU": "three gated tanh units on ASC, DESC, and their sum",
            "pixel_contrastive_temperature": TAU,
            "temperature_source": "unresolved in OGP-Net full paper; use cited PCD temperature once, no sweep",
            "gamma_exchange_threshold": GAMMA_THRESHOLD,
            "gamma_threshold_source": "unresolved in OGP-Net full paper; CEN-style near-zero cutoff, fixed once",
            "loss": "Eq.12 unweighted sum; no KD budget controller",
        }
        write_json(OUT / "progress.json", {"protocol": protocol, "history": history,
                    "best_epoch": best_epoch, "best_test_overall_auprc": best_score})
        print(f"OGP-Net-Port epoch={epoch} testAP={score:.6f} total={means['total']:.5f} "
              f"contrast={means['pixel_contrastive']:.5f} best={best_epoch}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("OGP-Net-Port did not produce a best checkpoint")
    model.load_state_dict(best_state)
    _, metrics, score = evaluate(model, test)
    summary = {"method": "OGP-Net-Port", "protocol": protocol,
               "best_epoch": best_epoch, "best_test_overall_auprc": best_score,
               "metrics": metrics, "history": history,
               "single_orbit_parameters": sum(p.numel() for p in model.branch_asc.parameters()),
               "two_branch_training_parameters_unique": sum(p.numel() for p in model.parameters()),
               "full_checkpoint": str(OUT / "OGP-Net-Port_seed42_best.pt")}
    write_json(OUT / "summary.json", summary)
    print(f"OGP-Net-Port COMPLETE best_epoch={best_epoch} bestAP={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
