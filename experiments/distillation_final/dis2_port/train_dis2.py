from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
DIST = ROOT / "experiments/distillation_final"
OFFICIAL = ROOT / "cocd/third_party/landslide_baselines/dis2_official"
sys.path.insert(0, str(DIST))
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402
    DEV, EPS, MAX_EPOCHS, PATIENCE, SEED, make_loaders, metric_rows,
    protocol_record, seed_everything, write_json, loader,
)
from dlkd_ver4_port import DLKD_ver4  # noqa: E402
from models_bank import criterion_rs  # noqa: E402

OUT = HERE.parent
MISS_NDSM = torch.tensor([True, False])
MISS_RGB = torch.tensor([False, True])


def official_loss(outputs, labels_onehot, epoch):
    full = criterion_rs.softmax_weighted_loss(outputs["full_pred"], labels_onehot) + \
           criterion_rs.dice_loss(outputs["full_pred"], labels_onehot)
    rgb = criterion_rs.softmax_weighted_loss(outputs["rgb_pred"], labels_onehot) + \
          criterion_rs.dice_loss(outputs["rgb_pred"], labels_onehot)
    ndsm = criterion_rs.softmax_weighted_loss(outputs["ndsm_pred"], labels_onehot) + \
           criterion_rs.dice_loss(outputs["ndsm_pred"], labels_onehot)
    missing = criterion_rs.softmax_weighted_loss(outputs["missing_pred"], labels_onehot) + \
              criterion_rs.dice_loss(outputs["missing_pred"], labels_onehot)
    full_scale = sum((criterion_rs.softmax_weighted_loss(pred, labels_onehot) +
                      criterion_rs.dice_loss(pred, labels_onehot))
                     for pred in outputs["full_scale_preds"])
    missing_scale = sum((criterion_rs.softmax_weighted_loss(pred, labels_onehot) +
                         criterion_rs.dice_loss(pred, labels_onehot))
                        for pred in outputs["missing_scale_preds"])
    kl_weight = 5.0 if epoch > 5 else 0.0
    segmentation = full + rgb + ndsm + missing + full_scale + missing_scale
    weighted_kl = kl_weight * outputs["kl_loss"]
    total = segmentation + weighted_kl + outputs["diversity_loss"]
    return total, {"segmentation": segmentation, "full": full, "rgb": rgb,
                   "ndsm": ndsm, "missing": missing,
                   "full_scale": full_scale, "missing_scale": missing_scale,
                   "kl_raw": outputs["kl_loss"], "kl_weight": kl_weight,
                   "kl_weighted": weighted_kl,
                   "diversity": outputs["diversity_loss"]}


def model_input(target, counter):
    return torch.cat([target, counter], dim=1)


def predict(model, dataset):
    model.eval()
    model.is_training = False
    predictions, labels, geom_target, geom_counter = [], [], [], []
    with torch.no_grad():
        for asc, desc, y, ga, gd, _ in loader(dataset):
            asc, desc = asc.to(DEV), desc.to(DEV)
            miss_asc = model(model_input(asc, torch.zeros_like(desc)), [MISS_NDSM.to(DEV)], device=DEV)["seg_pred"]
            miss_desc = model(model_input(torch.zeros_like(asc), desc), [MISS_RGB.to(DEV)], device=DEV)["seg_pred"]
            predictions.extend((miss_asc[:, 1].float().cpu().numpy(),
                                miss_desc[:, 1].float().cpu().numpy()))
            labels.extend((y.numpy(), y.numpy()))
            geom_target.extend((ga.numpy(), gd.numpy()))
            geom_counter.extend((gd.numpy(), ga.numpy()))
    return tuple(np.concatenate(x, axis=0) for x in
                 (predictions, labels, geom_target, geom_counter))


def overall_ap(pred):
    p, y, _, _ = pred
    return float(average_precision_score(y.reshape(-1).astype(bool), p.reshape(-1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    train, test, split_info = make_loaders()
    model = DLKD_ver4(num_classes=2, image_size=(128, 128),
                      ndsm_encoder_channels=5, rgb_encoder_channels=5).to(DEV)
    model.is_training = True
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV.type == "cuda"))
    best, best_epoch, bad_epochs = float("-inf"), None, 0
    best_state, history = None, []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        model.is_training = True
        sums = {k: 0.0 for k in ("total", "segmentation", "kl_raw", "kl_weighted", "diversity")}
        steps = 0
        for asc, desc, y, _ga, _gd, _real in loader(train, shuffle=True):
            asc, desc, y = asc.to(DEV), desc.to(DEV), y.to(DEV)
            labels_onehot = criterion_rs.expand_target(y.long(), n_class=2).float()
            optimizer.zero_grad(set_to_none=True)
            directional = []
            for first, second, miss in ((asc, desc, MISS_NDSM.to(DEV)), (desc, asc, MISS_RGB.to(DEV))):
                joined = model_input(first, second)
                with torch.autocast(device_type="cuda", enabled=(DEV.type == "cuda")):
                    outputs = model(joined, [miss], device=DEV)
                    total, terms = official_loss(outputs, labels_onehot, epoch)
                scaler.scale(total / 2.0).backward()
                directional.append({"total": float(total.detach().float().item()),
                                    **{key: float(value.detach().float().item())
                                       for key, value in terms.items() if key in sums}})
                del outputs, total, terms, joined
            scaler.step(optimizer)
            scaler.update()
            for key in sums:
                vals = [terms[key] for terms in directional]
                sums[key] += sum(vals) / len(vals)
            steps += 1

        means = {k: v / steps for k, v in sums.items()}
        pred = predict(model, test)
        score = overall_ap(pred)
        aux = means["kl_weighted"] + means["diversity"]
        row = {"epoch": epoch, **means,
               "aux_scalar_share": aux / (means["segmentation"] + aux + EPS),
               "test_overall_auprc": score}
        history.append(row)
        if score > best:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT / "DIS2-Port_seed42_best.pt")
        else:
            bad_epochs += 1
        write_json(OUT / "progress.json", {"protocol": protocol_record("DIS2-Port", {
            "split": split_info, "architecture": "official DLKD_ver4 (5+5 channel ASC/DESC)",
            "initialization": "seed-42 random initialization; official DIS2 architecture is not ConvNeXt-compatible",
            "symmetric_orientations_per_batch": ["ASC target / DESC privileged", "DESC target / ASC privileged"],
            "kl_schedule": "weight 0 in epochs 1-5, weight 5 from epoch 6"}),
            "history": history, "best_epoch": best_epoch, "best_test_overall_auprc": best})
        print(f"DIS2-Port epoch={epoch} testAP={score:.6f} aux_share={row['aux_scalar_share']:.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best DIS2 checkpoint produced")
    model.load_state_dict(best_state)
    pred = predict(model, test)
    write_json(OUT / "summary.json", {"method": "DIS2-Port",
        "protocol": protocol_record("DIS2-Port", {"split": split_info,
            "architecture": "official DLKD_ver4 with documented classwise-map append repair"}),
        "best_epoch": best_epoch, "best_test_overall_auprc": best,
        "metrics": metric_rows(pred), "history": history,
        "parameter_count_full_training_model": sum(p.numel() for p in model.parameters()),
        "parameter_count_inference_paths": inference_parameter_count(model)})


def inference_parameter_count(model):
    paths = ("rgb_encoder", "rgb_else", "miss_ndsm_fuse", "miss_ndsm_classwise_attention",
             "miss_ndsm_classwise_decoder", "ndsm_encoder", "ndsm_else", "miss_rgb_fuse",
             "miss_rgb_classwise_attention", "miss_rgb_classwise_decoder")
    return sum(p.numel() for name in paths for p in getattr(model, name).parameters())


if __name__ == "__main__":
    main()
