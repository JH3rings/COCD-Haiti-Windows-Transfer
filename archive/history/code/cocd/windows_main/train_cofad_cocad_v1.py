"""CO-FAD and CO-CAD: two paired-orbit distillation baselines, v4 protocol.

Follows the active Windows formal v4 contract exactly (train=1370, test=343,
no validation, test-AUPRC checkpoint monitor, seed 42, batch 16, Adam lr 5e-5,
wd 0, max 100, patience 2).  Protocol components are imported from the v4
runner so the two arms cannot drift from S0-clean.

CO-FAD  : low-level Haar high-frequency energy consensus on P2 (BothValid only).
CO-CAD  : high-level local affinity consensus on P3 (BothValid only).
Neither adds parameters; inference stays X_t -> Student -> segmentation.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))

from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.train_sgri_powc_masked_v4 import (  # noqa: E402
    SEED, BATCH, LR, MAX_EPOCHS, PATIENCE, WD, EPS,
    set_seed, masked_bce, evaluate, metric_rows, ap_all, write_json,
    make_student, loader)

OUT = ROOT / "experiments/windows_main/cofad_cocad_v1_seed42"
KD_SHARE = 0.075  # initial KD contribution target: 5-10% of seg, one-shot calibration
FAD_RATIO_THRESH = 0.9
CAD_WIN = 7
ARMS = ("CO-FAD", "CO-CAD")


def ratio_pool(mask01: torch.Tensor, size: int) -> torch.Tensor:
    """Valid-ratio pooling of a (B,H,W) 0/1 mask to a size x size feature grid."""
    return F.adaptive_avg_pool2d(mask01.unsqueeze(1), size)[:, 0]


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if x.numel() < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = (xm.norm() * ym.norm()).clamp_min(EPS)
    return float((xm * ym).sum() / denom)


# ---------------- CO-FAD ----------------

def haar_highfreq_energy(p2: torch.Tensor) -> torch.Tensor:
    """Channel-mean Haar high-frequency energy map. (B,C,H,W) -> (B,H/2,W/2)."""
    B, C, H, W = p2.shape
    x = p2.view(B, C, H // 2, 2, W // 2, 2)
    a = x[:, :, :, 0, :, 0]
    b = x[:, :, :, 0, :, 1]
    c = x[:, :, :, 1, :, 0]
    d = x[:, :, :, 1, :, 1]
    lh = 0.5 * (a + b - c - d)
    hl = 0.5 * (a - b + c - d)
    hh = 0.5 * (a - b - c + d)
    return torch.sqrt(lh * lh + hl * hl + hh * hh + EPS).mean(1)


def fad_terms(qa, qd, pair_mask):
    q_star = ((qa + qd) / 2).detach()
    lf = F.smooth_l1_loss(qa, q_star, reduction="none")
    ld = F.smooth_l1_loss(qd, q_star, reduction="none")
    m = pair_mask.float()
    l_a = (lf * m).sum() / m.sum().clamp_min(1.0)
    l_d = (ld * m).sum() / m.sum().clamp_min(1.0)
    return q_star, l_a, l_d


# ---------------- CO-CAD ----------------

def local_affinity(feat: torch.Tensor, win: int = CAD_WIN) -> torch.Tensor:
    """Local cosine affinity within a win x win window. (B,C,H,W) -> (B,H,W,win^2)."""
    B, C, H, W = feat.shape
    pad = win // 2
    fp = F.pad(feat, (pad, pad, pad, pad), mode="replicate")
    nb = fp.unfold(2, win, 1).unfold(3, win, 1)          # B,C,H,W,win,win
    nb = nb.contiguous().view(B, C, H, W, win * win)
    return torch.einsum("bchw,bchwN->bhwN", feat, nb)     # feat is L2-normalized -> cosine


def _unfold_mask(m: torch.Tensor, win: int) -> torch.Tensor:
    """Unfold a (B,H,W) 0/1 mask into (B,H,W,win^2) neighbor-cell values."""
    B, H, W = m.shape
    pad = win // 2
    mp = F.pad(m.unsqueeze(1), (pad, pad, pad, pad), mode="replicate")
    return mp.unfold(2, win, 1).unfold(3, win, 1).reshape(B, H, W, win * win)


def cad_terms(fa, fd, pair_mask, gt_cell):
    fa = F.normalize(fa, dim=1)
    fd = F.normalize(fd, dim=1)
    aa = local_affinity(fa)
    ad = local_affinity(fd)
    a_star = (0.5 * (aa + ad)).detach()
    diff = F.smooth_l1_loss(aa, a_star, reduction="none") + \
        F.smooth_l1_loss(ad, a_star, reduction="none")   # B,H,W,N
    anchor4 = pair_mask.unsqueeze(-1)                     # B,H,W,1
    nb_valid = _unfold_mask(pair_mask.float(), CAD_WIN) > 0.5
    nb_gt = _unfold_mask(gt_cell.float(), CAD_WIN) > 0.5
    gt_a = gt_cell.unsqueeze(-1)
    is_ll = anchor4 & nb_valid & gt_a & nb_gt
    is_lb = anchor4 & nb_valid & (gt_a ^ nb_gt)
    is_bb = anchor4 & nb_valid & ~gt_a & ~nb_gt
    class_losses, class_aff, class_counts = [], {}, {}
    for name, cm in (("ll", is_ll), ("lb", is_lb), ("bb", is_bb)):
        m = cm.float()
        s = m.sum()
        class_counts[name] = float(s)
        if s >= 1:
            class_losses.append(((diff * m).sum() / s.clamp_min(1.0)))
            av = 0.5 * (aa + ad).detach()
            class_aff[name] = float(((av * m).sum() / s.clamp_min(1.0)))
    if not class_losses:  # degenerate batch: no valid pairs
        zero = aa.sum() * 0.0
        class_losses = [zero]
    return a_star, aa, ad, class_losses, class_aff, class_counts


def masked_pairs_diff(x, y, m):
    d = (x - y).abs()
    return float(((d * m.float()).sum() / m.float().sum().clamp_min(1.0)))


# ---------------- training ----------------

def train_arm(name, train, test):
    out_ckpt = OUT / f"{name}_seed42_best.pt"
    prog_path = OUT / f"{name}_progress.json"
    if out_ckpt.exists() and prog_path.exists() and \
            json.loads(prog_path.read_text(encoding="utf-8")).get("complete"):
        print(f"SKIP complete {name}", flush=True)
        return
    set_seed()
    s = make_student()
    opt = torch.optim.Adam(s.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    lam = None
    calib = None
    # Consensus losses are ~0 at initialisation (shared encoder, both orbits seen
    # by identical weights), so a first-batch calibration collapses and the KD
    # share explodes.  Instead: epoch 1 is a pure-segmentation warm-up, then the
    # lambda is calibrated ONCE from the epoch-1 means and frozen for the rest
    # of training.  This still is a single calibration, no sweep.
    warmup_epochs = 1
    seg_sum, kd_sum, kd_batches = 0.0, 0.0, 0
    best, bad, best_epoch, best_state = -1.0, 0, None, None
    history = []

    for ep in range(1, MAX_EPOCHS + 1):
        s.train()
        totals = []
        mech = {}
        energy_pair_d = []

        def rec(key, val):
            mech.setdefault(key, []).append(float(val))

        for a, d, y, ga, gd, real in loader(train, True):
            a, d, y, ga, gd = [x.to(DEV) for x in (a, d, y, ga, gd)]
            real = real.to(DEV).bool()
            opt.zero_grad(set_to_none=True)

            oa, od = s(a), s(d)
            la, _ = masked_bce(oa["z"], y, ga == 0)
            ld_, _ = masked_bce(od["z"], y, gd == 0)
            seg = 0.5 * (la + ld_)

            pair_full = (ga == 0) & (gd == 0)
            bothvalid = pair_full.float()
            gt_real = (y > 0) & real.unsqueeze(-1).unsqueeze(-1)

            if name == "CO-FAD":
                qa = haar_highfreq_energy(oa["T"][0])
                qd = haar_highfreq_energy(od["T"][0])
                pm = ratio_pool(bothvalid, qa.shape[-1]) >= FAD_RATIO_THRESH
                q_star, l_fa, l_fd = fad_terms(qa, qd, pm)
                kd_raw = 0.5 * (l_fa + l_fd)
                rec("fad_loss", kd_raw)
                rec("qcorr_ad", pearson(qa[pm], qd[pm]))
                rec("qcorr_a_star", pearson(qa[pm], q_star[pm]))
                if gt_real.any():
                    gm = ratio_pool(gt_real.float(), qa.shape[-1]) > 0.5
                    if (gm & pm).any() and (pm & ~gm).any():
                        sep_a = (qa[gm & pm].mean() / qa[pm & ~gm].mean().clamp_min(EPS) - 1)
                        sep_d = (qd[gm & pm].mean() / qd[pm & ~gm].mean().clamp_min(EPS) - 1)
                        rec("energy_sep_landslide_rel_A", sep_a)
                        rec("energy_sep_landslide_rel_D", sep_d)
                kd_term = l_fa + l_fd
            else:  # CO-CAD
                pm_cell = ratio_pool(bothvalid, oa["T"][1].shape[-1]) >= FAD_RATIO_THRESH
                gt_cell = ratio_pool(gt_real.float(), oa["T"][1].shape[-1]) > 0.5
                a_star, aa, ad, cls_losses, cls_aff, cls_counts = cad_terms(
                    oa["T"][1], od["T"][1], pm_cell, gt_cell)
                kd_raw = sum(cls_losses) / len(cls_losses)
                rec("cad_loss", kd_raw)
                rec("acorr_ad", pearson(aa[pm_cell], ad[pm_cell]))
                for k in cls_aff:
                    rec(f"aff_{k}", cls_aff[k])
                mech.setdefault("pairs_ll", []).append(cls_counts.get("ll", 0.0))
                mech.setdefault("pairs_lb", []).append(cls_counts.get("lb", 0.0))
                mech.setdefault("pairs_bb", []).append(cls_counts.get("bb", 0.0))
                kd_term = sum(cls_losses)

            if lam is None and ep > warmup_epochs:
                raise RuntimeError("lambda must be calibrated after the warm-up epoch")
            if ep <= warmup_epochs:
                seg_sum += float(seg.detach())
                kd_sum += float(kd_raw.detach())
                kd_batches += 1
                kd_term = kd_term * 0.0  # warm-up: pure segmentation
            loss = seg + (0.0 if lam is None else lam * kd_term)
            if lam is not None:
                rec("kd_contribution", (lam * kd_term.detach()) / loss.detach().clamp_min(EPS))
            loss.backward()
            opt.step()
            totals.append(float(loss.detach()))

        if lam is None and ep >= warmup_epochs:
            mean_seg = seg_sum / max(kd_batches, 1)
            mean_kd = kd_sum / max(kd_batches, 1)
            lam = KD_SHARE * mean_seg / max(mean_kd, EPS)
            calib = {"seg_warmup_mean": mean_seg, "kd_warmup_mean": mean_kd,
                     "lambda": lam, "kd_share_target": KD_SHARE,
                     "warmup_epochs": warmup_epochs,
                     "note": "single calibration from epoch-1 means; frozen for all later epochs"}
            print(f"{name} calibration: {calib}", flush=True)

        pred = evaluate(s, test)
        score = ap_all(pred)
        row = {"epoch": ep, "train_loss": float(np.mean(totals)),
               "monitor_test_auprc_raw_overall": score,
               "valid_metrics": metric_rows(pred),
               "lambda": lam,
               "mechanism": {k: (float(np.nanmean(v)) if len(v) else None)
                             for k, v in mech.items()}}
        history.append(row)
        if score > best:
            best, bad, best_epoch = score, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in s.state_dict().items()}
        else:
            bad += 1
        payload = {"arm": name, "seed": SEED, "best_test_auprc_raw_overall": best,
                   "best_epoch": best_epoch, "epoch": ep, "patience": PATIENCE,
                   "max_epochs": MAX_EPOCHS,
                   "monitor_risk": "test AUPRC selects checkpoint under canonical project v4 rule",
                   "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8,
                                 "weight_decay": WD},
                   "batch_size": BATCH, "calibration": calib,
                   "mask_rule": {"seg": "per-orbit target geom==0", "kd": "BothValid ratio>=0.9 cells",
                                 "kd_share_target": KD_SHARE},
                   "history": history, "complete": False}
        write_json(prog_path, payload)
        print(f"{name} epoch={ep} monitor_AP={score:.6f} best={best:.6f}@{best_epoch} "
              f"kd%={100*row['mechanism'].get('kd_contribution', 0):.2f}", flush=True)
        if bad >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    torch.save(best_state, out_ckpt)
    # post-hoc region metrics at best checkpoint
    s.load_state_dict(best_state)
    final = {"arm": name, "best_epoch": best_epoch, "monitor_auprc": best,
             "best_metrics": metric_rows(evaluate(s, test)), "checkpoint": str(out_ckpt)}
    write_json(OUT / f"{name}_summary.json", final)
    payload = json.loads(prog_path.read_text(encoding="utf-8"))
    payload["complete"] = True
    payload["checkpoint"] = str(out_ckpt)
    write_json(prog_path, payload)
    print(f"COMPLETE {name} best={best:.6f}@{best_epoch}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tid, vid, eid = split()
    if len(vid) != 0:
        raise RuntimeError(f"Expected canonical no-validation split, got {len(vid)} val ids")
    train = HaitiPairs(tid, True)
    test = HaitiPairs(eid, False, train.stats)
    write_json(OUT / "protocol.json", {
        "protocol": "active Windows formal v4; train=1370, test=343, no validation; "
                    "test AUPRC monitor; max 100; patience 2",
        "seed": SEED, "train_ids": len(tid), "test_ids": len(eid), "val_ids": len(vid),
        "train_modes": len(train), "arms": ARMS,
        "method": {"CO-FAD": "P2 Haar high-freq energy consensus, BothValid ratio>=0.9, SmoothL1",
                   "CO-CAD": "P3 L2-normalized 7x7 local affinity consensus, BothValid cells, "
                             "SmoothL1 with ll/lb/bb balance",
                   "kd_share_target": KD_SHARE,
                   "calibration": "epoch-1 pure-seg warm-up; lambda calibrated once from "
                                  "epoch-1 mean seg and KD terms, then frozen (no sweep)",
                   "consensus": "stopgrad mean of the two orbits; symmetric per-orbit terms"},
        "baseline": "S0-clean = S0-mask in sgri_powc_masked_v4_seed42_20260925 (same contract)",
        "inference": "single-orbit student only; no KD branch exists in the model",
    })
    for arm in ARMS:
        train_arm(arm, train, test)
    # final comparison table against S0-clean
    s0p = ROOT / "experiments/windows_main/sgri_powc_masked_v4_seed42_20260925/S0-mask_progress.json"
    s0 = json.loads(s0p.read_text(encoding="utf-8"))
    s0_best = [h for h in s0["history"] if h["epoch"] == s0["best_epoch"]][0]
    table = {"S0-clean": {"best_epoch": s0["best_epoch"], **s0_best["valid_metrics"]}}
    for arm in ARMS:
        p = OUT / f"{arm}_summary.json"
        if p.exists():
            sm = json.loads(p.read_text(encoding="utf-8"))
            table[arm] = {"best_epoch": sm["best_epoch"], **sm["best_metrics"]}
    write_json(OUT / "final_comparison.json", table)
    print(json.dumps(table, indent=2), flush=True)
    print("ALL ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
