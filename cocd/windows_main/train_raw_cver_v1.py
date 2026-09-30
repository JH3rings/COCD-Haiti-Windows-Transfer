"""No-Mask / Raw double-orbit -> single-orbit formal experiment (v4 protocol).

Arms (sequential):
  S0-Raw   plain single-orbit Student, full-pixel BCE.
  S2-Raw   verbatim repro of the original successful Search-KD
           (cocd/windows_main/student_search_kd_v2.py arm S2) under the locked
           current protocol: policy JS + same-target-memory evidence cosine,
           G10/G01 query weighting (w = 1 + mask), first-batch calibration,
           KD fraction 0.10.
  CVER-A   Cross-View Evidence Ranking: fixed 3-level x 9-offset candidate pool;
           frozen CG Teacher emits task-relevance utility
           u_T = ReLU(level-attention x grad*activation removal benefit);
           Student light scorer learns p_S; listwise KL(p_T || p_S).
  CVER-B   same pool, but only Teacher-positive candidates (u_T_raw > 0) carry
           target mass in the listwise distribution.

S0-Raw and CVER use no geometry during training; historical S2-Raw calls
``qmask_raw`` and weights KD queries by G10/G01 geometry. Geometry is also
used for post-hoc region metrics (Overall / BothValid / TargetValid / G10 / G01).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))

from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_da_search import pyramid, pyramid_from_encoded  # noqa: E402
from windows_main.models_teacher_target_search_v2 import (  # noqa: E402
    TargetSearchTeacherV2, load_so_backbone)
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402
from windows_main.train_sgri_powc_masked_v4 import (  # noqa: E402
    SEED, BATCH, LR, MAX_EPOCHS, PATIENCE, WD, EPS,
    set_seed, write_json, make_student, loader, spatial_policy, js_distance)

OUT = ROOT / "experiments/windows_main/raw_cver_v1_seed42"
TEACHER = ROOT / "experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
INIT = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
KD_FRACTION = 0.10          # original S2 fraction
CVER_SHARE = 0.075          # CVER initial KD contribution target
TAU = 1.0
LEVELS, K_PER_LEVEL = 3, 9
K_TOTAL = LEVELS * K_PER_LEVEL
OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]  # feature-pixel units per level
ARMS = ("S0-Raw", "S2-Raw", "CVER-A", "CVER-B")


def bce_full(z, y):
    return F.binary_cross_entropy_with_logits(z[:, 0], y)


# ---------------- fixed candidate pool ----------------

def sample_candidates(values, b):
    """Sample the fixed K=27 candidate pool on the P2 query grid -> (b,K,C,32,32)."""
    hq, wq = values[0].shape[-2:]
    feats = []
    for lv, feat in enumerate(values):
        hm, wm = feat.shape[-2:]
        yy = (torch.arange(hq, device=feat.device, dtype=feat.dtype) + 0.5) / hq * hm - 0.5
        xx = (torch.arange(wq, device=feat.device, dtype=feat.dtype) + 0.5) / wq * wm - 0.5
        by, bx = torch.meshgrid(yy, xx, indexing="ij")
        by, bx = by[None].expand(b, -1, -1), bx[None].expand(b, -1, -1)
        for (dy, dx) in OFFS:
            y = (by + dy).clamp(0, hm - 1)
            x = (bx + dx).clamp(0, wm - 1)
            grid = torch.stack(((x + 0.5) / wm * 2.0 - 1.0, (y + 0.5) / hm * 2.0 - 1.0), dim=-1)
            feats.append(F.grid_sample(feat, grid, mode="bilinear",
                                       padding_mode="zeros", align_corners=False))
    return torch.stack(feats, dim=1)


# ---------------- Teacher utility (CVER) ----------------

def teacher_fixed_readout(t, pyr_t, pyr_c, b, y):
    """Uniform fixed-pool readout by the frozen teacher; grad*activation utility.

    Returns u_raw (b,K,hq,wq): level-attention x first-order removal benefit.
    """
    values = tuple(layer(f) for layer, f in zip(t.value_proj, pyr_t))
    with torch.no_grad():
        cand = sample_candidates(values, b)
    R = cand.mean(dim=1).detach().requires_grad_(True)
    z = t.forward_from_cached_R(R, y.shape[-2:])
    loss = bce_full(z, y)
    g, = torch.autograd.grad(loss, R)
    with torch.no_grad():
        out = t.from_pyramids(pyr_t, pyr_c, y.shape[-2:])
        marg = out["policy"]["weights"].sum(2)                    # b,LEVELS,hq,wq
    u = []
    for lv in range(LEVELS):
        m = marg[:, lv][:, None]
        for k in range(K_PER_LEVEL):
            e = cand[:, lv * K_PER_LEVEL + k]
            u.append(m * (g * e).sum(1, keepdim=True) / math.sqrt(g.shape[1]))
    return torch.cat(u, dim=1)                                    # b,K,h,w


def shuffled_counter_utility(t, a, d, y):
    """Teacher utility with batch-rolled (wrong) counters, for the agreement diag."""
    perm = torch.roll(torch.arange(a.shape[0], device=a.device), 1)
    enc = t.backbone.encode(torch.cat((a, d[perm]), dim=0))
    fa = tuple(x.chunk(2, dim=0)[0] for x in enc)
    fd = tuple(x.chunk(2, dim=0)[1] for x in enc)                 # features of d[perm]
    ua = teacher_fixed_readout(t, pyramid_from_encoded(t.backbone, fa), pyramid_from_encoded(t.backbone, fd), a.shape[0], y)
    enc = t.backbone.encode(torch.cat((d, a[perm]), dim=0))
    fd2 = tuple(x.chunk(2, dim=0)[0] for x in enc)
    fa2 = tuple(x.chunk(2, dim=0)[1] for x in enc)                # features of a[perm]
    ud = teacher_fixed_readout(t, pyramid_from_encoded(t.backbone, fd2), pyramid_from_encoded(t.backbone, fa2), d.shape[0], y)
    return torch.cat((ua, ud), 0)


# ---------------- CVER student ----------------

class CVERScorer(nn.Module):
    """Light query/candidate utility scorer (the only added parameters)."""

    def __init__(self, channels=128, hidden=64):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, hidden, 1)
        self.k_proj = nn.Conv2d(channels, hidden, 1)

    def scores(self, f_q, cand):
        b, K, C, h, w = cand.shape
        q = self.q_proj(f_q) / math.sqrt(self.q_proj.out_channels)
        k = self.k_proj(cand.reshape(b * K, C, h, w)).view(b, K, -1, h, w)
        return (q.unsqueeze(1) * k).sum(2)                        # b,K,h,w


def cver_forward(s, scorer, x, weights=None):
    pyr = pyramid(s.backbone, x)
    values = tuple(layer(f) for layer, f in zip(s.value_proj, pyr))
    b = x.shape[0]
    cand = sample_candidates(values, b)
    u = scorer.scores(pyr[0], cand)
    if weights is None:
        p = torch.softmax(u, dim=1)
    else:
        p = weights
    R = (cand * p.unsqueeze(2)).sum(1)
    z = s.decode_R(R, x.shape[-2:])
    return {"z": z, "u": u, "p": p, "R": R}


def kl_dist(logp, p):
    return (p * (p.add(EPS).log() - logp)).sum(1)                 # b,h,w


def spearman_rows(a, b_):
    ra = a.argsort(-1).argsort(-1).float() - a.shape[-1] / 2.0
    rb = b_.argsort(-1).argsort(-1).float() - b_.shape[-1] / 2.0
    num = (ra * rb).sum(-1)
    den = (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(EPS)
    return num / den


def topk_overlap(u1, u2, k=3):
    t1 = u1.topk(k, dim=1).indices
    t2 = u2.topk(k, dim=1).indices
    return t1.eq(t2).any(1).float().mean()


# ---------------- evaluation ----------------

def evaluate_dispatch(s, scorer, ds, arm, intervention=None, teacher=None):
    """Region-store evaluation; CVER arms aggregate through the learned scorer."""
    s.eval()
    if scorer is not None:
        scorer.eval()
    ps, ys, gas, gds = [], [], [], []
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1701)
    with torch.no_grad():
        for a, d, y, ga, gd, _ in loader(ds):
            a, d = a.to(DEV), d.to(DEV)
            if arm.startswith("CVER"):
                oa = cver_forward(s, scorer, a)
                od = cver_forward(s, scorer, d)
                if intervention is not None:
                    w = oa["p"]
                    if intervention == "random":
                        perm = torch.randperm(w.shape[1], generator=gen).to(w.device)
                        w = w[:, perm]
                    elif intervention == "uniform":
                        w = torch.full_like(w, 1.0 / w.shape[1])
                    oa = cver_forward(s, scorer, a, weights=w)
                    od = cver_forward(s, scorer, d, weights=w)
                za, zd = oa["z"], od["z"]
            else:
                za, zd = s(a)["z"], s(d)["z"]
            ps.extend((torch.sigmoid(za[:, 0]).cpu().numpy(),
                       torch.sigmoid(zd[:, 0]).cpu().numpy()))
            ys.extend((y.numpy(), y.numpy()))
            gas.extend((ga.numpy(), gd.numpy()))
            gds.extend((gd.numpy(), ga.numpy()))
    return tuple(np.concatenate(x, axis=0) for x in (ps, ys, gas, gds))


def metric_rows_raw(pred):
    p, y, g, c = pred
    out = {}
    regions = {"Overall": np.ones_like(g, bool),
               "BothValid": (g == 0) & (c == 0),
               "TargetValid": g == 0,
               "G10": (g > 0) & (c == 0),
               "G01": (g == 0) & (c > 0)}
    for name, m in regions.items():
        mm = m.reshape(-1)
        yy = y.reshape(-1)[mm].astype(bool)
        pp = p.reshape(-1)[mm]
        prob = pp >= .5
        tp = int((prob & yy).sum()); fp = int((prob & ~yy).sum()); fn = int((~prob & yy).sum())
        out[name] = {"AUPRC": float(average_precision_score(yy, pp)),
                     "IoU": tp / (tp + fp + fn + EPS),
                     "F1": 2 * tp / (2 * tp + fp + fn + EPS),
                     "pixels": int(mm.sum())}
    return out


def ap_all(pred):
    p, y, _, _ = pred
    return float(average_precision_score(y.reshape(-1).astype(bool), p.reshape(-1)))


# ---------------- S2-Raw (verbatim repro) ----------------

def qmask_raw(ga, gd):
    x = torch.cat(((ga > 0) & ~(gd > 0), (gd > 0) & ~(ga > 0))).float().to(DEV)[:, None]
    return F.max_pool2d(x, kernel_size=4, stride=4)[:, 0] > 0


def wmean(x, m):
    x = x.reshape(x.shape[0], -1); m = m.reshape(m.shape[0], -1)
    w = 1 + m.float()
    return (x * w).sum() / (w.sum() + EPS)


def s2_terms(s, t, sa, sd, ga, gd):
    with torch.no_grad():
        ta, td = t.forward_pair(sa, sd, "counter")
        tp = [ta["policy"], td["policy"]]
    outs = [s(sa), s(sd)]
    ms = [tuple(x.detach() for x in o["values"]) for o in outs]
    sp = [s.policy(s.context(tuple(x.detach() for x in o["T"]))) for o in outs]
    qm = qmask_raw(ga, gd)
    pl, el = [], []
    for m, p, pt in zip(ms, sp, tp):
        P = spatial_policy(p, m)
        T = spatial_policy({k: v.detach() for k, v in pt.items() if k in ("offsets", "weights")}, m)
        pl.append(js_distance(P, T))
        rs = s.search(m, p)
        rt = s.search(m, {k: v.detach() for k, v in pt.items() if k in ("offsets", "weights")})
        el.append(1 - F.cosine_similarity(rs, rt, dim=1))
    lp = .5 * (wmean(pl[0], qm[:len(sa)]) + wmean(pl[1], qm[len(sa):]))
    le = .5 * (wmean(el[0], qm[:len(sa)]) + wmean(el[1], qm[len(sa):]))
    return lp, le, outs


# ---------------- CVER training batch ----------------

def cver_terms(s, scorer, t, a, d, y, arm):
    oa = cver_forward(s, scorer, a)
    od = cver_forward(s, scorer, d)
    seg = .5 * (bce_full(oa["z"], y) + bce_full(od["z"], y))
    with torch.no_grad():
        enc = t.backbone.encode(torch.cat((a, d), dim=0))
        fa = tuple(x.chunk(2, dim=0)[0] for x in enc)
        fd = tuple(x.chunk(2, dim=0)[1] for x in enc)
        pya, pyd = pyramid_from_encoded(t.backbone, fa), pyramid_from_encoded(t.backbone, fd)
    b = a.shape[0]
    ua = teacher_fixed_readout(t, pya, pyd, b, y)
    ud = teacher_fixed_readout(t, pyd, pya, b, y)
    u_raw = torch.cat((ua, ud), 0)                                # 2b,K,h,w detached
    u_pos = F.relu(u_raw)
    p_t = torch.softmax(u_pos / TAU, dim=1)
    u_s = torch.cat((oa["u"], od["u"]), 0)
    logp_s = F.log_softmax(u_s, dim=1)
    if arm == "CVER-A":
        rank = kl_dist(logp_s, p_t).mean()
    else:  # CVER-B: positives only carry target mass
        pos = u_raw > 0
        any_pos = pos.any(1)
        # exp-masked normalisation (never -inf: an all -inf softmax row would
        # yield NaN whose backward pass poisons gradients via NaN*0)
        p_raw = torch.exp(u_raw / TAU) * pos.float()
        p_tb = p_raw / p_raw.sum(1, keepdim=True).clamp_min(EPS)
        kl = kl_dist(logp_s, p_tb)
        rank = kl[any_pos].mean() if bool(any_pos.any()) else (kl * 0).mean()
    diag = {"u_raw": u_raw, "u_s": u_s.detach()}
    return oa, od, seg, rank, diag


# ---------------- training ----------------

def train_arm(name, train, test, teacher):
    out_ckpt = OUT / f"{name}_seed42_best.pt"
    prog_path = OUT / f"{name}_progress.json"
    if out_ckpt.exists() and prog_path.exists() and \
            json.loads(prog_path.read_text(encoding="utf-8")).get("complete"):
        print(f"SKIP complete {name}", flush=True)
        return
    set_seed()
    s = make_student()
    scorer = CVERScorer().to(DEV) if name.startswith("CVER") else None
    params = list(s.parameters()) + (list(scorer.parameters()) if scorer else [])
    opt = torch.optim.Adam(params, lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=WD)
    lam = None
    calib = None
    best, bad, best_epoch, best_state = -1., 0, None, None
    history = []

    for ep in range(1, MAX_EPOCHS + 1):
        s.train()
        if scorer is not None:
            scorer.train()
        totals, mech = [], {}

        def rec(key, val):
            mech.setdefault(key, []).append(float(val))

        for bi, (a, d, y, ga, gd, _real) in enumerate(loader(train, True)):
            a, d, y, ga, gd = [x.to(DEV) for x in (a, d, y, ga, gd)]
            opt.zero_grad(set_to_none=True)

            if name == "S0-Raw":
                oa, od = s(a), s(d)
                seg = .5 * (bce_full(oa["z"], y) + bce_full(od["z"], y))
                loss = seg

            elif name == "S2-Raw":
                lp, le, outs = s2_terms(s, teacher, a, d, ga, gd)
                seg = .5 * (bce_full(outs[0]["z"], y) + bce_full(outs[1]["z"], y))
                if calib is None:
                    calib = {"lp": max(float(lp.detach()), EPS), "le": max(float(le.detach()), EPS),
                             "seg": float(seg.detach()),
                             "note": "original S2 first-batch calibration"}
                    print(f"{name} calibration: {calib}", flush=True)
                kd = .5 * lp / calib["lp"] + .5 * le / calib["le"]
                loss = seg + KD_FRACTION * calib["seg"] * kd
                rec("kd_contribution", (loss - seg).detach() / loss.detach().clamp_min(EPS))
                rec("policy_js", float(lp.detach()))
                rec("evidence_cos", float(le.detach()))

            else:  # CVER-A / CVER-B
                oa, od, seg, rank, diag = cver_terms(s, scorer, teacher, a, d, y, name)
                if calib is None:
                    lam = CVER_SHARE * float(seg.detach()) / max(float(rank.detach()), EPS)
                    calib = {"seg_first_batch": float(seg.detach()),
                             "kd_first_batch": float(rank.detach()), "lambda": lam,
                             "kd_share_target": CVER_SHARE,
                             "note": "single first-batch calibration; frozen"}
                    print(f"{name} calibration: {calib}", flush=True)
                loss = seg + lam * rank
                rec("kd_contribution", (lam * rank.detach()) / loss.detach().clamp_min(EPS))
                rec("rank_loss", float(rank.detach()))
                u_raw, u_s = diag["u_raw"], diag["u_s"]
                p_t = torch.softmax(F.relu(u_raw) / TAU, dim=1)
                rec("t_entropy", float(-(p_t.add(EPS).log() * p_t).sum(1).mean()))
                rec("t_top1", float(p_t.max(1).values.mean()))
                rec("t_top3", float(p_t.topk(3, dim=1).values.sum(1).mean()))
                rec("t_pos_frac", float((u_raw > 0).float().mean()))
                rec("s_spearman", float(spearman_rows(u_s, u_raw).mean()))
                rec("s_top3_overlap", float(topk_overlap(u_s, u_raw)))
                if bi % 4 == 0:
                    u_sh = shuffled_counter_utility(teacher, a, d, y)
                    rec("t_shuffle_spearman", float(spearman_rows(u_raw, u_sh).mean()))

            loss.backward()
            opt.step()
            totals.append(float(loss.detach()))

        pred = evaluate_dispatch(s, scorer, test, name)
        score = ap_all(pred)
        row = {"epoch": ep, "train_loss": float(np.mean(totals)),
               "monitor_test_auprc_raw_overall": score,
               "valid_metrics": metric_rows_raw(pred), "lambda": lam, "calibration": calib,
               "mechanism": {k: (float(np.nanmean(v)) if len(v) else None)
                             for k, v in mech.items()}}
        history.append(row)
        if score > best:
            best, bad, best_epoch = score, 0, ep
            st = {k: v.detach().cpu().clone() for k, v in s.state_dict().items()}
            best_state = st if scorer is None else \
                {"student": st, "scorer": {k: v.detach().cpu().clone()
                                           for k, v in scorer.state_dict().items()}}
        else:
            bad += 1
        payload = {"arm": name, "seed": SEED, "best": best, "best_epoch": best_epoch,
                   "epoch": ep, "patience": PATIENCE, "max_epochs": MAX_EPOCHS,
                   "monitor_risk": "test AUPRC selects checkpoint under canonical project v4 rule",
                   "optimizer": {"name": "Adam", "lr": LR, "betas": [.9, .999], "eps": 1e-8, "wd": WD},
                   "batch_size": BATCH, "setting": "No-Mask / Raw; geometry post-hoc only",
                   "teacher": str(TEACHER), "history": history, "complete": False}
        write_json(prog_path, payload)
        print(f"{name} epoch={ep} monitor_AP={score:.6f} best={best:.6f}@{best_epoch}", flush=True)
        if bad >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    if scorer is not None:
        s.load_state_dict(best_state["student"])
        scorer.load_state_dict(best_state["scorer"])
    else:
        s.load_state_dict(best_state)
    torch.save(best_state, out_ckpt)
    final = {"arm": name, "best_epoch": best_epoch, "monitor_auprc": best,
             "best_metrics": metric_rows_raw(evaluate_dispatch(s, scorer, test, name)),
             "checkpoint": str(out_ckpt)}
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
    teacher = TargetSearchTeacherV2().to(DEV)
    teacher.load_state_dict(torch.load(TEACHER, map_location=DEV, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    write_json(OUT / "protocol.json", {
        "protocol": "active Windows formal v4; train=1370, test=343, no validation; "
                    "test AUPRC monitor; max 100; patience 2",
        "setting": "S0-Raw/CVER: full-pixel segmentation and geometry post-hoc only. "
                   "Historical S2-Raw uses geometry-derived G10/G01 query weights in KD; "
                   "it is not compliant with the later geometry-post-hoc-only rule.",
        "seed": SEED, "train_ids": len(tid), "test_ids": len(eid), "val_ids": len(vid),
        "arms": ARMS,
        "teacher_provenance": {
            "used": str(TEACHER),
            "note": "CG10 was trained by teacher_target_search_v2.py with gamma=0.10 TGD "
                    "geometry-specific supervision. Historical S2-Raw uses CG10. Current "
                    "geometry-free Search-KD runs use CG0; counter enters policy context only.",
            "raw_contrast": {"CG0_vs_SG0": "0.8948 vs 0.8815 (+1.33pp AUPRC)",
                             "CG10_vs_SG10": "0.9012 vs 0.8808 (+2.04pp AUPRC)"}},
        "cver": {"pool": "3 FPN levels x 9 fixed offsets (3x3, feature-pixel units)",
                 "utility": "ReLU(level-attention x grad*activation removal benefit) "
                            "under teacher uniform fixed-pool readout, TAU=1",
                 "A": "listwise KL(p_T||p_S) over all candidates",
                 "B": "listwise KL with target mass on Teacher-positive candidates only",
                 "scorer": "light q/k 1x1-conv dot-product scorer (only added params)",
                 "kd_share_target": CVER_SHARE,
                 "calibration": "single first-batch calibration; frozen (no sweep)"},
        "s2_repro": "historical cocd/windows_main/student_search_kd_v2.py arm S2: policy JS + "
                    "evidence cosine on shared target memory, G10/G01 query weighting "
                    "(w=1+m) via qmask_raw(ga,gd), first-batch calibration, KD fraction 0.10; "
                    "retained as historical reference only",
        "s2_geometry_provenance_correction": "The previous generic No-Mask/Raw setting note "
                    "was too broad: code inspection shows S2 KD query weights depend on "
                    "target/counter geometry. S0-Raw and CVER do not use geometry in KD.",
    })
    for arm in ARMS:
        train_arm(arm, train, test, teacher)
    print("ALL ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
