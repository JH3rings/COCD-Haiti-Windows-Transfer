"""Mask-aware Search-KD v5 experiment; canonical model modules stay unchanged.

The existing Teacher/Student parameters are reused.  This runner supplies hard
geometry masks to FPN memory sampling, BothValid KD queries, and masked counter
context.  It deliberately does not add trainable modules or loss terms.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from paths import DEV
from windows_main.data import HaitiPairs, split
from windows_main.models_da_search import pyramid, pyramid_from_encoded
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2, load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.student_search_kd_v2 import js, EPS

OUT = ROOT / "experiments/windows_main/mask_aware_search_kd_v5_seed42"
INIT = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
TEACHER_CKPT = ROOT / "experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
SEED, BATCH, LR, MAX_EPOCHS, PATIENCE, WD = 42, 16, 5e-5, 100, 2, 0.0
KD_TOTAL_FRACTION = 0.10
METHODS = ("S0-clean", "S2-MaskAware")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pixel_valid(geom):
    return geom == 0


def valid_pyramid(geom, shapes):
    """Conservative FPN support: every source pixel in a cell must be valid."""
    invalid = (geom > 0).float().unsqueeze(1)
    result = []
    hp, wp = geom.shape[-2:]
    for h, w in shapes:
        if hp % h or wp % w:
            raise ValueError(f"non-integral geometry/FPN scale: {(hp, wp)} -> {(h, w)}")
        kh, kw = hp // h, wp // w
        bad = F.max_pool2d(invalid, kernel_size=(kh, kw), stride=(kh, kw))[:, 0] > 0
        result.append(~bad)
    return tuple(result)


def query_valid(pixel_mask):
    """A P2 query is valid only if all corresponding 4x4 pixels are valid."""
    return F.avg_pool2d(pixel_mask.float().unsqueeze(1), 4, 4)[:, 0] >= 1.0 - 1e-6


def masked_search(search, memory, policy, memory_valid):
    """Existing deformable search with strict sampled-support gating + renorm."""
    offsets = search.offset_bound * torch.tanh(policy["offsets"])
    weights = policy["weights"]
    hq, wq = memory[0].shape[-2:]
    numerator = torch.zeros_like(memory[0])
    denominator = torch.zeros_like(weights[:, 0, 0])
    total = 0
    valid_count = torch.zeros((), device=memory[0].device, dtype=torch.long)
    for level, feat in enumerate(memory):
        valid_map = memory_valid[level].to(feat.dtype).unsqueeze(1)
        hm, wm = feat.shape[-2:]
        for point in range(search.points):
            grid = search._grid(hq, wq, hm, wm, offsets[:, level, point])
            support = F.grid_sample(valid_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # Remove invalid source cells exactly, then renormalize the tiny
            # interpolation deficit for accepted (>0.999) support locations.
            value_num = F.grid_sample(feat * valid_map, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=False)
            value = value_num / support.clamp_min(EPS)
            reliable = support[:, 0] > 0.999
            w = weights[:, level, point] * reliable.to(weights.dtype)
            # Point 0 at every FPN level is the existing zero-offset centre.
            # Thus any valid centre candidate already participates normally;
            # if all candidates (including centres) fail, the query is marked
            # unusable and is excluded from KD below, with no invalid fallback.
            numerator = numerator + value * w.unsqueeze(1)
            denominator = denominator + w
            total += int(reliable.numel())
            valid_count += reliable.sum()
    valid_count_int = int(valid_count)
    usable = denominator > EPS
    retrieved = numerator / denominator.clamp_min(EPS).unsqueeze(1)
    stats = {
        "sampled_points": total,
        "valid_sampled_points": valid_count_int,
        "masked_sampled_points": total - valid_count_int,
        "mean_valid_samples_per_query": valid_count_int / max(int(usable.numel()), 1),
        "all_invalid_queries": int((~usable).sum()),
        "all_invalid_query_fraction": float((~usable).float().mean()),
        "usable_queries": int(usable.sum()),
    }
    return retrieved, usable, stats


def student_forward(model, x, geom):
    target = pyramid(model.backbone, x)
    masks = valid_pyramid(geom, [tuple(f.shape[-2:]) for f in target])
    context = model.context(target)
    policy = model.policy(context)
    values = tuple(layer(f) for layer, f in zip(model.value_proj, target))
    retrieved, usable, sample_stats = masked_search(model.search, values, policy, masks)
    logits = model.decode_R(retrieved, x.shape[-2:])
    return {"z": logits, "R": retrieved, "policy": policy, "values": values,
            "T": target, "memory_valid": masks, "search_usable": usable,
            "search_stats": sample_stats}


def teacher_from_pyramids(model, target, counter, target_x, target_geom, counter_geom):
    tv = valid_pyramid(target_geom, [tuple(f.shape[-2:]) for f in target])
    cv = valid_pyramid(counter_geom, [tuple(f.shape[-2:]) for f in counter])
    counter_masked = tuple(f * m.unsqueeze(1).to(f.dtype) for f, m in zip(counter, cv))
    context = model.context(target, counter_masked)
    policy = model.policy(context)
    values = tuple(layer(f) for layer, f in zip(model.value_proj, target))
    retrieved, usable, sample_stats = masked_search(model.search, values, policy, tv)
    logits = model._decode_from_search(retrieved, target_x.shape[-2:])
    ratios = [float(m.float().mean()) for m in cv]
    masked_nonzero = [float((f.abs().sum(1) > 0).float().mean()) for f in counter_masked]
    return {"z": logits, "R": retrieved, "policy": policy, "values": values,
            "T": target, "C": counter_masked, "memory_valid": tv,
            "search_usable": usable, "search_stats": sample_stats,
            "counter_valid_ratio": ratios, "counter_nonzero_ratio_after_mask": masked_nonzero}


def teacher_pair(model, a, d, ga, gd):
    encoded = model.backbone.encode(torch.cat((a, d), dim=0))
    fa = tuple(x.chunk(2, dim=0)[0] for x in encoded)
    fd = tuple(x.chunk(2, dim=0)[1] for x in encoded)
    pa, pd = pyramid_from_encoded(model.backbone, fa), pyramid_from_encoded(model.backbone, fd)
    ta = teacher_from_pyramids(model, pa, pd, a, ga, gd)
    td = teacher_from_pyramids(model, pd, pa, d, gd, ga)
    return ta, td


def masked_spatial_policy(policy, memory, memory_valid):
    """Project policy mass only from geometry-valid sampled supports."""
    offsets = 2 * torch.tanh(policy["offsets"])
    weights = policy["weights"]
    b, _, k, _, h, w = offsets.shape
    q = h * w
    yy, xx = torch.meshgrid(torch.arange(h, device=offsets.device),
                            torch.arange(w, device=offsets.device), indexing="ij")
    by, bx = yy.reshape(1, q, 1).float(), xx.reshape(1, q, 1).float()
    candidates, candidate_valid, candidate_xy = [], [], []
    for level, mem in enumerate(memory):
        hm, wm = mem.shape[-2:]
        scale = h / hm
        oy = offsets[:, level, :, 0].permute(0, 2, 3, 1).reshape(b, q, k) * scale
        ox = offsets[:, level, :, 1].permute(0, 2, 3, 1).reshape(b, q, k) * scale
        y = (by + oy).clamp(0, h - 1)
        x = (bx + ox).clamp(0, w - 1)
        ym = (y + 0.5) / h * hm - 0.5
        xm = (x + 0.5) / w * wm - 0.5
        gy = (ym + 0.5) / hm * 2 - 1
        gx = (xm + 0.5) / wm * 2 - 1
        grid = torch.stack((gx, gy), dim=-1).reshape(b, h, w * k, 2)
        vm = F.grid_sample(memory_valid[level][:, None].float(), grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False).reshape(b, h, w, k).reshape(b, q, k)
        candidates.append(weights[:, level].permute(0, 2, 3, 1).reshape(b, q, k))
        candidate_valid.append(vm > 0.999)
        candidate_xy.append((y, x))
    denom = sum((wt * vm.to(wt.dtype)).sum(-1, keepdim=True)
                for wt, vm in zip(candidates, candidate_valid))
    out = torch.zeros(b, q, q, device=offsets.device, dtype=weights.dtype)
    for wt, vm, (y, x) in zip(candidates, candidate_valid, candidate_xy):
        alpha = wt * vm.to(wt.dtype) / denom.clamp_min(EPS)
        y0, x0 = y.floor().long(), x.floor().long()
        fy, fx = y - y0, x - x0
        for yi, xi, coeff in ((y0, x0, (1-fy)*(1-fx)),
                              (y0, (x0+1).clamp(max=w-1), (1-fy)*fx),
                              ((y0+1).clamp(max=h-1), x0, fy*(1-fx)),
                              ((y0+1).clamp(max=h-1), (x0+1).clamp(max=w-1), fy*fx)):
            out.scatter_add_(2, yi*w + xi, alpha*coeff)
    return out, (denom[..., 0] > EPS)


def mask_and_normalize_policy(dist, p2_valid):
    # Spatial policy is [B, query, key]; invalid P2 cells cannot be KD keys.
    d = dist * p2_valid.flatten(1).unsqueeze(1).to(dist.dtype)
    norm = d.sum(-1, keepdim=True)
    row_ok = norm[..., 0] > EPS
    return d / norm.clamp_min(EPS), row_ok


def masked_kd(student, teacher, a, d, ga, gd, outputs=None):
    if outputs is None:
        outputs = [student_forward(student, a, ga), student_forward(student, d, gd)]
    with torch.no_grad():
        targets = teacher_pair(teacher, a, d, ga, gd)
    policies, evidences, kd_masks = [], [], []
    search_usable_pairs = []
    kd_search_stats = {k: 0 for k in ("sampled_points", "valid_sampled_points", "masked_sampled_points",
                                      "all_invalid_queries", "usable_queries")}
    for out, tout in zip(outputs, targets):
        memory = tuple(v.detach() for v in out["values"])
        sp = student.policy(student.context(tuple(v.detach() for v in out["T"])))
        tp = {k: v.detach() for k, v in tout["policy"].items() if k in ("offsets", "weights")}
        ps_dist, ps_support = masked_spatial_policy(sp, memory, out["memory_valid"])
        pt_dist, pt_support = masked_spatial_policy(tp, memory, out["memory_valid"])
        ps, ps_ok = mask_and_normalize_policy(ps_dist, out["memory_valid"][0])
        pt, pt_ok = mask_and_normalize_policy(pt_dist, out["memory_valid"][0])
        ps_ok = ps_ok & ps_support
        pt_ok = pt_ok & pt_support
        policy_map = js(ps, pt)
        rs, rs_usable, rs_stats = masked_search(student.search, memory, sp, out["memory_valid"])
        rt, rt_usable, rt_stats = masked_search(student.search, memory, tp, out["memory_valid"])
        for key in kd_search_stats:
            kd_search_stats[key] += rs_stats[key] + rt_stats[key]
        evidence_map = 1 - F.cosine_similarity(rs, rt.detach(), dim=1)
        policies.append(policy_map)
        evidences.append(evidence_map)
        search_usable_pairs.append((rs_usable, rt_usable))
        kd_masks.append((ps_ok & pt_ok).detach())
    both = query_valid(pixel_valid(ga) & pixel_valid(gd)).flatten(1)
    masks = []
    for i, row_support in enumerate(kd_masks):
        rs_ok, rt_ok = search_usable_pairs[i]
        masks.append(both & row_support & rs_ok.flatten(1) & rt_ok.flatten(1))
    if any(not m.any() for m in masks):
        raise RuntimeError("empty BothValid usable-query set in KD batch")
    p_terms = [p[m].mean() for p, m in zip(policies, masks)]
    e_terms = [e.flatten(1)[m].mean() for e, m in zip(evidences, masks)]
    return (0.5 * (p_terms[0] + p_terms[1]), 0.5 * (e_terms[0] + e_terms[1]),
            outputs, masks, targets, kd_search_stats)


def masked_bce(logits, labels, valid):
    per_pixel = F.binary_cross_entropy_with_logits(logits[:, 0], labels, reduction="none")
    mask = valid.bool()
    return per_pixel[mask].mean() if mask.any() else per_pixel.sum() * 0.0


def segmentation_loss(outputs, labels, geoms):
    return 0.5 * sum(masked_bce(o["z"], labels, pixel_valid(g))
                     for o, g in zip(outputs, geoms))


def make_student():
    model = StudentTargetSearchV2().to(DEV)
    load_so_backbone(model, str(INIT))
    return model


def make_teacher():
    model = TargetSearchTeacherV2().to(DEV)
    model.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEV, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def metric_rows(pred):
    probs, labels, gt, gc = pred
    rows = {}
    masks = {
        "BothValid": (gt == 0) & (gc == 0),
        "TargetValid": gt == 0,
        "OverallValid": (gt == 0) | (gc == 0),
        "TGD_G10": (gt > 0) & (gc == 0),
        "Overall": np.ones_like(gt, dtype=bool),
    }
    for name, mask in masks.items():
        m = mask.reshape(-1)
        y = labels.reshape(-1)[m].astype(bool)
        p = probs.reshape(-1)[m]
        binary = p >= 0.5
        tp = int((binary & y).sum()); fp = int((binary & ~y).sum()); fn = int((~binary & y).sum())
        rows[name] = {"AUPRC": float(average_precision_score(y, p)) if y.any() else None,
                      "IoU": tp / (tp + fp + fn + EPS),
                      "F1": 2 * tp / (2 * tp + fp + fn + EPS),
                      "pixels": int(m.sum())}
    return rows


@torch.no_grad()
def evaluate_student(model, ds):
    model.eval(); ps=[]; ys=[]; gs=[]; cs=[]
    for a, d, y, ga, gd, _ in loader(ds):
        a, d = a.to(DEV), d.to(DEV)
        oa = student_forward(model, a, ga.to(DEV)); od = student_forward(model, d, gd.to(DEV))
        ps.extend((torch.sigmoid(oa["z"][:, 0]).cpu().numpy(), torch.sigmoid(od["z"][:, 0]).cpu().numpy()))
        ys.extend((y.numpy(), y.numpy())); gs.extend((ga.numpy(), gd.numpy())); cs.extend((gd.numpy(), ga.numpy()))
    return tuple(np.concatenate(x, axis=0) for x in (ps, ys, gs, cs))


@torch.no_grad()
def evaluate_diagnostics(model, ds, teacher=None, calibration=None):
    """One final test pass with region, query, sample, KD, and masked-loss stats."""
    model.eval(); ps=[]; ys=[]; gs=[]; cs=[]
    q = {k: 0 for k in ("total", "target_valid", "both_valid", "kd_used", "all_invalid_search")}
    s = {k: 0 for k in ("sampled_points", "valid_sampled_points", "masked_sampled_points", "usable_queries")}
    segs=[]; policies=[]; evidences=[]; counter_ratios=np.zeros(3); counter_nonzero=np.zeros(3); n_counter=0
    kd_samples={k: 0 for k in ("sampled_points", "valid_sampled_points", "masked_sampled_points", "all_invalid_queries", "usable_queries")}
    contrib={k: [] for k in ("seg", "policy", "evidence", "total")}
    for a, d, y, ga, gd, _ in loader(ds):
        a, d, y, ga_d, gd_d = a.to(DEV), d.to(DEV), y.to(DEV), ga.to(DEV), gd.to(DEV)
        outs=[student_forward(model, a, ga_d), student_forward(model, d, gd_d)]
        seg=segmentation_loss(outs, y.to(DEV), (ga_d, gd_d)); segs.append(float(seg))
        for out, geom in zip(outs, (ga_d, gd_d)):
            qv=query_valid(pixel_valid(geom)); q["total"]+=qv.numel(); q["target_valid"]+=int(qv.sum())
            q["all_invalid_search"]+=out["search_stats"]["all_invalid_queries"]
            for key in ("sampled_points", "valid_sampled_points", "masked_sampled_points", "usable_queries"):
                s[key]+=out["search_stats"][key]
        both=query_valid(pixel_valid(ga_d)&pixel_valid(gd_d));q["both_valid"]+=2*int(both.sum())
        if teacher is not None:
            lp, le, _, masks, touts, kd_stats = masked_kd(model, teacher, a, d, ga_d, gd_d, outs)
            policies.append(float(lp)); evidences.append(float(le)); q["kd_used"]+=sum(int(m.sum()) for m in masks)
            for key in kd_samples: kd_samples[key] += kd_stats[key]
            for to in touts:
                counter_ratios += np.asarray(to["counter_valid_ratio"])
                counter_nonzero += np.asarray(to["counter_nonzero_ratio_after_mask"]); n_counter+=1
            if calibration is not None:
                kn=.5*lp/calibration["policy_init"]+.5*le/calibration["evidence_init"]
                lam=(KD_TOTAL_FRACTION/(1-KD_TOTAL_FRACTION))*seg.detach()/(kn.detach()+EPS)
                pc=lam*.5*lp/calibration["policy_init"]; ec=lam*.5*le/calibration["evidence_init"]
                contrib["seg"].append(float(seg));contrib["policy"].append(float(pc))
                contrib["evidence"].append(float(ec));contrib["total"].append(float(seg+pc+ec))
        ps.extend((torch.sigmoid(outs[0]["z"][:,0]).cpu().numpy(),torch.sigmoid(outs[1]["z"][:,0]).cpu().numpy()))
        ys.extend((y.cpu().numpy(),y.cpu().numpy()));gs.extend((ga.cpu().numpy(),gd.cpu().numpy()));cs.extend((gd.cpu().numpy(),ga.cpu().numpy()))
    pred=tuple(np.concatenate(x,axis=0) for x in (ps,ys,gs,cs))
    total=float(np.mean(contrib["total"])) if contrib["total"] else float(np.mean(segs))
    mechanism={"seg_raw_mean":float(np.mean(segs)),"policy_js_raw_mean":float(np.mean(policies)) if policies else None,
               "evidence_distance_raw_mean":float(np.mean(evidences)) if evidences else None,
               "weighted_contribution_mean":{k:float(np.mean(v)) for k,v in contrib.items() if v},
               "weighted_contribution_percent_of_total":({k:100*float(np.mean(contrib[k]))/total for k in ("seg","policy","evidence") if contrib[k]} if contrib["total"] else None)}
    return {"metrics":metric_rows(pred),"query_counts":q,"search_sampling":s,
            "kd_evidence_search_sampling":kd_samples if teacher is not None else None,
            "mean_valid_samples_per_usable_query":s["valid_sampled_points"]/max(s["usable_queries"],1),
            "all_invalid_search_query_fraction":q["all_invalid_search"]/max(q["total"],1),
            "counter_fpn_valid_ratio_by_level":(counter_ratios/max(n_counter,1)).tolist() if n_counter else None,
            "counter_fpn_nonzero_after_mask_by_level":(counter_nonzero/max(n_counter,1)).tolist() if n_counter else None,
            "mechanism":mechanism}


@torch.no_grad()
def evaluate_teacher(model, ds):
    model.eval(); ps=[]; ys=[]; gs=[]; cs=[]
    for a, d, y, ga, gd, _ in loader(ds):
        a, d, ga_d, gd_d = a.to(DEV), d.to(DEV), ga.to(DEV), gd.to(DEV)
        oa, od = teacher_pair(model, a, d, ga_d, gd_d)
        ps.extend((torch.sigmoid(oa["z"][:, 0]).cpu().numpy(), torch.sigmoid(od["z"][:, 0]).cpu().numpy()))
        ys.extend((y.numpy(), y.numpy())); gs.extend((ga.numpy(), gd.numpy())); cs.extend((gd.numpy(), ga.numpy()))
    return tuple(np.concatenate(x, axis=0) for x in (ps, ys, gs, cs))


def overall_ap(pred):
    p, y, _, _ = pred
    return float(average_precision_score(y.reshape(-1).astype(bool), p.reshape(-1)))


def train_arm(name, train, test, teacher=None):
    final = OUT / f"{name}_seed{SEED}_best.pt"
    progress = OUT / f"{name}_progress.json"
    if final.exists() and progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("complete"):
            student = make_student()
            student.load_state_dict(torch.load(final, map_location=DEV, weights_only=True))
            return student, saved
    seed_all()
    student = make_student()
    opt = torch.optim.Adam(student.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    calibration = None; best = -1.0; best_epoch = 0; bad = 0; history = []; best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        student.train(); totals=[]; segs=[]; pols=[]; evidences=[]; weighted=[]; lambdas=[]
        qdiag = {k: 0 for k in ("total", "target_valid", "both_valid", "kd_used", "all_invalid_search")}
        sdiag = {k: 0 for k in ("sampled_points", "valid_sampled_points", "masked_sampled_points", "usable_queries")}
        kd_sdiag = {k: 0 for k in ("sampled_points", "valid_sampled_points", "masked_sampled_points", "all_invalid_queries", "usable_queries")}
        context_valid_sum = np.zeros(3, dtype=np.float64); context_nonzero_sum = np.zeros(3, dtype=np.float64); context_count = 0
        for a, d, y, ga, gd, _ in loader(train, True):
            a, d, y, ga, gd = [x.to(DEV) for x in (a, d, y, ga, gd)]
            outs = [student_forward(student, a, ga), student_forward(student, d, gd)]
            seg = segmentation_loss(outs, y, (ga, gd)); loss = seg
            for out, g in zip(outs, (ga, gd)):
                q = query_valid(pixel_valid(g)); qdiag["total"] += q.numel(); qdiag["target_valid"] += int(q.sum())
                sdiag["sampled_points"] += out["search_stats"]["sampled_points"]
                sdiag["valid_sampled_points"] += out["search_stats"]["valid_sampled_points"]
                sdiag["masked_sampled_points"] += out["search_stats"]["masked_sampled_points"]
                sdiag["usable_queries"] += out["search_stats"]["usable_queries"]
                qdiag["all_invalid_search"] += out["search_stats"]["all_invalid_queries"]
            both = query_valid(pixel_valid(ga) & pixel_valid(gd))
            qdiag["both_valid"] += 2 * int(both.sum())
            p_raw = e_raw = None
            if name == "S2-MaskAware":
                # Both teacher and student use identical hard target-memory rules.
                p_raw, e_raw, _, kd_masks, teacher_outputs, kd_stats = masked_kd(student, teacher, a, d, ga, gd, outs)
                for key in kd_sdiag: kd_sdiag[key] += kd_stats[key]
                for to in teacher_outputs:
                    context_valid_sum += np.asarray(to["counter_valid_ratio"])
                    context_nonzero_sum += np.asarray(to["counter_nonzero_ratio_after_mask"])
                    context_count += 1
                qdiag["kd_used"] += sum(int(m.sum()) for m in kd_masks)
                if calibration is None:
                    calibration = {"seg_init": float(seg.detach()), "policy_init": max(float(p_raw.detach()), EPS),
                                   "evidence_init": max(float(e_raw.detach()), EPS),
                                   "target_kd_total_fraction": KD_TOTAL_FRACTION}
                normalized_policy = p_raw / calibration["policy_init"]
                normalized_evidence = e_raw / calibration["evidence_init"]
                # Retain the two original KD terms and dynamically calibrate
                # their single scalar weight from this training batch only.
                # The detached lambda keeps the KD share at 10% of total.
                kd_norm = 0.5 * normalized_policy + 0.5 * normalized_evidence
                lam = (KD_TOTAL_FRACTION / (1.0 - KD_TOTAL_FRACTION)) * seg.detach() / (kd_norm.detach() + EPS)
                lambdas.append(float(lam.detach()))
                policy_contrib = lam * 0.5 * normalized_policy
                evidence_contrib = lam * 0.5 * normalized_evidence
                loss = seg + policy_contrib + evidence_contrib
                pols.append(float(p_raw.detach())); evidences.append(float(e_raw.detach()))
                weighted.append({"seg": float(seg.detach()), "policy": float(policy_contrib.detach()),
                                     "evidence": float(evidence_contrib.detach()), "total": float(loss.detach()),
                                     "seg_pct": float(100 * seg.detach() / loss.detach()),
                                     "policy_pct": float(100 * policy_contrib.detach() / loss.detach()),
                                     "evidence_pct": float(100 * evidence_contrib.detach() / loss.detach())})
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            totals.append(float(loss.detach())); segs.append(float(seg.detach()))
        pred = evaluate_student(student, test); score = overall_ap(pred)
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            torch.save(best_state, final)
        else:
            bad += 1
        rec = {"epoch": epoch, "train_total_loss": float(np.mean(totals)),
               "train_seg_raw": float(np.mean(segs)), "monitor_test_auprc_overall": score,
               "best_test_auprc_overall": best, "best_epoch": best_epoch, "bad_epochs": bad,
               "query_counts": qdiag, "search_sampling": sdiag,
               "kd_lambda_mean": float(np.mean(lambdas)) if lambdas else None,
               "kd_evidence_search_sampling": kd_sdiag if name == "S2-MaskAware" else None,
               "search_mean_valid_samples_per_query": sdiag["valid_sampled_points"] / max(sdiag["usable_queries"], 1),
               "all_invalid_search_query_fraction": qdiag["all_invalid_search"] / max(qdiag["total"], 1),
               "counter_fpn_valid_ratio_by_level": (context_valid_sum / max(context_count, 1)).tolist() if context_count else None,
               "counter_fpn_nonzero_after_mask_by_level": (context_nonzero_sum / max(context_count, 1)).tolist() if context_count else None}
        if name == "S2-MaskAware":
            rec.update({"policy_js_raw": float(np.mean(pols)), "evidence_distance_raw": float(np.mean(evidences)),
                        "first_batch_weighted_contributions": weighted[0] if epoch == 1 else None,
                        "weighted_contribution_epoch_mean": {k: float(np.mean([x[k] for x in weighted])) for k in ("seg", "policy", "evidence", "total")},
                        "weighted_contribution_epoch_percent": {k: 100*float(np.mean([x[k] for x in weighted]))/max(float(np.mean([x["total"] for x in weighted])), EPS) for k in ("seg", "policy", "evidence")},
                        "kd_lambda_mean": float(np.mean(lambdas)),
                        "calibration": calibration})
        history.append(rec)
        payload = {"method": name, "seed": SEED, "epoch": epoch, "history": history,
                   "best_test_auprc_overall": best, "best_epoch": best_epoch, "complete": False,
                   "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": WD},
                   "batch_size": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "checkpoint_selection": "current canonical protocol: test raw overall AUPRC; no validation split",
                   "calibration": calibration}
        write_json(progress, payload)
        print(f"{name} epoch={epoch} train={rec['train_total_loss']:.6f} test_AP={score:.6f} best={best:.6f}@{best_epoch}", flush=True)
        if bad >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"no checkpoint selected: {name}")
    student.load_state_dict(best_state)
    payload.update({"complete": True, "checkpoint": str(final), "checkpoint_sha256": sha256(final)})
    write_json(progress, payload)
    return student, payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, test_ids = split()
    if val_ids:
        raise RuntimeError(f"canonical no-validation protocol expected; found {len(val_ids)} val IDs")
    train = HaitiPairs(train_ids, True)
    test = HaitiPairs(test_ids, False, train.stats)
    teacher = make_teacher()
    protocol = {"method": "Mask-Aware Search-KD v5", "seed": SEED, "batch_size": BATCH,
                "optimizer": {"name": "Adam", "lr": LR, "weight_decay": WD, "betas": [0.9, 0.999], "eps": 1e-8},
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "train_ids": len(train_ids),
                "validation_ids": len(val_ids), "test_ids": len(test_ids),
                "split_policy": "reuse pinned split(); no validation created; test used per canonical monitoring rule",
                "init_checkpoint": str(INIT), "teacher_checkpoint": str(TEACHER_CKPT),
                "geometry_codes": {"0": "valid", "1": "layover", "2": "shadow", "3": "layover+shadow"},
                "segmentation_mask": "per-orbit target geometry == 0; BCE denominator is valid pixel count",
                "kd_query_mask": "strict BothValid 4x4 blocks; additionally both policy distributions and masked searches must have support",
                "memory_mask": "all corresponding geometry pixels in each FPN cell must be valid; bilinear support >0.999; mask sampled weights then jointly renormalize",
                "counter_context": "counter FPN feature maps multiplied by conservative counter geometry-valid masks before Teacher context",
                "kd": "original policy JS + same Student target-memory evidence distance only; a single detached scalar lambda is batch-calibrated from training losses to maintain 10% KD contribution; no test values enter lambda",
                "no_new_parameters_or_loss_terms": True}
    write_json(OUT / "protocol.json", protocol)
    results = {}
    s2_calibration = None
    for name in METHODS:
        model, report = train_arm(name, train, test, teacher if name == "S2-MaskAware" else None)
        if name == "S2-MaskAware": s2_calibration = report.get("calibration")
        diag = evaluate_diagnostics(model, test, teacher if name == "S2-MaskAware" else None,
                                    s2_calibration if name == "S2-MaskAware" else None)
        results[name] = {"best_epoch": report["best_epoch"],
                         "best_test_auprc_overall": report["best_test_auprc_overall"],
                         **diag, "checkpoint": report["checkpoint"]}
    teacher_diag = evaluate_teacher(teacher, test)
    teacher_diag_metrics = metric_rows(teacher_diag)
    results["CG-Teacher-masked-inference"] = {"best_epoch": None, "metrics": teacher_diag_metrics,
                                              "checkpoint": str(TEACHER_CKPT),
                                              "note": "existing canonical CG checkpoint; evaluated with v5 memory and counter masks, not retrained"}
    s0 = results["S0-clean"]["metrics"]; s2 = results["S2-MaskAware"]["metrics"]
    results["delta_S2_minus_S0"] = {r: {m: s2[r][m] - s0[r][m] for m in ("AUPRC", "IoU", "F1")}
                                    for r in ("BothValid", "TargetValid")}
    write_json(OUT / "final_metrics.json", {"protocol": protocol, "results": results,
                                            "test_selection_risk": "test AUPRC is used for stopping/best checkpoint under the requested canonical project protocol"})
    print(json.dumps({"output": str(OUT), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
