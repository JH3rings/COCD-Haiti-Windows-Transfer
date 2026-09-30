"""Re-evaluate frozen checkpoints and build auditable GRSL tables/candidates.

This script performs inference only. It never trains, selects checkpoints, or
changes the fixed split. Candidate ranking is a screening aid; case selection
columns remain blank for human review.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion, label
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
COCD = ROOT / "cocd"
OUT = ROOT / "experiments/GRSL_complete_paper_package"
sys.path.insert(0, str(COCD))
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
sys.path.insert(0, str(ROOT / "experiments/distillation_final" / "multi2uni_port"))
sys.path.insert(0, str(OUT / "baselines" / "mmokd_port"))

from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2  # noqa: E402
from multi2uni_model import Multi2UniPort  # noqa: E402
from mmokd_model import SingleOrbitBranch, load_haiti_initialization  # noqa: E402
from models.landslide_cocd import ConvNeXtTinyFPN  # noqa: E402


BATCH = 16
SEED = 42
INIT = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
S0_CKPT = ROOT / "experiments/windows_main/student_target_search_v2/S0_seed42.pt"
S2_CKPT = ROOT / "experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt"
SG_CKPT = ROOT / "experiments/windows_main/teacher_target_search_v2/SG10_seed42.pt"
CG_CKPT = ROOT / "experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
MULTI_CKPT = ROOT / "experiments/distillation_final/multi2uni_port/Multi2Uni-Port_deploy_student_seed42_best.pt"
MMOKD_A = OUT / "baselines/mmokd_port/student_asc_best.pt"
MMOKD_D = OUT / "baselines/mmokd_port/student_desc_best.pt"
EPS = 1e-8

GREEN = ListedColormap(["#111827", "#16a34a"])
ERRORS = ListedColormap(["#111827", "#16a34a", "#dc2626", "#facc15"])
GEOM = ListedColormap(["#d1fae5", "#f97316", "#7c3aed", "#be123c"])


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(p: np.ndarray, y: np.ndarray, m: np.ndarray) -> dict:
    yy = y[m].astype(bool)
    pp = p[m].astype(np.float32)
    hh = pp >= 0.5
    tp = int((hh & yy).sum())
    fp = int((hh & ~yy).sum())
    fn = int((~hh & yy).sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return {
        "AUPRC": float(average_precision_score(yy, pp)) if yy.any() else None,
        "IoU": float(tp / (tp + fp + fn + EPS)),
        "F1": float(2 * precision * recall / (precision + recall + EPS)),
        "pixels": int(m.sum()), "positive_pixels": int(yy.sum()),
    }


def region_masks(g: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Overall": np.ones_like(g, dtype=bool),
        "TargetValid": g == 0,
        "BothValid": (g == 0) & (c == 0),
        "OverallValid": (g == 0) | (c == 0),
        "G10": (g > 0) & (c == 0),
        "G01": (g == 0) & (c > 0),
    }


def model_metric_rows(preds, y, g, c):
    rows = []
    for method, p in preds.items():
        for region, mask in region_masks(g, c).items():
            rows.append({"Method": method, "Region": region, **metrics(p, y, mask)})
    return rows


def as_prob_single(model, x, two_class=False):
    out = model(x)
    logits = out["logits"] if "logits" in out else out["z"]
    if two_class:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits[:, 0])


def load_branch(path: Path) -> SingleOrbitBranch:
    template = ConvNeXtTinyFPN()
    load_haiti_initialization(template, INIT)
    model = SingleOrbitBranch(template).to(DEV).eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def js_per_query(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(1) + (q * (q.log() - m.log())).sum(1))


def strict_query_mask(pixel_mask: torch.Tensor, hq: int, wq: int) -> torch.Tensor:
    if pixel_mask.ndim == 3:
        pixel_mask = pixel_mask[:, None]
    return F.adaptive_avg_pool2d(pixel_mask.float(), (hq, wq)) >= 1.0 - 1e-7


def aggregate_policy(policy, mask: torch.Tensor) -> tuple[list[float], float]:
    weights = policy["weights"]
    offsets = 2.0 * torch.tanh(policy["offsets"])
    if mask.shape[-2:] != weights.shape[-2:]:
        mask = F.adaptive_avg_pool2d(mask[:, None].float(), weights.shape[-2:]) >= 1.0 - 1e-7
        mask = mask[:, 0]
    mass = []
    for level in range(weights.shape[1]):
        level_mass = weights[:, level].sum(1)
        selected = level_mass[mask]
        mass.append(float(selected.mean().item()) if selected.numel() else float("nan"))
    mag = torch.linalg.vector_norm(offsets[:, :, 1:], dim=3).mean(2)
    selected_mag = mag.permute(0, 2, 3, 1)[mask]
    return mass, float(selected_mag.mean().item()) if selected_mag.numel() else float("nan")


def policy_alignment(teacher: dict, student: dict, shuffled: dict, mask: torch.Tensor,
                     rt: torch.Tensor, rshuffle: torch.Tensor, rs: torch.Tensor) -> dict:
    wt = teacher["weights"].flatten(1, 2)
    ws = student["weights"].flatten(1, 2)
    wh = shuffled["weights"].flatten(1, 2)
    jts = js_per_query(wt, ws)
    jth = js_per_query(wt, wh)
    if mask.shape[-2:] != jts.shape[-2:]:
        mask = strict_query_mask(mask, *jts.shape[-2:])
    cosine = F.cosine_similarity(rt, rs, dim=1, eps=1e-8)
    cosine_shuffle = F.cosine_similarity(rshuffle, rs, dim=1, eps=1e-8)
    values = {
        "policy_js_CG_vs_S2": float(jts[mask].mean().item()) if mask.any() else None,
        "policy_js_CG_vs_shuffled": float(jth[mask].mean().item()) if mask.any() else None,
        "evidence_distance_CG_vs_S2": float((1 - cosine[mask]).mean().item()) if mask.any() else None,
        "evidence_cosine_CG_vs_S2": float(cosine[mask].mean().item()) if mask.any() else None,
        "evidence_distance_shuffled_vs_S2": float((1 - cosine_shuffle[mask]).mean().item()) if mask.any() else None,
    }
    mt, ot = aggregate_policy(teacher, mask)
    ms, os_ = aggregate_policy(student, mask)
    mh, oh = aggregate_policy(shuffled, mask)
    values.update({f"CG_weight_mass_P{i+2}": x for i, x in enumerate(mt)})
    values.update({f"S2_weight_mass_P{i+2}": x for i, x in enumerate(ms)})
    values.update({f"Shuffle_weight_mass_P{i+2}": x for i, x in enumerate(mh)})
    values.update({"CG_offset_magnitude": ot, "S2_offset_magnitude": os_,
                   "Shuffle_offset_magnitude": oh})
    return values


@torch.inference_mode()
def infer_all(ds):
    models = {
        "S0": StudentTargetSearchV2().to(DEV).eval(),
        "S2-2Stage": StudentTargetSearchV2().to(DEV).eval(),
        "SG Teacher": TargetSearchTeacherV2().to(DEV).eval(),
        "CG Teacher": TargetSearchTeacherV2().to(DEV).eval(),
        "Multi2Uni-Port": Multi2UniPort(str(INIT)).to(DEV).eval(),
        "MMOKD ASC": load_branch(MMOKD_A),
        "MMOKD DESC": load_branch(MMOKD_D),
    }
    models["S0"].load_state_dict(torch.load(S0_CKPT, map_location="cpu", weights_only=True), strict=True)
    models["S2-2Stage"].load_state_dict(torch.load(S2_CKPT, map_location="cpu", weights_only=True), strict=True)
    models["SG Teacher"].load_state_dict(torch.load(SG_CKPT, map_location="cpu", weights_only=True), strict=True)
    models["CG Teacher"].load_state_dict(torch.load(CG_CKPT, map_location="cpu", weights_only=True), strict=True)
    multi_state = torch.load(MULTI_CKPT, map_location="cpu", weights_only=True)
    models["Multi2Uni-Port"].student.load_state_dict(multi_state, strict=True)

    pred_a = {name: [] for name in ("S0", "S2-2Stage", "Multi2Uni-Port", "MMOKD-Port", "SG Teacher", "CG Teacher", "CG shuffle")}
    pred_d = {name: [] for name in pred_a}
    labels, ga_rows, gd_rows = [], [], []
    align_accum: dict[str, list[float]] = {}
    gen = torch.Generator().manual_seed(SEED + 1701)
    for batch in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0):
        a, d, y, ga, gd = batch[:5]
        a_dev, d_dev = a.to(DEV), d.to(DEV)
        for name in ("S0", "S2-2Stage"):
            pred_a[name].append(as_prob_single(models[name], a_dev).cpu().numpy())
            pred_d[name].append(as_prob_single(models[name], d_dev).cpu().numpy())
        multi = models["Multi2Uni-Port"]
        pred_a["Multi2Uni-Port"].append(torch.sigmoid(multi.predict_single(a_dev)[:, 0]).cpu().numpy())
        pred_d["Multi2Uni-Port"].append(torch.sigmoid(multi.predict_single(d_dev)[:, 0]).cpu().numpy())
        pred_a["MMOKD-Port"].append(as_prob_single(models["MMOKD ASC"], a_dev, True).cpu().numpy())
        pred_d["MMOKD-Port"].append(as_prob_single(models["MMOKD DESC"], d_dev, True).cpu().numpy())

        sg_a, sg_d = models["SG Teacher"].forward_pair(a_dev, d_dev, "self")
        cg_a, cg_d = models["CG Teacher"].forward_pair(a_dev, d_dev, "counter")
        perm = torch.randperm(len(a), generator=gen)
        if torch.equal(perm, torch.arange(len(a))):
            perm = torch.roll(perm, 1)
        sh_a = models["CG Teacher"](a_dev, d_dev[perm.to(DEV)])
        sh_d = models["CG Teacher"](d_dev, a_dev[perm.to(DEV)])
        for name, oa, od in (("SG Teacher", sg_a, sg_d), ("CG Teacher", cg_a, cg_d),
                             ("CG shuffle", sh_a, sh_d)):
            za = oa["z"] if "z" in oa else oa
            zd = od["z"] if "z" in od else od
            pred_a[name].append(torch.sigmoid(za[:, 0]).cpu().numpy())
            pred_d[name].append(torch.sigmoid(zd[:, 0]).cpu().numpy())

        # Mechanism statistics: target-query-only masks; pooled 4x4 patches are
        # included only if every underlying target pixel is geometry-valid.
        with torch.no_grad():
            s2_a = models["S2-2Stage"](a_dev)
            s2_d = models["S2-2Stage"](d_dev)
        for target_geom, counter_geom, teacher, student, shuffled in (
            (ga, gd, cg_a, s2_a, sh_a), (gd, ga, cg_d, s2_d, sh_d)):
            target_valid = (target_geom.to(DEV) == 0)
            both_valid = target_valid & (counter_geom.to(DEV) == 0)
            for region, pixel_mask in (("TargetValid", target_valid), ("BothValid", both_valid)):
                hq, wq = teacher["policy"]["weights"].shape[-2:]
                qmask = strict_query_mask(pixel_mask, hq, wq)[:, 0]
                # Match the canonical same-target-memory evidence objective:
                # compare student-policy and teacher-policy retrievals using
                # the student's own target memory, never counter-side values.
                student_memory = tuple(x.detach() for x in student["values"])
                r_teacher_on_student_memory = models["S2-2Stage"].search(student_memory, teacher["policy"])
                r_shuffle_on_student_memory = models["S2-2Stage"].search(student_memory, shuffled["policy"])
                values = policy_alignment(teacher["policy"], student["policy"],
                                          shuffled["policy"], qmask,
                                          r_teacher_on_student_memory, r_shuffle_on_student_memory,
                                          student["R"])
                for key, value in values.items():
                    if value is not None and np.isfinite(value):
                        align_accum.setdefault(f"{region}/{key}", []).append(float(value))

        labels.append(y.numpy()); ga_rows.append(ga.numpy()); gd_rows.append(gd.numpy())

    p = {name: (np.concatenate(pred_a[name], axis=0), np.concatenate(pred_d[name], axis=0)) for name in pred_a}
    y = np.concatenate(labels, axis=0)
    ga, gd = np.concatenate(ga_rows, axis=0), np.concatenate(gd_rows, axis=0)
    n = len(ds.ids)
    ids = np.asarray(list(ds.ids) + list(ds.ids), dtype=np.int32)
    orbit = np.asarray(["asc"] * n + ["desc"] * n)
    yy = np.concatenate((y, y), axis=0)
    g = np.concatenate((ga, gd), axis=0)
    c = np.concatenate((gd, ga), axis=0)
    preds = {name: np.concatenate((pa, pd), axis=0).astype(np.float32) for name, (pa, pd) in p.items()}
    # Verify this post-hoc evaluation agrees with the recorded summaries.
    metric_rows = model_metric_rows(preds, yy, g, c)
    write_csv(OUT / "04_tables" / "reevaluated_region_metrics.csv", metric_rows)
    write_json(OUT / "03_search_behavior" / "search_statistics.json",
               {key: float(np.mean(values)) for key, values in align_accum.items()})
    np.savez_compressed(OUT / "02_qualitative" / "final_case_data" / "test_prediction_cache.npz",
                        **{f"p_{name.replace(' ', '_').replace('-', '_')}": value
                           for name, value in preds.items() if name in ("S0", "S2-2Stage", "Multi2Uni-Port", "MMOKD-Port")},
                        y=yy.astype(np.uint8), target_geometry=g.astype(np.uint8),
                        counter_geometry=c.astype(np.uint8), sample_id=ids, orbit=orbit)
    return ds, preds, yy.astype(np.uint8), g.astype(np.uint8), c.astype(np.uint8), ids, orbit, align_accum


def load_patch(ds, index: int, orbit: str):
    asc, desc, y, ga, gd, _ = ds[index]
    target, counter = (asc, desc) if orbit == "asc" else (desc, asc)
    gt, tg, cg = y.numpy(), (ga.numpy() if orbit == "asc" else gd.numpy()), (gd.numpy() if orbit == "asc" else ga.numpy())
    # Input stack is [VV_pre,VH_pre,VV_post,VH_post,orbit_id].
    return target.numpy(), counter.numpy(), gt.astype(np.uint8), tg.astype(np.uint8), cg.astype(np.uint8)


def calc_case_metrics(p, y, mask):
    return metrics(p, y, mask)


def geometry_display(tg: np.ndarray, cg: np.ndarray) -> np.ndarray:
    # 0 both-valid; 1 target invalid only; 2 counter invalid only; 3 both invalid.
    ti, ci = tg > 0, cg > 0
    return (ti.astype(np.uint8) + 2 * ci.astype(np.uint8)).astype(np.uint8)


def display_image(stack: np.ndarray):
    img = 0.5 * (stack[0] + stack[2])
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def show_prediction(ax, p, tg):
    h = p >= 0.5
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    rgb[h] = (0.09, 0.64, 0.31)
    invalid = tg > 0
    rgb[invalid] = (0.72, 0.74, 0.78)
    ax.imshow(rgb)


def make_candidate_rows(ds, preds, y, g, c, ids, orbit, external):
    rows_by_category = {"A_bothvalid": [], "B_targetvalid_weak": [], "C_geometry_diagnostic": []}
    # Candidate arrays are 686 rows (all ASC, then all DESC).
    n = len(ds.ids)
    # Map stable sample id and orbit to the corresponding target input stack.
    index = {int(sample): i for i, sample in enumerate(ds.ids)}
    for row in range(len(y)):
        sid = int(ids[row]); orb = str(orbit[row]); yi = y[row] > 0
        tg, cg = g[row], c[row]
        tv, bv = tg == 0, (tg == 0) & (cg == 0)
        p0, po = preds["S0"][row], preds["S2-2Stage"][row]
        pe = preds[external][row]
        pm = preds["MMOKD-Port"][row]
        gt_fraction = float(yi.mean())
        ti_ratio, ci_ratio = float((tg > 0).mean()), float((cg > 0).mean())
        weak_change = float("nan")
        target_stack, counter_stack, gt, tg_patch, cg_patch = load_patch(ds, index[sid], orb)
        change = np.abs(target_stack[0] - target_stack[2]) + np.abs(target_stack[1] - target_stack[3])
        if yi.any():
            weak_change = float(np.mean(change[yi]))
        boundary = yi & ~binary_erosion(yi, iterations=2, border_value=0)
        components, count = label(yi)
        largest_component = max([int((components == i).sum()) for i in range(1, count + 1)] or [0])
        def candidate_record(category, region, mask, ranking_score):
            base = calc_case_metrics(p0, yi, mask)
            ours = calc_case_metrics(po, yi, mask)
            multi = calc_case_metrics(preds["Multi2Uni-Port"][row], yi, mask)
            mm = calc_case_metrics(pm, yi, mask)
            rescue = int((mask & yi & (p0 < 0.5) & (po >= 0.5)).sum())
            new_fn = int((mask & yi & (p0 >= 0.5) & (po < 0.5)).sum())
            new_fp = int((mask & ~yi & (p0 < 0.5) & (po >= 0.5)).sum())
            corrected_fp = int((mask & ~yi & (p0 >= 0.5) & (po < 0.5)).sum())
            fn_to_tp = mask & yi & (p0 < 0.5) & (po >= 0.5)
            return {
                "patch_id": f"{sid:06d}_{orb}", "sample_id": sid, "target_orbit": orb,
                "category": category, "evaluation_region": region,
                "GT_fraction": gt_fraction, "target_invalid_ratio": ti_ratio,
                "counter_invalid_ratio": ci_ratio,
                "S0_AP": base["AUPRC"], "S0_IoU": base["IoU"],
                "Multi2Uni_AP": multi["AUPRC"], "Multi2Uni_IoU": multi["IoU"],
                "MMOKD_AP": mm["AUPRC"], "MMOKD_IoU": mm["IoU"],
                "Ours_AP": ours["AUPRC"], "Ours_IoU": ours["IoU"], "Ours_F1": ours["F1"],
                "Ours_S0_rescue_pixels": rescue, "S0_TP_to_Ours_FN": new_fn,
                "new_error_pixels": new_fn + new_fp,
                "corrected_error_pixels": rescue + corrected_fp,
                "boundary_rescue_pixels": int((fn_to_tp & boundary).sum()),
                "weak_change_score": (1.0 / (weak_change + EPS)) if np.isfinite(weak_change) else None,
                "mean_abs_change_in_GT": weak_change,
                "largest_GT_component_pixels": largest_component,
                "GT_invalid_target_fraction": float((yi & ~tv).sum() / (yi.sum() + EPS)),
                "ranking_score": float(ranking_score),
                "manual_selected": "", "manual_notes": "",
            }

        # A: finite landslide support on the intersection of both valid views,
        # and a positive net error transition from S0 to Ours.
        both_gt = int((yi & bv).sum())
        if 40 <= both_gt <= 3500:
            rescue = int((bv & yi & (p0 < 0.5) & (po >= 0.5)).sum())
            lost = int((bv & yi & (p0 >= 0.5) & (po < 0.5)).sum())
            fp_fix = int((bv & ~yi & (p0 >= 0.5) & (po < 0.5)).sum())
            fp_new = int((bv & ~yi & (p0 < 0.5) & (po >= 0.5)).sum())
            net = rescue + fp_fix - lost - fp_new
            boundary_rescue = int((bv & boundary & (p0 < 0.5) & (po >= 0.5)).sum())
            if net > 0:
                rows_by_category["A_bothvalid"].append(candidate_record(
                    "A_bothvalid", "BothValid", bv, net + 0.15 * boundary_rescue))

        # B: every GT pixel is target-valid; rank actual target-valid error gain
        # first, then favor weaker normalized pre/post SAR change evidence.
        gt_count = int(yi.sum())
        if 20 <= gt_count <= 1200 and not (yi & ~tv).any():
            rescue = int((tv & yi & (p0 < 0.5) & (po >= 0.5)).sum())
            lost = int((tv & yi & (p0 >= 0.5) & (po < 0.5)).sum())
            fp_fix = int((tv & ~yi & (p0 >= 0.5) & (po < 0.5)).sum())
            fp_new = int((tv & ~yi & (p0 < 0.5) & (po >= 0.5)).sum())
            net = rescue + fp_fix - lost - fp_new
            if net > 0:
                weak_score = 1.0 / (weak_change + 0.2) if np.isfinite(weak_change) else 0.0
                rows_by_category["B_targetvalid_weak"].append(candidate_record(
                    "B_targetvalid_weak", "TargetValid", tv, net + 5.0 * weak_score))
        if ti_ratio >= 0.15 and ci_ratio <= 0.15 and (ti_ratio - ci_ratio) >= 0.15:
            row_record=candidate_record("C_geometry_diagnostic","G10 diagnostic",np.ones_like(tv),
                                         ti_ratio-ci_ratio+0.02*float(yi.any()))
            rows_by_category["C_geometry_diagnostic"].append(row_record)

    for category, rows in rows_by_category.items():
        # De-duplicate opposite target direction from the same source location.
        rows.sort(key=lambda r: float(r.get("ranking_score", -1e9)), reverse=True)
        unique, seen = [], set()
        for row in rows:
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"]); unique.append(row)
            if len(unique) >= 10:
                break
        rows_by_category[category] = unique
    return rows_by_category


def make_candidate_pdf(ds, rows_by_category, preds, y, g, c, ids, orbit, external):
    cat_dir = OUT / "02_qualitative" / "candidates"
    names = {"S0": "S0", "Multi2Uni-Port": "Multi2Uni", "MMOKD-Port": "MMOKD", "S2-2Stage": "Ours S2-2Stage"}
    ext_key = external
    pdf_path = OUT / "02_qualitative" / "fig2_candidates_contact_sheet.pdf"
    all_map = {(int(s), str(o)): i for i, (s, o) in enumerate(zip(ids, orbit))}
    with PdfPages(pdf_path) as pdf:
        for category, rows in rows_by_category.items():
            target = cat_dir / category
            target.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(max(1, len(rows)), 7, figsize=(14, max(2.15, 1.7 * max(1, len(rows)))))
            if len(rows) == 1: axes = np.expand_dims(axes, 0)
            headers = ["Target SAR", "Counter SAR", "Geometry", "GT", "S0", names[ext_key], names["S2-2Stage"]]
            for j, text in enumerate(headers): axes[0, j].set_title(text, fontsize=8)
            for i, row in enumerate(rows):
                pi = all_map[(int(row["sample_id"]), row["target_orbit"])]
                stack, counter_stack, gt, tg, cg = load_patch(ds, int(np.where(np.asarray(ds.ids) == row["sample_id"])[0][0]), row["target_orbit"])
                axes[i, 0].imshow(display_image(stack), cmap="gray")
                axes[i, 1].imshow(display_image(counter_stack), cmap="gray")
                axes[i, 2].imshow(geometry_display(tg, cg), cmap=GEOM, vmin=0, vmax=3)
                axes[i, 3].imshow(gt > 0, cmap=GREEN, vmin=0, vmax=1)
                show_prediction(axes[i, 4], preds["S0"][pi], tg)
                show_prediction(axes[i, 5], preds[ext_key][pi], tg)
                show_prediction(axes[i, 6], preds["S2-2Stage"][pi], tg)
                axes[i, 0].text(2, 9, f"{i+1}. {row['patch_id']}", color="white", fontsize=6,
                                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})
                for ax in axes[i]: ax.axis("off")
            fig.suptitle(f"{category}: automatically ranked candidates (manual selection pending)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, bbox_inches="tight")
            for i, row in enumerate(rows):
                one, ax = plt.subplots(1, 7, figsize=(14, 2.2))
                pi = all_map[(int(row["sample_id"]), row["target_orbit"])]
                stack, counter_stack, gt, tg, cg = load_patch(ds, int(np.where(np.asarray(ds.ids) == row["sample_id"])[0][0]), row["target_orbit"])
                ax[0].imshow(display_image(stack), cmap="gray"); ax[1].imshow(display_image(counter_stack), cmap="gray")
                ax[2].imshow(geometry_display(tg, cg), cmap=GEOM, vmin=0, vmax=3); ax[3].imshow(gt > 0, cmap=GREEN)
                show_prediction(ax[4], preds["S0"][pi], tg); show_prediction(ax[5], preds[ext_key][pi], tg); show_prediction(ax[6], preds["S2-2Stage"][pi], tg)
                for a in ax: a.axis("off")
                one.tight_layout(); one.savefig(target / f"{i+1:02d}_{row['patch_id']}.png", dpi=180); plt.close(one)
            plt.close(fig)
    return pdf_path


def build_tables(preds, y, g, c, align_accum):
    source_path = ROOT / "experiments/distillation_final/comparison/final_table.csv"
    with source_path.open(newline="", encoding="utf-8-sig") as f:
        source = {r["Method"]: r for r in csv.DictReader(f)}
    source["MMOKD-Port"] = {
        "Method": "MMOKD-Port", "Status": "COMPLETE_MECHANISM_FAITHFUL_PORT",
        "BestEpoch": 6, "InferenceParams": "28865218", "InferenceSupportedGFLOPsPerOrbit": "1.599889408",
        "DeploymentInput": "ASC-only branch or DESC-only branch; no counter at inference",
    }
    summary = json.loads((OUT / "baselines/mmokd_port/summary.json").read_text(encoding="utf-8"))
    full_rows = []
    method_map = {"S0-Raw": "S0", "Hinton-KD-Port": None, "DIS2-Port": None,
                  "Multi2Uni-Port": "Multi2Uni-Port", "S2-Raw-native": None,
                  "S2-2Stage": "S2-2Stage", "MMOKD-Port": "MMOKD-Port"}
    for method in ("S0-Raw", "Hinton-KD-Port", "DIS2-Port", "Multi2Uni-Port", "MMOKD-Port", "S2-Raw-native", "S2-2Stage"):
        row = source.get(method, {})
        if method == "MMOKD-Port":
            met = summary["metrics"]
            meta = {"best_epoch": summary["best_epoch"], "inference_params": summary["inference_parameters_per_orbit"],
                    "inference_gflops": summary["compute_cost"]["student_asc_inference"]["supported_forward_gflops_per_orbit"],
                    "training_only_branches": "; ".join(summary["compute_cost"]["training_only_branches"]),
                    "deployment_input": summary["compute_cost"]["inference_only_inputs"]}
        elif method == "S2-2Stage":
            sj = json.loads((ROOT / "experiments/distillation_final/ours_s2_2stage/summary.json").read_text(encoding="utf-8")); met=sj["metrics"]
            meta={"best_epoch":sj["best_epoch"],"inference_params":row.get("InferenceParams"),"inference_gflops":row.get("InferenceSupportedGFLOPsPerOrbit"),"training_only_branches":"counter encoder/context, paired Teacher","deployment_input":"target orbit only"}
        elif method == "S0-Raw":
            sj = json.loads((ROOT / "experiments/windows_main/raw_cver_v1_seed42/S0-Raw_summary.json").read_text(encoding="utf-8")); met=sj.get("metrics",{})
            if not met: met={r:{"AUPRC":float(row[f"{r}_AUPRC"]),"IoU":float(row[f"{r}_IoU"]),"F1":float(row[f"{r}_F1"])} for r in ("Overall","BothValid","TargetValid","G10","G01")}
            meta={"best_epoch":row.get("BestEpoch"),"inference_params":row.get("InferenceParams"),"inference_gflops":row.get("InferenceSupportedGFLOPsPerOrbit"),"training_only_branches":"none","deployment_input":"target orbit only"}
        else:
            # The locked final comparison CSV is the exact recorded source for frozen methods.
            met={region:{"AUPRC":row.get(f"{region}_AUPRC"),"IoU":row.get(f"{region}_IoU"),"F1":row.get(f"{region}_F1")} for region in ("Overall","BothValid","TargetValid","G10","G01")}
            meta={"best_epoch":row.get("BestEpoch"),"inference_params":row.get("InferenceParams"),"inference_gflops":row.get("InferenceSupportedGFLOPsPerOrbit"),"training_only_branches":"see source run / baseline notes","deployment_input":row.get("DeploymentInput")}
        out={"Method":method,"Status":row.get("Status","COMPLETE"),**meta}
        for region in ("Overall","TargetValid","BothValid","G10","G01"):
            for metric in ("AUPRC","IoU","F1"):
                val=met.get(region,{}).get(metric)
                if isinstance(val,str):
                    try: val=float(val)
                    except ValueError: pass
                out[f"{region}_{metric}"]=val
        full_rows.append(out)
    write_csv(OUT / "04_tables/full_metrics.csv", full_rows)
    by_name = {r["Method"]: r for r in full_rows}
    main_methods = ["S0-Raw", "Hinton-KD-Port", "DIS2-Port", "Multi2Uni-Port", "MMOKD-Port", "S2-2Stage"]
    t1=[]
    for method in main_methods:
        r=by_name[method]
        t1.append({"Method": {"DIS2-Port":"DIS2-adapted","S2-2Stage":"Ours S2-2Stage"}.get(method,method),
                   "Overall AP":r["Overall_AUPRC"],"TargetValid AP":r["TargetValid_AUPRC"],
                   "TargetValid IoU":r["TargetValid_IoU"],"TargetValid F1":r["TargetValid_F1"],
                   "BothValid AP":r["BothValid_AUPRC"],"BothValid IoU":r["BothValid_IoU"]})
    write_csv(OUT / "04_tables/table1_main.csv", t1)

    strict = {}
    for row in model_metric_rows(preds, y, g, c):
        strict.setdefault(row["Method"], {})[row["Region"]] = row
    teacher_names={"SG Teacher":"SG Teacher","CG Teacher":"CG Teacher","CG shuffle":"CG + shuffled counter"}
    t2=[]
    for label, source_method in [("SG Teacher","SG10"),("CG Teacher","CG10"),("CG + shuffled counter","CG10-counter-shuffle")]:
        # strict region metrics are recomputed from the locked Teacher checkpoints.
        lookup="SG Teacher" if source_method=="SG10" else ("CG Teacher" if source_method=="CG10" else "CG shuffle")
        rrows=strict[lookup]
        t2.append({"Method":label,"Overall AP":rrows["Overall"]["AUPRC"],"TargetValid AP":rrows["TargetValid"]["AUPRC"],"BothValid AP":rrows["BothValid"]["AUPRC"],
                   "Overall IoU":rrows["Overall"]["IoU"],"TargetValid IoU":rrows["TargetValid"]["IoU"],"BothValid IoU":rrows["BothValid"]["IoU"]})
    for label, source_method, met in [("S0 Student","S0-Raw",None),("S2-native","S2-Raw-native",None),("S2-2Stage","S2-2Stage",None)]:
        r=by_name[source_method]
        t2.append({"Method":label,"Overall AP":r["Overall_AUPRC"],"TargetValid AP":r["TargetValid_AUPRC"],"BothValid AP":r["BothValid_AUPRC"],
                   "Overall IoU":r["Overall_IoU"],"TargetValid IoU":r["TargetValid_IoU"],"BothValid IoU":r["BothValid_IoU"]})
    write_csv(OUT / "04_tables/table2_ablation.csv", t2)

    gain=[]
    s0, ours=by_name["S0-Raw"],by_name["S2-2Stage"]
    for region in ("Overall","TargetValid","BothValid","G10","G01"):
        rec={"Region":region}
        for metric in ("AUPRC","IoU","F1"):
            rec[f"S0_{metric}"]=s0[f"{region}_{metric}"];rec[f"Ours_{metric}"]=ours[f"{region}_{metric}"]
            rec[f"Ours_minus_S0_{metric}"]=ours[f"{region}_{metric}"]-s0[f"{region}_{metric}"]
        gain.append(rec)
    write_csv(OUT / "05_analysis/region_gain.csv", gain)

    # Pixel transition counts for the clean target-only comparison. Region
    # classes overlap (small component/boundary/weak-change tags); the raw
    # rescue/regression totals are the primary quantities.
    rescue_rows=[]
    p0,po=preds["S0"],preds["S2-2Stage"]
    for region,mask in region_masks(g,c).items():
        if region not in ("TargetValid","BothValid"): continue
        yy=y>0;h0=p0>=.5;ho=po>=.5
        rescued=int((mask&yy&~h0&ho).sum());lost=int((mask&yy&h0&~ho).sum())
        fp_corrected=int((mask&~yy&h0&~ho).sum());fp_new=int((mask&~yy&~h0&ho).sum())
        rescue_rows.append({"Region":region,"S0_FN_to_Ours_TP":rescued,"S0_TP_to_Ours_FN":lost,
                            "S0_FP_to_Ours_TN":fp_corrected,"S0_TN_to_Ours_FP":fp_new,
                            "net_positive_pixel_transitions":rescued-lost,
                            "net_error_reduction":rescued+fp_corrected-lost-fp_new})
    write_csv(OUT / "05_analysis/rescue_analysis.csv",rescue_rows)

    search=[]
    for region in ("TargetValid","BothValid"):
        for key,values in align_accum.items():
            if key.startswith(region+"/"):
                search.append({"Region":region,"Metric":key.split("/",1)[1],"Mean":float(np.mean(values)),"BatchesOrOrbitRecords":len(values)})
    write_csv(OUT / "03_search_behavior/search_statistics.csv",search)

    # Compact LaTeX versions with per-column best/second formatting.
    make_latex_table(OUT / "04_tables/table1_main.csv", OUT / "04_tables/table1_main.tex", "Main comparison")
    make_latex_table(OUT / "04_tables/table2_ablation.csv", OUT / "04_tables/table2_ablation.tex", "Mechanism and ablation")
    return full_rows, t1, t2


def make_latex_table(csv_path: Path, tex_path: Path, caption: str):
    with csv_path.open(newline="", encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    cols=list(rows[0]); numeric=cols[1:]
    stats={}
    for col in numeric:
        vals=[]
        for r in rows:
            try: vals.append((float(r[col]),r["Method"]))
            except (ValueError,KeyError): pass
        if vals: stats[col]=sorted(vals,reverse=True)
    lines=[r"\begin{table}[t]",r"\centering",r"\caption{"+caption+r"}",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{l"+"c"*(len(cols)-1)+"}",r"\toprule", " & ".join(cols)+r" \\",r"\midrule"]
    for r in rows:
        cells=[r["Method"]]
        for col in numeric:
            try: val=float(r[col]); cell=f"{val:.4f}"
            except (ValueError,TypeError): cell=r.get(col,"")
            order=stats.get(col,[])
            rank=next((i+1 for i,(v,m) in enumerate(order) if m==r["Method"]),None)
            if rank==1: cell=r"\textbf{"+cell+"}"
            elif rank==2: cell=r"\underline{"+cell+"}"
            cells.append(cell)
        lines.append(" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    tex_path.write_text("\n".join(lines),encoding="utf-8")


def checklist(full_rows, t1, t2, rows_by_category):
    by={r["Method"]:r for r in full_rows}
    ours,s0=by["S2-2Stage"],by["S0-Raw"]
    teacher= {r["Method"]:r for r in t2}
    vals={
        "paired CG Teacher > SG Teacher":teacher["CG Teacher"]["Overall AP"]>teacher["SG Teacher"]["Overall AP"],
        "shuffled counter degrades Teacher":teacher["CG + shuffled counter"]["Overall AP"]<teacher["CG Teacher"]["Overall AP"],
        "Ours > S0 on Overall":ours["Overall_AUPRC"]>s0["Overall_AUPRC"],
        "Ours > S0 on TargetValid":ours["TargetValid_AUPRC"]>s0["TargetValid_AUPRC"],
        "Ours > S0 on BothValid":ours["BothValid_AUPRC"]>s0["BothValid_AUPRC"],
        "Ours compared with Multi2Uni":True,
        "Ours compared with MMOKD":True,
        "BothValid qualitative case selected":False,
        "TargetValid weak-evidence case selected":False,
        "G10/G01 diagnostic case selected":False,
        "Search-behavior visualization available":False,
        "inference is target-only":True,
        "geometry-invalid region not used as primary claim":True,
        "final Table I ready":True,
        "final Table II ready":True,
    }
    lines=["# GRSL evidence checklist",""]
    for name,value in vals.items(): lines.append(f"- [{'x' if value else ' '}] {name}")
    lines += ["", "Automated candidate lists are screening outputs, not the final human case selection. The three case-selection boxes remain unchecked until visual review. Search behavior is computed in `search_statistics.csv`; the small figure candidate still requires a legibility review before it is counted as available.",""]
    (OUT/"grsl_evidence_checklist.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    tid,vid,eid=split()
    if vid: raise RuntimeError(f"Expected locked train+test protocol with no validation, found {len(vid)} validation locations")
    train=HaitiPairs(tid,True)
    ds=HaitiPairs(eid,False,train.stats)
    ds, preds, y, g, c, ids, orbit, align_accum = infer_all(ds)
    # Compare re-evaluation with final_table and run summaries; no values are
    # silently replaced if there is an unexplained mismatch.
    full_rows, t1, t2 = build_tables(preds,y,g,c,align_accum)
    summary=json.loads((OUT/"baselines/mmokd_port/summary.json").read_text(encoding="utf-8"))
    mm_metric={r["Region"]:r for r in model_metric_rows(preds,y,g,c) if r["Method"]=="MMOKD-Port"}
    multi_metric={r["Region"]:r for r in model_metric_rows(preds,y,g,c) if r["Method"]=="Multi2Uni-Port"}
    for region in ("Overall","TargetValid","BothValid","G10","G01"):
        expected=summary["metrics"][region]["AUPRC"]
        actual=mm_metric[region]["AUPRC"]
        if abs(expected-actual)>2e-5: raise RuntimeError(f"MMOKD {region} AP mismatch: recorded={expected}, re-eval={actual}")
    ext_metric=json.loads((ROOT/"experiments/distillation_final/multi2uni_port/summary.json").read_text(encoding="utf-8"))["metrics"]
    for region in ("Overall","TargetValid","BothValid","G10","G01"):
        expected=ext_metric[region]["AUPRC"]; actual=multi_metric[region]["AUPRC"]
        if abs(expected-actual)>2e-5: raise RuntimeError(f"Multi2Uni {region} AP mismatch: recorded={expected}, re-eval={actual}")
    score_multi=0.5*(ext_metric["TargetValid"]["AUPRC"]+ext_metric["BothValid"]["AUPRC"])
    score_mm=0.5*(summary["metrics"]["TargetValid"]["AUPRC"]+summary["metrics"]["BothValid"]["AUPRC"])
    external="Multi2Uni-Port" if score_multi+0.001>=score_mm else "MMOKD-Port"
    rows_by_category=make_candidate_rows(ds,preds,y,g,c,ids,orbit,external)
    case_rows=[]
    for category,rows in rows_by_category.items():
        case_rows.extend(rows)
        folder={"A_bothvalid":"A_bothvalid","B_targetvalid_weak":"B_targetvalid_weak","C_geometry_diagnostic":"C_geometry_diagnostic"}[category]
        write_csv(OUT/"02_qualitative"/"candidates"/folder/"candidates.csv",rows)
    write_csv(OUT/"02_qualitative"/"case_selection.csv",case_rows)
    pdf=make_candidate_pdf(ds,rows_by_category,preds,y,g,c,ids,orbit,external)
    cols=["category","patch_id","sample_id","target_orbit","GT_fraction","target_invalid_ratio","counter_invalid_ratio","S0_AP","S0_IoU","Multi2Uni_AP","Multi2Uni_IoU","MMOKD_AP","MMOKD_IoU","Ours_AP","Ours_IoU","Ours_F1","Ours_S0_rescue_pixels","new_error_pixels","weak_change_score","manual_selected","manual_notes"]
    write_csv(OUT/"02_qualitative"/"case_selection.csv",case_rows,cols)
    write_json(OUT/"02_qualitative"/"candidate_manifest.json",{
        "source": "fixed test split; canonical frozen checkpoints; predictions are target-only except teacher mechanism rows",
        "method": "automatic deterministic screening only; a human must choose final three cases",
        "strong_external_used": external,
        "external_choice_rule": "higher mean of TargetValid and BothValid AUPRC; Multi2Uni retained if within 0.001",
        "categories": {k:[r["patch_id"] for r in v] for k,v in rows_by_category.items()},
        "geometry_panel_codes": {"0":"both valid","1":"target invalid only","2":"counter invalid only","3":"both invalid"},
        "target-invalid prediction pixels": "displayed gray as unreliable; no invalid-region performance claim",
        "contact_sheet": str(pdf),
    })
    template=[{"category":k,"selected_patch_id":"","reviewer":"","notes":""} for k in rows_by_category]
    write_csv(OUT/"02_qualitative/final_case_data/manual_case_selection_template.csv",template)
    full_rows, t1, t2 = build_tables(preds,y,g,c,align_accum)
    checklist(full_rows,t1,t2,rows_by_category)
    print(f"INFERENCE_AND_PACKAGE_OK external={external}; candidate_counts="+
          str({k:len(v) for k,v in rows_by_category.items()}),flush=True)


if __name__=="__main__": main()
