"""Read-only final-v2 four-case recovery screen. Illustrative, not an unbiased test.

The S0/S2 checkpoints were selected with test monitoring in their training
script. This screen never retrains or changes the split/checkpoints; it records
all candidates and selects geographically disjoint cases by a fixed rule.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import rasterio
from scipy.ndimage import label
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from paths import DEV, DATASET_ROOT  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2  # noqa: E402

OUT = ROOT / "experiments/GRSL_complete_paper_package/99_historical_qualitative/four_region_rescue"
S_DIR = ROOT / "experiments/windows_main/student_target_search_v2"
T_PATH = ROOT / "experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
SEED = 20260928
THR = 0.5
SIDE = 96
MIN_COMPONENT = 12
MIN_REGION_GT = 25
MIN_REGION_PIXELS = 100
MIN_G00_CROP_FRAC = 0.70


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box(y: int, x: int) -> tuple[int, int, int, int]:
    return max(0, min(128 - SIDE, y - SIDE // 2)), max(0, min(128 - SIDE, x - SIDE // 2)), SIDE, SIDE


def overlaps(a: dict, b: dict) -> bool:
    # Same map CRS for the four raw acquisitions at each location.
    ax0, ay0, ax1, ay1 = a["bounds"]
    bx0, by0, bx1, by1 = b["bounds"]
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def error_map(p: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h = p >= THR
    z = np.zeros(gt.shape, np.uint8)
    z[h & gt] = 1  # TP
    z[~h & gt] = 2  # FN
    z[h & ~gt] = 3  # FP
    return z


def infer() -> tuple[list[dict], dict, dict]:
    train_ids, val_ids, test_ids = split()
    train = HaitiPairs(train_ids, True)
    ds = HaitiPairs(test_ids, False, train.stats)
    models = {}
    for name in ("S0", "S2"):
        model = StudentTargetSearchV2().to(DEV).eval()
        model.load_state_dict(torch.load(S_DIR / f"{name}_seed42.pt", map_location=DEV, weights_only=True))
        models[name] = model
    teacher = TargetSearchTeacherV2().to(DEV).eval()
    teacher.load_state_dict(torch.load(T_PATH, map_location=DEV, weights_only=True))

    rows: list[dict] = []
    scenes: dict[str, dict] = {}
    with torch.inference_mode():
        for i, sid in enumerate(test_ids):
            aa, dd, yy, ga, gd, _ = ds[i]
            a, d = aa[None].to(DEV), dd[None].to(DEV)
            ta, td = teacher.forward_pair(a, d, "counter")
            outputs = {}
            for orbit, x, t in (("asc", a, ta), ("desc", d, td)):
                outputs[orbit] = {name: torch.sigmoid(model(x)["z"])[0, 0].cpu().numpy()
                                  for name, model in models.items()}
                outputs[orbit]["CG10"] = torch.sigmoid(t["z"])[0, 0].cpu().numpy()
            gt = yy.numpy() > 0
            gm = {"asc": ga.numpy() > 0, "desc": gd.numpy() > 0}
            with rasterio.open(DATASET_ROOT / "Pre_event/S1_ASC_20210805" / f"{sid}.tif") as src:
                bounds = tuple(src.bounds)
            for orbit, other in (("asc", "desc"), ("desc", "asc")):
                p = outputs[orbit]
                pc = outputs[other]
                sidkey = f"{sid:06d}_{orbit}"
                scenes[sidkey] = dict(sample_id=int(sid), orbit=orbit, other=other,
                                      gt=gt, ga=ga.numpy(), gd=gd.numpy(),
                                      p=p, pc=pc, bounds=bounds)
                categories = {
                    "distorted_complement": gm[orbit] & ~gm[other],
                    "both_valid_complement": ~gm[orbit] & ~gm[other],
                }
                err0 = (p["S0"] >= THR) != gt
                err2 = (p["S2"] >= THR) != gt
                rescue_core = gt & err0 & ~err2 & (pc["S0"] >= THR) & (p["CG10"] >= THR)
                for category, region in categories.items():
                    connected, n = label(rescue_core & region)
                    for k in range(1, n + 1):
                        ys, xs = np.where(connected == k)
                        if len(ys) < MIN_COMPONENT:
                            continue
                        top, left, h, w = box(int(round(ys.mean())), int(round(xs.mean())))
                        sl = np.s_[top:top+h, left:left+w]
                        reg = region[sl]
                        g = gt[sl]
                        if (g & reg).sum() < MIN_REGION_GT or reg.sum() < MIN_REGION_PIXELS:
                            continue
                        if category == "both_valid_complement" and reg.mean() < MIN_G00_CROP_FRAC:
                            continue
                        e0, e2 = err0[sl], err2[sl]
                        won = int((e0 & ~e2 & reg).sum())
                        lost = int((~e0 & e2 & reg).sum())
                        crop_won = int((e0 & ~e2).sum())
                        crop_lost = int((~e0 & e2).sum())
                        if won <= lost or crop_won <= crop_lost:
                            continue
                        strict = int((rescue_core[sl] & reg).sum())
                        rows.append(dict(scene_id=sidkey, sample_id=int(sid), target_orbit=orbit,
                                         category=category, top=top, left=left, height=h, width=w,
                                         focus_y=int(round(ys.mean())), focus_x=int(round(xs.mean())),
                                         component_pixels=len(ys), strict_rescue_pixels=strict,
                                         target_region_pixels=int(reg.sum()), target_region_GT_pixels=int((g & reg).sum()),
                                         target_region_fraction=float(reg.mean()),
                                         S0_target_errors=int((e0 & reg).sum()),
                                         S2_target_errors=int((e2 & reg).sum()),
                                         region_won=won, region_lost=lost, region_net=won-lost,
                                         crop_S0_errors=int(e0.sum()), crop_S2_errors=int(e2.sum()),
                                         crop_net=crop_won-crop_lost,
                                         counter_S0_correct_on_strict=int((rescue_core[sl] & reg).sum()),
                                         teacher_correct_on_strict=strict,
                                         bounds=bounds))
            if (i + 1) % 25 == 0:
                print(f"inferred {i+1}/{len(test_ids)} locations", flush=True)
    provenance = dict(split="data/splits_v3; train stats recomputed from train IDs only",
                      train_locations=len(train_ids), validation_locations=len(val_ids), test_locations=len(test_ids),
                      threshold=THR, seed=SEED, box_pixels=SIDE,
                      source_checkpoints={p.name: sha(p) for p in [S_DIR/"S0_seed42.pt", S_DIR/"S2_seed42.pt", T_PATH]},
                      warning="S0/S2 checkpoints were selected using monitored test AUPRC in student_search_kd_v2.py; cases are illustrative development evidence, not independent held-out validation")
    return rows, scenes, provenance


def aggregate(scenes: dict[str, dict]) -> list[dict]:
    """Exact fixed-threshold regional counts across all screened test scenes."""
    groups: dict[tuple[str, str, str], dict] = {}
    for scene in scenes.values():
        orbit, other = scene["orbit"], scene["other"]
        target = (scene["ga"] if orbit == "asc" else scene["gd"]) > 0
        counter = (scene["gd"] if orbit == "asc" else scene["ga"]) > 0
        masks = {"G00": ~target & ~counter, "G10": target & ~counter,
                 "G01": ~target & counter, "G11": target & counter}
        y = scene["gt"]
        for region, m in masks.items():
            for method in ("S0", "S2"):
                key = (orbit, region, method)
                out = groups.setdefault(key, dict(pixels=0, gt_pixels=0, tp=0, fp=0, fn=0,
                                                   target_S0_FN_rescued_by_S2=0,
                                                   target_S0_FN_rescued_counter_S0_teacher_S2=0,
                                                   new_S2_errors=0))
                h = scene["p"][method] >= THR
                s0 = scene["p"]["S0"] >= THR
                s2 = scene["p"]["S2"] >= THR
                other_s0 = scene["pc"]["S0"] >= THR
                teacher = scene["p"]["CG10"] >= THR
                out["pixels"] += int(m.sum())
                out["gt_pixels"] += int((y & m).sum())
                out["tp"] += int((h & y & m).sum())
                out["fp"] += int((h & ~y & m).sum())
                out["fn"] += int((~h & y & m).sum())
                out["target_S0_FN_rescued_by_S2"] += int((y & m & ~s0 & s2).sum())
                out["target_S0_FN_rescued_counter_S0_teacher_S2"] += int((y & m & ~s0 & s2 & other_s0 & teacher).sum())
                out["new_S2_errors"] += int((m & (s0 == y) & (s2 != y)).sum())
    rows = []
    for (orbit, region, method), v in sorted(groups.items()):
        tp, fp, fn = (v[x] for x in ("tp", "fp", "fn"))
        rows.append(dict(target_orbit=orbit, region=region, method=method, **v,
                         IoU=tp/(tp+fp+fn) if tp+fp+fn else None,
                         F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    return rows


def select(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    # Rule fixes category/orbit composition, disjoint source footprints and rank.
    ordered = sorted(rows, key=lambda r: (-r["region_net"], -r["strict_rescue_pixels"],
                                          -r["crop_net"], r["sample_id"], r["top"], r["left"]))
    targets = [("distorted_complement", "asc"), ("distorted_complement", "desc"),
               ("both_valid_complement", "asc"), ("both_valid_complement", "desc")]
    chosen = []
    for category, orbit in targets:
        # Keep the 40-px detail window away from the 128-px tile boundary.
        pool = [r for r in ordered if r["category"] == category and r["target_orbit"] == orbit
                and 20 <= r["focus_y"] <= 107 and 20 <= r["focus_x"] <= 107]
        hit = next((r for r in pool if all(not overlaps(r, s) for s in chosen)), None)
        if hit is not None:
            chosen.append(hit)
    return chosen, ordered


def draw_case(row: dict, scene: dict, path: Path) -> None:
    sid, orbit, other = row["sample_id"], row["target_orbit"], scene["other"]
    raw = {}
    for o in (orbit, other):
        for e, sub in (("pre", f"Pre_event/S1_{o.upper()}_{'20210805' if o=='asc' else '20210803'}"),
                       ("post", f"Post_event/S1_{o.upper()}_{'20210817' if o=='asc' else '20210815'}")):
            with rasterio.open(DATASET_ROOT / sub / f"{sid}.tif") as src:
                raw[o, e] = src.read(2)
    gt, p, pc = scene["gt"], scene["p"], scene["pc"]
    geom = scene["ga"] if orbit == "asc" else scene["gd"]
    geom_other = scene["gd"] if orbit == "asc" else scene["ga"]
    target_valid, counter_valid = geom == 0, geom_other == 0
    g00 = target_valid & counter_valid
    g10 = ~target_valid & counter_valid
    regions = np.zeros(gt.shape, np.uint8)
    regions[g10] = 1
    regions[target_valid & ~counter_valid] = 2
    regions[~target_valid & ~counter_valid] = 3
    focus_region = g10 if row["category"] == "distorted_complement" else g00
    s0, s2 = p["S0"] >= THR, p["S2"] >= THR
    strict = gt & focus_region & ~s0 & s2 & (pc["S0"] >= THR) & (p["CG10"] >= THR)
    e0, e2 = s0 != gt, s2 != gt
    change = np.zeros(gt.shape, np.uint8)
    change[e0 & ~e2] = 1
    change[~e0 & e2] = 2
    masked_error = error_map(p["S2"], gt) + 1
    masked_error[~target_valid] = 0
    color_err = ListedColormap(["#1d2430", "#31ad6a", "#e04747", "#f4bd3b"])
    norm_err = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    geom_cmap = ListedColormap(["#eeeeee", "#ed8a35", "#4775bd", "#b34daa"])
    geom_norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    fig, ax = plt.subplots(3, 5, figsize=(18, 12), constrained_layout=True)
    kind = "G10: target distorted, counter valid" if row["category"] == "distorted_complement" else "G00: both valid"
    fig.suptitle(f"{sid:06d} | target {orbit.upper()} | {kind} | full 128 x 128 source patch (no crop)", fontsize=13)
    def panel(a, x, title, cmap="gray", vmin=None, vmax=None, norm=None, gt_outline=False):
        a.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest")
        if gt_outline and gt.any():
            a.contour(gt, levels=[.5], colors="#f000ff", linewidths=.7)
        a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])
    for j, o in enumerate((orbit, other)):
        lo, hi = np.percentile(np.r_[raw[o, "pre"].ravel(), raw[o, "post"].ravel()], [2, 98])
        panel(ax[0, j*2], raw[o, "pre"], f"{o.upper()} pre VV + GT outline", vmin=lo, vmax=hi, gt_outline=True)
        panel(ax[0, j*2+1], raw[o, "post"], f"{o.upper()} post VV + GT outline", vmin=lo, vmax=hi, gt_outline=True)
    panel(ax[0,4], gt, "GT: white = landslide", ListedColormap(["#1d2430", "#ffffff"]))
    panel(ax[1,0], geom, "Target geometry", geom_cmap, norm=geom_norm)
    panel(ax[1,1], geom_other, "Counter geometry", geom_cmap, norm=geom_norm)
    panel(ax[1,2], regions, "Joint geometry G00/G10/G01/G11", geom_cmap, norm=geom_norm)
    panel(ax[1,3], error_map(p["S0"], gt), "S0 target (all pixels)", color_err, norm=norm_err)
    panel(ax[1,4], error_map(p["S2"], gt), "S2 target (all pixels)", color_err, norm=norm_err)
    panel(ax[2,0], error_map(pc["S0"], gt), "S0 counter (all pixels)", color_err, norm=norm_err)
    panel(ax[2,1], error_map(p["CG10"], gt), "CG10 teacher (all pixels)", color_err, norm=norm_err)
    panel(ax[2,2], strict, "Strict rescue in selected region", ListedColormap(["#1d2430", "#43d17d"]))
    panel(ax[2,3], change, "S0 to S2: fixed / new errors", ListedColormap(["#1d2430", "#43d17d", "#d4509e"]), norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
    panel(ax[2,4], masked_error, "S2 evaluated on target-valid only", ListedColormap(["#b8bec7", "#1d2430", "#31ad6a", "#e04747", "#f4bd3b"]), norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5],5))
    fig.supxlabel("Magenta outlines on SAR = GT landslide; geometry gray=valid, orange=layover, blue=shadow, purple=both. Joint map gray=G00, orange=G10, blue=G01, purple=G11.\nError maps: green TP, red FN, yellow FP, dark TN. Last panel grays out target-invalid pixels; those pixels must still be assessed separately as G10/G11. Fixed/new map: green=fixed, pink=new error. Test-monitored illustrative evidence.", fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=230)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def case_region_metrics(row: dict, scene: dict) -> list[dict]:
    gt = scene["gt"]
    target = scene["ga"] if scene["orbit"] == "asc" else scene["gd"]
    counter = scene["gd"] if scene["orbit"] == "asc" else scene["ga"]
    masks = {"all": np.ones(gt.shape, bool), "target_valid_only": target == 0,
             "G00_both_valid": (target == 0) & (counter == 0),
             "G10_target_distorted_counter_valid": (target > 0) & (counter == 0),
             "G11_both_distorted": (target > 0) & (counter > 0)}
    result = []
    for name, mask in masks.items():
        for model in ("S0", "S2"):
            pred = scene["p"][model] >= THR
            tp = int((pred & gt & mask).sum())
            fp = int((pred & ~gt & mask).sum())
            fn = int((~pred & gt & mask).sum())
            result.append(dict(scene_id=row["scene_id"], target_orbit=row["target_orbit"],
                               region=name, method=model, pixels=int(mask.sum()),
                               gt_pixels=int((gt & mask).sum()), tp=tp, fp=fp, fn=fn,
                               F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, scenes, provenance = infer()
    regional = aggregate(scenes)
    write_csv(OUT/"regional_metrics.csv", regional)
    if not rows:
        (OUT/"screen_status.json").write_text(json.dumps({"status":"no candidates",**provenance},indent=2),encoding="utf-8")
        return
    selected, ranked = select(rows)
    write_csv(OUT/"all_candidate_crops.csv", ranked)
    write_csv(OUT/"selected_cases.csv", selected)
    case_metrics = []
    for idx,row in enumerate(selected,1):
        draw_case(row, scenes[row["scene_id"]], OUT/f"case_{idx}_{row['category']}_{row['target_orbit']}")
        case_metrics.extend(case_region_metrics(row, scenes[row["scene_id"]]))
    write_csv(OUT/"case_region_metrics.csv", case_metrics)
    status=dict(status="four cases found" if len(selected)==4 else "fewer than four evidence-matched cases",
                selected=len(selected), candidates=len(rows), categories=[(r["category"],r["target_orbit"]) for r in selected],
                **provenance)
    (OUT/"screen_status.json").write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:status[k] for k in ("status","selected","candidates","categories")},indent=2),flush=True)


if __name__ == "__main__":
    main()
