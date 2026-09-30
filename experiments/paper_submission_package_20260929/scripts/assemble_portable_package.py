from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
OUT = Path(os.environ.get("HAITI_PAPER_PACKAGE_OUT", str(EXP / "paper_submission_package_20260929")))
PAPER = EXP / "GRSL_complete_paper_package"
MASTER = EXP / "summary" / "MASTER_METRICS.csv"

REGIONS = ["Overall", "BothValid", "TargetValid", "G10", "G01"]
METRICS = ["AUPRC", "IoU", "F1"]
BASELINES = {
    "boehm": "Boehm SAR U-Net++",
    "cdnette": "CDNetE Early Fusion",
    "deeplabv3": "DeepLabV3",
    "mfewf": "MFEWF adapted",
    "unet": "Standard U-Net",
}
CORE_METHODS = ["S0-Raw", "Hinton-KD-Port", "DIS2-Port", "Multi2Uni-Port",
                "MMOKD-Port", "S2-Raw-native", "S2-2Stage"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: Path, rel: str | None = None) -> None:
    if not src.exists() or not src.is_file():
        return
    dest = OUT / (rel or str(src.relative_to(EXP)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_tree_files(src: Path, dest_rel: str, extensions: set[str] | None = None,
                    exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    for p in src.rglob("*"):
        if not p.is_file() or p.name in exclude_names:
            continue
        if extensions is not None and p.suffix.lower() not in extensions:
            continue
        rel = p.relative_to(src)
        copy_file(p, str(Path(dest_rel) / rel))


def metric_row(method: str, run_id: str, region: str, y: np.ndarray,
               p: np.ndarray, pixels: int, source: str) -> dict:
    y = np.asarray(y, dtype=bool).reshape(-1)
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    pred = p >= 0.5
    tp = int(np.logical_and(pred, y).sum())
    fp = int(np.logical_and(pred, ~y).sum())
    fn = int(np.logical_and(~pred, y).sum())
    denom = 2 * tp + fp + fn
    iou_denom = tp + fp + fn
    return {
        "RunID": run_id, "Method": method, "TrainingRegime": "Raw/no geometry mask",
        "Protocol": "single-orbit baseline v3; geometry used post hoc only",
        "BestEpoch": "", "Region": region,
        "AUPRC": float(average_precision_score(y, p)) if y.any() else None,
        "IoU": float(tp / iou_denom) if iou_denom else 0.0,
        "F1": float(2 * tp / denom) if denom else 0.0,
        "Pixels": int(pixels), "TrainingGeometryMask": "No",
        "TrainIDs": "see source progress.json", "ValidationIDs": "see source progress.json",
        "TestIDs": 343, "CheckpointSelection": "see source progress.json",
        "SourceMetrics": source,
    }


def evaluate_external_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict] = []
    audit_rows: list[dict] = []
    reference = None
    for key, label in BASELINES.items():
        path = EXP / "single_orbit_baselines_v3" / key / "test_predictions.npz"
        with np.load(path) as z:
            p = z["p"].astype(np.float32)
            y = z["y"] > 0
            gt = z["gt"].astype(np.uint8)  # target-orbit geometry code
            gc = z["gc"].astype(np.uint8)  # counter-orbit geometry code
        if p.shape != y.shape or p.shape != gt.shape or p.shape != gc.shape:
            raise ValueError(f"shape mismatch in {path}: {p.shape}, {y.shape}, {gt.shape}, {gc.shape}")
        codes = set(np.unique(gt).tolist()) | set(np.unique(gc).tolist())
        if not codes.issubset({0, 1, 2, 3}):
            raise ValueError(f"unexpected geometry codes in {path}: {codes}")
        signature = (y, gt, gc)
        if reference is None:
            reference = signature
        elif not (np.array_equal(reference[0], y) and np.array_equal(reference[1], gt)
                  and np.array_equal(reference[2], gc)):
            raise ValueError(f"test labels or geometry order differs in {path}")
        masks = {
            "Overall": np.ones_like(y, dtype=bool),
            "OverallValid": (gt == 0) | (gc == 0),
            "TargetValid": gt == 0,
            "BothValid": (gt == 0) & (gc == 0),
            "G10": (gt > 0) & (gc == 0),
            "G01": (gt == 0) & (gc > 0),
        }
        run_id = f"single_orbit_baselines_v3/{key}"
        source = str(path.relative_to(EXP)).replace("\\", "/")
        for region, mask in masks.items():
            metrics_rows.append(metric_row(label, run_id, region, y[mask], p[mask],
                                           int(mask.sum()), source))
        total = int(y.size)
        target_codes = ";".join(map(str, sorted(np.unique(gt).tolist())))
        counter_codes = ";".join(map(str, sorted(np.unique(gc).tolist())))
        overall = next(r for r in metrics_rows if r["RunID"] == run_id and r["Region"] == "Overall")
        metrics_path = path.parent / "metrics.csv"
        stored = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        src_overall = stored[stored["partition"].astype(str).str.lower() == "overall"] if len(stored) else pd.DataFrame()
        if len(src_overall):
            row = src_overall.iloc[0]
            for field in ("auprc", "iou", "f1"):
                recorded = float(row[field])
                metric_name = {"auprc": "AUPRC", "iou": "IoU", "f1": "F1"}[field]
                calculated = float(overall[metric_name])
                audit_rows.append({"Method": label, "Metric": field, "Recorded": recorded,
                                   "Reevaluated": calculated, "AbsoluteDifference": abs(recorded-calculated),
                                   "Agreement_1e-5": abs(recorded-calculated) <= 1e-5,
                                   "Source": f"{key}/metrics.csv vs test_predictions.npz"})
        progress_path = path.parent / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        for r in metrics_rows[-len(masks):]:
            r["BestEpoch"] = progress.get("best_epoch", progress.get("selected_epoch", ""))
            r["Protocol"] = f"single-orbit baseline v3; training geometry mask not applied; source {key}/progress.json"
        if not total:
            raise ValueError(f"empty predictions: {path}")
    return pd.DataFrame(metrics_rows), pd.DataFrame(audit_rows)


def to_wide(long: pd.DataFrame, index_fields: list[str] | None = None) -> pd.DataFrame:
    index_fields = index_fields or ["Method"]
    wide = long.pivot_table(index=index_fields, columns="Region", values=METRICS, aggfunc="first")
    wide.columns = [f"{region}_{metric}" for metric, region in wide.columns]
    return wide.reset_index()


def copy_curated_sources() -> None:
    # Current GRSL results, plots, table sources, and reproduction notes.
    copy_file(PAPER / "README.md", "07_provenance/current_grsl_package_readme.md")
    for rel in [
        "01_method/fig1_method.png", "01_method/fig1_method.svg",
        "02_qualitative/fig2_candidates_contact_sheet.pdf",
        "02_qualitative/candidate_manifest.json", "02_qualitative/case_selection.csv",
        "02_qualitative/qualitative_summary.md",
        "03_search_behavior/figure_search_behavior_candidate.png",
        "03_search_behavior/figure_search_behavior_candidate.pdf",
        "03_search_behavior/search_statistics.csv", "03_search_behavior/search_statistics.json",
        "03_search_behavior/search_query_metadata.json",
        "04_tables/full_metrics.csv", "04_tables/reevaluated_region_metrics.csv",
        "04_tables/table1_main.csv", "04_tables/table1_main.tex",
        "04_tables/table2_ablation.csv", "04_tables/table2_ablation.tex",
        "05_analysis/region_gain.csv", "05_analysis/rescue_analysis.csv",
        "05_analysis/s2_2stage_source_summary.json", "grsl_evidence_checklist.md",
        "07_reproduction/baseline_notes.md", "07_reproduction/mmokd_spec.md",
        "07_reproduction/related_multiview_methods.md",
    ]:
        src = PAPER / rel
        if rel.startswith("02_qualitative/") and src.parent.name == "final_case_data":
            continue
        copy_file(src, str(Path("07_provenance/current_grsl_package") / rel))
    for name in ["fig2_candidates_contact_sheet.pdf", "candidate_manifest.json", "case_selection.csv"]:
        copy_file(PAPER / "02_qualitative" / name, f"03_qualitative/{name}")
    # Candidate scenes are included as candidates, with no claim of human approval.
    qsrc = PAPER / "02_qualitative" / "candidates"
    if qsrc.exists():
        copy_tree_files(qsrc, "03_qualitative/candidates", {".png", ".csv"})

    # The package owns these model/result cards; only compact summaries are duplicated.
    for sub in ["hinton_kd", "dis2_port", "multi2uni_port", "ours_s2_2stage"]:
        src = EXP / "distillation_final" / sub
        copy_tree_files(src, f"07_provenance/run_cards/distillation_final/{sub}",
                        {".json", ".md", ".csv"},
                        {"test_predictions.npz"})
    mm = PAPER / "baselines" / "mmokd_port"
    copy_tree_files(mm, "07_provenance/run_cards/mmokd_port", {".json", ".md", ".csv"},
                    {"test_predictions.npz"})
    for key in BASELINES:
        src = EXP / "single_orbit_baselines_v3" / key
        copy_tree_files(src, f"07_provenance/run_cards/single_orbit_baselines_v3/{key}",
                        {".json", ".csv", ".md"}, {"test_predictions.npz"})
    for f in ["README.md", "method_reproduction_spec.md", "comparison/final_report.md"]:
        copy_file(EXP / "distillation_final" / f, f"07_provenance/distillation_final/{f}")

    # Dataset geometry/physics evidence needed to interpret mask partitions.
    audit = EXP / "dataset_audit_final"
    for rel in ["README.md", "final_dataset_audit.md", "00_看图入口.md"]:
        copy_file(audit / rel, f"06_geometry_audit/{rel}")
    for name in ["geometry_summary.csv", "joint_geometry_summary.csv",
                 "g10_g01_statistics.csv", "g10_g01_patch_statistics.csv",
                 "patch_distribution_summary.csv", "file_integrity.csv"]:
        copy_file(audit / "01_statistics" / name, f"06_geometry_audit/statistics/{name}")
    for name in ["aspect_geometry_heatmap.png", "gt_elevation_aspect.png", "slope_gt_geometry.png"]:
        copy_file(audit / "01_statistics" / "figures" / name,
                  f"02_figures/geometry_audit/{name}")

    # All existing final comparison plots, explicitly segregated by protocol family.
    copy_tree_files(EXP / "summary" / "figures", "02_figures/project_comparisons",
                    {".png", ".pdf"})
    # TGD and external checkpoint retests: compact metric/report files only.
    tgd = EXP / "tgd_robustness_analysis"
    copy_tree_files(tgd, "05_diagnostics/tgd_robustness", {".csv", ".json", ".md"})
    # Keep the quantitative mask runs as report cards, not their model weights/log bulk.
    mask_dirs = [
        "windows_main/geometry_transfer_restored_v1",
        "windows_main/mask_aware_search_kd_v5_seed42",
        "windows_main/sgri_powc_masked_v4_seed42_20260925",
    ]
    for rel in mask_dirs:
        src = EXP / rel
        copy_tree_files(src, f"05_diagnostics/{Path(rel).name}",
                        {".csv", ".json", ".md"}, {"test_predictions.npz"})
    # Matched KD-budget reports and curves; the three controllers are kept separate.
    kd_dirs = [p for p in (EXP / "windows_main").glob("multitoone_kd_10pct*") if p.is_dir()]
    for src in kd_dirs:
        copy_tree_files(src, f"05_diagnostics/{src.name}", {".csv", ".json", ".md"})


def create_figures(core: pd.DataFrame, external: pd.DataFrame, all_long: pd.DataFrame) -> None:
    figdir = OUT / "02_figures" / "generated"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})

    # Main deployment benchmark: AUPRC on five fixed geometry partitions.
    methods = core["Method"].tolist()
    vals = core.set_index("Method")
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    colors = ["#243b53", "#2a9d8f", "#e9c46a", "#e76f51", "#6c63a8"]
    x = np.arange(len(methods))
    for region, color in zip(REGIONS, colors):
        ax.plot(x, [vals.loc[m, f"{region}_AUPRC"] for m in methods], marker="o",
                lw=1.8, label=region, color=color)
    ax.set_xticks(x, methods, rotation=28, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1)
    ax.set_title("Final target-only benchmark by geometry region")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "final_benchmark_auprc_regions.png", bbox_inches="tight")
    fig.savefig(figdir / "final_benchmark_auprc_regions.pdf", bbox_inches="tight")
    plt.close(fig)

    # External single-orbit baselines by region, all re-evaluated from saved logits.
    ext = external[external.Region.isin(["Overall", "BothValid", "TargetValid"])].copy()
    pivot = ext.pivot(index="Method", columns="Region", values="AUPRC")
    order = [m for m in BASELINES.values() if m in pivot.index]
    pivot = pivot.loc[order, ["Overall", "TargetValid", "BothValid"]]
    ax = pivot.plot(kind="bar", figsize=(10.5, 4.8), color=["#243b53", "#2a9d8f", "#e9c46a"])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1)
    ax.set_title("External single-orbit baselines by geometry region")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=3)
    ax.figure.tight_layout()
    ax.figure.savefig(figdir / "external_single_orbit_auprc_regions.png", bbox_inches="tight")
    ax.figure.savefig(figdir / "external_single_orbit_auprc_regions.pdf", bbox_inches="tight")
    plt.close(ax.figure)

    # A protocol-separated raw/masked comparison; labels expose non-matched training variants.
    selected = all_long[
        ((all_long.Method == "S0") & all_long.TrainingRegime.isin(["Raw/no geometry mask", "Masked geometry"]))
        | ((all_long.Method == "S0-mask"))
        | ((all_long.Method == "S2-Raw-native"))
        | ((all_long.Method == "S2-mask"))
        | ((all_long.Method == "S2-MaskAware"))
    ].copy()
    selected = selected[selected.Region.isin(["BothValid", "TargetValid", "Overall"])].copy()
    label_map = {
        "S0": "S0 / Raw",
        "S0-mask": "S0 / geometry-masked v4",
        "S2-Raw-native": "S2 native / Raw",
        "S2-mask": "S2 / geometry-masked v4",
        "S2-MaskAware": "S2 MaskAware / legacy split",
    }
    selected["FullLabel"] = selected["Method"] + " | " + selected["TrainingRegime"].astype(str)
    selected["Label"] = selected["Method"].map(label_map).fillna(selected["Method"]) \
        + " | " + selected["TrainingRegime"].astype(str).str.replace("Raw/no geometry mask", "Raw", regex=False) \
        .str.replace("Masked geometry", "mask", regex=False)
    selected = selected.drop_duplicates(["Label", "Region"])
    selected.to_csv(OUT / "01_results" / "mask_protocol_plot_data.csv", index=False)
    pivot = selected.pivot(index="Label", columns="Region", values="AUPRC")
    pivot = pivot.loc[:, [c for c in ["Overall", "TargetValid", "BothValid"] if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11.5, 5.4), color=["#243b53", "#2a9d8f", "#e9c46a"][:len(pivot.columns)])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1)
    ax.set_title("Raw and geometry-masked runs (protocols shown separately)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=3)
    ax.figure.tight_layout()
    ax.figure.savefig(figdir / "raw_vs_masked_protocols_auprc.png", bbox_inches="tight")
    ax.figure.savefig(figdir / "raw_vs_masked_protocols_auprc.pdf", bbox_inches="tight")
    plt.close(ax.figure)

    # Raw/masked comparison on fixed partitions, with all three segmentation measures.
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.0), sharey=True)
    for ax, metric in zip(axes, METRICS):
        pvt = selected.pivot(index="Label", columns="Region", values=metric)
        pvt = pvt.loc[:, [c for c in ["BothValid", "TargetValid", "Overall"] if c in pvt.columns]]
        pvt.plot(kind="barh", ax=ax, color=["#e9c46a", "#2a9d8f", "#243b53"][:len(pvt.columns)])
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.2)
        ax.get_legend().remove()
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("")
    fig.suptitle("Geometry-mask diagnostics — not a matched main-method comparison")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .07, 1, .93])
    fig.savefig(figdir / "raw_vs_masked_protocols_triplet.png", bbox_inches="tight")
    fig.savefig(figdir / "raw_vs_masked_protocols_triplet.pdf", bbox_inches="tight")
    plt.close(fig)

    # Training convergence for the final benchmark arms, using their saved histories.
    curves = {
        "S0-Raw": EXP / "windows_main/raw_cver_v1_seed42/S0-Raw_progress.json",
        "S2-Raw-native": EXP / "windows_main/raw_cver_v1_seed42/S2-Raw_progress.json",
        "S2-2Stage": EXP / "distillation_final/ours_s2_2stage/progress.json",
        "Hinton-KD-Port": EXP / "distillation_final/hinton_kd/progress.json",
        "DIS2-Port": EXP / "distillation_final/dis2_port/progress.json",
        "Multi2Uni-Port": EXP / "distillation_final/multi2uni_port/progress.json",
        "MMOKD-Port": PAPER / "baselines/mmokd_port/progress.json",
    }
    rows = []
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for label, path in curves.items():
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for item in d.get("history", []):
            score = item.get("test_overall_auprc", item.get("monitor_test_auprc_raw_overall"))
            if score is None:
                continue
            epoch = int(item.get("epoch", 0))
            rows.append({"Method": label, "Epoch": epoch, "TestOverallAUPRC": float(score),
                         "Source": str(path.relative_to(EXP)).replace("\\", "/")})
        sub = [r for r in rows if r["Method"] == label]
        if sub:
            ax.plot([r["Epoch"] for r in sub], [r["TestOverallAUPRC"] for r in sub],
                    marker=".", ms=3, lw=1.2, label=label)
    pd.DataFrame(rows).to_csv(OUT / "01_results" / "final_training_curves.csv", index=False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Overall AUPRC (project monitor)")
    ax.set_title("Recorded convergence curves for final benchmark arms")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "final_training_curves.png", bbox_inches="tight")
    fig.savefig(figdir / "final_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def build() -> None:
    existing_files = [p for p in OUT.rglob("*") if p.is_file() and p.resolve() != Path(__file__).resolve()]
    if existing_files:
        raise FileExistsError(f"Refusing to overwrite package files: {existing_files[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["00_manifest", "01_results", "02_figures", "03_qualitative",
              "04_geometry_mask", "05_diagnostics", "06_geometry_audit",
              "07_provenance", "scripts"]:
        (OUT / d).mkdir(exist_ok=True)

    master = pd.read_csv(MASTER)
    external, audit = evaluate_external_baselines()
    all_long = pd.concat([master, external], ignore_index=True, sort=False)
    all_long.to_csv(OUT / "01_results" / "all_methods_all_regions_long.csv", index=False)
    to_wide(all_long, ["RunID", "Method", "TrainingRegime", "Protocol", "BestEpoch"]).to_csv(
        OUT / "01_results" / "all_methods_all_regions_wide.csv", index=False)
    audit.to_csv(OUT / "01_results" / "external_baseline_metric_crosscheck.csv", index=False)

    full = pd.read_csv(PAPER / "04_tables" / "full_metrics.csv")
    full = full[full["Method"].isin(CORE_METHODS)].copy()
    full.to_csv(OUT / "01_results" / "final_core_benchmark.csv", index=False)
    ext_long = external[external.Region.isin(REGIONS)].copy()
    ext_wide = to_wide(ext_long)
    core_rows = []
    for _, row in full.iterrows():
        r = {"Method": row["Method"], "Status": row["Status"], "BestEpoch": row["best_epoch"],
             "InferenceParams": row["inference_params"], "InferenceGFLOPs": row["inference_gflops"],
             "DeploymentInput": row["deployment_input"], "Source": "GRSL_complete_paper_package/04_tables/full_metrics.csv"}
        for region in REGIONS:
            for metric in METRICS:
                col = f"{region}_{metric}"
                r[col] = row.get(col, np.nan)
        core_rows.append(r)
    for _, row in ext_wide.iterrows():
        base_key = next(k for k, v in BASELINES.items() if v == row["Method"])
        meta_path = EXP / "single_orbit_baselines_v3" / base_key / "metrics.csv"
        meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        overall_row = meta[meta["partition"].astype(str).str.lower() == "overall"] if len(meta) else pd.DataFrame()
        recorded = pd.read_csv(EXP / "single_orbit_baselines_v3" / "comparison_table.csv")
        recorded_row = recorded[recorded["method"] == row["Method"]] if len(recorded) else pd.DataFrame()
        row_out = {"Method": row["Method"], "Status": "EXTERNAL_SINGLE_ORBIT_BASELINE",
                   "BestEpoch": int(overall_row.iloc[0]["selected_epoch"]) if len(overall_row) else "not recorded",
                   "InferenceParams": (float(recorded_row.iloc[0]["parameters_M"]) * 1e6
                                       if len(recorded_row) else "not recorded"),
                   "InferenceGFLOPs": "not recorded in source table", "DeploymentInput": "target orbit only",
                   "Source": f"single_orbit_baselines_v3/{base_key}/test_predictions.npz; re-evaluated"}
        for region in REGIONS:
            for metric in METRICS:
                row_out[f"{region}_{metric}"] = row.get(f"{region}_{metric}", np.nan)
        core_rows.append(row_out)
    final_table = pd.DataFrame(core_rows)
    final_table.to_csv(OUT / "01_results" / "paper_main_comparison_all_regions.csv", index=False)

    mask_methods = {"S0-clean", "S2-MaskAware", "S0-mask", "S2-mask", "SGRI-A1", "SGRI-A2",
                    "POWC-B1", "POWC-B2", "SG-Teacher", "CG-Teacher", "CG-shuffle",
                    "SG Teacher", "CG Teacher", "CG shuffle",
                    "S0", "S2", "SG", "CG", "CG_shuffle", "ReliabilityKD", "StableEvidenceKD"}
    mask_protocol = master[(master["Method"].isin(mask_methods)) |
                           (master["Method"].isin(["S2-Raw-native", "S2-2Stage"]))].copy()
    mask_protocol.to_csv(OUT / "04_geometry_mask" / "mask_and_nomask_diagnostics_all.csv", index=False)
    mask_protocol[mask_protocol["TrainingRegime"].astype(str).str.contains("Masked geometry", na=False)].to_csv(
        OUT / "04_geometry_mask" / "masked_runs_only.csv", index=False)
    master[master["TrainingRegime"].astype(str).str.contains("matched KD-budget", na=False)].to_csv(
        OUT / "05_diagnostics" / "equal_kd_budget_all_completed_runs.csv", index=False)

    # Protocol/run map with stable paths relative to the experiment hub.
    run_map = [
        ["S0-Raw", "windows_main/raw_cver_v1_seed42/S0-Raw_seed42_best.pt", "final core; no geometry training mask"],
        ["S2-Raw-native", "windows_main/raw_cver_v1_seed42/S2-Raw_seed42_best.pt", "historical native Search-KD; geometry-weighted queries"],
        ["S2-2Stage", "distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt", "final core; target-only deployment"],
        ["Hinton-KD-Port", "distillation_final/hinton_kd/Hinton-KD-Port_seed42_best.pt", "external comparison"],
        ["DIS2-Port", "distillation_final/dis2_port/DIS2-Port_seed42_best.pt", "external comparison; low result retained"],
        ["Multi2Uni-Port", "distillation_final/multi2uni_port/Multi2Uni-Port_seed42_best.pt", "external comparison; mechanism-faithful port"],
        ["MMOKD-Port", "GRSL_complete_paper_package/baselines/mmokd_port", "external comparison; checkpoint and progress in source directory"],
        ["SG/CG teachers", "windows_main/teacher_target_search_v2/", "teacher/search diagnostics"],
        ["single-orbit baselines", "single_orbit_baselines_v3/", "Boehm, CDNetE, DeepLabV3, MFEWF, U-Net"],
        ["masked diagnostics", "windows_main/{geometry_transfer_restored_v1,mask_aware_search_kd_v5_seed42,sgri_powc_masked_v4_seed42_20260925}", "metrics/report cards included; source folders remain on disk and cleanup candidates are recorded in 00_curation/purge_manifest_20260929.csv"],
        ["qualitative prediction cache", "GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz", "retained source cache for the existing qualitative candidate/contact-sheet assets"],
    ]
    pd.DataFrame(run_map, columns=["MethodGroup", "SourcePath_relative_to_experiments", "Role"]).to_csv(
        OUT / "00_manifest" / "retained_run_map.csv", index=False)

    # Table and report files; all copied source content keeps the original protocol labels.
    copy_file(MASTER, "01_results/MASTER_METRICS_source_snapshot.csv")
    copy_file(EXP / "summary" / "MASTER_RUN_CATALOG.csv", "00_manifest/MASTER_RUN_CATALOG_source_snapshot.csv")
    copy_file(EXP / "single_orbit_baselines_v3" / "comparison_table.csv", "01_results/single_orbit_baseline_recorded_metrics.csv")
    copy_curated_sources()
    create_figures(final_table, external, all_long)

    # Store the builder itself for repeatable assembly; no network or training needed.
    builder_src = Path(__file__)
    if builder_src.resolve() != (OUT / "scripts" / builder_src.name).resolve():
        shutil.copy2(builder_src, OUT / "scripts" / builder_src.name)

    readme = """# Haiti COCD paper submission package — 2026-09-29\n\n"""
    readme += """## Start here\n\nThis is the new paper-facing package. Move the entire project `experiments/` folder to preserve run checkpoints and source outputs; this folder is the compact paper bundle with tables, plots, protocol cards, and links back to retained runs. Nothing here requires re-running training.\n\n"""
    readme += """The qualitative source prediction cache used by the current candidate panels is retained in `../GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz`.\n\n"""
    readme += """## Recommended paper table\n\nUse `01_results/paper_main_comparison_all_regions.csv`. It contains the 7 current GRSL core rows plus 5 single-orbit external baselines, with Overall, BothValid, TargetValid, G10 and G01 AUPRC/IoU/F1. External single-orbit baseline region metrics were re-evaluated from saved test predictions using the project metric definition; see `01_results/external_baseline_metric_crosscheck.csv`. The matching complete source predictions remain under `../single_orbit_baselines_v3/`.\n\n`S2-Raw-native` is explicitly marked historical because its native KD query weighting uses geometry-derived G10/G01 information. Do not fold it into the geometry-free S2-2Stage claim. DIS2-Port remains in the table despite its low score. Multi2Uni-Port and MMOKD-Port are mechanism-faithful ports, not official numerical reproductions.\n\n## Protocol limits\n\nThe locked project protocol has no validation split and uses fixed-test Overall AUPRC as the per-epoch checkpoint monitor. These are test-monitored development results, not independent test estimates. Raw/no-mask and masked-training rows have distinct protocols; geometry is post-hoc for the raw main benchmark. `G10` and `G01` are diagnostic partitions.\n\n## Package integrity\n\n`00_manifest/package_sha256.csv` lists every asset in this folder except the checksum file itself. The build script is under `scripts/`; it only copies existing artifacts and re-evaluates saved baseline predictions, and does not train models.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    inventory = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "package_sha256.csv":
            continue
        inventory.append({"Path": p.relative_to(OUT).as_posix(), "Bytes": p.stat().st_size,
                          "SHA256": sha256(p), "Role": "figure" if p.suffix.lower() in {".png", ".pdf", ".svg"} else "data_or_report"})
    pd.DataFrame(inventory).to_csv(OUT / "00_manifest" / "package_sha256.csv", index=False)
    print(json.dumps({"package": str(OUT), "files": len(inventory),
                      "bytes": sum(x["Bytes"] for x in inventory),
                      "main_methods": len(final_table), "external_baseline_rechecks": len(audit)}, indent=2))


if __name__ == "__main__":
    build()
