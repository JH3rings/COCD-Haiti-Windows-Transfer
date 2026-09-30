"""Correct/refresh derived temporal-proxy artifacts from the completed audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
summary_path = OUT / "audit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
patches = pd.read_csv(OUT / "dataset_patch_audit.csv")
eligible = patches[(patches.gt_frac >= 0.02) & (patches.gt_bothvalid_pixels > 0)].copy()
a = pd.to_numeric(eligible.asc_vv_contrast_bothvalid, errors="coerce")
d = pd.to_numeric(eligible.desc_vv_contrast_bothvalid, errors="coerce")
ratio = a / d.clip(lower=0.01)
reverse_ratio = d / a.clip(lower=0.01)
asc_only = (a >= 1.20) & (d <= 1.05) & (ratio >= 1.5)
desc_only = (d >= 1.20) & (a <= 1.05) & (reverse_ratio >= 1.5)

def ids(mask):
    return sorted(eligible.loc[mask, "sample_id"].astype(int).tolist())

summary["possible_temporal_mismatch_proxy"] = {
    "candidate_rule": "GT fraction >=2%, at least one BothValid GT pixel; one orbit GT/BG |ΔVV| contrast >=1.20, other <=1.05, ratio >=1.5",
    "candidate_count_ASC_only": int(asc_only.sum()),
    "candidate_count_DESC_only": int(desc_only.sum()),
    "candidate_total_union": int((asc_only | desc_only).sum()),
    "eligible_with_bothvalid_GT": int(len(eligible)),
    "candidate_sample_ids_ASC_only": ids(asc_only),
    "candidate_sample_ids_DESC_only": ids(desc_only),
    "interpretation": "possible temporal/orbit-view discordance; not a mislabeled-polygon determination",
}
summary["temporal_proxy_filter_correction"] = {
    "note": "The first audit pass ranked Type5 anomalies before applying the GT>=2% eligibility filter. The primary audit statistics were unaffected. This finalizer applied GT>=2% and BothValid GT eligibility before computing candidate counts and the Type5 top-k list.",
    "eligible_count": int(len(eligible)),
    "eligible_rule": "gt_frac >= 0.02 and gt_bothvalid_pixels > 0",
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

anomalies = pd.read_csv(OUT / "anomaly_topk.csv")
anomalies = anomalies[anomalies.anomaly_type != "Type5_possible_temporal_mismatch_proxy"]
candidates = eligible.loc[asc_only | desc_only].copy()
candidates["temporal_discordance"] = np.abs(np.log(a.loc[candidates.index].clip(lower=0.01) /
                                                       d.loc[candidates.index].clip(lower=0.01)))
top = candidates.sort_values("temporal_discordance", ascending=False).head(10)
rows = []
for rank, (_, r) in enumerate(top.iterrows(), 1):
    rows.append({
        "anomaly_type": "Type5_possible_temporal_mismatch_proxy",
        "rank": rank,
        "sample_id": int(r.sample_id),
        "gt_frac": float(r.gt_frac),
        "gt_bothvalid_frac": float(r.gt_bothvalid_frac),
        "asc_gt_in_bad": float(r.asc_gt_in_bad),
        "desc_gt_in_bad": float(r.desc_gt_in_bad),
        "gt_slope_under10_frac": float(r.gt_slope_under10_frac),
        "asc_slope_mask_spearman": float(r.asc_slope_mask_spearman),
        "desc_slope_mask_spearman": float(r.desc_slope_mask_spearman),
        "asc_vv_contrast_bothvalid": float(r.asc_vv_contrast_bothvalid),
        "desc_vv_contrast_bothvalid": float(r.desc_vv_contrast_bothvalid),
        "selection_is_review_candidate_not_error": True,
    })
anomalies = pd.concat([anomalies, pd.DataFrame(rows)], ignore_index=True)
boundary_path = OUT / "gt_boundary_change_alignment.csv"
if boundary_path.exists():
    boundary = pd.read_csv(boundary_path)
    anomalies = anomalies.drop(columns=[c for c in boundary.columns if c != "sample_id" and c in anomalies.columns], errors="ignore")
    anomalies = anomalies.merge(boundary, on="sample_id", how="left", validate="many_to_one")
diagnostic_columns = [
    "slope_mean", "asc_bad_frac", "desc_bad_frac", "asc_gt_in_bad", "desc_gt_in_bad",
    "asc_dvv_abs_gt_bothvalid", "desc_dvv_abs_gt_bothvalid",
    "asc_dvh_abs_gt_bothvalid", "desc_dvh_abs_gt_bothvalid",
    "asc_vv_contrast_bothvalid", "desc_vv_contrast_bothvalid",
    "asc_dvv_abs_matched_diff", "desc_dvv_abs_matched_diff",
    "asc_dvh_abs_matched_diff", "desc_dvh_abs_matched_diff",
    "asc_joint_abs_matched_diff", "desc_joint_abs_matched_diff",
    "orbit_contrast_log_disagreement",
]
diagnostic_columns = [c for c in diagnostic_columns if c in patches.columns]
anomalies = anomalies.drop(columns=[c for c in diagnostic_columns if c in anomalies.columns], errors="ignore")
anomalies = anomalies.merge(patches[["sample_id", *diagnostic_columns]], on="sample_id",
                            how="left", validate="many_to_one")
anomalies.to_csv(OUT / "anomaly_topk.csv", index=False)
print(json.dumps(summary["possible_temporal_mismatch_proxy"], indent=2))
print(f"Updated Type5 rows: {len(rows)}")
