"""Where does optical help, and does SAR carry evidence there?  (train patches only)

Uses the official-split out-of-fold Teachers (each trained on the other event
group, so predictions are not memorised labels):
  p_sar  = SAR-only OOF Teacher,  p_fus = SAR+optical OOF Teacher.
For label-positive pixels: recall of each by bins of slope, pre-event NDVI,
SAR change magnitude, landslide object size, boundary/interior.
For label-negative pixels: false-positive rate by the same bins.
"optical helps" = fus right & sar wrong.  If those pixels still have a strong
SAR change signal, SAR *could* learn them (learnable headroom); if SAR change is
~0 there, they are SAR-invisible.
"""
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

import common as C
import dev as D
import run as R


def main():
    C.setup()
    f = C.frames()
    R.TRAIN, R.VAL = C.Inputs(f["train"]), C.Inputs(f["val"])
    D.TEACHER_ROOT = C.HERE / "results_final" / "teachers"
    C.RESULTS = C.HERE / "results_final" / "seed42"
    zs, zf = D.oof_teachers("sar"), D.oof_teachers("fusion")
    rows = []
    for it in R.TRAIN.cache:
        sid = it["sample_id"]
        v = it["valid"][0].numpy()
        y = it["mask"][0].numpy() > .5
        x = it["x"].numpy()
        ps = torch.sigmoid(zs[sid][0]).numpy()
        pf = torch.sigmoid(zf[sid][0]).numpy()
        dvv, dvh = x[2] - x[0], x[3] - x[1]
        sar_change = np.sqrt(dvv ** 2 + dvh ** 2)
        lab, n = ndimage.label(y)
        size = np.zeros_like(lab, float)
        if n:
            counts = np.bincount(lab.ravel())
            size = counts[lab].astype(float)
            size[lab == 0] = 0
        interior = ndimage.binary_erosion(y, iterations=2)
        boundary = y & ~interior
        ndvi_pre = it["ndvi"][0].numpy()
        d = pd.DataFrame({"event": it["event_id"], "y": y[v], "ps": ps[v] >= .5, "pf": pf[v] >= .5,
                          "slope": x[5][v], "ndvi_pre": ndvi_pre[v], "dndvi": (it["ndvi"][1] - it["ndvi"][0]).numpy()[v],
                          "sar_change": sar_change[v], "size": size[v], "boundary": boundary[v]})
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    pos, neg = d[d.y], d[~d.y]
    out = {}

    def table(frame, col, bins, labels, what):
        g = frame.groupby(pd.cut(frame[col], bins, labels=labels), observed=False)
        if what == "recall":
            t = g.agg(n=("y", "size"), SAR=("ps", "mean"), FUSION=("pf", "mean"))
            hl = g.apply(lambda s: (s.pf & ~s.ps).mean(), include_groups=False)
            t["optical_helps"] = hl
        else:
            t = g.agg(n=("y", "size"), SAR_FP=("ps", "mean"), FUSION_FP=("pf", "mean"))
        return t.round(3)

    q = lambda s: [-np.inf] + list(np.quantile(s, [.25, .5, .75])) + [np.inf]
    quart = ["Q1 low", "Q2", "Q3", "Q4 high"]
    print("label-positive pixels:", len(pos), " recall SAR", round(pos.ps.mean(), 3), " FUSION", round(pos.pf.mean(), 3))
    for col in ("sar_change", "slope", "ndvi_pre", "dndvi"):
        t = table(pos, col, q(pos[col]), quart, "recall")
        print(f"\n[positives] by {col}\n{t.to_string()}")
        out[f"pos_{col}"] = t.reset_index().astype(str).to_dict("records")
    t = table(pos, "size", [0, 50, 200, 1000, np.inf], ["<50 px", "50-200", "200-1000", ">1000"], "recall")
    print(f"\n[positives] by object size\n{t.to_string()}")
    out["pos_size"] = t.reset_index().astype(str).to_dict("records")
    t = pos.groupby("boundary").agg(n=("y", "size"), SAR=("ps", "mean"), FUSION=("pf", "mean")).round(3)
    print(f"\n[positives] boundary(True) vs interior\n{t.to_string()}")
    for col in ("sar_change", "ndvi_pre", "slope"):
        t = table(neg, col, q(neg[col]), quart, "fp")
        print(f"\n[negatives] FP rate by {col}\n{t.to_string()}")
        out[f"neg_{col}"] = t.reset_index().astype(str).to_dict("records")
    helps = pos[pos.pf & ~pos.ps]
    both = pos[pos.pf & pos.ps]
    miss = pos[~pos.pf & ~pos.ps]
    print("\nSAR change magnitude (median) on positives: optical-helps", round(helps.sar_change.median(), 3),
          " both-right", round(both.sar_change.median(), 3), " both-miss", round(miss.sar_change.median(), 3),
          " background", round(neg.sar_change.median(), 3))
    thr = np.quantile(neg.sar_change, .9)
    print("share of optical-helps positives with SAR change above the background 90th pct:",
          round(float((helps.sar_change > thr).mean()), 3),
          "(both-right:", round(float((both.sar_change > thr).mean()), 3), ")")
    out["summary"] = {"recall_SAR": float(pos.ps.mean()), "recall_FUSION": float(pos.pf.mean()),
                      "fp_SAR": float(neg.ps.mean()), "fp_FUSION": float(neg.pf.mean()),
                      "optical_helps_share_of_pos": float((pos.pf & ~pos.ps).mean()),
                      "optical_hurts_share_of_pos": float((~pos.pf & pos.ps).mean()),
                      "helps_with_strong_sar_change": float((helps.sar_change > thr).mean()),
                      "bothright_with_strong_sar_change": float((both.sar_change > thr).mean())}
    ev = d.groupby("event").apply(lambda s: pd.Series({
        "pos": int(s.y.sum()), "recall_SAR": (s.ps & s.y).sum() / max(s.y.sum(), 1),
        "recall_FUS": (s.pf & s.y).sum() / max(s.y.sum(), 1),
        "fp_SAR": (s.ps & ~s.y).sum() / max((~s.y).sum(), 1), "fp_FUS": (s.pf & ~s.y).sum() / max((~s.y).sum(), 1)}),
        include_groups=False).round(3)
    print(f"\nper event\n{ev.to_string()}")
    out["per_event"] = ev.reset_index().to_dict("records")
    print("\nsummary", out["summary"])
    C.write(C.HERE / "results" / "diag_regions.json", out)


if __name__ == "__main__":
    main()
