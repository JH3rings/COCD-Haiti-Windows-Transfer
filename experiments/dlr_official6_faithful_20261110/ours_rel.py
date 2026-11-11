"""Our relational method (2026-11-11), official split, seed 42, internal comparison.

Goal: make the SAR network organise ITS OWN features so that real landslides
and landslide-like background are separated, using optical only to say what
should be separated, never what it looks like.

Evidence it builds on (diag_pairs.py): PKDOT does not work by copying Teacher
relations (pairwise agreement with the Teacher drops 0.42 -> 0.28); 91.5% of
its constrained pairs are cross-event and 75% landslide-vs-background, i.e. it
acts as a cross-event separation constraint.  Half of the pairs SAR still
confuses are background-vs-background: which background is a confounder is
information labels do not contain and optical does.

Tokens: decoder dec[1] (32 px), 4x4 region tokens per patch, pooled over the
batch (as PKDOT).

  Point 1  cross-event relation structure: PKDOT's Sinkhorn on relation
           profiles, but every token's partners are restricted to tokens of
           OTHER events (30 the optical Teacher finds least similar).
  Point 2  optical-defined confounder separation: background tokens with SAR
           change above the batch median (SAR sees something) vs landslide
           tokens of other events.  One-sided: penalise only where the SAR
           student finds the pair MORE similar than the optical Teacher does
           (similarities standardised per batch over cross-event pairs).

Arms
  PKDOT_LBL  control: PKDOT unchanged except the Teacher relation is replaced
             by label-only relations (no optical anywhere).
  OursCE     Point 1 (lambda .6, PKDOT's) + Point 2 (lambda 1).
  OursP1     Point 1 only (ablation, run only if requested).

    python ours_rel.py --arms PKDOT_LBL
    python ours_rel.py --arms OursCE
    python ours_rel.py --test PKDOT_LBL OursCE
    python ours_rel.py --check
"""
import argparse
import shutil

import numpy as np
import torch
from torch.nn import functional as F

import common as C
import methods as M
import run as R

K = 30
LAMBDA_P1, LAMBDA_P2 = .6, 1.
FG = .05
ROOT = C.HERE / "results_rel"


def prepare():
    C.RESULTS = ROOT
    ROOT.mkdir(exist_ok=True)
    src = C.HERE / "results"
    if not (ROOT / "initial_student.pt").exists():
        shutil.copy2(src / "initial_student.pt", ROOT / "initial_student.pt")
    (ROOT / "T_fusion").mkdir(exist_ok=True)
    for f in ("best.pt", "selection.json"):
        if not (ROOT / "T_fusion" / f).exists():
            shutil.copy2(src / "T_fusion" / f, ROOT / "T_fusion" / f)
    assert C.sha(ROOT / "initial_student.pt") == C.sha(src / "initial_student.pt")


def tokens(fmap, valid):
    cover = F.adaptive_avg_pool2d(F.interpolate(valid.float(), size=fmap.shape[-2:], mode="area"), 4)
    v = F.interpolate(valid.float(), size=fmap.shape[-2:], mode="area")
    pooled = F.adaptive_avg_pool2d(fmap.float() * v, 4) / cover.clamp_min(1e-6)
    support = cover.flatten() >= .75
    return F.normalize(pooled.permute(0, 2, 3, 1).flatten(0, 2)[support], dim=1), support


def token_meta(b, support):
    v = b["valid"].float()
    cov = F.adaptive_avg_pool2d(v, 4).clamp_min(1e-6)
    fg = (F.adaptive_avg_pool2d(b["mask"] * v, 4) / cov).flatten()[support]
    x = b["x"]
    ch = torch.sqrt((x[:, 2:3] - x[:, 0:1]) ** 2 + (x[:, 3:4] - x[:, 1:2]) ** 2)
    sar = (F.adaptive_avg_pool2d(ch * v, 4) / cov).flatten()[support]
    codes = {e: i for i, e in enumerate(sorted(set(b["event_id"])))}
    ev = torch.tensor([codes[e] for e in b["event_id"]], device=fg.device).repeat_interleave(16)[support]
    return fg, sar, ev


def cross_event_relations(s, t, ev, sinkhorn):
    """Point 1: PKDOT relation profiles restricted to other-event partners."""
    cross = ev[:, None] != ev[None, :]
    rows = cross.sum(1) >= K
    if rows.sum() < 3:
        return s.sum() * 0, cross
    ss, st = s @ s.T, t @ t.T
    big = torch.finfo(st.dtype).max
    idx = st.masked_fill(~cross, big).topk(K, largest=False).indices[rows]
    st_min, ss_min = st[cross].min(), ss[cross].min()
    return sinkhorn((st - st_min)[rows].gather(1, idx), (ss - ss_min)[rows].gather(1, idx)), cross


def confounder_separation(s, t, fg, sar, cross):
    """Point 2: SAR-visible background (SAR change >= batch median, no label)
    vs other-event landslide tokens; one-sided, standardised similarities."""
    bg = (fg == 0) & (sar >= sar.median())
    pos = fg >= FG
    pair = bg[:, None] & pos[None, :] & cross
    if pair.sum() < 4:
        return s.sum() * 0, 0
    ss, st = s @ s.T, t @ t.T
    ref_s, ref_t = ss[cross].detach(), st[cross]
    zs = (ss - ref_s.mean()) / ref_s.std().clamp_min(1e-6)
    zt = (st - ref_t.mean()) / ref_t.std().clamp_min(1e-6)
    excess = F.relu(zs - zt)[pair]
    return excess.square().mean(), int(pair.sum())


def build(arm):
    from geomloss import SamplesLoss
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=.1, backend="tensorized")
    s = R.initial_student()
    T = R.teacher_fusion() if arm != "PKDOT_LBL" else None

    def objective(b, epoch):
        z = s(b["x"])
        seg = C.segmentation(z, b)
        if arm == "PKDOT_LBL":
            lab = torch.cat((b["mask"], 1 - b["mask"]), 1)
            lab = F.interpolate(lab, size=s.dec[1].shape[-2:], mode="area")
            r = M.pkdot_relations(s.dec[1], lab, b["valid"], sinkhorn)
            return seg + .6 * r, {"seg": seg.detach(), "sinkhorn": r.detach()}
        with torch.no_grad():
            T(M.fusion_input(b))
        st, sup = tokens(T.dec[1], b["valid"])
        sv, _ = tokens(s.dec[1], b["valid"])
        fg, sar, ev = token_meta(b, sup)
        p1, cross = cross_event_relations(sv, st, ev, sinkhorn)
        loss, logs = seg + LAMBDA_P1 * p1, {"seg": seg.detach(), "p1": p1.detach()}
        if arm == "OursCE":
            p2, n = confounder_separation(sv, st, fg, sar, cross)
            loss = loss + LAMBDA_P2 * p2
            logs.update({"p2": p2.detach(), "p2_pairs": torch.tensor(float(n))})
        return loss, logs
    return s, (), objective, R.student_predict(s), {}


def check():
    C.setup()
    prepare()
    f = C.frames()
    R.TRAIN = C.Inputs(f["train"].iloc[::4], augment=True)
    gen = torch.Generator().manual_seed(0)
    batches = [C.cuda(b) for i, b in zip(range(4), C.loader(R.TRAIN, True, gen))]
    for arm in ("PKDOT_LBL", "OursP1", "OursCE"):
        s, _, objective, _, _ = build(arm)
        params = list(s.parameters())
        flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])
        cos, ratio, logs_all = [], [], []
        for b in batches:
            loss, logs = objective(b, 1)
            seg = logs["seg"]
            g_tot = flat(torch.autograd.grad(loss, params, allow_unused=True))
            z = s(b["x"])
            g_seg = flat(torch.autograd.grad(C.segmentation(z, b), params, allow_unused=True))
            g_aux = g_tot - g_seg
            cos.append(float(F.cosine_similarity(g_seg, g_aux, 0)))
            ratio.append(float(g_aux.norm() / g_seg.norm()))
            logs_all.append({k: round(float(v), 4) for k, v in logs.items()})
        print(arm, "cos", np.round(cos, 3).tolist(), "aux/seg", np.round(ratio, 2).tolist(), logs_all[0], flush=True)


def test(arms):
    data = C.Inputs(C.frames()["test"], optical=False)
    for arm in arms:
        out = ROOT / arm
        if not (out / "test_metrics.json").exists():
            m = R.frozen(arm, C.student())
            C.write(out / "test_started.json", {"arm": arm})
            C.write(out / "test_metrics.json", C.evaluate(R.student_predict(m), data,
                                                          save=out / "test_predictions.npz", full=True))
    rows = {}
    for arm, path in [("B0", C.HERE / "results" / "B0"), ("PKDOT", C.HERE / "results" / "PKDOT")] + \
            [(a, ROOT / a) for a in arms]:
        r = C.read(path / "test_metrics.json")
        rows[arm] = {"all120": round(100 * r["IoU"], 2), "ind44": round(100 * r["per_official_subset"]["testind"]["IoU"], 2),
                     "spt76": round(100 * r["per_official_subset"]["testspt"]["IoU"], 2),
                     "P": round(100 * r["Precision"], 1), "R": round(100 * r["Recall"], 1),
                     "AUPRC": round(100 * r["AUPRC"], 2), "val": round(100 * C.read(path / "selection.json")["validation_IoU"], 2)}
        print(arm, rows[arm], flush=True)
    C.write(ROOT / "comparison_seed42.json", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        return
    C.setup()
    prepare()
    if a.test is not None:
        test(a.test)
        return
    f = C.frames()
    R.TRAIN, R.VAL = C.Inputs(f["train"], augment=True), C.Inputs(f["val"])
    for arm in a.arms:
        out = ROOT / arm
        C.write(out / "config.json", {"arm": arm, "K": K, "lambda_p1": LAMBDA_P1, "lambda_p2": LAMBDA_P2,
                                      "fg_token": FG, "tokens": "dec[1] 4x4 per patch, batch-pooled",
                                      "seed": C.SEED, "doc": __doc__})
        model, extra, objective, predict, kw = build(arm)
        C.fit(arm, model, R.TRAIN, R.VAL, objective, predict, extra_params=extra, **kw)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
