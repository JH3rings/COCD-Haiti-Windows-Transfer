"""Zero-training diagnostic: which region relations does PKDOT transfer, and are
they relations SAR already has?   (frozen seed-42 models, train/val only)

PKDOT (run.py / methods.pkdot_relations): decoder map dec[1] (32 px, 256 ch),
4x4 region tokens per patch, pooled over the WHOLE batch (16 patches, so pairs
span patches and events), cosine-similarity matrix, for every token the 30
pairs the Teacher finds least similar, Sinkhorn between Teacher and Student
relation profiles.

Per batch (training order, no augmentation) we measure for Teacher T, B0 and
the trained PKDOT student:
  overlap@30   share of T's 30 least-similar pairs that the student also ranks
               among its own 30 least similar (1 = SAR already separates them)
  confused     share of T's 30 pairs that the student puts in the upper half
               of its similarity row (SAR thinks they are alike)
  pair types   landslide/background token mix, same vs cross event,
               SAR-change evidence of the tokens
"""
import numpy as np
import torch
from torch.nn import functional as F

import common as C
import methods as M
import run as R

K = 30
FG = .05          # token is "landslide" if >=5% of its valid pixels are labelled


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
    ok = F.adaptive_avg_pool2d(b["optical_ok"].float(), 4).flatten()[support]
    ev = np.repeat(np.array(b["event_id"]), 16)[support.cpu().numpy()]
    return fg.cpu().numpy(), sar.cpu().numpy(), ok.cpu().numpy(), ev


@torch.no_grad()
def run(ds, models, T, shuffle_seed=7):
    stats = {n: {"overlap": [], "confused": [], "spearman": []} for n in models}
    types = {"T30": [], "conf_B0": [], "all": []}
    gen = torch.Generator().manual_seed(shuffle_seed)
    ds.augment = False
    for b in C.loader(ds, True, gen):
        b = C.cuda(b)
        T(M.fusion_input(b))
        t, sup = tokens(T.dec[1], b["valid"])
        if len(t) < K + 2:
            continue
        fg, sar, ok, ev = token_meta(b, sup)
        st = t @ t.T
        st.fill_diagonal_(2)                       # exclude self from "least similar"
        idx = st.topk(K, largest=False).indices    # N,K
        n = len(t)
        lab = fg >= FG
        for name, m in models.items():
            m(b["x"])
            s, _ = tokens(m.dec[1], b["valid"])
            ss = s @ s.T
            ss_self = ss.clone()
            ss_self.fill_diagonal_(2)
            own = ss_self.topk(K, largest=False).indices
            hit = (idx[:, :, None] == own[:, None, :]).any(-1).float().mean()
            rank = ss.argsort(1).argsort(1).float() / (n - 1)          # 0 = least similar
            conf = (rank.gather(1, idx) > .5).float()
            stats[name]["overlap"].append(float(hit))
            stats[name]["confused"].append(float(conf.mean()))
            iu = torch.triu_indices(n, n, 1, device=st.device)
            a, c = st[iu[0], iu[1]], ss[iu[0], iu[1]]
            ra, rc = a.argsort().argsort().float(), c.argsort().argsort().float()
            stats[name]["spearman"].append(float(torch.corrcoef(torch.stack((ra, rc)))[0, 1]))
            if name == "B0":
                ii = np.repeat(np.arange(n), K)
                jj = idx.flatten().cpu().numpy()
                cm = conf.flatten().cpu().numpy() > .5
                for key, sel in (("T30", np.ones_like(cm)), ("conf_B0", cm)):
                    types[key].append(np.stack([lab[ii][sel], lab[jj][sel], ev[ii][sel] == ev[jj][sel],
                                                np.minimum(sar[ii], sar[jj])[sel],
                                                np.minimum(ok[ii], ok[jj])[sel]], 1))
                ai, aj = np.triu_indices(n, 1)
                types["all"].append(np.stack([lab[ai], lab[aj], ev[ai] == ev[aj],
                                              np.minimum(sar[ai], sar[aj]), np.minimum(ok[ai], ok[aj])], 1))
    out = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in stats.items()}
    sar_thr = None
    summary = {}
    for key, rows in types.items():
        a = np.concatenate(rows)
        if sar_thr is None:
            sar_thr = float(np.quantile(np.concatenate(types["all"])[:, 3], .5))
        l1, l2 = a[:, 0] > .5, a[:, 1] > .5
        summary[key] = {"pairs": int(len(a)), "fg-fg": float((l1 & l2).mean()), "fg-bg": float((l1 ^ l2).mean()),
                        "bg-bg": float((~l1 & ~l2).mean()), "same_event": float(a[:, 2].mean()),
                        "both_SAR_change_above_median": float((a[:, 3] > sar_thr).mean()),
                        "optical_ok>=0.99": float((a[:, 4] >= .99).mean())}
    return out, summary


def main():
    C.setup()
    f = C.frames()
    R.TRAIN, R.VAL = C.Inputs(f["train"]), C.Inputs(f["val"])
    T = R.teacher_fusion()
    models = {"B0": R.frozen("B0", C.student()), "PKDOT": R.frozen("PKDOT", C.student()),
              "DIS2": R.frozen("DIS2", C.student())}
    res = {}
    for name, ds in (("train", R.TRAIN), ("val", R.VAL)):
        rel, typ = run(ds, models, T)
        res[name] = {"relations_vs_teacher": rel, "pair_types": typ}
        print(f"\n== {name}")
        for k, v in rel.items():
            print(f"  {k:6s}", {kk: round(vv, 3) for kk, vv in v.items()})
        for k, v in typ.items():
            print(f"  {k:8s}", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    C.write(C.RESULTS / "diag_pairs.json", res)


if __name__ == "__main__":
    main()
