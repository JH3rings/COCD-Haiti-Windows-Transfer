"""Which cross-event relations did each frozen seed-42 student actually learn?
(validation + training patches; no test).  For every region, the optical
Teacher's 30 least-similar OTHER-EVENT regions; share of them the student also
puts among its own 30 least-similar other-event regions, split by pair type
(landslide-background vs background-background).  Compared with test IoU."""
import numpy as np
import torch
from torch.nn import functional as F

import common as C
import methods as M
import ours_rel as O
import pr as P
import run as R

K = 30
C.setup(); O.prepare()
f = C.frames()
T = R.teacher_fusion()
unet = {"B0": C.HERE / "results" / "B0", "PKDOT": C.HERE / "results" / "PKDOT",
        "PKDOT_LBL": C.RESULTS / "PKDOT_LBL", "OursP1": C.RESULTS / "OursP1"}
eprm = ("EPRM", "EPRM_CEraw", "EPRM_CEraw_repro", "EPRM_RD", "EPRM_STR", "EPRM_LC", "EPRM_SEPnb", "EPRM_SEP", "EPRM_CC")
models = {}
for a, p in unet.items():
    m = C.student()
    m.load_state_dict(torch.load(p / "best.pt", map_location="cpu", weights_only=True))
    models[a] = m.cuda().eval()
for a in eprm:
    models[a] = P.frozen(a)
test = {a: C.read((unet.get(a) or C.RESULTS / a) / "test_metrics.json")["IoU"] for a in models}
out = {}
for split in ("train", "val"):
    ds = C.Inputs(f[split])
    acc = {a: {"fg-bg": [], "bg-bg": []} for a in models}
    with torch.no_grad():
        for b in C.loader(ds, True, torch.Generator().manual_seed(7)):
            b = C.cuda(b)
            T(M.fusion_input(b))
            tt, sup = O.tokens(T.dec[1], b["valid"])
            fg, _, ev = O.token_meta(b, sup)
            cross = ev[:, None] != ev[None, :]
            if (cross.sum(1) < K).any():
                continue
            big = 9.
            it = (tt @ tt.T).masked_fill(~cross, big).topk(K, largest=False).indices
            lab = fg >= .05
            kind_fb = (lab[:, None] != lab[it])
            kind_bb = (~lab[:, None]) & (~lab[it])
            for a, m in models.items():
                m(b["x"])
                ts, _ = O.tokens(m.dec[1], b["valid"])
                own = (ts @ ts.T).masked_fill(~cross, big).topk(K, largest=False).indices
                hit = (it[:, :, None] == own[:, None, :]).any(-1)
                acc[a]["fg-bg"].append(float(hit[kind_fb].float().mean()))
                acc[a]["bg-bg"].append(float(hit[kind_bb].float().mean()))
    out[split] = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
print(f"{'arm':11s} {'test':>6s} | train fg-bg  bg-bg | val fg-bg  bg-bg")
for a in sorted(models, key=lambda a: -test[a]):
    print(f"{a:11s} {100*test[a]:6.2f} | {out['train'][a]['fg-bg']:.3f}  {out['train'][a]['bg-bg']:.3f}"
          f" | {out['val'][a]['fg-bg']:.3f}  {out['val'][a]['bg-bg']:.3f}")
for k in ("fg-bg", "bg-bg"):
    for s in ("train", "val"):
        x = [out[s][a][k] for a in models]; y = [test[a] for a in models]
        print(f"corr(test IoU, {s} {k} overlap) = {np.corrcoef(x, y)[0, 1]:.2f}")
C.write(C.RESULTS / "diag_learned.json", {"test_IoU": test, "overlap": out})
