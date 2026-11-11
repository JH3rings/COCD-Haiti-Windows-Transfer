"""Is the prototype module used, and are its concepts shared across events?
(frozen seed-42 EPRM arms; validation + test patches, after all arms froze)"""
import numpy as np
import torch
from torch.nn import functional as F

import common as C
import ours_rel as O
import pr as P

C.setup(); O.prepare()
f = C.frames()
for split in ("val", "test"):
    ds = C.Inputs(f[split], optical=False)
    for arm in ("EPRM", "EPRM_CEraw", "EPRM_CE"):
        m = P.frozen(arm)
        ratio, ent, dom = [], [], {}
        with torch.no_grad():
            for b in C.loader(ds):
                b = C.cuda(b)
                cap = {}
                h = m.eprm.out.register_forward_hook(lambda mod, i, o: cap.__setitem__("o", o))
                h2 = m.eprm.register_forward_hook(lambda mod, i, o: cap.__setitem__("f", i[0]))
                m(b["x"]); h.remove(); h2.remove()
                ratio.append(float(cap["o"].norm() / cap["f"].norm()))
                a = m.eprm.assign
                ent.append(float(-(a * a.clamp_min(1e-8).log()).sum(1).mean() / np.log(P.PROTOS)))
                y = F.adaptive_avg_pool2d(b["mask"], a.shape[-1]) > .5
                v = F.adaptive_avg_pool2d(b["valid"].float(), a.shape[-1]) > .99
                top = a.argmax(1)
                for i, e in enumerate(b["event_id"]):
                    d = dom.setdefault(e, [np.zeros(P.PROTOS), np.zeros(P.PROTOS)])
                    for cls in (0, 1):
                        sel = top[i][(y[i, 0] == bool(cls)) & v[i, 0]]
                        d[cls] += np.bincount(sel.cpu().numpy(), minlength=P.PROTOS)
        # per prototype: share of its pixels that are landslide (purity), and how many events it serves for landslides
        fg = sum(d[1] for d in dom.values()); bg = sum(d[0] for d in dom.values())
        purity = fg / np.maximum(fg + bg, 1)
        ls_protos = np.where(purity > 3 * fg.sum() / (fg.sum() + bg.sum()))[0]
        ev_with_fg = [e for e, d in dom.items() if d[1].sum() > 50]
        cover = np.mean([d[1][ls_protos].sum() / d[1].sum() for e, d in dom.items() if e in ev_with_fg]) if len(ls_protos) else 0
        print(f"{split:4s} {arm:11s} |O(ctx)|/|f| {np.mean(ratio):.3f}  assign entropy {np.mean(ent):.3f}  "
              f"used protos {(fg + bg > 0).sum()}  landslide-enriched protos {len(ls_protos)}  "
              f"share of each event's landslide px on them {cover:.3f} (events {len(ev_with_fg)})", flush=True)
