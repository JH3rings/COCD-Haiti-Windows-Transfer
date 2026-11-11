"""Final-method candidates (2026-11-12): structure additions around the
event-shared prototype module (EPM) + cross-event Wasserstein relation
distillation (L_CEW).  Official split, seed 42, one test after freezing.

Problem being solved: the SAR U-Net memorises disaster EVENTS (same-event
validation 0.91 vs new-event test 0.52); the optical Teacher's specific
pairwise relations are tied to the training events, only the SHAPE of its
cross-event relation distribution transfers (pr.py, ANALYSIS_ZH.md s.18-19).

Structures tested here (each serves "do not memorise events"):
  MS  multi-scale EPM + L_CEW: decoder stage 2 (32 px, 4x4 regions per
      patch) and stage 3 (64 px, 8x8 finer regions).  Landslide sizes span
      orders of magnitude; SAR's weak spots are small objects and boundaries.
  EA  event-adversarial separation branch (training only): a small event
      classifier reads the student's stage-2 region features through a
      gradient-reversal layer; the classifier learns to tell which disaster
      event a region came from, the reversed gradient makes the student's
      features event-indistinguishable.  Removed at deployment.

Arms:  F_MS, F_EA, F_MSEA   (single-scale reference = pr.py EPRM_W)
    python final_method.py --check
    python final_method.py --arms F_MS
    python final_method.py --test F_MS F_EA F_MSEA
"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import common as C
import methods as M
import ours_rel as O
import pr as P
import run as R

LEVELS = {"F_MS": (1, 2), "F_EA": (1,), "F_MSEA": (1, 2)}
GRID = {1: 4, 2: 8}                 # regions per patch side at each decoder stage
ADV = {"F_MS": False, "F_EA": True, "F_MSEA": True}
LAMBDA_W = 30.                      # pr.EPRM_W value; multi-scale averages the levels
LAMBDA_ADV = 5.                     # adversarial gradient ~= segmentation gradient at init (0.21x at 1)
SEED_NEW = 20261113


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class EventClassifier(nn.Module):
    def __init__(self, c, n_events):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, n_events))

    def forward(self, tokens):
        return self.net(GradReverse.apply(tokens))


class FinalNet(C.UNet):
    def __init__(self, levels):
        super().__init__(6, 1)
        self.levels = levels
        widths = self.widths[-2::-1]
        self.epm = nn.ModuleDict({str(l): P.EPRM(widths[l]) for l in levels})

    def features(self, x):
        enc = []
        for i, block in enumerate(self.enc_blocks):
            x = block(F.max_pool2d(x, 2) if i else x)
            enc.append(x)
        dec = []
        for i, (up, block, skip) in enumerate(zip(self.up, self.dec_blocks, enc[-2::-1])):
            x = block(torch.cat((skip, up(x)), 1))
            if str(i) in self.epm:
                x = self.epm[str(i)](x)
            dec.append(x)
        self.enc, self.dec = enc, dec
        return x


def make(arm):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED_NEW)
        m = FinalNet(LEVELS[arm])
    return m


def initial(arm):
    m = make(arm)
    state = torch.load(C.RESULTS / "initial_student.pt", map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("epm.") for k in missing), (missing, unexpected)
    return m.cuda()


def frozen(arm):
    m = make(arm)
    sel = C.read(C.RESULTS / arm / "selection.json")
    assert C.sha(C.RESULTS / arm / "best.pt") == sel["checkpoint_sha256"]
    m.load_state_dict(torch.load(C.RESULTS / arm / "best.pt", map_location="cpu", weights_only=True))
    return m.cuda().eval().requires_grad_(False)


def tokens(fmap, valid, g):
    v = F.interpolate(valid.float(), size=fmap.shape[-2:], mode="area")
    cover = F.adaptive_avg_pool2d(v, g)
    pooled = F.adaptive_avg_pool2d(fmap.float() * v, g) / cover.clamp_min(1e-6)
    support = cover.flatten() >= .75
    return F.normalize(pooled.permute(0, 2, 3, 1).flatten(0, 2)[support], dim=1), support


def event_codes(b, support, g, index):
    ev = torch.tensor([index[e] for e in b["event_id"]], device=support.device)
    return ev.repeat_interleave(g * g)[support]


def build(arm, event_index):
    s = initial(arm)
    T = R.teacher_fusion()
    extra = ()
    clf = None
    if ADV[arm]:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(SEED_NEW + 1)
            clf = EventClassifier(s.widths[-2::-1][1], len(event_index)).cuda()
        extra = (clf,)

    def objective(b, epoch):
        z = s(b["x"])
        seg = C.segmentation(z, b)
        with torch.no_grad():
            T(M.fusion_input(b))
        logs = {"seg": seg.detach()}
        w_terms = []
        for l in LEVELS[arm]:
            g = GRID[l]
            tt, sup = tokens(T.dec[l], b["valid"], g)
            ts, _ = tokens(s.dec[l], b["valid"], g)
            ev = event_codes(b, sup, g, event_index)
            w = P.cross_event_wasserstein(ts, tt, ev)
            logs[f"w{l}"] = w.detach()
            w_terms.append(w)
        loss = seg + LAMBDA_W * sum(w_terms) / len(w_terms)
        if clf is not None:
            g = GRID[1]
            ts, sup = tokens(s.dec[1], b["valid"], g)
            ev = event_codes(b, sup, g, event_index)
            adv = F.cross_entropy(clf(ts), ev)
            with torch.no_grad():
                acc = (clf.net(ts).argmax(1) == ev).float().mean()
            loss = loss + LAMBDA_ADV * adv
            logs.update({"adv_ce": adv.detach(), "event_acc": acc})
        return loss, logs
    return s, extra, objective, (lambda b: s(b["x"]).sigmoid()), {}


def event_index_of(frame):
    return {e: i for i, e in enumerate(sorted(frame.event_id.unique()))}


def check():
    C.setup()
    O.prepare()
    f = C.frames()
    idx = event_index_of(f["train"])
    R.TRAIN = C.Inputs(f["train"].iloc[::4], augment=True)
    batches = [C.cuda(b) for _, b in zip(range(3), C.loader(R.TRAIN, True, torch.Generator().manual_seed(0)))]
    base = R.initial_student().eval()
    for arm in LEVELS:
        s, extra, objective, _, _ = build(arm, idx)
        with torch.no_grad():
            print(arm, "identity at init:", float((s.eval()(batches[0]["x"]) - base(batches[0]["x"])).abs().max()),
                  "extra deployed params:", sum(p.numel() for n, p in s.named_parameters() if n.startswith("epm.")))
        s.train()
        params = list(s.parameters())
        flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])
        for b in batches:
            loss, logs = objective(b, 1)
            g_tot = flat(torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True))
            z = s(b["x"])
            g_seg = flat(torch.autograd.grad(C.segmentation(z, b), params, allow_unused=True))
            g = g_tot - g_seg
            print(f"  aux/seg {float(g.norm() / g_seg.norm()):.2f}", {k: round(float(v), 4) for k, v in logs.items()})
        if ADV[arm]:
            b = batches[0]
            s(b["x"])
            ts, sup = tokens(s.dec[1], b["valid"], GRID[1])
            ev = event_codes(b, sup, GRID[1], idx)
            ga = flat(torch.autograd.grad(F.cross_entropy(extra[0](ts), ev), params, allow_unused=True))
            z = s(b["x"])
            gs = flat(torch.autograd.grad(C.segmentation(z, b), params, allow_unused=True))
            print(f"  adversarial-only grad / seg grad at lambda 1: {float(ga.norm() / gs.norm()):.3f}")


def test(arms):
    data = C.Inputs(C.frames()["test"], optical=False)
    for arm in arms:
        out = C.RESULTS / arm
        if not (out / "test_metrics.json").exists():
            m = frozen(arm)
            C.write(out / "test_started.json", {"arm": arm})
            C.write(out / "test_metrics.json", C.evaluate(lambda b: m(b["x"]).sigmoid(), data,
                                                          save=out / "test_predictions.npz", full=True))
    rows = {}
    for arm, path in [("B0", C.HERE / "results" / "B0"), ("PKDOT", C.HERE / "results" / "PKDOT"),
                      ("EPRM_W", C.RESULTS / "EPRM_W")] + [(a, C.RESULTS / a) for a in arms]:
        r = C.read(path / "test_metrics.json")
        rows[arm] = {"all120": round(100 * r["IoU"], 2), "ind44": round(100 * r["per_official_subset"]["testind"]["IoU"], 2),
                     "spt76": round(100 * r["per_official_subset"]["testspt"]["IoU"], 2),
                     "P": round(100 * r["Precision"], 1), "R": round(100 * r["Recall"], 1),
                     "AUPRC": round(100 * r["AUPRC"], 2), "val": round(100 * C.read(path / "selection.json")["validation_IoU"], 2)}
        print(arm, rows[arm], flush=True)
    C.write(C.RESULTS / "comparison_final_structures_seed42.json", rows)


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
    O.prepare()
    if a.test is not None:
        test(a.test)
        return
    f = C.frames()
    idx = event_index_of(f["train"])
    R.TRAIN, R.VAL = C.Inputs(f["train"], augment=True), C.Inputs(f["val"])
    for arm in a.arms:
        C.write(C.RESULTS / arm / "config.json", {"arm": arm, "levels": LEVELS[arm], "grid": GRID, "adversarial": ADV[arm],
                                                  "lambda_w": LAMBDA_W, "lambda_adv": LAMBDA_ADV, "prototypes": P.PROTOS,
                                                  "events": idx, "seed": C.SEED, "doc": __doc__})
        model, extra, objective, predict, kw = build(arm, idx)
        C.fit(arm, model, R.TRAIN, R.VAL, objective, predict, extra_params=extra, **kw)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
