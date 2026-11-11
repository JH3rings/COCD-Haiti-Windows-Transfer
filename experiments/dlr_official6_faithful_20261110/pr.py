"""Event-shared Prototype Relation network + Cross-Event Optical Relation
Distillation (2026-11-11).  Official split, seed 42, internal comparison.

Problem (our diagnostics): landslide inventories are clustered by event; the
SAR U-Net memorises events (same-event val IoU 0.909 vs new-event test 54.6).
Optical cannot be imitated (~70% of what optical adds is SAR-invisible), but
the optical Teacher's CROSS-EVENT region relations transfer (OursP1 56.27 >
PKDOT 55.74 > label-only relations 54.09).

PKDOT (Aslam et al., pain estimation from video, BioVid): a TransformNet
hallucinates the missing physiological feature; OT aligns sample-level
(whole-clip) relation matrices.  Here instead:

STRUCTURE  Event-shared Prototype Relation Module (EPRM) on decoder dec[1]
  (32 px, 256 ch).  M learnable prototypes shared by every event.  Each pixel
  is softly assigned to the prototypes, a = softmax(cos(q(f), P) / tau), and
  refined by them: f <- f + O(a @ V(P)) (O zero-init: identical to B0 at
  step 0).  SAR does not imitate optical; it organises its own features into
  an event-independent concept space, kept at deployment (SAR-only input).
LOSS       Cross-Event Optical Relation Distillation (CE-ORD) in that space:
  region tokens (4x4 per patch, batch-pooled); Teacher relation = cosine of
  optical-Teacher region features; Student relation = cosine of the
  region's prototype-assignment distributions; each region is compared only
  with regions of OTHER events (30 the Teacher finds least similar); profiles
  aligned with Sinkhorn OT (PKDOT's core).  Optical therefore decides which
  regions of different events must share / not share a concept, SAR decides
  what the concepts are.

Arms
  EPRM        structure only (no optical)
  EPRM_CE     structure + CE-ORD in prototype space        (proposed)
  EPRM_CEraw  structure + CE-ORD on raw dec[1] features    (variant)
  EPRM_CC     structure + cross-event CORRESPONDENCE distillation (2026-11-11
              redesign, no OT / no hard pairs): KL between Teacher and Student
              soft distributions over all other-event regions

    python pr.py --check
    python pr.py --arms EPRM_CE
    python pr.py --test EPRM EPRM_CE EPRM_CEraw
"""
import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import common as C
import methods as M
import ours_rel as O
import run as R

PROTOS, PDIM, TAU = 16, 64, .1
TAU_T, TAU_S, LAMBDA_CC = .1, .1, 2.5   # 2.5: same initial gradient budget as EPRM_CEraw (~30x seg)
LEVEL = 1
LAMBDA = .6
SEED_NEW = 20261112


class EPRM(nn.Module):
    def __init__(self, c, m=PROTOS, d=PDIM):
        super().__init__()
        self.q = nn.Conv2d(c, d, 1)
        self.protos = nn.Parameter(torch.randn(m, d))
        self.v = nn.Linear(d, c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.assign = None

    def forward(self, f):
        q = F.normalize(self.q(f), dim=1)
        p = F.normalize(self.protos, dim=1)
        a = torch.einsum("bdhw,md->bmhw", q, p).div(TAU).softmax(1)          # B,M,H,W
        ctx = torch.einsum("bmhw,mc->bchw", a, self.v(p))
        self.assign = a
        return f + self.out(ctx)


class EPRMNet(C.UNet):
    def __init__(self):
        super().__init__(6, 1)
        self.eprm = EPRM(self.widths[-2 - LEVEL])

    def features(self, x):
        enc = []
        for i, block in enumerate(self.enc_blocks):
            x = block(F.max_pool2d(x, 2) if i else x)
            enc.append(x)
        dec = []
        for i, (up, block, skip) in enumerate(zip(self.up, self.dec_blocks, enc[-2::-1])):
            x = block(torch.cat((skip, up(x)), 1))
            if i == LEVEL:
                x = self.eprm(x)
            dec.append(x)
        self.enc, self.dec = enc, dec
        return x


def make():
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED_NEW)
        m = EPRMNet()
    return m


def initial():
    m = make()
    state = torch.load(C.RESULTS / "initial_student.pt", map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("eprm.") for k in missing), (missing, unexpected)
    return m.cuda()


def frozen(arm):
    m = make()
    sel = C.read(C.RESULTS / arm / "selection.json")
    assert C.sha(C.RESULTS / arm / "best.pt") == sel["checkpoint_sha256"]
    m.load_state_dict(torch.load(C.RESULTS / arm / "best.pt", map_location="cpu", weights_only=True))
    return m.cuda().eval().requires_grad_(False)


def correspondence_loss(ts, tt, ev):
    """Cross-event correspondence distillation.  For every region, the optical
    Teacher gives a soft distribution over ALL regions of OTHER events in the
    batch ("which regions of other disasters are my kind"); the SAR student's
    (concept-bank refined) regions give the same distribution; KL(T || S).
    No optimal transport, no hard pair selection."""
    cross = ev[:, None] != ev[None, :]
    rows = cross.any(1)
    if rows.sum() < 2:
        return ts.sum() * 0
    neg = torch.finfo(ts.dtype).min
    lt = (tt @ tt.T / TAU_T).masked_fill(~cross, neg)[rows]
    ls = (ts @ ts.T / TAU_S).masked_fill(~cross, neg)[rows]
    pt = lt.softmax(1)
    kl = (pt * (pt.clamp_min(1e-12).log() - ls.log_softmax(1))).masked_fill(~cross[rows], 0).sum(1)
    return kl.mean()


def assignment_tokens(a, valid):
    """Region tokens of the prototype-assignment map (unit-normalised)."""
    return O.tokens(a, valid)


def build(arm):
    from geomloss import SamplesLoss
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=.1, backend="tensorized")
    s = initial()
    T = R.teacher_fusion() if arm != "EPRM" else None

    def objective(b, epoch):
        z = s(b["x"])
        seg = C.segmentation(z, b)
        if arm == "EPRM":
            return seg, {"seg": seg.detach()}
        with torch.no_grad():
            T(M.fusion_input(b))
        tt, sup = O.tokens(T.dec[LEVEL], b["valid"])
        if arm == "EPRM_CE":
            ts, _ = assignment_tokens(s.eprm.assign, b["valid"])
        else:
            ts, _ = O.tokens(s.dec[LEVEL], b["valid"])
        _, _, ev = O.token_meta(b, sup)
        if arm == "EPRM_CC":
            cc = correspondence_loss(ts, tt, ev)
            return seg + LAMBDA_CC * cc, {"seg": seg.detach(), "cc": cc.detach()}
        rel, _ = O.cross_event_relations(ts, tt, ev, sinkhorn)
        logs = {"seg": seg.detach(), "ce": rel.detach()}
        if arm == "EPRM_CE":
            usage = s.eprm.assign.mean((0, 2, 3))
            logs["proto_perplexity"] = torch.exp(-(usage * usage.clamp_min(1e-8).log()).sum()).detach()
        return seg + LAMBDA * rel, logs
    return s, (), objective, (lambda b: s(b["x"]).sigmoid()), {}


def check():
    C.setup()
    O.prepare()
    f = C.frames()
    R.TRAIN = C.Inputs(f["train"].iloc[::4], augment=True)
    batches = [C.cuda(b) for _, b in zip(range(3), C.loader(R.TRAIN, True, torch.Generator().manual_seed(0)))]
    base = R.initial_student().eval()
    m = initial().eval()
    with torch.no_grad():
        print("identity at init:", float((m(batches[0]["x"]) - base(batches[0]["x"])).abs().max()),
              "extra params:", sum(p.numel() for n, p in m.named_parameters() if n.startswith("eprm.")))
    for arm in ("EPRM_CEraw", "EPRM_CC"):
        s, _, objective, _, _ = build(arm)
        params = list(s.parameters())
        flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])
        for b in batches:
            loss, logs = objective(b, 1)
            g_tot = flat(torch.autograd.grad(loss, params, allow_unused=True))
            g_seg = flat(torch.autograd.grad(C.segmentation(s(b["x"]), b), params, allow_unused=True))
            g = g_tot - g_seg
            print(arm, "cos", round(float(F.cosine_similarity(g, g_seg, 0)), 3), "aux/seg",
                  round(float(g.norm() / g_seg.norm()), 2), {k: round(float(v), 3) for k, v in logs.items()})


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
                      ("OursP1", C.RESULTS / "OursP1")] + [(a, C.RESULTS / a) for a in arms]:
        r = C.read(path / "test_metrics.json")
        rows[arm] = {"all120": round(100 * r["IoU"], 2), "ind44": round(100 * r["per_official_subset"]["testind"]["IoU"], 2),
                     "spt76": round(100 * r["per_official_subset"]["testspt"]["IoU"], 2),
                     "P": round(100 * r["Precision"], 1), "R": round(100 * r["Recall"], 1),
                     "AUPRC": round(100 * r["AUPRC"], 2), "val": round(100 * C.read(path / "selection.json")["validation_IoU"], 2)}
        print(arm, rows[arm], flush=True)
    C.write(C.RESULTS / "comparison_eprm_seed42.json", rows)


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
    R.TRAIN, R.VAL = C.Inputs(f["train"], augment=True), C.Inputs(f["val"])
    for arm in a.arms:
        C.write(C.RESULTS / arm / "config.json", {"arm": arm, "prototypes": PROTOS, "proto_dim": PDIM, "tau": TAU,
                                                  "level": LEVEL, "lambda": LAMBDA, "tau_t": TAU_T, "tau_s": TAU_S, "lambda_cc": LAMBDA_CC, "K": O.K, "seed": C.SEED,
                                                  "doc": __doc__})
        model, extra, objective, predict, kw = build(arm)
        C.fit(arm, model, R.TRAIN, R.VAL, objective, predict, extra_params=extra, **kw)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
