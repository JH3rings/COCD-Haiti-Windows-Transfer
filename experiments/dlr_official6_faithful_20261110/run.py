"""Queue runner: B0 -> Teachers -> every distillation / guidance arm -> one test.

    python run.py --smoke        one forward/backward per arm, no files kept
    python run.py                train every arm (resumable), then test once
"""
import argparse
import traceback

import torch
from torch import nn
from torch.nn import functional as F

import common as C
import methods as M

TEACHERS = ("T_fusion", "T_optical", "T_MEF")
STUDENTS = ("KD", "AT", "WKD", "DIS2", "ACAM", "MFSAD", "M2U", "Nair", "PKDOT", "ProtoKD",
            "OursS", "OursS_SAR", "OursR")
QUEUE = ("B0",) + TEACHERS + STUDENTS
NAIR_PRETEXT_EPOCHS = 100


def initial_student(out=1):
    path = C.RESULTS / "initial_student.pt"
    if not path.exists():
        C.seed_all()
        torch.save(C.student().state_dict(), path)
    if out != 1:
        C.seed_all(C.SEED + out)
    model = C.student(out)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if out != 1:
        fresh = model.state_dict()
        state["head.weight"], state["head.bias"] = fresh["head.weight"], fresh["head.bias"]
    model.load_state_dict(state)
    return model.cuda()


def frozen(name, model):
    sel = C.read(C.RESULTS / name / "selection.json")
    path = C.RESULTS / name / "best.pt"
    assert C.sha(path) == sel["checkpoint_sha256"]
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.cuda().eval().requires_grad_(False)


def teacher_fusion():
    return frozen("T_fusion", C.UNet(18))


def student_predict(model, binary=None):
    def predict(b):
        z = model(b["x"])
        return (binary(z) if binary else z).sigmoid()
    return predict


def seg_only(model):
    def objective(b, epoch):
        loss = C.segmentation(model(b["x"]), b)
        return loss, {"seg": loss.detach()}
    return objective


# --------------------------------------------------------------------------- #
def build(arm):
    """-> (model, extra_modules, objective, predict, fit_kwargs)"""
    if arm == "B0":
        m = initial_student()
        return m, (), seg_only(m), student_predict(m), {}

    if arm in ("T_fusion", "T_optical"):
        C.seed_all()
        m = C.UNet(18 if arm == "T_fusion" else 12).cuda()
        view = M.fusion_input if arm == "T_fusion" else (lambda b: b["optical"])

        def objective(b, epoch):
            loss = C.segmentation(m(view(b)), b)
            return loss, {"seg": loss.detach()}
        return m, (), objective, lambda b: m(view(b)).sigmoid(), {}

    if arm == "T_MEF":
        C.seed_all()
        m = M.MEFTeacher().cuda()

        def objective(b, epoch):
            out = m(b["optical"], b["x"])
            parts = {"main": C.segmentation(out["main"], b), "opt": C.segmentation(out["opt"], b),
                     "sar": C.segmentation(out["sar"], b),
                     "aux": sum(C.segmentation(a, b) for a in out["aux"])}
            return sum(parts.values()), {k: v.detach() for k, v in parts.items()}
        return m, (), objective, lambda b: m(b["optical"], b["x"])["main"].sigmoid(), {}

    # ----------------------------- students -------------------------------
    if arm in ("KD", "AT", "WKD", "DIS2", "ACAM", "MFSAD", "PKDOT", "ProtoKD"):
        T = teacher_fusion()
    if arm == "KD":
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(M.fusion_input(b))
            z = s(b["x"])
            seg, kd = C.segmentation(z, b), M.hinton_kd(z, zt, b["valid"].float(), 4.)
            return seg + kd, {"seg": seg.detach(), "kd": kd.detach()}
        return s, (), objective, student_predict(s), {}

    if arm == "AT":
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            att = 1000 / 2 * sum(M.at_loss(a, t) for a, t in zip(s.enc[1:], T.enc[1:]))
            return seg + att, {"seg": seg.detach(), "at": att.detach()}
        return s, (), objective, student_predict(s), {}

    if arm == "WKD":
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                pt = T(M.fusion_input(b)).sigmoid()
            z = s(b["x"])
            seg, w = C.segmentation(z, b), M.wavelet_kd(z.sigmoid(), pt, b["valid"].float())
            return seg + w, {"seg": seg.detach(), "wkd": w.detach()}
        return s, (), objective, student_predict(s), {}

    if arm == "DIS2":
        s = initial_student()
        proj = nn.ModuleList(M.identity_1x1(c) for c in (128, 256, 512, 1024, 64)).cuda()

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            pooled_s = [proj[i](f).mean((2, 3)) for i, f in enumerate(s.enc[1:])]
            pooled_t = [f.mean((2, 3)) for f in T.enc[1:]]
            feat = sum(M.l2_kd_loss(a, t) for a, t in zip(pooled_s, pooled_t))
            pen = M.l2_kd_loss(proj[4](s.dec[3]), T.dec[3], spatial=True)
            logit = M.dis2_logit_kl(z, zt, 2)
            kl = feat + pen + logit
            weight = 5.0 if epoch > 5 else 0.0
            return seg + weight * kl, {"seg": seg.detach(), "feat": feat.detach(), "pen": pen.detach(),
                                       "logit": logit.detach()}
        return s, (proj,), objective, student_predict(s), {}

    if arm == "ACAM":
        s = initial_student()
        acam = M.ACAM().cuda()

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            lc, lsp, div = acam(M.two_class(z), M.two_class(zt), b["valid"])
            return seg + 1. * (lc + lsp) + 1. * div, {"seg": seg.detach(), "c": lc.detach(),
                                                       "s": lsp.detach(), "div": div.detach()}
        return s, (acam,), objective, student_predict(s), {}

    if arm == "MFSAD":
        s = initial_student()

        def adapter(c):
            return nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        enc_ad = nn.ModuleList(adapter(c) for c in (256, 512, 1024)).cuda()
        dec_ad = nn.ModuleList(adapter(c) for c in (512, 256, 128)).cuda()
        lam = {"bk": 100., "freq": 80., "jscad": 10., "out": .12}

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(M.fusion_input(b))
            z = s(b["x"])
            valid = b["valid"].float()
            parts = {"bk": [], "freq": [], "jscad": []}
            for i in range(3):
                a, t = enc_ad[i](s.enc[i + 2]), T.enc[i + 2]
                mk = M.down(valid, a)
                parts["bk"].append(((a - t).square() * mk).sum() / (mk.sum().clamp_min(1) * a.shape[1]))
                f, j = M.mfsad_neck(dec_ad[i](s.dec[i]), T.dec[i], M.down(valid, s.dec[i]))
                parts["freq"].append(f)
                parts["jscad"].append(j)
            terms = {k: torch.stack(v).mean() for k, v in parts.items()}
            terms["out"] = M.hinton_kd(z, zt, valid, 10.)
            seg = C.segmentation(z, b)
            return seg + sum(lam[k] * v for k, v in terms.items()), {"seg": seg.detach(),
                                                                     **{k: v.detach() for k, v in terms.items()}}
        return s, (enc_ad, dec_ad), objective, student_predict(s), {}

    if arm == "M2U":
        T = frozen("T_MEF", M.MEFTeacher())
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                T(b["optical"], b["x"])
            z = s(b["x"])
            seg = C.segmentation(z, b)
            valid = b["valid"].float()
            f = M.fad(s.dec[2], T.fused[2], M.down(valid, s.dec[2]))
            c = M.cad(s.dec[0], T.fused[0])
            return seg + f + c, {"seg": seg.detach(), "fad": f.detach(), "cad": c.detach()}
        return s, (), objective, student_predict(s), {}

    if arm == "Nair":
        T = frozen("T_optical", C.UNet(12))
        s = initial_student()
        pre = C.RESULTS / "Nair_pretext" / "final.pt"
        if pre.exists():
            s.load_state_dict(torch.load(pre, map_location="cpu", weights_only=True))
        else:
            raise RuntimeError("Nair pretext must run first")
        return s, (), seg_only(s), student_predict(s), {}

    if arm == "PKDOT":
        from geomloss import SamplesLoss
        sinkhorn = SamplesLoss("sinkhorn", p=2, blur=.1, backend="tensorized")
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            r = M.pkdot_relations(s.dec[1], T.dec[1], b["valid"], sinkhorn)
            return seg + .6 * r, {"seg": seg.detach(), "sinkhorn": r.detach()}
        return s, (), objective, student_predict(s), {}

    if arm == "ProtoKD":
        s = initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            kd = M.softmax_kl_loss(M.two_class(z) / 10, M.two_class(zt) / 10).mean()
            proto = M.prototype_loss_2d(s.dec[3], T.dec[3], b["mask"].long())
            return seg + 10 * kd + .1 * proto, {"seg": seg.detach(), "kd": kd.detach(), "proto": proto.detach()}
        return s, (), objective, student_predict(s), {}

    if arm in ("OursS", "OursS_SAR"):
        spec_path = C.RESULTS / arm / "subtypes.json"
        if spec_path.exists():
            spec = C.read(spec_path)
        else:
            spec = M.fit_subtypes(TRAIN, "optical" if arm == "OursS" else "sar")
            C.write(spec_path, spec)
        s = initial_student(out=5)

        def objective(b, epoch):
            z = s(b["x"])
            seg = C.segmentation(M.binary_from_subtypes(z), b)
            ce = F.cross_entropy(z, M.subtype_labels(b, spec), ignore_index=-100)
            return seg + ce, {"seg": seg.detach(), "subtype_ce": ce.detach()}
        return s, (), objective, student_predict(s, M.binary_from_subtypes), {}

    if arm == "OursR":
        s = initial_student()
        head = nn.Conv2d(64, 8, 1).cuda()

        def objective(b, epoch):
            z = s(b["x"])
            seg = C.segmentation(z, b)
            out = head(s.dec[3])
            mu, logvar = out[:, :4], out[:, 4:].clamp(-6, 6)
            y = M.optical_change(b)
            ok = (b["valid"] & b["optical_ok"]).float()
            nll = M.masked_mean((.5 * (logvar + (y - mu).square() * torch.exp(-logvar))).mean(1, keepdim=True), ok)
            return seg + nll, {"seg": seg.detach(), "nll": nll.detach()}
        return s, (head,), objective, student_predict(s), {}
    raise KeyError(arm)


def nair_pretext():
    """Label-free cosine reconstruction of optical-teacher features (3 enc + 3 dec)."""
    T = frozen("T_optical", C.UNet(12))
    s = initial_student()
    tstate = torch.load(C.RESULTS / "T_optical" / "best.pt", map_location="cpu", weights_only=True)
    sstate = s.state_dict()
    for k, v in tstate.items():
        if sstate[k].shape == v.shape:
            sstate[k] = v
    s.load_state_dict(sstate)

    def objective(b, epoch):
        with torch.no_grad():
            T(b["optical"])
        s(b["x"])
        loss = M.nair_cosine(s.enc[2:5] + s.dec[0:3], T.enc[2:5] + T.dec[0:3], b["valid"])
        return loss, {"cosine": loss.detach()}
    C.fit("Nair_pretext", s, TRAIN, VAL, objective, student_predict(s),
          epochs=NAIR_PRETEXT_EPOCHS, val_every=NAIR_PRETEXT_EPOCHS)


def train(arm):
    C.write(C.RESULTS / arm / "config.json", {"arm": arm, "fidelity": M.FIDELITY[arm], "seed": C.SEED,
            "epochs": C.EPOCHS, "validation_every": C.VAL_EVERY, "batch": C.BATCH, "drop_last": True,
            "deterministic": True, "student_input": list(C.SAR_FIELDS + C.TERRAIN_FIELDS),
            "privileged_input": list(C.OPTICAL_FIELDS), "split_sha256": C.sha(C.SPLIT)})
    if arm == "Nair":
        nair_pretext()
    C.write(C.RESULTS / "status.json", {"state": "TRAINING", "arm": arm})
    model, extra, objective, predict, kw = build(arm)
    sel = C.fit(arm, model, TRAIN, VAL, objective, predict, extra_params=extra, **kw)
    del model, extra
    torch.cuda.empty_cache()
    return sel


def test_all():
    C.write(C.RESULTS / "status.json", {"state": "TESTING"})
    test = C.Inputs(C.frames()["test"])
    table = {}
    for arm in QUEUE:
        out = C.RESULTS / arm
        if (out / "test_metrics.json").exists():
            table[arm] = C.read(out / "test_metrics.json")
            continue
        if arm in ("T_fusion", "T_optical"):
            m = frozen(arm, C.UNet(18 if arm == "T_fusion" else 12))
            view = M.fusion_input if arm == "T_fusion" else (lambda b: b["optical"])
            predict = lambda b, m=m, view=view: m(view(b)).sigmoid()
        elif arm == "T_MEF":
            m = frozen(arm, M.MEFTeacher())
            predict = lambda b, m=m: m(b["optical"], b["x"])["main"].sigmoid()
        elif arm in ("OursS", "OursS_SAR"):
            m = frozen(arm, C.student(5))
            predict = student_predict(m, M.binary_from_subtypes)
        else:
            m = frozen(arm, C.student())
            predict = student_predict(m)
        C.write(out / "test_started.json", {"arm": arm})
        result = C.evaluate(predict, test, save=out / "test_predictions.npz", full=True)
        C.write(out / "test_metrics.json", result)
        table[arm] = result
        del m
    rows = {a: {"all120_IoU": r["IoU"], "ind44_IoU": r["per_official_subset"]["testind"]["IoU"],
                "spt76_IoU": r["per_official_subset"]["testspt"]["IoU"], "AUPRC": r["AUPRC"],
                "macro_event_IoU": r["macro_event_IoU"]} for a, r in table.items()}
    C.write(C.RESULTS / "comparison.json", rows)
    C.write(C.RESULTS / "status.json", {"state": "COMPLETE"})


def smoke():
    global TRAIN, VAL
    f = C.frames()
    TRAIN, VAL = C.Inputs(f["train"].iloc[:32], augment=True), C.Inputs(f["val"].iloc[:16])
    batch = C.cuda(next(iter(C.loader(TRAIN, True, torch.Generator().manual_seed(0)))))
    real = C.RESULTS
    C.RESULTS = C.HERE / "smoke_results"
    # stand-in frozen teachers so the student arms can be built
    for name, model in (("T_fusion", C.UNet(18)), ("T_optical", C.UNet(12)), ("T_MEF", M.MEFTeacher())):
        d = C.RESULTS / name
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "best.pt")
        C.write(d / "selection.json", {"checkpoint_sha256": C.sha(d / "best.pt")})
    d = C.RESULTS / "Nair_pretext"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(C.student().state_dict(), d / "final.pt")
    for arm in QUEUE:
        model, extra, objective, predict, _ = build(arm)
        params = list(model.parameters()) + [p for m in extra for p in m.parameters()]
        loss, logs = objective(batch, 10)
        loss.backward()
        g = sum(p.grad.square().sum() for p in params if p.grad is not None).sqrt()
        with torch.no_grad():
            p = predict(batch)
        assert torch.isfinite(loss) and torch.isfinite(g) and p.shape == batch["mask"].shape
        print(f"{arm:10s} loss {float(loss):.4f} grad {float(g):.3f} " +
              " ".join(f"{k}={float(v):.4g}" for k, v in logs.items()), flush=True)
        del model, extra
        torch.cuda.empty_cache()
    C.RESULTS = real
    print("SMOKE OK")


def main():
    global TRAIN, VAL
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    f = C.frames()
    TRAIN, VAL = C.Inputs(f["train"], augment=True), C.Inputs(f["val"])
    C.write(C.RESULTS / "queue.json", {"queue": QUEUE, "fidelity": M.FIDELITY})
    for arm in QUEUE:
        train(arm)
    test_all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    C.setup()
    try:
        smoke() if args.smoke else main()
    except Exception:
        if not args.smoke:
            C.write(C.RESULTS / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise
