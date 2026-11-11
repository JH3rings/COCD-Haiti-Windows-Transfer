"""Network-internal diagnostics of the frozen seed-42 students (train/val only,
the test set is not read).

1. Teacher-feature predictability (modality-shared information): ridge
   regression from a student's layer features to the fusion Teacher's same
   layer, fitted on training pixels, R^2 on validation pixels.  B0's R^2 is the
   share of Teacher information a SAR network already has without any
   distillation; a method that truly transfers raises it.
2. Linear CKA student vs Teacher per layer.
3. Landslide linear probe per layer (logistic, train -> val AUPRC): where in the
   network the discriminative information sits.
4. Gradient conflict at the shared initial weights and at each arm's final
   weights: cosine between the segmentation gradient and the distillation
   gradient, and their norm ratio (8 training batches, no augmentation).
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import common as C
import methods as M
import run as R

ARMS = ("B0", "KD", "AT", "DIS2", "PKDOT", "OursS", "OursR", "ACAM", "MFSAD")
LAYERS = ("enc1", "enc2", "enc3", "enc4", "dec0", "dec1", "dec2", "dec3")
PER_PATCH = 24
rng = np.random.default_rng(0)


def layer(model, name):
    kind, i = name[:3], int(name[3:])
    return (model.enc if kind == "enc" else model.dec)[i]


def load_student(arm):
    if arm in ("OursS", "OursS_SAR"):
        return R.frozen(arm, C.student(5))
    return R.frozen(arm, C.student())


@torch.no_grad()
def collect(models, ds, positions=None):
    """Features at fixed random valid positions (same positions for every model)."""
    feats = {k: {l: [] for l in LAYERS} for k in models}
    labels = {l: [] for l in LAYERS}
    pos_out = [] if positions is None else positions
    k = 0
    for b in C.loader(ds):
        b = C.cuda(b)
        for name, (m, view) in models.items():
            m(view(b))
        for i in range(len(b["x"])):
            if positions is None:
                v = b["valid"][i, 0].cpu().numpy()
                y = b["mask"][i, 0].cpu().numpy() > .5
                # stratified: half landslide, half background where available
                fg, bg = np.argwhere(v & y), np.argwhere(v & ~y)
                pick = []
                for arr in (fg, bg):
                    if len(arr):
                        pick.append(arr[rng.choice(len(arr), min(PER_PATCH // 2, len(arr)), replace=False)])
                pts = np.concatenate(pick) if pick else np.zeros((0, 2), int)
                pos_out.append(pts)
            pts = pos_out[k]
            k += 1
            for l in LAYERS:
                ref = layer(next(iter(models.values()))[0], l)
                s = 128 // ref.shape[-1]
                r, c = torch.as_tensor(pts[:, 0] // s), torch.as_tensor(pts[:, 1] // s)
                labels[l].append(b["mask"][i, 0].cpu().numpy()[pts[:, 0], pts[:, 1]])
                for name, (m, _) in models.items():
                    f = layer(m, l)[i].float().cpu()
                    feats[name][l].append(f[:, r, c].T.numpy())
    feats = {n: {l: np.concatenate(v) for l, v in d.items()} for n, d in feats.items()}
    labels = {l: np.concatenate(v) > .5 for l, v in labels.items()}
    return feats, labels, pos_out


def standardize(a, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return (a - mu) / sd


def ridge_r2(xtr, ytr, xva, yva, lam=10.):
    xtr_s, xva_s = standardize(xtr, xtr), standardize(xva, xtr)
    ytr_s, yva_s = standardize(ytr, ytr), standardize(yva, ytr)
    x = torch.as_tensor(xtr_s, dtype=torch.float64)
    w = torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1], dtype=torch.float64),
                           x.T @ torch.as_tensor(ytr_s, dtype=torch.float64))
    pred = torch.as_tensor(xva_s, dtype=torch.float64) @ w
    resid = ((torch.as_tensor(yva_s) - pred) ** 2).sum(0)
    total = ((torch.as_tensor(yva_s) - torch.as_tensor(yva_s).mean(0)) ** 2).sum(0)
    keep = total > 1e-6
    return float(1 - resid[keep].sum() / total[keep].sum())


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    num = np.linalg.norm(a.T @ b) ** 2
    return float(num / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def probe(xtr, ytr, xva, yva):
    clf = LogisticRegression(max_iter=300, C=1.)
    clf.fit(standardize(xtr, xtr), ytr)
    return float(average_precision_score(yva, clf.predict_proba(standardize(xva, xtr))[:, 1]))


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])


def gradient_conflict(arm, batches, final):
    model, extra, objective, predict, _ = R.build(arm)
    if final:
        model.load_state_dict(load_student(arm).state_dict())
    model.train()
    params = [p for p in model.parameters()]
    binary = M.binary_from_subtypes if arm in ("OursS", "OursS_SAR") else (lambda z: z)
    cos, ratio = [], []
    for b in batches:
        model.zero_grad(set_to_none=True)
        loss, _ = objective(b, 10)
        loss.backward()
        g_total = flat_grad(params).clone()
        model.zero_grad(set_to_none=True)
        C.segmentation(binary(model(b["x"])), b).backward()
        g_seg = flat_grad(params).clone()
        g_kd = g_total - g_seg
        cos.append(float(torch.nn.functional.cosine_similarity(g_seg, g_kd, 0)))
        ratio.append(float(g_kd.norm() / g_seg.norm().clamp_min(1e-12)))
    del model, extra
    torch.cuda.empty_cache()
    return {"cos_seg_vs_distill": float(np.mean(cos)), "norm_distill_over_seg": float(np.mean(ratio))}


def main():
    C.setup()
    f = C.frames()
    R.TRAIN, R.VAL = C.Inputs(f["train"]), C.Inputs(f["val"])
    T = R.teacher_fusion()
    models = {"T": (T, M.fusion_input)}
    for a in ARMS:
        models[a] = (load_student(a), lambda b: b["x"])
    tr_feats, tr_y, _ = collect(models, R.TRAIN)
    va_feats, va_y, _ = collect(models, R.VAL)
    out = {"predictability_R2": {}, "cka": {}, "probe_AUPRC": {}, "gradient": {}}
    for a in ("T",) + ARMS:
        out["probe_AUPRC"][a] = {l: probe(tr_feats[a][l], tr_y[l], va_feats[a][l], va_y[l]) for l in LAYERS}
        if a == "T":
            continue
        out["predictability_R2"][a] = {l: ridge_r2(tr_feats[a][l], tr_feats["T"][l], va_feats[a][l], va_feats["T"][l])
                                       for l in LAYERS}
        out["cka"][a] = {l: cka(va_feats[a][l], va_feats["T"][l]) for l in LAYERS}
        print(a, "R2", {l: round(v, 3) for l, v in out["predictability_R2"][a].items()}, flush=True)
    # teacher-feature subspace: is the SAR-predictable part the landslide-relevant part?
    sub = {}
    for l in LAYERS:
        x, y = tr_feats["B0"][l], tr_feats["T"][l]
        xs, ys = standardize(x, x), standardize(y, y)
        w = np.linalg.solve(xs.T @ xs + 10 * np.eye(xs.shape[1]), xs.T @ ys)
        xv, yv = standardize(va_feats["B0"][l], x), standardize(va_feats["T"][l], y)
        pred_tr, pred_va = xs @ w, xv @ w
        sub[l] = {"probe_T_full": out["probe_AUPRC"]["T"][l],
                  "probe_T_SAR_predictable_part": probe(pred_tr, tr_y[l], pred_va, va_y[l]),
                  "probe_T_residual_part": probe(ys - pred_tr, tr_y[l], yv - pred_va, va_y[l])}
    out["teacher_subspace_probe"] = sub
    batches = []
    gen = torch.Generator().manual_seed(7)
    for i, b in enumerate(C.loader(R.TRAIN, True, gen)):
        if i == 8:
            break
        batches.append(C.cuda(b))
    for a in ARMS[1:]:
        out["gradient"][a] = {"initial": gradient_conflict(a, batches, False),
                              "final": gradient_conflict(a, batches, True)}
        print(a, "grad", out["gradient"][a], flush=True)
    C.write(C.RESULTS / "diag_repr.json", out)
    for key in ("probe_AUPRC", "predictability_R2", "cka"):
        print("\n" + key)
        for a, d in out[key].items():
            print(f"{a:7s}", " ".join(f"{d[l]:.3f}" for l in LAYERS))
    print("\nteacher subspace probe (full / SAR-predictable / residual)")
    for l, d in sub.items():
        print(l, " ".join(f"{v:.3f}" for v in d.values()))


if __name__ == "__main__":
    main()
