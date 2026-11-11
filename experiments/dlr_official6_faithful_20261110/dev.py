"""Event-held-out DEVELOPMENT folds for designing our method without the test set.

Why: the official validation set (88 patches) comes from the same 23 events as
training and saturates (B0 0.909, most arms within +-1 pp), so it cannot rank
methods for NEW events, which is what the independent test subset measures.
Here whole events are removed from train+val, every model (Teacher included,
so no held-out label leaks through distillation) is trained on the remaining
events with the unchanged recipe, the official-val patches of the remaining
events select the checkpoint, and the held-out events are scored once per arm.
The official test set is never read by this script.

    python dev.py --fold A --arms B0 T_fusion KD SelfKD OursR REL
"""
import argparse
import shutil
import traceback

import pandas as pd
import torch
from torch import nn

import common as C
import methods as M
import run as R

# KG0002 (160 patches) and the two other large events stay in training.
FOLDS = {
    "A": ("CA0001", "IS0001", "PH0001", "IR0001", "NZ0001", "IN0002"),
    "B": ("PK0001", "CN0002", "IN0003", "CL0002", "GT0001", "US0001", "KG0001", "IT0001"),
}
DEV = None


def prepare(fold):
    global DEV
    f = C.frames()
    held = list(FOLDS[fold])
    tr = f["train"][~f["train"].event_id.isin(held)]
    va = f["val"][~f["val"].event_id.isin(held)]
    dv = pd.concat([f["train"][f["train"].event_id.isin(held)], f["val"][f["val"].event_id.isin(held)]])
    C.RESULTS = C.HERE / "results_dev" / fold
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    init = C.RESULTS / "initial_student.pt"
    if not init.exists():
        shutil.copy2(C.HERE / "results" / "initial_student.pt", init)
    C.write(C.RESULTS / "fold.json", {"fold": fold, "held_out_events": held, "train": len(tr),
                                      "selection_val": len(va), "dev": len(dv)})
    R.TRAIN, R.VAL = C.Inputs(tr, augment=True), C.Inputs(va)
    DEV = C.Inputs(dv)


# --------------------------------------------------------------------------- #
# additional development arms (comparison arms come from run.build)
# --------------------------------------------------------------------------- #
def affinity(f, valid, grid=16):
    """Cosine affinity of grid x grid region tokens; tokens outside SAR-valid dropped."""
    cover = torch.nn.functional.adaptive_avg_pool2d(valid.float(), grid)
    pooled = torch.nn.functional.adaptive_avg_pool2d(f.float(), grid)
    tok = torch.nn.functional.normalize(pooled.flatten(2), dim=1)       # B,C,N
    keep = (cover.flatten(1) >= .75).float()                             # B,N
    return tok.transpose(1, 2) @ tok, keep[:, :, None] * keep[:, None, :]


def relation_loss(fs, ft, valid):
    """Relative affinity error: ||A_s - A_t||^2 / ||A_t - mean(A_t)||^2 (O(1) at init)."""
    a_s, m = affinity(fs, valid)
    a_t, _ = affinity(ft.detach(), valid)
    num = ((a_s - a_t).square() * m).sum((1, 2))
    mean_t = (a_t * m).sum((1, 2), keepdim=True) / m.sum((1, 2), keepdim=True).clamp_min(1)
    den = ((a_t - mean_t).square() * m).sum((1, 2)).clamp_min(1e-6)
    return (num / den).mean()


def oof_groups(frame):
    """Two event groups of the current training set, balanced by patch count
    (greedy: largest event first, each to the currently smaller group)."""
    groups, load = [[], []], [0, 0]
    for event, n in frame.event_id.value_counts().items():
        g = int(load[1] < load[0])
        groups[g].append(event)
        load[g] += int(n)
    return groups


def oof_teachers(kind):
    """Out-of-fold Teachers: each is trained on one event group and labels the other
    group's training patches, so its soft targets are NOT memorised labels.
    kind = 'fusion' (SAR+terrain+optical, 18 ch) or 'sar' (control, 6 ch)."""
    tr, va = R.TRAIN.frame, R.VAL.frame
    groups = oof_groups(tr)
    logits = {}
    home, seed = C.RESULTS, C.SEED
    if TEACHER_ROOT is not None:          # teachers shared across student seeds
        C.RESULTS, C.SEED = TEACHER_ROOT, 42
        C.RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        _train_oof(kind, tr, va, groups, logits)
    finally:
        C.RESULTS, C.SEED = home, seed
    assert set(logits) == set(tr.sample_id)
    C.write(C.RESULTS / f"oof_{kind}.json", {"groups": groups})
    return logits


TEACHER_ROOT = None


def _train_oof(kind, tr, va, groups, logits):
    for g, events in enumerate(groups):
        name = f"Toof_{kind}_g{g}"
        C.seed_all()
        m = C.UNet(18 if kind == "fusion" else 6).cuda()
        view = M.fusion_input if kind == "fusion" else (lambda b: b["x"])

        def objective(b, epoch, m=m, view=view):
            loss = C.segmentation(m(view(b)), b)
            return loss, {"seg": loss.detach()}
        sub_tr = C.Inputs(tr[tr.event_id.isin(events)], augment=True)
        sub_va = C.Inputs(va[va.event_id.isin(events)])
        C.fit(name, m, sub_tr, sub_va, objective, lambda b, m=m, view=view: m(view(b)).sigmoid())
        m = R.frozen(name, C.UNet(18 if kind == "fusion" else 6))
        other = C.Inputs(tr[~tr.event_id.isin(events)])
        with torch.no_grad():
            for b in C.loader(other):
                z = m(view(C.cuda(b))).float().cpu()
                for i, sid in enumerate(b["sample_id"]):
                    logits[sid] = z[i]
    assert set(logits) == set(tr.sample_id)
    C.write(C.RESULTS / f"oof_{kind}.json", {"groups": groups})
    return logits


def attach(key, logits):
    for item in R.TRAIN.cache:
        item[key] = logits[item["sample_id"]]


def dev_build(arm):
    if arm in ("OOF_KD", "OOF_SELF"):
        kind = "fusion" if arm == "OOF_KD" else "sar"
        attach("z_oof", oof_teachers(kind))
        s = R.initial_student()

        def objective(b, epoch):
            z = s(b["x"])
            seg, kd = C.segmentation(z, b), M.hinton_kd(z, b["z_oof"], b["valid"].float(), 4.)
            return seg + kd, {"seg": seg.detach(), "kd": kd.detach()}
        return s, (), objective, R.student_predict(s), {}

    if arm == "SelfKD":
        # control: identical output KD, but the Teacher is the SAR-only B0 of this fold
        T = R.frozen("B0", C.student())
        s = R.initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                zt = T(b["x"])
            z = s(b["x"])
            seg, kd = C.segmentation(z, b), M.hinton_kd(z, zt, b["valid"].float(), 4.)
            return seg + kd, {"seg": seg.detach(), "kd": kd.detach()}
        return s, (), objective, R.student_predict(s), {}

    if arm == "REL":
        T = R.teacher_fusion()
        s = R.initial_student()

        def objective(b, epoch):
            with torch.no_grad():
                T(M.fusion_input(b))
            z = s(b["x"])
            seg = C.segmentation(z, b)
            rel = sum(relation_loss(a, t, b["valid"]) for a, t in zip(s.dec[1:3], T.dec[1:3])) / 2
            return seg + rel, {"seg": seg.detach(), "rel": rel.detach()}
        return s, (), objective, R.student_predict(s), {}
    return R.build(arm)


def predictor(arm):
    if arm in ("T_fusion", "T_optical"):
        m = R.frozen(arm, C.UNet(18 if arm == "T_fusion" else 12))
        view = M.fusion_input if arm == "T_fusion" else (lambda b: b["optical"])
        return lambda b: m(view(b)).sigmoid()
    if arm in ("OursS", "OursS_SAR"):
        return R.student_predict(R.frozen(arm, C.student(5)), M.binary_from_subtypes)
    return R.student_predict(R.frozen(arm, C.student()))


def train_and_score(arm):
    out = C.RESULTS / arm
    if (out / "dev_metrics.json").exists():
        return C.read(out / "dev_metrics.json")
    C.write(out / "config.json", {"arm": arm, "fold": C.read(C.RESULTS / "fold.json"),
                                  "fidelity": M.FIDELITY.get(arm, "development arm, see dev.py")})
    if arm == "Nair":
        R.nair_pretext()
    model, extra, objective, predict, kw = dev_build(arm)
    C.fit(arm, model, R.TRAIN, R.VAL, objective, predict, extra_params=extra, **kw)
    del model, extra
    torch.cuda.empty_cache()
    result = C.evaluate(predictor(arm), DEV, save=out / "dev_predictions.npz", full=True)
    C.write(out / "dev_metrics.json", result)
    print(f"DEV {arm} IoU {result['IoU']:.4f} macro {result['macro_event_IoU']:.4f} "
          f"AUPRC {result['AUPRC']:.4f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=sorted(FOLDS))
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    C.setup()
    prepare(args.fold)
    for arm in args.arms:
        try:
            train_and_score(arm)
        except Exception:
            C.write(C.RESULTS / arm / "FAILED.json", {"error": traceback.format_exc()})
            print(f"FAILED {arm}", traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
