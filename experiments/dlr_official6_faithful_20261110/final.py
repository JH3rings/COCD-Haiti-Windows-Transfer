"""Official-split FINAL runs, multi-seed (42/43/44), for the arms chosen on the
event-held-out development folds.  Training/selection only; the test set is
read by ``final.py --test`` once every requested arm/seed is frozen.

Seed s changes the student initial weights (seed_all(s) then student(),
identical to results_noise B0 replicates) and the data order (SEED + 81).
Teachers are the seed-42 ones (T_fusion from results/, OOF teachers shared
under results_final/teachers), so the seed only varies the student.

    python final.py --prepare-oof
    python final.py --arms OOF_KD OOF_SELF --seeds 42 43 44
    python final.py --test --arms B0 OOF_KD OOF_SELF --seeds 42 43 44

Active comparison set from 2026-11-11 (user): DIS2, PKDOT (best two) + KD, AT
(classic).  The other comparison arms are retired from future runs; their
frozen results stay in results/ as historical evidence.
"""
import argparse
import shutil

import numpy as np
import torch

import common as C
import methods as M
import dev as D
import run as R

ROOT = C.HERE / "results_final"
D.TEACHER_ROOT = ROOT / "teachers"


def official():
    f = C.frames()
    R.TRAIN, R.VAL = C.Inputs(f["train"], augment=True), C.Inputs(f["val"])
    D.DEV = None
    return f


def seed_dir(s):
    C.SEED = s
    C.RESULTS = ROOT / f"seed{s}"
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    init = C.RESULTS / "initial_student.pt"
    if not init.exists():
        C.seed_all(s)
        torch.save(C.student().state_dict(), init)
    dst = C.RESULTS / "T_fusion"          # the frozen seed-42 fusion Teacher (copy, verified by sha)
    if not (dst / "selection.json").exists():
        dst.mkdir(exist_ok=True)
        for name in ("best.pt", "selection.json"):
            shutil.copy2(C.HERE / "results" / "T_fusion" / name, dst / name)
    return C.RESULTS


def train(arms, seeds):
    for s in seeds:
        seed_dir(s)
        for arm in arms:
            if (C.RESULTS / arm / "selection.json").exists():
                continue
            C.write(C.RESULTS / arm / "config.json", {"arm": arm, "seed": s, "split": "official",
                                                      "teachers": "seed 42"})
            model, extra, objective, predict, kw = D.dev_build(arm)
            C.fit(arm, model, R.TRAIN, R.VAL, objective, predict, extra_params=extra, **kw)
            del model, extra
            torch.cuda.empty_cache()


MAIN_RUN_ARMS = ("DIS2", "PKDOT", "OursS", "KD", "AT")   # seed 42 already frozen+tested in results/


def b0_source(s):
    return C.HERE / ("results" if s == 42 else f"results_noise/seed{s}") / "B0"


def source(arm, s):
    if arm == "B0":
        return b0_source(s)
    if s == 42 and arm in MAIN_RUN_ARMS:
        return C.HERE / "results" / arm
    return ROOT / f"seed{s}" / arm


def test(arms, seeds):
    data = C.Inputs(C.frames()["test"], optical=False)
    rows = {}
    for arm in arms:
        for s in seeds:
            out = source(arm, s)
            if not (out / "test_metrics.json").exists():
                sel = C.read(out / "selection.json")
                assert C.sha(out / "best.pt") == sel["checkpoint_sha256"]
                subtypes = arm in ("OursS", "OursS_SAR")
                m = C.student(5 if subtypes else 1)
                m.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
                m = m.cuda().eval()
                predict = R.student_predict(m, M.binary_from_subtypes if subtypes else None)
                C.write(out / "test_started.json", {"arm": arm, "seed": s})
                C.write(out / "test_metrics.json", C.evaluate(predict, data,
                                                              save=out / "test_predictions.npz", full=True))
            r = C.read(out / "test_metrics.json")
            rows.setdefault(arm, []).append({"seed": s, "all120": r["IoU"],
                                             "ind44": r["per_official_subset"]["testind"]["IoU"],
                                             "spt76": r["per_official_subset"]["testspt"]["IoU"],
                                             "AUPRC": r["AUPRC"], "macro_event": r["macro_event_IoU"]})
    summary = {}
    for arm, rs in rows.items():
        summary[arm] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs], ddof=1))
                            if len(rs) > 1 else None, "per_seed": [r[k] for r in rs]}
                        for k in ("all120", "ind44", "spt76", "AUPRC", "macro_event")}
        print(arm, {k: f"{100 * v['mean']:.2f}" for k, v in summary[arm].items()}, flush=True)
    C.write(ROOT / "final_comparison.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-oof", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    a = ap.parse_args()
    C.setup()
    official()
    if a.prepare_oof:
        seed_dir(42)
        for kind in ("fusion", "sar"):
            D.oof_teachers(kind)
    elif a.test:
        test(a.arms, a.seeds)
    else:
        train(a.arms, a.seeds)


if __name__ == "__main__":
    main()
