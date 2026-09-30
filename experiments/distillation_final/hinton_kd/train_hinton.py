from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import (  # noqa: E402
    DEV, MAX_EPOCHS, PATIENCE, SEED, bce, canonical_optimizer, load_cg0_teacher,
    make_loaders, make_student, metric_rows, overall_ap, protocol_record,
    seed_everything, student_predictions, write_json, loader,
)

OUT = HERE.parent
T = 2.0


def soft_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    # A single binary logit z is represented as the symmetric two-class logits [-z, z].
    s = torch.stack((-student_logits, student_logits), dim=1) / T
    t = torch.stack((-teacher_logits, teacher_logits), dim=1) / T
    log_q = F.log_softmax(s, dim=1)
    p = F.softmax(t, dim=1)
    return F.kl_div(log_q, p, reduction="none").sum(dim=1).mean() * (T * T)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    train, test, split_info = make_loaders()
    teacher = load_cg0_teacher()
    student = make_student()
    optimizer = canonical_optimizer(student)
    best = float("-inf")
    bad_epochs = 0
    history = []
    best_state = None
    best_epoch = None

    for epoch in range(1, MAX_EPOCHS + 1):
        student.train()
        means = torch.zeros(4, dtype=torch.float64)
        n = 0
        for asc, desc, y, _ga, _gd, _real in loader(train, shuffle=True):
            asc, desc, y = asc.to(DEV), desc.to(DEV), y.to(DEV)
            with torch.no_grad():
                ta, td = teacher.forward_pair(asc, desc, counter_mode="counter")
            oa, od = student(asc), student(desc)
            seg = 0.5 * (bce(oa["z"], y) + bce(od["z"], y))
            kd = 0.5 * (soft_kd(oa["z"][:, 0], ta["z"][:, 0]) +
                        soft_kd(od["z"][:, 0], td["z"][:, 0]))
            loss = kd + 0.5 * seg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            means += torch.tensor([seg.detach().item(), kd.detach().item(),
                                   (0.5 * seg).detach().item(), loss.detach().item()], dtype=torch.float64)
            n += 1

        pred = student_predictions(student, test)
        score = overall_ap(pred)
        row = {"epoch": epoch, "seg_raw": means[0].item() / n,
               "soft_kd_raw": means[1].item() / n,
               "seg_weighted": means[2].item() / n,
               "kd_weighted": means[1].item() / n,
               "kd_scalar_share": (means[1].item() / (means[1].item() + means[2].item() + 1e-12)),
               "train_total": means[3].item() / n, "test_overall_auprc": score}
        history.append(row)
        if score > best:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            torch.save(best_state, OUT / "Hinton-KD-Port_seed42_best.pt")
        else:
            bad_epochs += 1
        write_json(OUT / "progress.json", {"protocol": protocol_record("Hinton-KD-Port", {
            "split": split_info, "temperature": T, "objective": "L_soft + 0.5 * L_seg",
            "teacher": str(ROOT / "experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt")}),
            "history": history, "best_epoch": best_epoch, "best_test_overall_auprc": best})
        print(f"Hinton-KD-Port epoch={epoch} testAP={score:.6f} kd_share={row['kd_scalar_share']:.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best Hinton checkpoint produced")
    student.load_state_dict(best_state)
    report = {"method": "Hinton-KD-Port", "protocol": protocol_record("Hinton-KD-Port", {
        "split": split_info, "temperature": T, "objective": "L_soft + 0.5 * L_seg"}),
        "best_epoch": best_epoch, "best_test_overall_auprc": best,
        "metrics": metric_rows(student_predictions(student, test)), "history": history}
    write_json(OUT / "summary.json", report)


if __name__ == "__main__":
    main()
