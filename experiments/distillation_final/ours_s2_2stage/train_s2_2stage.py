from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))
from common import (  # noqa: E402
    DEV, EPS, MAX_EPOCHS, PATIENCE, SEED, bce, canonical_optimizer,
    load_cg0_teacher, make_loaders, make_student, metric_rows, overall_ap,
    protocol_record, seed_everything, student_predictions, write_json, loader,
)

OUT = HERE.parent
LEVELS = 3
POINTS = 4


def spatial_policy(policy, memory):
    """Splat LxK policy samples onto the target P2 query grid, as in canonical S2."""
    offsets, weights = 2.0 * torch.tanh(policy["offsets"]), policy["weights"]
    b, _lv, _k, _xy, h, w = offsets.shape
    q = h * w
    yy, xx = torch.meshgrid(torch.arange(h, device=offsets.device),
                            torch.arange(w, device=offsets.device), indexing="ij")
    by, bx = yy.reshape(1, q, 1).float(), xx.reshape(1, q, 1).float()
    out = torch.zeros(b, q, q, device=offsets.device, dtype=offsets.dtype)
    for level, mem in enumerate(memory):
        scale = h / mem.shape[-2]
        off_y = offsets[:, level, :, 0].permute(0, 2, 3, 1).reshape(b, q, POINTS)
        off_x = offsets[:, level, :, 1].permute(0, 2, 3, 1).reshape(b, q, POINTS)
        y = (by + off_y * scale).clamp(0, h - 1)
        x = (bx + off_x * scale).clamp(0, w - 1)
        y0, x0 = y.floor().long(), x.floor().long()
        fy, fx = y - y0, x - x0
        a = weights[:, level].permute(0, 2, 3, 1).reshape(b, q, POINTS)
        for yi, xi, coeff in ((y0, x0, (1-fy)*(1-fx)),
                              (y0, (x0+1).clamp(max=w-1), (1-fy)*fx),
                              ((y0+1).clamp(max=h-1), x0, fy*(1-fx)),
                              ((y0+1).clamp(max=h-1), (x0+1).clamp(max=w-1), fy*fx)):
            out.scatter_add_(2, yi*w+xi, a*coeff)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(EPS)


def js_divergence(p, q):
    mid = 0.5 * (p + q)
    return 0.5 * ((p * (p.clamp_min(EPS).log() - mid.clamp_min(EPS).log())).sum(-1) +
                  (q * (q.clamp_min(EPS).log() - mid.clamp_min(EPS).log())).sum(-1))


def kd_terms(student, teacher, asc, desc, student_outputs=None):
    with torch.no_grad():
        teacher_a, teacher_d = teacher.forward_pair(asc, desc, counter_mode="counter")
    student_a, student_d = student_outputs if student_outputs is not None else (student(asc), student(desc))
    policy_losses, evidence_losses = [], []
    for so, to in ((student_a, teacher_a), (student_d, teacher_d)):
        memory = tuple(v.detach() for v in so["values"])
        # Match canonical S2's stop-gradient target-memory policy KD definition.
        ps = student.policy(student.context(tuple(x.detach() for x in so["T"])))
        pt = {k: v.detach() for k, v in to["policy"].items()}
        distr_s, distr_t = spatial_policy(ps, memory), spatial_policy(pt, memory)
        policy_losses.append(js_divergence(distr_s, distr_t).mean())
        rs = student.search(memory, ps)
        rt = student.search(memory, pt)
        evidence_losses.append((1.0 - F.cosine_similarity(rs, rt, dim=1)).mean())
    return (0.5 * (policy_losses[0] + policy_losses[1]),
            0.5 * (evidence_losses[0] + evidence_losses[1]), student_a, student_d)


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    present = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    if not present:
        return 0.0
    return float(torch.sqrt(torch.stack(present).sum()).item())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    train, test, split_info = make_loaders()
    teacher = load_cg0_teacher()
    student = make_student()
    optimizer = canonical_optimizer(student)
    backbone_params = list(student.backbone.parameters())
    best, best_epoch, bad_epochs = float("-inf"), None, 0
    best_state, history = None, []
    seg_cal = policy_cal = evidence_cal = None
    lambda0 = None

    for epoch in range(1, MAX_EPOCHS + 1):
        student.train()
        sums = {k: 0.0 for k in ("seg", "policy", "evidence", "weighted_policy", "weighted_evidence", "total")}
        grad_record = None
        batches = 0
        for asc, desc, y, _ga, _gd, _real in loader(train, shuffle=True):
            asc, desc, y = asc.to(DEV), desc.to(DEV), y.to(DEV)
            if epoch <= 4:
                oa, od = student(asc), student(desc)
                lp, le = oa["z"].new_zeros(()), oa["z"].new_zeros(())
            else:
                lp, le, oa, od = kd_terms(student, teacher, asc, desc)
            seg = 0.5 * (bce(oa["z"], y) + bce(od["z"], y))
            if epoch <= 5:
                loss, wp, we = seg, seg.new_zeros(()), seg.new_zeros(())
            else:
                progress = (epoch - 6) / max(1, MAX_EPOCHS - 6)
                multiplier = 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * progress))
                wp = 0.5 * lambda0 * multiplier * lp / max(policy_cal, EPS)
                we = 0.5 * lambda0 * multiplier * le / max(evidence_cal, EPS)
                loss = seg + wp + we
            if batches == 0:
                grad_record = {"seg_to_shared_backbone": grad_norm(seg, backbone_params)}
                if epoch <= 4:
                    grad_record.update({"policy_to_shared_backbone": 0.0,
                                        "evidence_to_shared_backbone": 0.0})
                else:
                    grad_record.update({"policy_to_shared_backbone": grad_norm(lp, backbone_params),
                                        "evidence_to_shared_backbone": grad_norm(le, backbone_params)})
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in (("seg", seg), ("policy", lp), ("evidence", le),
                               ("weighted_policy", wp), ("weighted_evidence", we), ("total", loss)):
                sums[key] += float(value.detach().item())
            batches += 1

        means = {key: value / batches for key, value in sums.items()}
        if epoch == 5:
            seg_cal, policy_cal, evidence_cal = means["seg"], means["policy"], means["evidence"]
            lambda0 = 0.25 * seg_cal
        pred = student_predictions(student, test)
        score = overall_ap(pred)
        kd_weighted = means["weighted_policy"] + means["weighted_evidence"]
        row = {"epoch": epoch, **means,
               "kd_scalar_share": kd_weighted / (means["seg"] + kd_weighted + EPS),
               "lambda0": lambda0, "seg_calibration": seg_cal,
               "policy_calibration": policy_cal, "evidence_calibration": evidence_cal,
               "gradient_norms_first_batch": grad_record,
               "test_overall_auprc": score}
        history.append(row)
        if score > best:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            torch.save(best_state, OUT / "S2-2Stage_seed42_best.pt")
        else:
            bad_epochs += 1
        write_json(OUT / "progress.json", {"protocol": protocol_record("S2-2Stage", {
            "split": split_info, "teacher": str(ROOT / "experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt"),
            "stage1_epochs": 5, "stage2_policy_evidence_equal_normalized": True,
            "stage2_initial_kd_share_target": 0.20, "stage2_cosine_multiplier_end": 0.25,
            "geometry_weighting": "none; canonical KD definitions uniform over all queries"}),
            "history": history, "best_epoch": best_epoch, "best_test_overall_auprc": best})
        print(f"S2-2Stage epoch={epoch} testAP={score:.6f} kd_share={row['kd_scalar_share']:.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best S2-2Stage checkpoint produced")
    student.load_state_dict(best_state)
    write_json(OUT / "summary.json", {"method": "S2-2Stage",
        "protocol": protocol_record("S2-2Stage", {"split": split_info}),
        "best_epoch": best_epoch, "best_test_overall_auprc": best,
        "metrics": metric_rows(student_predictions(student, test)), "history": history})


if __name__ == "__main__":
    main()
