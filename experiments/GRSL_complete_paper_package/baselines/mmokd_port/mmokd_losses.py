from __future__ import annotations

import torch
import torch.nn.functional as F


TEMPERATURE = 4.0


def ce_plus_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    labels = target.long()
    ce = F.cross_entropy(logits, labels)
    probability = torch.softmax(logits, dim=1)[:, 1]
    truth = target.float()
    dims = (1, 2)
    intersection = (probability * truth).sum(dims)
    denominator = probability.sum(dims) + truth.sum(dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce + dice


def cosine_relations(feature: torch.Tensor) -> torch.Tensor:
    pixels = F.normalize(feature.flatten(2), p=2, dim=1)
    return pixels.transpose(1, 2) @ pixels


def response_kd(student: torch.Tensor, teacher: torch.Tensor,
                temperature: float = TEMPERATURE) -> torch.Tensor:
    target = F.softmax(teacher.detach() / temperature, dim=1)
    predicted = F.log_softmax(student / temperature, dim=1)
    per_pixel = F.kl_div(predicted, target, reduction="none").sum(dim=1)
    return per_pixel.mean() * (temperature ** 2)


def mmokd_terms(outputs: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = outputs["teacher"]
    student_a = outputs["student_asc"]
    student_d = outputs["student_desc"]

    seg_t = ce_plus_dice(teacher["logits"], target)
    seg_a = ce_plus_dice(student_a["logits"], target)
    seg_d = ce_plus_dice(student_d["logits"], target)
    seg = (seg_t + seg_a + seg_d) / 3.0

    # JD-MFK encoding-level MSE: one low-level and one high-level map.
    feature_pairs = []
    for student in (student_a, student_d):
        feature_pairs.extend((
            F.mse_loss(student["pseudo_fusion"][0], teacher["fused"][0].detach()),
            F.mse_loss(student["pseudo_fusion"][3], teacher["fused"][3].detach()),
        ))
    feature = torch.stack(feature_pairs).mean()

    # JD-MFK decoding-level full pixel-to-pixel cosine relation matching.
    relation_target = cosine_relations(teacher["decoder"].detach())
    relation = 0.5 * (
        F.mse_loss(cosine_relations(student_a["decoder"]), relation_target) +
        F.mse_loss(cosine_relations(student_d["decoder"]), relation_target)
    )

    response = 0.5 * (
        response_kd(student_a["logits"], teacher["logits"]) +
        response_kd(student_d["logits"], teacher["logits"])
    )
    distill = (feature + relation + response) / 3.0
    total = seg + distill  # fixed symmetric beta=1, alpha=1; no sweep/controller
    return {"total": total, "seg": seg, "seg_teacher": seg_t,
            "seg_student_asc": seg_a, "seg_student_desc": seg_d,
            "feature_kd": feature, "relation_kd": relation,
            "response_kd": response, "distill": distill}


__all__ = ["ce_plus_dice", "cosine_relations", "mmokd_terms", "response_kd"]
