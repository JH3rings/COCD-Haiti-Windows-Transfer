from __future__ import annotations

import torch
import torch.nn.functional as F


def haar_high_frequency(x: torch.Tensor) -> torch.Tensor:
    """One-level orthonormal Haar DWT detail bands (LH, HL, HH), differentiably."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = x[..., :x.shape[-2] // 2 * 2, :x.shape[-1] // 2 * 2]
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.stack((lh, hl, hh), dim=2)


def fad_loss(teacher_features, student_features) -> torch.Tensor:
    terms = []
    for level in (0, 1):  # P2/P3-equivalent lowest-level encoder maps
        target = haar_high_frequency(teacher_features[level].detach())
        prediction = haar_high_frequency(student_features[level])
        terms.append(F.mse_loss(prediction, target))
    return torch.stack(terms).mean()


def affinity_matrix(features: torch.Tensor) -> torch.Tensor:
    # Full spatial cross-affinity, with no local window or spatial cropping.
    vectors = F.normalize(features.flatten(2), dim=1)
    return vectors.transpose(1, 2) @ vectors


def cad_loss(teacher_asc, teacher_desc, student_asc, student_desc) -> torch.Tensor:
    # Full cross-view teacher affinity maps supervise self-affinity of each deployable view.
    ta = F.normalize(teacher_asc.detach().flatten(2), dim=1)
    td = F.normalize(teacher_desc.detach().flatten(2), dim=1)
    teacher_cross = ta.transpose(1, 2) @ td
    student_a = affinity_matrix(student_asc)
    student_d = affinity_matrix(student_desc)
    return 0.5 * (F.l1_loss(student_a, teacher_cross) +
                  F.l1_loss(student_d, teacher_cross.transpose(1, 2)))

