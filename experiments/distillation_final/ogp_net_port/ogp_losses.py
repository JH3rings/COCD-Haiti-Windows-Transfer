from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    target = y.long()
    ce = F.cross_entropy(logits, target)
    p = torch.softmax(logits, dim=1)[:, 1]
    t = target.float()
    dice = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return ce + dice


def dist_response(student_logits: torch.Tensor, teacher_binary_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DIST inter-class and intra-class Pearson relations with explicit two-class logits."""
    teacher_logits = torch.cat((-teacher_binary_logit, teacher_binary_logit), dim=1)
    if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
        teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:],
                                       mode="bilinear", align_corners=False)
    s = student_logits.flatten(2)
    t = teacher_logits.flatten(2)

    def one_minus_corr(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
        a = a - a.mean(dim=dim, keepdim=True)
        b = b - b.mean(dim=dim, keepdim=True)
        numerator = (a * b).sum(dim=dim)
        denominator = torch.sqrt((a * a).sum(dim=dim) * (b * b).sum(dim=dim)).clamp_min(1e-6)
        return 1.0 - numerator / denominator

    # Inter-class relation at each pixel; intra-class relation over all pixels.
    inter = one_minus_corr(s.transpose(1, 2), t.transpose(1, 2), dim=-1).mean()
    intra = one_minus_corr(s, t, dim=-1).mean()
    return inter, intra


def _sample_stage(projected_a: torch.Tensor, projected_d: torch.Tensor, y: torch.Tensor,
                  samples_per_class_per_view: int = 16):
    b, _, h, w = projected_a.shape
    labels = F.interpolate(y[:, None].float(), size=(h, w), mode="nearest")[:, 0].long()
    vectors, classes = [], []
    for feature in (projected_a, projected_d):
        for bi in range(b):
            flat = feature[bi].permute(1, 2, 0).reshape(-1, feature.shape[1])
            lab = labels[bi].reshape(-1)
            for cls in (0, 1):
                idx = torch.where(lab == cls)[0]
                if idx.numel() == 0:
                    continue
                if idx.numel() > samples_per_class_per_view:
                    idx = idx[torch.randperm(idx.numel(), device=idx.device)[:samples_per_class_per_view]]
                vectors.append(flat[idx])
                classes.append(torch.full((idx.numel(),), cls, device=idx.device, dtype=torch.long))
    if not vectors:
        return projected_a.new_zeros((0, 64)), torch.empty(0, device=y.device, dtype=torch.long)
    return torch.cat(vectors, dim=0), torch.cat(classes, dim=0)


def semi_hard_pixel_contrastive(projected_pairs, y: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Supervised pixel contrastive objective with hardest positive and closest-negative top 10%."""
    losses = []
    for projected_a, projected_d in projected_pairs:
        vectors, labels = _sample_stage(projected_a, projected_d, y)
        if vectors.shape[0] < 3:
            continue
        embeddings = F.normalize(vectors, dim=1)
        similarity = embeddings @ embeddings.t()
        n = similarity.shape[0]
        self_mask = torch.eye(n, dtype=torch.bool, device=similarity.device)
        same = labels[:, None].eq(labels[None, :]) & ~self_mask
        different = labels[:, None].ne(labels[None, :])
        has_pos = same.any(dim=1)
        has_neg = different.any(dim=1)
        # The farthest positive is the difficult positive; retain closest-negative top 10%.
        pos_sim = similarity.masked_fill(~same, float("inf")).min(dim=1).values
        neg_count = different.sum(dim=1)
        k_max = max(1, math.ceil(0.10 * max(1, n - 1)))
        top_values = similarity.masked_fill(~different, float("-inf")).topk(
            min(k_max, n), dim=1, largest=True).values
        ranks = torch.arange(top_values.shape[1], device=similarity.device)[None, :]
        selected_negative = ranks < torch.ceil(0.10 * neg_count.float())[:, None].clamp_min(1)
        neg_sim = top_values.masked_fill(~selected_negative, float("-inf"))
        valid = has_pos & has_neg
        if valid.any():
            positive = (pos_sim[valid] / temperature)[:, None]
            negatives = neg_sim[valid] / temperature
            log_denominator = torch.logsumexp(torch.cat((positive, negatives), dim=1), dim=1)
            losses.append((log_denominator - positive[:, 0]).mean())
    if not losses:
        return projected_pairs[0][0].sum() * 0.0
    return torch.stack(losses).mean()
