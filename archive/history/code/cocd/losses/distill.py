"""Distillation terms for the Haiti structural change-decoding study.

Two families live here and they deliberately score the *same* set of taps, so an
ablation can swap the rule without touching the network:

``l2_kd_loss`` / ``dis2_logit_kl`` / ``dis2_multilevel_kd``
    Byte-faithful ports of the DIS2 snapshot in
    ``third_party/landslide_baselines/dis2/models_bank/DLKD_ver4.py``: the four
    fused levels become attention-pooled tokens compared by a channel-normalised
    mean squared difference, the penultimate map is compared the same way, and
    the classifier logits are matched by a temperature-scaled KL.  The teacher
    side is always detached, as in the snapshot.

``selective_multilevel_kd``
    The same taps, rescored.  Every term is weighted per pixel by
    ``gain x protect`` and the correction *effect* is matched directly instead of
    the pre-correction activation.

Nothing in this module reads geometry, and nothing here runs at inference time.
"""
import torch
from torch.nn import functional as F

EPS = 1e-7
T_DIS2 = 2.0

__all__ = [
    'l2_kd_loss', 'orthogonality_loss', 'dis2_logit_kl', 'dis2_multilevel_kd',
    'dis2_logit_only', 'gain_protect', 'selective_multilevel_kd', 'pool_tokens',
]


# --------------------------------------------------------------------------- #
# DIS2, verbatim
# --------------------------------------------------------------------------- #
def l2_kd_loss(a, b, spatial=False):
    """Channel-normalised mean squared difference, DIS2's ``l2_kd_loss``.

    ``F.normalize(dim=1)`` scales each channel to unit norm before the
    difference, so the term is a cosine-type distance and is insensitive to
    channel gain.  Both branches of the snapshot's ``if spatial`` reduce every
    element of the tensor, so the flag only changes the expression, not the
    value; it is kept so the port can be diffed against the original.
    """
    a_norm = F.normalize(a, p=2, dim=1, eps=1e-6)
    b_norm = F.normalize(b, p=2, dim=1, eps=1e-6)
    if spatial:
        return (a_norm - b_norm).pow(2).mean(dim=(0, 1, 2, 3))
    return F.mse_loss(a_norm, b_norm, reduction='mean')


def orthogonality_loss(a, b):
    """Squared cosine similarity between two ``(B, D)`` token sets, DIS2 verbatim."""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return ((a_norm * b_norm).sum(dim=-1) ** 2).mean()


def pool_tokens(x):
    """``(B, C, H, W)`` to a ``(B, C)`` token.

    DIS2 pools with a learnable ``AttnPoolToToken``; mean pooling is used here so
    the distilled arm carries exactly the parameter count of its control and the
    arm differs only in the objective.
    """
    return x.mean(dim=(2, 3))


def _two_class(z):
    """``(B, 1, H, W)`` logits to ``(B, 2, H, W)`` with a fixed zero background logit."""
    return torch.cat((torch.zeros_like(z), z), 1)


def dis2_logit_kl(s_logits, t_logits, T=T_DIS2):
    """DIS2's logits KL: ``log_softmax(student / T)`` against ``softmax(teacher / T)``, times ``T**2``."""
    q = F.log_softmax(_two_class(s_logits) / T, dim=1)
    with torch.no_grad():
        p = F.softmax(_two_class(t_logits) / T, dim=1)
    return F.kl_div(q, p, reduction='mean') * T * T


def dis2_multilevel_kd(stu, tea, lam_feat=1.0, lam_pen=1.0, lam_logit=1.0, lam_div=1.0):
    """The DIS2 distillation block, unchanged apart from where the taps come from.

    All three pieces of ``DLKD_ver4.py`` are present: the four pooled fused levels
    and the penultimate map compared by ``l2_kd_loss`` against a detached teacher,
    the temperature-scaled logits KL, and the diversity term that keeps the
    correction dissimilar to the state it corrects.

    Returns ``(total, parts)``; ``parts`` keeps every term separate so the run log
    shows which one is actually moving.

    ``stu`` / ``tea`` are tap dictionaries with ``levels`` (a 4-tuple, coarse to
    fine), ``pen`` (the penultimate map), ``logits`` and, on the student side,
    ``r`` (the two corrections, fine to coarse).
    """
    # ``r`` is (fine, coarse) while ``levels`` runs coarse to fine, so the
    # correction pairs with the reversed tail of ``levels``: r3 with p3, r4 with
    # p4.  Without this term the cheapest correction available to the student is
    # to re-encode the target track, which would make the correction vacuous.
    corrected = stu['levels'][1:3][::-1]
    parts = {
        'feat': sum(l2_kd_loss(pool_tokens(stu['levels'][k]), pool_tokens(tea['levels'][k]))
                    for k in range(len(stu['levels']))),
        'pen': l2_kd_loss(stu['pen'], tea['pen'], spatial=True),
        'logit': dis2_logit_kl(stu['logits'], tea['logits']),
        'div': sum(orthogonality_loss(pool_tokens(r), pool_tokens(p))
                   for r, p in zip(stu['r'], corrected)),
    }
    return (lam_feat * parts['feat'] + lam_pen * parts['pen'] + lam_logit * parts['logit']
            + lam_div * parts['div']), parts


def dis2_logit_only(stu, tea, lam_logit=1.0):
    """DIS2's logits KL alone; the probe that asks whether the feature terms carry anything."""
    parts = {'logit': dis2_logit_kl(stu['logits'], tea['logits'])}
    return lam_logit * parts['logit'], parts


# --------------------------------------------------------------------------- #
# Selective rescoring of the same taps
# --------------------------------------------------------------------------- #
def gain_protect(t_self, t_dual, s_out, y):
    """Per-pixel weight: where the privileged track helps and the student is not already better.

    ``gain`` is the relative BCE improvement of the dual track over the self
    track, clamped to ``[0, 1]``; ``protect`` zeroes the weight wherever the
    student's own prediction already beats the dual track.  Both factors are
    detached, so the weight never receives gradient.
    """
    with torch.no_grad():
        e0 = F.binary_cross_entropy_with_logits(t_self[:, 0], y, reduction='none')
        et = F.binary_cross_entropy_with_logits(t_dual[:, 0], y, reduction='none')
        es = F.binary_cross_entropy_with_logits(s_out[:, 0], y, reduction='none')
        gain = ((e0 - et) / (e0 + EPS)).clamp_(0, 1)
        protect = (et < es).float()
    return gain * protect


def _weighted_norm_mse(a, b, w):
    """Channel-normalised squared difference, spatially resolved, averaged under ``w``.

    Unlike ``l2_kd_loss`` this never pools, so the term keeps the location of the
    disagreement, and ``w`` decides how much each location counts.  Dividing by
    ``sum(w)`` rather than by the pixel count makes levels of different
    resolution directly comparable.
    """
    d = (F.normalize(a, p=2, dim=1, eps=1e-6) - F.normalize(b.detach(), p=2, dim=1, eps=1e-6)).pow(2).mean(1)
    wd = F.adaptive_avg_pool2d(w[:, None], d.shape[-2:])[:, 0]
    return (d * wd).sum() / (wd.sum() + EPS)


def _balanced_mean(x, y):
    """Foreground/background means with full-class denominators, never ``sum(w)``.

    This is the aggregation the ``gain`` control already uses, so reusing it here
    makes the selective arm's output term identical to the control's rather than
    merely similar.
    """
    values = [x[y == cls].mean() for cls in (0., 1.) if (y == cls).any()]
    return torch.stack(values).mean() if values else x.new_zeros(())


def _selective_logit_kl(s_logits, t_logits, y, w):
    """Stable Bernoulli KL at T=1, aggregated exactly as the ``gain`` control does it.

    The classifier logits live at the working resolution while ``y`` and ``w`` are
    at the label resolution, so the per-pixel KL is resampled up before the
    weighting rather than the label being resampled down.
    """
    s, t = s_logits[:, 0], t_logits[:, 0]
    with torch.no_grad():
        target_p = torch.sigmoid(t)
        entropy = F.binary_cross_entropy(target_p, target_p, reduction='none')
    soft_ce = F.binary_cross_entropy_with_logits(s, target_p, reduction='none')
    per_pixel = soft_ce - entropy
    if per_pixel.shape[-2:] != y.shape[-2:]:
        per_pixel = F.interpolate(per_pixel[:, None], size=y.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    return _balanced_mean(w * per_pixel, y)


def selective_multilevel_kd(stu, tea, w, y, lam_feat=1.0, lam_eff=1.0, lam_pen=1.0, lam_logit=1.0):
    """The multi-level design with both of its failure modes addressed.

    Three changes against ``dis2_multilevel_kd``:

    * every tap is scored through ``w``, so supervision concentrates where the
      counter track demonstrably helps rather than spreading uniformly;
    * ``eff`` matches the correction the two tracks actually produced
      (``r_k``, the teacher's Omega output) instead of the raw activation, which
      is the representable part of the privileged signal;
    * ``levels`` and ``pen`` are scored spatially, so a pooled token cannot hide a
      localised disagreement.

    The ``logit`` term is the ``gain`` control's own output term, evaluated on the
    same weights and the same tensors, so this arm differs from that control by
    exactly the three terms above and by nothing else.
    """
    parts = {
        'feat': sum(_weighted_norm_mse(stu['levels'][k], tea['levels'][k], w)
                    for k in range(len(stu['levels']))) / len(stu['levels']),
        'eff': sum(_weighted_norm_mse(s, t, w) for s, t in zip(stu['r'], tea['r'])),
        'pen': _weighted_norm_mse(stu['pen'], tea['pen'], w),
        'logit': .5 * (_selective_logit_kl(stu['z4'], tea['z4'], y, w)
                       + _selective_logit_kl(stu['z34'], tea['z34'], y, w)),
    }
    total = (lam_feat * parts['feat'] + lam_eff * parts['eff']
             + lam_pen * parts['pen'] + lam_logit * parts['logit'])
    return total, parts
