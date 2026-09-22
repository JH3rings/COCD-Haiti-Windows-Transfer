"""Counter-Guided target-orbit evidence search.

This module is deliberately additive.  It reuses the project's five-channel
ConvNeXt-Tiny/FPN and the existing DCA implementation, but makes the only
new information path a change to the query used to search *target* P4.  Keys,
values, the analysis block, and the segmentation head are shared between the
self and dual readings.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN
from windows_main.models_complement import DCA

FPN_CHANNELS = 128
SEARCH_DIM = 32


def pyramid(backbone: ConvNeXtTinyFPN, x: torch.Tensor):
    """Return the existing decoder states p2, p3 and p4."""
    c2, c3, c4, c5 = backbone.encode(x)
    p5 = backbone.l5(c5)
    p4 = backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
    p3 = backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
    p2 = backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
    return p2, p3, p4


class TargetSearch(nn.Module):
    """One cross-scale search over all target P4 locations."""

    def __init__(self, channels: int = FPN_CHANNELS, dim: int = SEARCH_DIM):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.q = nn.Conv2d(channels, dim, 1)
        self.k = nn.Conv2d(channels, dim, 1)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                guidance: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        b, c, h3, w3 = p3.shape
        if p4.shape[1] != c:
            raise ValueError(f"P3/P4 channel mismatch: {tuple(p3.shape)} vs {tuple(p4.shape)}")
        q0 = self.q(p3)
        key = self.k(p4)
        # [B, Nq, d], [B, Nk, d], [B, Nk, C]. Values are raw target P4.
        q0f = q0.flatten(2).transpose(1, 2)
        kf = key.flatten(2).transpose(1, 2)
        vf = p4.flatten(2).transpose(1, 2)
        a0 = torch.softmax(torch.bmm(q0f, kf.transpose(1, 2)) / math.sqrt(self.dim), dim=-1)
        q = q0 if guidance is None else q0 + guidance
        qf = q.flatten(2).transpose(1, 2)
        a = torch.softmax(torch.bmm(qf, kf.transpose(1, 2)) / math.sqrt(self.dim), dim=-1)
        r = torch.bmm(a, vf).transpose(1, 2).reshape(b, c, h3, w3)
        r0 = torch.bmm(a0, vf).transpose(1, 2).reshape(b, c, h3, w3)
        return {
            "q0": q0, "q": q, "key": key, "value": p4,
            "A0": a0, "A": a, "rho": a - a0,
            "R0": r0, "R": r,
        }


class AnalysisBlock(nn.Module):
    """Compare P3 with retrieved target context and produce a P3 correction."""

    def __init__(self, channels: int = FPN_CHANNELS):
        super().__init__()
        hidden = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, p3: torch.Tensor, retrieved: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self.net(torch.cat((p3, retrieved, p3 - retrieved), dim=1))
        return p3 + delta, delta


class _CGBase(nn.Module):
    def __init__(self, channels: int = FPN_CHANNELS, dim: int = SEARCH_DIM):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.backbone = ConvNeXtTinyFPN()
        self.search = TargetSearch(channels, dim)
        self.analysis = AnalysisBlock(channels)

    def _logits(self, p2: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.backbone.head(p2), size=output_size,
                             mode="bilinear", align_corners=False)

    def _decode(self, p2: torch.Tensor, delta3: torch.Tensor,
                output_size: Tuple[int, int]) -> torch.Tensor:
        p2c = p2 + F.interpolate(delta3, size=p2.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return self._logits(p2c, output_size)


class CGSearchTeacher(_CGBase):
    """Training-time dual-orbit teacher; counter changes queries only."""

    def __init__(self, channels: int = FPN_CHANNELS, dim: int = SEARCH_DIM):
        super().__init__(channels, dim)
        self.dca = DCA(channels)
        self.guide_t = nn.Conv2d(channels, dim, 1)
        nn.init.zeros_(self.guide_t.weight)
        nn.init.zeros_(self.guide_t.bias)

    def from_features(self, target, counter, output_size: Tuple[int, int] = (128, 128)) -> dict:
        p2t, p3t, p4t = pyramid(self.backbone, target)
        _p2c, p3c, _p4c = pyramid(self.backbone, counter)
        b_t = self.guide_t(self.dca(p3t, p3c))
        self_read = self.search(p3t, p4t, None)
        dual_read = self.search(p3t, p4t, b_t)
        _h_self, delta_self = self.analysis(p3t, self_read["R0"])
        _h_dual, delta_dual = self.analysis(p3t, dual_read["R"])
        return {
            "z_self": self._decode(p2t, delta_self, output_size),
            "z_dual": self._decode(p2t, delta_dual, output_size),
            "A0": dual_read["A0"], "AT": dual_read["A"],
            "rho": dual_read["rho"], "rho_T": dual_read["rho"],
            "K": dual_read["key"], "V": dual_read["value"],
            "R0": self_read["R0"], "RT": dual_read["R"],
            "bT": b_t, "P2": p2t, "P3": p3t, "P4": p4t,
            "delta_self": delta_self, "delta_dual": delta_dual,
            "shapes": {"P2": tuple(p2t.shape), "P3": tuple(p3t.shape),
                       "P4": tuple(p4t.shape), "A": tuple(dual_read["A"].shape)},
        }

    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        if target.shape != counter.shape:
            raise ValueError("target and counter must have identical shapes")
        return self.from_features(target, counter, tuple(target.shape[-2:]))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        features = self.backbone.encode(torch.cat((asc, desc), dim=0))
        fa = tuple(t.chunk(2, dim=0)[0] for t in features)
        fd = tuple(t.chunk(2, dim=0)[1] for t in features)
        # Reuse encoded features while retaining the public pair API.
        return self.from_encoded(fa, fd, tuple(asc.shape[-2:])), self.from_encoded(fd, fa, tuple(desc.shape[-2:]))

    def from_encoded(self, ft, fc, output_size=(128, 128)) -> dict:
        def pyr(features):
            c2, c3, c4, c5 = features
            p5 = self.backbone.l5(c5)
            p4 = self.backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
            p3 = self.backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
            p2 = self.backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
            return p2, p3, p4
        p2t, p3t, p4t = pyr(ft)
        _p2c, p3c, _p4c = pyr(fc)
        b_t = self.guide_t(self.dca(p3t, p3c))
        self_read = self.search(p3t, p4t, None)
        dual_read = self.search(p3t, p4t, b_t)
        _, delta_self = self.analysis(p3t, self_read["R0"])
        _, delta_dual = self.analysis(p3t, dual_read["R"])
        return {
            "z_self": self._decode(p2t, delta_self, output_size),
            "z_dual": self._decode(p2t, delta_dual, output_size),
            "A0": dual_read["A0"], "AT": dual_read["A"],
            "rho": dual_read["rho"], "rho_T": dual_read["rho"],
            "K": dual_read["key"], "V": dual_read["value"],
            "R0": self_read["R0"], "RT": dual_read["R"], "bT": b_t,
            "P2": p2t, "P3": p3t, "P4": p4t,
            "delta_self": delta_self, "delta_dual": delta_dual,
            "shapes": {"P2": tuple(p2t.shape), "P3": tuple(p3t.shape),
                       "P4": tuple(p4t.shape), "A": tuple(dual_read["A"].shape)},
        }


class GuideS(nn.Module):
    def __init__(self, channels: int = FPN_CHANNELS, dim: int = SEARCH_DIM):
        super().__init__()
        hidden = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        p4u = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat((p3, p4u), dim=1))


class CGSearchStudent(_CGBase):
    """Target-only deployment student with a learned query guidance."""

    def __init__(self, channels: int = FPN_CHANNELS, dim: int = SEARCH_DIM):
        super().__init__(channels, dim)
        self.guide_s = GuideS(channels, dim)

    def _forward(self, x: torch.Tensor, force_zero_guidance: bool = False) -> dict:
        p2, p3, p4 = pyramid(self.backbone, x)
        b_s = torch.zeros((x.shape[0], self.dim, p3.shape[-2], p3.shape[-1]),
                          dtype=p3.dtype, device=p3.device) if force_zero_guidance else self.guide_s(p3, p4)
        read = self.search(p3, p4, b_s)
        _, delta_self = self.analysis(p3, read["R0"])
        _, delta = self.analysis(p3, read["R"])
        return {
            "z": self._decode(p2, delta, tuple(x.shape[-2:])),
            "z_self": self._decode(p2, delta_self, tuple(x.shape[-2:])),
            "A0": read["A0"], "AS": read["A"], "rho": read["rho"],
            "K": read["key"], "V": read["value"],
            "R0": read["R0"], "RS": read["R"], "bS": b_s,
            "P2": p2, "P3": p3, "P4": p4,
            "delta_self": delta_self, "delta": delta,
            "shapes": {"P2": tuple(p2.shape), "P3": tuple(p3.shape),
                       "P4": tuple(p4.shape), "A": tuple(read["A"].shape)},
        }

    def forward(self, x: torch.Tensor) -> dict:
        return self._forward(x, False)

    @torch.no_grad()
    def forward_zero_guidance(self, x: torch.Tensor) -> dict:
        return self._forward(x, True)


def copy_teacher_to_student(student: CGSearchStudent, teacher: CGSearchTeacher) -> None:
    """Copy the shared self/search/analysis path; leave GuideS at zero."""
    student.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)
    student.search.load_state_dict(teacher.search.state_dict(), strict=True)
    student.analysis.load_state_dict(teacher.analysis.state_dict(), strict=True)


def load_so_backbone(model: nn.Module, checkpoint: str | None = None) -> dict:
    """Load the existing SO checkpoint into a CGSearch model's backbone."""
    if checkpoint is None:
        checkpoint = "experiments/ours_v2/SO_wm_seed42.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    source = {k.removeprefix("backbone."): v for k, v in state.items()
              if k.startswith("backbone.")}
    result = model.backbone.load_state_dict(source, strict=True)
    return {"checkpoint": str(checkpoint), "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys)}


def parameter_counts(teacher: CGSearchTeacher | None = None,
                     student: CGSearchStudent | None = None) -> dict:
    base = ConvNeXtTinyFPN()
    ans = {"SO_total": sum(p.numel() for p in base.parameters())}
    if teacher is not None:
        ans["Teacher_total"] = sum(p.numel() for p in teacher.parameters())
        ans["Teacher_extra_vs_SO"] = ans["Teacher_total"] - ans["SO_total"]
    if student is not None:
        ans["Student_total"] = sum(p.numel() for p in student.parameters())
        ans["Student_extra_vs_SO"] = ans["Student_total"] - ans["SO_total"]
    return ans


def action_distance(rho_student: torch.Tensor, rho_teacher: torch.Tensor) -> torch.Tensor:
    """Per-query mean absolute search-action distance, with teacher detached."""
    return (rho_student - rho_teacher.detach()).abs().mean(dim=-1)


__all__ = [
    "CGSearchTeacher", "CGSearchStudent", "TargetSearch", "AnalysisBlock",
    "copy_teacher_to_student", "load_so_backbone", "parameter_counts",
    "action_distance", "pyramid",
]
