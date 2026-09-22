"""Shared search primitives for the final target-evidence method.

The final Teacher and Student share the same FPN construction, policy context,
and target-memory deformable search. Counter features are supplied to
``SearchContext`` only by the Teacher. Superseded search architectures and
their initialization helpers are archived under ``archive/history/code``.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN

CHANNELS = 128
LEVELS = 3
POINTS = 4


def pyramid_from_encoded(backbone: ConvNeXtTinyFPN, features):
    """Build the three-level target memory pyramid from encoded features."""
    c2, c3, c4, c5 = features
    p5 = backbone.l5(c5)
    p4 = backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
    p3 = backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
    p2 = backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
    return (p2, p3, p4)


def pyramid(backbone: ConvNeXtTinyFPN, x: torch.Tensor):
    """Encode one orbit and return its target memory pyramid."""
    return pyramid_from_encoded(backbone, backbone.encode(x))


class SearchContext(nn.Module):
    """Fuse all scales into a policy context.

    With ``dual=False`` this is target-only. With ``dual=True`` the caller must
    provide the counter-derived context features as the second argument. This
    module never supplies values to the decoder; it only produces policy input.
    """

    def __init__(self, channels: int = CHANNELS, dual: bool = False):
        super().__init__()
        self.dual = dual
        self.target_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(LEVELS)
        ])
        if dual:
            self.dual_proj = nn.ModuleList([
                nn.Conv2d(channels, channels, 1) for _ in range(LEVELS)
            ])

    def forward(self, target: Sequence[torch.Tensor], dual_features=None) -> torch.Tensor:
        size = target[0].shape[-2:]
        context = None
        for i, t in enumerate(target):
            term = self.target_proj[i](t)
            if self.dual:
                if dual_features is None:
                    raise ValueError("dual policy context requires fused features")
                term = term + self.dual_proj[i](dual_features[i])
            term = F.interpolate(term, size=size, mode="bilinear", align_corners=False)
            context = term if context is None else context + term
        return context / math.sqrt(LEVELS)


class DeformableSearch(nn.Module):
    """Apply one policy to a three-level target memory pyramid."""

    def __init__(self, points: int = POINTS, offset_bound: float = 2.0):
        super().__init__()
        self.points = points
        self.offset_bound = offset_bound

    @staticmethod
    def _grid(hq: int, wq: int, hm: int, wm: int, offset: torch.Tensor) -> torch.Tensor:
        dtype, device = offset.dtype, offset.device
        yy = (torch.arange(hq, device=device, dtype=dtype) + 0.5) / hq * hm - 0.5
        xx = (torch.arange(wq, device=device, dtype=dtype) + 0.5) / wq * wm - 0.5
        by, bx = torch.meshgrid(yy, xx, indexing="ij")
        y = by[None] + offset[:, 0]
        x = bx[None] + offset[:, 1]
        gx = (x + 0.5) / wm * 2.0 - 1.0
        gy = (y + 0.5) / hm * 2.0 - 1.0
        return torch.stack((gx, gy), dim=-1)

    def forward(self, memory: Sequence[torch.Tensor], policy: dict) -> torch.Tensor:
        offsets = self.offset_bound * torch.tanh(policy["offsets"])
        weights = policy["weights"]
        hq, wq = memory[0].shape[-2:]
        result = torch.zeros_like(memory[0])
        for level, feat in enumerate(memory):
            hm, wm = feat.shape[-2:]
            for point in range(self.points):
                grid = self._grid(hq, wq, hm, wm, offsets[:, level, point])
                sampled = F.grid_sample(
                    feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                result = result + sampled * weights[:, level, point, None]
        return result


class _TargetBackbone(nn.Module):
    """Small private adapter for the ordinary target-only reference."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.channels = channels

    def logits(self, p2: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            self.backbone.head(p2), size=output_size,
            mode="bilinear", align_corners=False,
        )


class TargetOnlyBaseline(_TargetBackbone):
    """Ordinary target-only reference used by final evaluation and figures."""

    def forward(self, x: torch.Tensor) -> dict:
        p2, _, _ = pyramid(self.backbone, x)
        return {"z": self.logits(p2, tuple(x.shape[-2:]))}


__all__ = [
    "CHANNELS", "LEVELS", "POINTS", "pyramid", "pyramid_from_encoded",
    "SearchContext", "DeformableSearch", "TargetOnlyBaseline",
]
