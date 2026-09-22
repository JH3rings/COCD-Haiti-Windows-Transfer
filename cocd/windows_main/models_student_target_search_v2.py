"""Target-only Student whose decoder is reachable only through searched evidence."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN
from windows_main.models_da_search import (CHANNELS, LEVELS, POINTS, DeformableSearch,
                                           SearchContext, pyramid)
from windows_main.models_teacher_target_search_v2 import AnchoredDeformablePolicy


class StudentTargetSearchV2(nn.Module):
    """Single-orbit analogue of Teacher v2; raw target features never reach decoder."""
    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.context = SearchContext(channels, dual=False)
        self.policy = AnchoredDeformablePolicy(channels)
        self.value_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(LEVELS)])
        self.search = DeformableSearch(points=POINTS, offset_bound=2.0)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def decode_R(self, R: torch.Tensor, output_size) -> torch.Tensor:
        return F.interpolate(self.backbone.head(self.out_proj(R)), size=output_size,
                             mode="bilinear", align_corners=False)

    def from_pyramid(self, target: Sequence[torch.Tensor], output_size) -> dict:
        policy = self.policy(self.context(target))
        values = tuple(layer(x) for layer, x in zip(self.value_proj, target))
        R = self.search(values, policy)
        return {"z": self.decode_R(R, output_size), "R": R, "policy": policy,
                "values": values, "T": tuple(target)}

    def forward(self, target: torch.Tensor) -> dict:
        return self.from_pyramid(pyramid(self.backbone, target), target.shape[-2:])

    def decode_from_R_for_test(self, R: torch.Tensor, output_size) -> torch.Tensor:
        return self.decode_R(R, output_size)
