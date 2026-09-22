"""Counter-guided, target-value-only deformable-search Teacher (v2).

Counter features may affect only the policy context.  The task decoder consumes
only the searched target-value representation; it has no target, counter, or
dual-fusion skip input.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from windows_main.models_da_search import (CHANNELS, LEVELS, POINTS, DeformableSearch,
                                           SearchContext, pyramid, pyramid_from_encoded)
from models.landslide_cocd import ConvNeXtTinyFPN


class AnchoredDeformablePolicy(nn.Module):
    """One fixed centre point and three learnable points at every FPN level.

    Offsets are emitted in (y, x) feature-pixel units before bounded tanh in
    ``DeformableSearch``.  Weights are normalized jointly over all L x K
    candidates for each query; there is one head in this compact implementation.
    """
    def __init__(self, channels: int = CHANNELS, points: int = POINTS):
        super().__init__()
        if points != 4:
            raise ValueError("v2 fixes one centre plus three learnable points (POINTS=4)")
        self.points = points
        hidden = channels // 2
        self.body = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU())
        self.offset = nn.Conv2d(hidden, LEVELS * (points - 1) * 2, 1)
        self.weight = nn.Conv2d(hidden, LEVELS * points, 1)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight); nn.init.zeros_(self.weight.bias)

    def forward(self, context: torch.Tensor) -> dict:
        b, _, h, w = context.shape
        hidden = self.body(context)
        learned = self.offset(hidden).view(b, LEVELS, self.points - 1, 2, h, w)
        centre = torch.zeros((b, LEVELS, 1, 2, h, w), device=context.device, dtype=context.dtype)
        offsets = torch.cat((centre, learned), dim=2)
        logits = self.weight(hidden).view(b, LEVELS * self.points, h, w)
        weights = torch.softmax(logits, dim=1).view(b, LEVELS, self.points, h, w)
        return {"offsets": offsets, "weights": weights, "weight_logits": logits}


class TargetSearchTeacherV2(nn.Module):
    """Teacher with mandatory target-memory search on the decoder path.

    ``SearchContext(dual=True)`` is reused as a compact multi-scale context
    module.  Its second argument is *counter only* here and has no route to the
    decoder.  ``value_proj`` sees target FPN values only; ``out_proj(R)`` is the
    sole image representation passed to the inherited segmentation head.
    """
    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.context = SearchContext(channels, dual=True)
        self.policy = AnchoredDeformablePolicy(channels)
        self.value_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(LEVELS)])
        self.search = DeformableSearch(points=POINTS, offset_bound=2.0)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def _decode_from_search(self, retrieved: torch.Tensor, output_size) -> torch.Tensor:
        # No raw target/counter/context/policy feature is concatenated or added here.
        return F.interpolate(self.backbone.head(self.out_proj(retrieved)), size=output_size,
                             mode="bilinear", align_corners=False)

    def from_pyramids(self, target: Sequence[torch.Tensor], counter: Sequence[torch.Tensor], output_size) -> dict:
        # Counter enters only this context-to-policy computation.
        context = self.context(target, counter)
        policy = self.policy(context)
        values = tuple(layer(feature) for layer, feature in zip(self.value_proj, target))
        retrieved = self.search(values, policy)
        logits = self._decode_from_search(retrieved, output_size)
        return {"z": logits, "R": retrieved, "policy": policy, "values": values,
                "T": tuple(target), "C": tuple(counter), "memory_shapes": [list(x.shape[-2:]) for x in values]}

    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        return self.from_pyramids(pyramid(self.backbone, target), pyramid(self.backbone, counter), target.shape[-2:])

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor, counter_mode: str = "counter"):
        encoded = self.backbone.encode(torch.cat((asc, desc), dim=0))
        fa = tuple(x.chunk(2, dim=0)[0] for x in encoded)
        fd = tuple(x.chunk(2, dim=0)[1] for x in encoded)
        ta, td = pyramid_from_encoded(self.backbone, fa), pyramid_from_encoded(self.backbone, fd)
        if counter_mode == "counter":
            ca, cd = td, ta
        elif counter_mode == "self":
            ca, cd = ta, td
        else:
            raise ValueError(counter_mode)
        return self.from_pyramids(ta, ca, asc.shape[-2:]), self.from_pyramids(td, cd, desc.shape[-2:])

    def forward_from_cached_R(self, retrieved: torch.Tensor, output_size) -> torch.Tensor:
        """Test-only decoder entry point: proves decoder dependence on R alone."""
        return self._decode_from_search(retrieved, output_size)


def load_so_backbone(model: TargetSearchTeacherV2, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    source = ({k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
              if any(k.startswith("backbone.") for k in state) else state)
    model.backbone.load_state_dict(source, strict=True)


__all__ = ["TargetSearchTeacherV2", "AnchoredDeformablePolicy", "load_so_backbone"]
