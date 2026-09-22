"""Normal dual fusion plus target-memory deformable search.

The module is intentionally separate from the earlier CGSearch experiment.
Normal dual fusion is an explicit path in the Teacher; the deformable search
policy only chooses samples, and every sampled value comes from the target
feature pyramid.  Students have the same search/write/decode path but no
counter input.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN

CHANNELS = 128
LEVELS = 3
POINTS = 4


def pyramid_from_encoded(backbone: ConvNeXtTinyFPN, features):
    c2, c3, c4, c5 = features
    p5 = backbone.l5(c5)
    p4 = backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
    p3 = backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
    p2 = backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
    return (p2, p3, p4)


def pyramid(backbone: ConvNeXtTinyFPN, x: torch.Tensor):
    return pyramid_from_encoded(backbone, backbone.encode(x))


class FusionPyramid(nn.Module):
    """Independent, same-channel 1x1 target/counter fusion at all FPN levels."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.fuse = nn.ModuleList([nn.Conv2d(2 * channels, channels, 1)
                                   for _ in range(LEVELS)])

    def forward(self, target: Sequence[torch.Tensor], counter: Sequence[torch.Tensor]):
        return tuple(layer(torch.cat((t, c), dim=1))
                     for layer, t, c in zip(self.fuse, target, counter))


class SearchContext(nn.Module):
    """Fuse all scales into a common policy context without fixed semantics."""

    def __init__(self, channels: int = CHANNELS, dual: bool = False):
        super().__init__()
        self.dual = dual
        self.target_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1)
                                          for _ in range(LEVELS)])
        if dual:
            self.dual_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1)
                                            for _ in range(LEVELS)])

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


class DeformablePolicy(nn.Module):
    """Predict offsets and normalized weights at the finest FPN grid."""

    def __init__(self, channels: int = CHANNELS, points: int = POINTS):
        super().__init__()
        self.points = points
        hidden = channels // 2
        self.body = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU())
        self.offset = nn.Conv2d(hidden, LEVELS * points * 2, 1)
        self.weight = nn.Conv2d(hidden, LEVELS * points, 1)
        # Start at a stable uniform, zero-offset search.  This keeps T1/S0
        # close to their inherited single/dual baselines at iteration zero.
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight); nn.init.zeros_(self.weight.bias)

    def forward(self, context: torch.Tensor) -> dict:
        b, _, h, w = context.shape
        hidden = self.body(context)
        offsets = self.offset(hidden).view(b, LEVELS, self.points, 2, h, w)
        weights = self.weight(hidden).view(b, LEVELS * self.points, h, w)
        weights = torch.softmax(weights, dim=1).view(b, LEVELS, self.points, h, w)
        return {"offsets": offsets, "weights": weights}


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
                sampled = F.grid_sample(feat, grid, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)
                result = result + sampled * weights[:, level, point, None]
        return result


class SearchWrite(nn.Module):
    """Write target/search evidence back into the existing P2 head input."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, target_p2: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat((target_p2, retrieved), dim=1))


class _Base(nn.Module):
    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.channels = channels

    def logits(self, p2: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.backbone.head(p2), size=output_size,
                             mode="bilinear", align_corners=False)


class DAStudent(_Base):
    """Target-only DA search network used by the active S0 Student."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__(channels)
        self.context = SearchContext(channels, dual=False)
        self.policy = DeformablePolicy(channels)
        self.search = DeformableSearch()
        self.write = SearchWrite(channels)

    def from_pyramid(self, target, policy=None, output_size=(128, 128)) -> dict:
        p2, p3, p4 = target
        if policy is None:
            policy = self.policy(self.context(target))
        retrieved = self.search(target, policy)
        correction = self.write(p2, retrieved)
        return {"z": self.logits(p2 + correction, output_size),
                "z_base": self.logits(p2, output_size),
                "R": retrieved, "policy": policy,
                "P2": p2, "P3": p3, "P4": p4,
                "offsets": policy["offsets"], "weights": policy["weights"]}

    def forward(self, x: torch.Tensor) -> dict:
        return self.from_pyramid(pyramid(self.backbone, x), output_size=tuple(x.shape[-2:]))

    def forward_with_policy(self, x: torch.Tensor, policy: dict) -> dict:
        return self.from_pyramid(pyramid(self.backbone, x), policy=policy,
                                 output_size=tuple(x.shape[-2:]))


class TargetOnlyBaseline(_Base):
    """Ordinary target-only reference used only as a non-student baseline.

    This is the current DA-search FPN/head path with the search branch absent.
    It replaces the retired q3/q4 ``R0`` implementation for reporting the
    single-orbit reference.  It is not a Student model and is never used to
    initialise S0.
    """

    def forward(self, x: torch.Tensor) -> dict:
        p2, _, _ = pyramid(self.backbone, x)
        return {"z": self.logits(p2, tuple(x.shape[-2:]))}


class NormalDualTeacher(_Base):
    """T0: normal dual fusion, without any DA module."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__(channels)
        self.fusion = FusionPyramid(channels)

    def from_pyramids(self, target, counter, output_size=(128, 128)) -> dict:
        dual = self.fusion(target, counter)
        return {"z_dual": self.logits(dual[0], output_size),
                "T": target, "C": counter, "D": dual}

    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        return self.from_pyramids(pyramid(self.backbone, target),
                                  pyramid(self.backbone, counter), tuple(target.shape[-2:]))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        features = self.backbone.encode(torch.cat((asc, desc), dim=0))
        fa = tuple(t.chunk(2, dim=0)[0] for t in features)
        fd = tuple(t.chunk(2, dim=0)[1] for t in features)
        return (self.from_pyramids(pyramid_from_encoded(self.backbone, fa),
                                   pyramid_from_encoded(self.backbone, fd), tuple(asc.shape[-2:])),
                self.from_pyramids(pyramid_from_encoded(self.backbone, fd),
                                   pyramid_from_encoded(self.backbone, fa), tuple(desc.shape[-2:])))


class DASearchTeacher(_Base):
    """T1: explicit dual fusion plus target-memory deformable search."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__(channels)
        self.fusion = FusionPyramid(channels)
        self.context = SearchContext(channels, dual=True)
        self.policy = DeformablePolicy(channels)
        self.search = DeformableSearch()
        self.write = SearchWrite(channels)

    def from_pyramids(self, target, counter, output_size=(128, 128)) -> dict:
        dual = self.fusion(target, counter)
        policy = self.policy(self.context(target, dual))
        retrieved = self.search(target, policy)
        correction = self.write(target[0], retrieved)
        return {
            "z_dual": self.logits(dual[0] + correction, output_size),
            "z_dual_noda": self.logits(dual[0], output_size),
            "z_search": self.logits(target[0] + correction, output_size),
            "R": retrieved, "policy": policy,
            "T": target, "C": counter, "D": dual,
            "offsets": policy["offsets"], "weights": policy["weights"],
        }

    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        return self.from_pyramids(pyramid(self.backbone, target),
                                  pyramid(self.backbone, counter), tuple(target.shape[-2:]))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        features = self.backbone.encode(torch.cat((asc, desc), dim=0))
        fa = tuple(t.chunk(2, dim=0)[0] for t in features)
        fd = tuple(t.chunk(2, dim=0)[1] for t in features)
        ta, td = pyramid_from_encoded(self.backbone, fa), pyramid_from_encoded(self.backbone, fd)
        return (self.from_pyramids(ta, td, tuple(asc.shape[-2:])),
                self.from_pyramids(td, ta, tuple(desc.shape[-2:])))


def load_so_backbone(model: nn.Module, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    # Active runs use a compact bare-backbone initialisation checkpoint.  The
    # prefixed branch keeps compatibility with a packaged checkpoint that
    # still wraps the same state under ``model``/``backbone``.
    source = ({k.removeprefix("backbone."): v for k, v in state.items()
              if k.startswith("backbone.")}
              if any(k.startswith("backbone.") for k in state) else state)
    model.backbone.load_state_dict(source, strict=True)


def init_t1_from_t0(t1: DASearchTeacher, t0: NormalDualTeacher) -> None:
    t1.backbone.load_state_dict(t0.backbone.state_dict(), strict=True)
    t1.fusion.load_state_dict(t0.fusion.state_dict(), strict=True)


def init_student_from_teacher(student: DAStudent, teacher: DASearchTeacher) -> None:
    student.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)
    student.write.load_state_dict(teacher.write.state_dict(), strict=True)
    student.context.target_proj.load_state_dict(teacher.context.target_proj.state_dict(), strict=True)
    student.policy.load_state_dict(teacher.policy.state_dict(), strict=True)


def normalized_feature_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1, eps=1e-6)
    b = F.normalize(b, dim=1, eps=1e-6)
    return (a - b.detach()).abs().mean()


def parameter_counts(t0=None, t1=None, student=None) -> dict:
    base = ConvNeXtTinyFPN()
    ans = {"SO_total": sum(p.numel() for p in base.parameters())}
    for name, model in (("T0", t0), ("T1", t1), ("Student", student)):
        if model is not None:
            total = sum(p.numel() for p in model.parameters())
            ans[f"{name}_total"] = total
            ans[f"{name}_extra_vs_SO"] = total - ans["SO_total"]
    return ans


__all__ = [
    "CHANNELS", "LEVELS", "POINTS", "NormalDualTeacher", "DASearchTeacher",
    "DAStudent", "TargetOnlyBaseline", "DeformableSearch", "DeformablePolicy",
    "init_t1_from_t0",
    "init_student_from_teacher", "load_so_backbone", "normalized_feature_l1",
    "parameter_counts", "pyramid", "pyramid_from_encoded",
]
