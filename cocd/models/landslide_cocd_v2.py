"""Ours-V2: two-scale, geometry-gain selected task-response distillation.

The backbone is deliberately the existing five-channel ConvNeXt-Tiny/FPN so an
S0 checkpoint can initialize both the Teacher and Student without conversion.
Geometry is intentionally absent from every model forward method.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .landslide_cocd import ConvNeXtTinyFPN


class Correction(nn.Module):
    """Lightweight 1x1--3x3--1x1 correction, with a zero-output start."""

    def __init__(self, in_channels: int, channels: int = 128):
        super().__init__()
        width = max(1, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Driver(nn.Module):
    """Predicts a counter-feature surrogate from the target state alone.

    Two 3x3 convolutions through a narrow bottleneck, last layer zero-initialised,
    so a freshly built driver emits zero and the inherited operator is bypassed.
    """

    def __init__(self, channels: int = 128, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def omega(op: nn.Module, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Counter contribution of the correction operator, referenced to zero counter.

    ``op`` is a Teacher ``Correction`` treated as a reusable transform: it maps
    (target state, counter feature) to a state change.  Subtracting the
    zero-counter forward pass makes ``omega(op, a, 0) == 0`` hold exactly, so the
    returned tensor is the marginal contribution of the extra track rather than a
    fused feature.
    """
    return op(torch.cat((a, c), 1)) - op(torch.cat((a, torch.zeros_like(c)), 1))


class FPNStateDecoder(nn.Module):
    """Expose the existing FPN at 1/8 (P3) and 1/16 (P4) correction sites."""

    def __init__(self, backbone: ConvNeXtTinyFPN):
        super().__init__()
        self.backbone = backbone

    def pyramid4(self, features: tuple[torch.Tensor, ...]):
        """As ``pyramid`` but also returning P5, which only the distillation taps need."""
        c2, c3, c4, c5 = features
        p5 = self.backbone.l5(c5)
        p4 = self.backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return c2, p5, p4, p3

    def pyramid(self, features: tuple[torch.Tensor, ...]):
        c2, _, p4, p3 = self.pyramid4(features)
        return c2, p3, p4

    def decode(self, c2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        p2 = self.backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return F.interpolate(self.backbone.head(p2), size=(128, 128), mode="bilinear", align_corners=False)

    def taps(self, c2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
        """The three distillation taps that ``decode`` computes and then discards.

        Returns the pre-head P2 state, the penultimate map and the classifier
        logits, all at the 1/4 working resolution.  The arithmetic is the same
        expression ``decode`` evaluates, so a tap-based logits field upsamples to
        exactly the deployed prediction.
        """
        p2 = self.backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        pen = self.backbone.head[:-1](p2)
        return p2, pen, self.backbone.head[-1](pen)

    def states(self, features: tuple[torch.Tensor, ...], r3=None, r4=None):
        """Return self, P4-only, and P3+P4 logits with one shared decoder."""
        c2, p3, p4 = self.pyramid(features)
        z0 = self.decode(c2, p3, p4)
        r4 = torch.zeros_like(p4) if r4 is None else r4
        p4c = p4 + r4
        p3c = self.backbone.l3(features[1]) + F.interpolate(p4c, size=p3.shape[-2:], mode="nearest")
        z4 = self.decode(c2, p3c, p4c)
        r3 = torch.zeros_like(p3) if r3 is None else r3
        z34 = self.decode(c2, p3c + r3, p4c)
        return z0, z4, z34


class OursV2Teacher(nn.Module):
    """Shared target/counter encoder and counter-dependent P3/P4 corrections."""

    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.decoder = FPNStateDecoder(self.backbone)
        self.a3 = Correction(256)
        self.a4 = Correction(256)

    def load_s0(self, state: dict) -> None:
        """Load a legacy ``SingleOrbitStudent(correction=False)`` checkpoint."""
        state = state.get("model", state)
        source = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(source, strict=False)
        if unexpected or missing:
            raise RuntimeError(f"S0 checkpoint is incompatible: missing={missing}, unexpected={unexpected}")

    def from_features(self, ft: tuple[torch.Tensor, ...], fc: tuple[torch.Tensor, ...]):
        """Compute one target state from already-encoded target/counter features."""
        _, p3t, p4t = self.decoder.pyramid(ft)
        _, p3c, p4c = self.decoder.pyramid(fc)
        # Explicit zero-counter reference makes correction exactly zero when Fc=0.
        r3 = self.a3(torch.cat((p3t, p3c), 1)) - self.a3(torch.cat((p3t, torch.zeros_like(p3c)), 1))
        r4 = self.a4(torch.cat((p4t, p4c), 1)) - self.a4(torch.cat((p4t, torch.zeros_like(p4c)), 1))
        z0, z4, z34 = self.decoder.states(ft, r3, r4)
        return z0, z4, z34, (r3, r4)

    def forward(self, target: torch.Tensor, counter: torch.Tensor):
        return self.from_features(self.backbone.encode(target), self.backbone.encode(counter))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        """Both target directions with exactly two encoder calls, not four."""
        fa, fd = self.backbone.encode(asc), self.backbone.encode(desc)
        return self.from_features(fa, fd), self.from_features(fd, fa)

    def from_features_taps(self, ft: tuple[torch.Tensor, ...], fc: tuple[torch.Tensor, ...]) -> dict:
        """``from_features`` plus every intermediate a distillation term may score.

        ``levels`` runs coarse to fine and is the four-scale pyramid of the
        counter-corrected branch; ``r`` is the Omega output per scale, i.e. the
        marginal contribution the counter track actually made; ``z0`` is the
        uncorrected self path, kept because the selective weighting needs it.
        """
        c2, p5, p4t, p3t = self.decoder.pyramid4(ft)
        _, _, p4c, p3c = self.decoder.pyramid4(fc)
        r3 = self.a3(torch.cat((p3t, p3c), 1)) - self.a3(torch.cat((p3t, torch.zeros_like(p3c)), 1))
        r4 = self.a4(torch.cat((p4t, p4c), 1)) - self.a4(torch.cat((p4t, torch.zeros_like(p4c)), 1))
        p4d = p4t + r4
        p3u = self.backbone.l3(ft[1]) + F.interpolate(p4d, size=p3t.shape[-2:], mode="nearest")
        p3d = p3u + r3
        p2, pen, logits = self.decoder.taps(c2, p3d, p4d)
        z = F.interpolate(self.backbone.head(p2), size=(128, 128), mode="bilinear", align_corners=False)
        return {'levels': (p5, p4d, p3d, p2), 'pen': pen, 'logits': logits, 'r': (r3, r4),
                'z0': self.decoder.decode(c2, p3t, p4t), 'z4': self.decoder.decode(c2, p3u, p4d), 'z34': z}

    def forward_pair_taps(self, asc: torch.Tensor, desc: torch.Tensor):
        """``forward_pair`` with taps, still exactly two encoder calls per direction."""
        fa, fd = self.backbone.encode(asc), self.backbone.encode(desc)
        return self.from_features_taps(fa, fd), self.from_features_taps(fd, fa)


class OursV2Student(nn.Module):
    """Target-only deployment network. It never accepts counter or geometry."""

    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.decoder = FPNStateDecoder(self.backbone)
        self.q3 = Correction(128)
        self.q4 = Correction(128)

    def load_teacher_self(self, teacher: OursV2Teacher) -> None:
        self.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)

    def forward(self, target: torch.Tensor):
        ft = self.backbone.encode(target)
        _, p3, p4 = self.decoder.pyramid(ft)
        r3, r4 = self.q3(p3), self.q4(p4)
        _, z4, z34 = self.decoder.states(ft, r3, r4)
        return z4, z34, (r3, r4)


class OursV3Student(nn.Module):
    """Target-only deployment network for the structural ablation.

    ``R0``  no correction at all (identity); the KD-free floor.
    ``R1``  v2 behaviour: a free residual head per scale, driven by the
            uncorrected pyramid state.
    ``R2``  driver ``Q_k`` feeds the Teacher's inherited operator ``Omega_k``,
            still driven by the uncorrected P3 state.
    ``R3``  as ``R2``, but P3 is driven by the state that has already absorbed
            the P4 correction, so the two scales are no longer independent.

    Only R1 learns the residual transform itself.  R2/R3 inherit ``Omega`` from
    the Teacher and learn the driver that feeds it, which is the single variable
    this ablation isolates.
    """

    VARIANTS = ('R0', 'R1', 'R2', 'R3')

    def __init__(self, variant: str = 'R3'):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f'unknown variant {variant!r}, expected one of {self.VARIANTS}')
        self.variant = variant
        self.backbone = ConvNeXtTinyFPN()
        self.decoder = FPNStateDecoder(self.backbone)
        if variant == 'R1':
            self.q3 = Correction(128)
            self.q4 = Correction(128)
        elif variant in ('R2', 'R3'):
            self.op3 = Correction(256)
            self.op4 = Correction(256)
            self.d3 = Driver(128)
            self.d4 = Driver(128)

    def load_teacher_self(self, teacher: OursV2Teacher) -> None:
        """Copy the self path, and for R2/R3 the correction operator as well."""
        self.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)
        if self.variant in ('R2', 'R3'):
            self.op3.load_state_dict(teacher.a3.state_dict(), strict=True)
            self.op4.load_state_dict(teacher.a4.state_dict(), strict=True)

    def forward(self, target: torch.Tensor):
        ft = self.backbone.encode(target)
        c2, p3, p4 = self.decoder.pyramid(ft)
        if self.variant == 'R0':
            z = self.decoder.decode(c2, p3, p4)
            return z, z, (torch.zeros_like(p3), torch.zeros_like(p4))
        if self.variant == 'R1':
            r4 = self.q4(p4)
            p4c = p4 + r4
            a3 = self.backbone.l3(ft[1]) + F.interpolate(p4c, size=p3.shape[-2:], mode='nearest')
            r3 = self.q3(p3)
        else:
            r4 = omega(self.op4, p4, self.d4(p4))
            p4c = p4 + r4
            # a3 is the P3 state after the P4 correction has been propagated into it.
            a3 = self.backbone.l3(ft[1]) + F.interpolate(p4c, size=p3.shape[-2:], mode='nearest')
            drive3 = p3 if self.variant == 'R2' else a3
            r3 = omega(self.op3, drive3, self.d3(drive3))
        z4 = self.decoder.decode(c2, a3, p4c)
        z34 = self.decoder.decode(c2, a3 + r3, p4c)
        return z4, z34, (r3, r4)

    def forward_taps(self, target: torch.Tensor) -> dict:
        """``forward`` plus the distillation taps, from a single encoder pass.

        The returned ``z4`` / ``z34`` are the same tensors ``forward`` produces,
        so a distilled arm keeps an identical segmentation loss and an identical
        deployment path; the extra keys only feed the distillation terms and are
        never used after training.  ``z0`` is the uncorrected reading added for
        the decision-change term (``z34 - z0``); it is one decoder call on the
        uncorrected pyramid and nothing in the deployment path reads it.
        """
        ft = self.backbone.encode(target)
        c2, p5, p4, p3 = self.decoder.pyramid4(ft)
        z0 = self.decoder.decode(c2, p3, p4)
        if self.variant == 'R0':
            p2, pen, logits = self.decoder.taps(c2, p3, p4)
            return {'levels': (p5, p4, p3, p2), 'pen': pen, 'logits': logits,
                    'r': (torch.zeros_like(p3), torch.zeros_like(p4)),
                    'z0': z0, 'z4': z0, 'z34': z0}
        if self.variant == 'R1':
            r4 = self.q4(p4)
            p4c = p4 + r4
            a3 = self.backbone.l3(ft[1]) + F.interpolate(p4c, size=p3.shape[-2:], mode='nearest')
            r3 = self.q3(p3)
        else:
            r4 = omega(self.op4, p4, self.d4(p4))
            p4c = p4 + r4
            a3 = self.backbone.l3(ft[1]) + F.interpolate(p4c, size=p3.shape[-2:], mode='nearest')
            drive3 = p3 if self.variant == 'R2' else a3
            r3 = omega(self.op3, drive3, self.d3(drive3))
        p3c = a3 + r3
        p2, pen, logits = self.decoder.taps(c2, p3c, p4c)
        z4 = self.decoder.decode(c2, a3, p4c)
        z34 = self.decoder.decode(c2, p3c, p4c)
        return {'levels': (p5, p4c, p3c, p2), 'pen': pen, 'logits': logits,
                'r': (r3, r4), 'z0': z0, 'z4': z4, 'z34': z34}
