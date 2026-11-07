"""Single-scale residual SAR sampling and its local bilinear frequency response."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from sar_evidence_control import EvidenceUNet


class SearchEvidenceUNet(EvidenceUNet):
    def __init__(self, backbone, optical_teacher=False):
        super().__init__(backbone, optical_teacher)
        self.controller[-1] = nn.Conv2d(32, 11, 1)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.register_buffer("anchors", torch.tensor([[0., 0.], [.75, 0.], [0., .75], [-.75, -.75]]))

    def intervene(self, evidence, context):
        b, _, h, w = evidence.shape
        raw = self.controller(torch.cat((evidence, context), 1)).permute(0, 2, 3, 1)
        strength = .5 * raw[..., :1].tanh()
        weights = raw[..., 1:5].softmax(-1)
        learned = .75 * raw[..., 5:].reshape(b, h, w, 3, 2).tanh()
        offsets = torch.cat((torch.zeros_like(learned[..., :1, :]), learned), dim=-2) + self.anchors
        y, x = torch.meshgrid(torch.arange(h, device=raw.device, dtype=raw.dtype),
                              torch.arange(w, device=raw.device, dtype=raw.dtype), indexing="ij")
        grid = torch.stack((x, y), -1)[None, :, :, None, :] + offsets
        grid = (grid + .5) * grid.new_tensor([2 / w, 2 / h]) - 1
        # Keep grid_sample's channel output small by accumulating one tap at a time.
        retrieved = sum(F.grid_sample(evidence, grid[..., k, :], align_corners=False)
                        * weights[..., k].unsqueeze(1) for k in range(4))
        delta = strength.permute(0, 3, 1, 2) * (retrieved - evidence)
        return evidence + delta, {"offsets": offsets, "weights": weights, "strength": strength,
                                  "evidence": evidence, "delta": delta}


class JointEvidenceUNet(SearchEvidenceUNet):
    """Search SAR evidence first, then modulate its channels before decoding."""

    def __init__(self, backbone, optical_teacher=False):
        super().__init__(backbone, optical_teacher)
        self.modulator = nn.Sequential(
            nn.Conv2d(256 + 16, 32, 1), nn.GELU(), nn.Conv2d(32, 256, 1))
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def intervene(self, evidence, context):
        searched, control = super().intervene(evidence, context)
        gain = .5 * self.modulator(torch.cat((searched, context), 1)).tanh()
        return searched * (1 + gain), {**control, "searched": searched, "gain": gain}


def amplitude(result, radial_frequencies=(.1, .2, .35)):
    """Exact interior response of bilinear taps plus the residual centre."""
    offsets = result["offsets"].float()
    angles = torch.arange(4, device=offsets.device) * (math.pi / 4)
    directions = torch.stack((angles.cos(), angles.sin()), -1)
    radial = offsets.new_tensor(radial_frequencies)
    omega = (2 * math.pi * radial[:, None, None] * directions[None]).reshape(-1, 2)
    low = offsets.floor()
    frac = offsets - low
    # Each fractional tap is separable interpolation between adjacent samples.
    phase = -low[..., None, :] * omega
    z = torch.polar(torch.ones_like(phase), phase)
    step = torch.polar(torch.ones_like(omega), -omega)
    response = (z * (1 + frac[..., None, :] * (step - 1))).prod(dim=-1)
    kernel = (response * result["weights"][..., None]).sum(dim=-2)
    return (1 + result["strength"] * (kernel - 1)).abs()


def covariance(result):
    offsets, weights, strength = (result[k].float() for k in ("offsets", "weights", "strength"))
    mean = strength * (offsets * weights[..., None]).sum(dim=-2)
    second = (offsets.unsqueeze(-1) * offsets.unsqueeze(-2))
    frac = offsets - offsets.floor()
    second = second + torch.diag_embed(frac * (1 - frac))
    second = strength.unsqueeze(-1) * (second * weights[..., None, None]).sum(dim=-3)
    return (second - mean.unsqueeze(-1) * mean.unsqueeze(-2)).flatten(-2)
