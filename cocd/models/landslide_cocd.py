"""Shared ConvNeXt-Tiny/FPN backbone for the final GRSL method."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTinyFPN(nn.Module):
    """Five-channel (pre VV/VH, post VV/VH, orbit) ConvNeXt-Tiny with FPN."""

    def __init__(self, in_channels=5):
        super().__init__()
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        old = base.features[0][0]
        stem = nn.Conv2d(in_channels, 96, 4, 4)
        with torch.no_grad():
            # Four radiometric channels inherit the mean ImageNet response;
            # orbit is metadata, so its initial contribution is neutral.
            stem.weight.zero_()
            stem.weight[:, :4].copy_(
                old.weight.mean(1, keepdim=True).repeat(1, 4, 1, 1) * (3 / 4)
            )
            stem.bias.copy_(old.bias)
        base.features[0][0] = stem
        self.features = base.features
        self.l2 = nn.Conv2d(96, 128, 1)
        self.l3 = nn.Conv2d(192, 128, 1)
        self.l4 = nn.Conv2d(384, 128, 1)
        self.l5 = nn.Conv2d(768, 128, 1)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1)
        )

    def encode(self, x):
        x = self.features[0](x)
        c2 = self.features[1](x)
        x = self.features[2](c2)
        c3 = self.features[3](x)
        x = self.features[4](c3)
        c4 = self.features[5](x)
        x = self.features[6](c4)
        c5 = self.features[7](x)
        return c2, c3, c4, c5

    def decode(self, c2, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return F.interpolate(
            self.head(p2), size=(128, 128), mode="bilinear", align_corners=False
        )


__all__ = ["ConvNeXtTinyFPN"]
