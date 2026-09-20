"""MFEWF — Multi-level Features Effective Weighting and Fusion.

Chen et al. (2024), *Automatic detection of earthquake triggered landslides
using Sentinel-1 SAR imagery based on deep learning*, International Journal
of Digital Earth 17(1), DOI 10.1080/17538947.2024.2393261.

NO OFFICIAL CODE EXISTS. Verified 2026-09-19: the author profile named in the
paper (ChenLifu2022) exposes only an unrelated ``Aircraft-detection``
repository, and a GitHub repository search for "MFEWF" returns nothing. This
file is therefore an adaptation driven by the module descriptions in the
paper, and must be reported as ``adapted`` in any comparison table.

Modules, per the paper's ablation section:

    DRN          a residual backbone, argued to extract landslide features
                 better across levels (replaces the DeepLabV3+ trunk)
    AMM          Attention-based Multi-level weighting Module — selects useful
                 LOW-level features. Implemented as channel + spatial attention.
    CAASP        Contextual Atrous Attention and Semantic Pyramid — extracts
                 contextual and semantic information at HIGH level. Implemented
                 as parallel dilated branches (rates 1/2/4) plus a global-pool
                 branch.
    MFFRM        Multi-level Feature Fusion and Refinement Module — allocates
                 weights to low- and high-level features adaptively before
                 refinement.

Capacity: ResNet-34 trunk, kept close to the other seats (S0 is 28.08 M,
Boehm-adapted 26.09 M, CDNetE-adapted 24.44 M). ImageNet weights are used when
available, per the protocol's initialisation rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class AMM(nn.Module):
    """Channel + spatial attention over a low-level feature map."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, max(ch // reduction, 4), 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // reduction, 4), ch, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        ca = torch.sigmoid(avg + mx)
        x = x * ca
        sm = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        return x * torch.sigmoid(self.spatial(sm))


class CAASP(nn.Module):
    """Contextual atrous attention + semantic pooling on a high-level map."""

    def __init__(self, ch, out):
        super().__init__()
        self.b1 = nn.Conv2d(ch, out, 3, padding=1, dilation=1, bias=False)
        self.b2 = nn.Conv2d(ch, out, 3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(ch, out, 3, padding=4, dilation=4, bias=False)
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(ch, out, 1, bias=False))
        self.bn = nn.BatchNorm2d(out * 4)
        self.gate = nn.Sequential(nn.Conv2d(out * 4, out * 4, 1, bias=False), nn.Sigmoid())
        self.out = nn.Conv2d(out * 4, out, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x),
                       self.b4(x).expand(-1, -1, h, w)], dim=1)
        y = self.bn(y)
        return self.out(y * self.gate(y))


class MFFRM(nn.Module):
    """Adaptive weighting of low vs high level, then refinement."""

    def __init__(self, low_ch, high_ch, mid=192, out=96):
        super().__init__()
        # the two levels have different widths, so each is projected to a common
        # width before they can be mixed; the gate then weights the two levels
        self.low_ch, self.high_ch, self.mid = low_ch, high_ch, mid
        self.proj_low = nn.Conv2d(low_ch, mid, 1, bias=False)
        self.proj_high = nn.Conv2d(high_ch, mid, 1, bias=False)
        self.w = nn.Conv2d(2 * mid, 2 * mid, 1, bias=True)
        self.refine = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out), nn.GELU(),
        )

    def forward(self, low, high):
        high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        l = self.proj_low(low)
        h = self.proj_high(high)
        w = torch.sigmoid(self.w(torch.cat([l, h], dim=1)))
        wl, wh = w[:, :self.mid], w[:, self.mid:]
        return self.refine(wl * l + wh * h)


class MFEWF(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        trunk = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        # 5-channel input: the ImageNet stem is retiled, then the extra channels
        # are zero-initialised so the pretrained response is preserved exactly.
        old = self.stem[0]
        new = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :3] = old.weight
        self.stem[0] = new
        self.layer1, self.layer2 = trunk.layer1, trunk.layer2
        self.layer3, self.layer4 = trunk.layer3, trunk.layer4

        self.amm = AMM(64)
        self.caasp = CAASP(512, 256)
        self.mffrm = MFFRM(64, 256)
        self.head = nn.Conv2d(96, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)                      # 64 @ 1/4
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)                     # 512 @ 1/32
        low = self.amm(low)
        high = self.caasp(high)
        f = self.mffrm(low, high)
        return F.interpolate(self.head(f), size=(h, w), mode='bilinear', align_corners=False)
