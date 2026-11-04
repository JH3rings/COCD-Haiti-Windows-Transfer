"""One SAR-only dual-temporal student graph plus optical-change teachers.

Every arm in this study uses the *same* student class, input tensor, initial
state and decoder. Distillation only adds training-time adapters and losses;
none of them exist at deployment.

SAR tensor layout (built by ``data.DLRInputs``):
    0 VV pre, 1 VH pre, 2 VV post, 3 VH post, 4 OrbitID
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cocd"))
from windows_main.pico_optimization_v2 import Student as Base  # noqa: E402

WIDTH = 128


# --------------------------------------------------------------------------- #
# Encoder / segmentation graph
# --------------------------------------------------------------------------- #
class PicoEncoder(nn.Module):
    """ConvNeXtV2-Pico stem+stages with an input-channel adaptation.

    The pretrained RGB stem is folded to a mean kernel; per-channel weights then
    follow the convention already used by the project's 5-channel model
    (``2 * sar_column`` on VV/VH, ``2/3 * sar_column`` per optical band), so the
    effective input-response magnitude stays comparable across modalities.
    """

    def __init__(self, initial, in_ch):
        super().__init__()
        base = Base()
        base.load_state_dict(initial, strict=True)
        self.stem = base.backbone.stem
        self.stages = base.backbone.stages
        old = self.stem[0]
        assert old.in_channels == 5
        new = nn.Conv2d(in_ch, 64, 4, 4)
        with torch.no_grad():
            col = old.weight[:, :1]
            if in_ch == 3:
                w = 2.0 * col.repeat(1, 3, 1, 1)
                w[:, 2].zero_()  # OrbitID is metadata, not a radiometric band
            elif in_ch == 6:
                w = (2.0 / 3.0) * col.repeat(1, 6, 1, 1)
            else:
                raise ValueError(in_ch)
            new.weight.copy_(w)
            new.bias.copy_(old.bias)
        self.stem[0] = new
        self.out_channels = (64, 128, 256, 512)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # strides 4, 8, 16, 32


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class DualTemporalNet(nn.Module):
    """Shared-weight two-pass encoder, per-level change head, FPN, decoder."""

    def __init__(self, initial, in_ch):
        super().__init__()
        self.backbone = PicoEncoder(initial, in_ch)
        ch = self.backbone.out_channels
        self.change = nn.ModuleList([_block(4 * c, WIDTH) for c in ch])
        self.lateral = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([_block(WIDTH, WIDTH) for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.Conv2d(WIDTH, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def _fpn(self, fp, fq):
        change = []
        for i, (a, b) in enumerate(zip(fp, fq)):
            d = b - a
            change.append(self.change[i](torch.cat([a, b, d, d.abs()], 1)))
        fused = [None] * 4
        fused[3] = self.smooth[3](self.lateral[3](change[3]))
        for i in (2, 1, 0):
            up = F.interpolate(fused[i + 1], size=change[i].shape[-2:], mode="nearest")
            fused[i] = self.smooth[i](self.lateral[i](change[i]) + up)
        return fused

    def encode_change(self, pre, post):
        fp = self.backbone(pre)
        fq = self.backbone(post)
        return self._fpn(fp, fq), fp, fq

    def forward(self, pre, post):
        fused, fp, fq = self.encode_change(pre, post)
        z = self.decoder(fused[0])
        z = F.interpolate(z, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return {"z": z, "change": fused, "feat_pre": fp, "feat_post": fq}


class OpticalTeacher(DualTemporalNet):
    """Privileged PRE/POST optical teacher (training only)."""

    def __init__(self, initial):
        super().__init__(initial, 6)

    def forward(self, optical_pre, optical_post):
        return super().forward(optical_pre, optical_post)


class MultimodalTeacher(DualTemporalNet):
    """SAR + optical teacher: fuses both modalities before the change head."""

    def __init__(self, initial):
        super().__init__(initial, 6)
        self.sar_backbone = PicoEncoder(initial, 3)
        ch = self.backbone.out_channels
        self.fuse = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.GroupNorm(8, c), nn.GELU()) for c in ch]
        )

    def encode_change(self, pre, post):
        fp_s, fq_s = self.sar_backbone(pre["sar"]), self.sar_backbone(post["sar"])
        fp_o, fq_o = self.backbone(pre["opt"]), self.backbone(post["opt"])
        fp = [self.fuse[i](torch.cat([fp_s[i], fp_o[i]], 1)) for i in range(4)]
        fq = [self.fuse[i](torch.cat([fq_s[i], fq_o[i]], 1)) for i in range(4)]
        return self._fpn(fp, fq), fp, fq

    def forward(self, optical_pre, optical_post, sar):
        pre = {"sar": torch.cat([sar[:, :2], sar[:, 4:5]], 1), "opt": optical_pre}
        post = {"sar": torch.cat([sar[:, 2:4], sar[:, 4:5]], 1), "opt": optical_post}
        fused, fp, fq = self.encode_change(pre, post)
        z = self.decoder(fused[0])
        z = F.interpolate(z, size=sar.shape[-2:], mode="bilinear", align_corners=False)
        return {"z": z, "change": fused, "feat_pre": fp, "feat_post": fq}


class SARStudent(DualTemporalNet):
    """Deployed model: one orbit's PRE/POST VV+VH (+OrbitID). No optical path."""

    def __init__(self, initial):
        super().__init__(initial, 3)

    def forward(self, sar):
        return super().forward(torch.cat([sar[:, :2], sar[:, 4:5]], 1),
                               torch.cat([sar[:, 2:4], sar[:, 4:5]], 1))


# --------------------------------------------------------------------------- #
# GOLD-style modules
# --------------------------------------------------------------------------- #
def difference_attention(x):
    """Parameter-free difference-aware attention map.

    GOLD's transferable quantity is where the temporal difference is
    discriminative. Using a parameter-free normalised change magnitude keeps the
    signal transferable from a *frozen* teacher and works for both modalities.
    """
    a = x.abs().mean(1, keepdim=True)
    scale = a.flatten(2).amax(2).clamp_min(1e-6)[:, :, None, None]
    return (a / scale).clamp(0, 1)


class BackboneAdapter(nn.Module):
    """Bk: shallow per-time projection into the teacher channel space."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return F.silu(self.conv(x))


class NeckAdapter(nn.Module):
    def __init__(self, cin=WIDTH, cout=WIDTH):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def seg_loss(z, mask, valid):
    m = valid.float()
    p = z.sigmoid()
    dims = (1, 2, 3)
    bce = (F.binary_cross_entropy_with_logits(z, mask, reduction="none") * m).sum() / m.sum().clamp_min(1)
    dice = 1 - (2 * (p * mask * m).sum(dims) + 1e-6) / ((p * m).sum(dims) + (mask * m).sum(dims) + 1e-6)
    return bce + 0.5 * dice.mean()


def logit_kd(zs, zt, valid, temperature=2.0):
    mask = valid.float()[:, 0]
    ls = F.log_softmax(torch.cat([torch.zeros_like(zs), zs], 1) / temperature, 1)
    pt = F.softmax(torch.cat([torch.zeros_like(zt), zt], 1) / temperature, 1)
    d = F.kl_div(ls, pt, reduction="none").sum(1) * temperature ** 2
    return (d * mask).sum() / mask.sum().clamp_min(1)


def masked_mse(a, b, valid):
    m = F.interpolate(valid.float(), size=a.shape[-2:], mode="nearest")
    return ((a - b) ** 2 * m).sum() / m.sum().clamp_min(1) / a.shape[1]


def cosine_feature_loss(a, b, valid):
    m = F.interpolate(valid.float(), size=a.shape[-2:], mode="nearest")
    cos = (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1, keepdim=True)
    return ((1 - cos) * m).sum() / m.sum().clamp_min(1)


def attention_transfer(a_s, a_t, valid):
    """Difference-aware attention transfer weighted by teacher change mass."""
    m = F.interpolate(valid.float(), size=a_s.shape[-2:], mode="nearest")
    w = a_t * m
    w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return ((a_s - a_t) ** 2 * w).sum() / (w * m).sum().clamp_min(1)


def gaussian_lowpass(h, w, sigma, device):
    """Radial Gaussian mask centred on DC over the rfft2 half-spectrum grid."""
    fy = torch.fft.fftfreq(h, device=device)      # cycles / pixel
    fx = torch.fft.rfftfreq(w, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def freq_loss(s, t, valid, sigma=0.25):
    """M-FSAD Freq: low-band amplitude-spectrum MSE, high band left free."""
    S = torch.fft.rfft2(s, norm="ortho")
    T = torch.fft.rfft2(t, norm="ortho")
    M = gaussian_lowpass(s.shape[-2], s.shape[-1], sigma, s.device)[None, None]
    diff = (S.abs() - T.abs()).square()
    # The paper's Eq.(5) normalises only by the mask; a spatial valid mask cannot
    # be applied in the half-spectrum domain, so the (1.3%) zero-filled pixels are
    # left as a global perturbation rather than per-pixel masked out.
    return (diff * M).sum() / M.expand_as(diff).sum().clamp_min(1) / s.shape[1]


def jscad_loss(s, t, valid):
    """M-FSAD JSCAD: joint spatial x channel teacher-saliency mask on MSE."""
    sp = t.abs().mean(1, keepdim=True)
    ch = t.abs().mean((2, 3), keepdim=True)
    sp = (sp - sp.amin()) / (sp.amax() - sp.amin()).clamp_min(1e-6)
    ch = (ch - ch.amin()) / (ch.amax() - ch.amin()).clamp_min(1e-6)
    M = sp * ch
    m = F.interpolate(valid.float(), size=s.shape[-2:], mode="nearest")
    M = M * m
    return ((s - t).square() * M).sum() / M.sum().clamp_min(1) / s.shape[1]


# --------------------------------------------------------------------------- #
# Arm container
# --------------------------------------------------------------------------- #
def make_teacher(kind, initial):
    return MultimodalTeacher(initial) if kind == "multimodal" else OpticalTeacher(initial)


class StudentSystem(nn.Module):
    """Student plus training-only KD modules; the student graph is identical."""

    def __init__(self, initial, arm):
        super().__init__()
        self.arm = arm
        self.student = SARStudent(initial)
        if arm == "gold":
            self.proj = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1, bias=False) for _ in range(3)])
        if arm.startswith("mfsad"):
            ch = self.student.backbone.out_channels
            self.bk = nn.ModuleList([BackboneAdapter(ch[i]) for i in (1, 2, 3)])
            self.neck = nn.ModuleList([NeckAdapter() for _ in range(3)])

    def forward(self, sar):
        return self.student(sar)

    def bk_loss(self, so, to, valid):
        loss = 0.0
        for k, idx in enumerate((1, 2, 3)):
            for key in ("feat_pre", "feat_post"):
                loss = loss + masked_mse(self.bk[k](so[key][idx]), to[key][idx], valid)
        return loss / 6.0
