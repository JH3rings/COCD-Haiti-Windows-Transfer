"""Every arm of the 2026-11-10 comparison.

Source fidelity is declared per method in ``FIDELITY`` and copied into each
arm's config.json.  "author code" means the loss is a line-level port of the
authors' released code (commit/path given); "paper" means equations from the
paper text; "reconstruction" means the paper was not accessible and the
formulation follows the abstract/earlier local record -- such arms must not
be reported as exact reproductions.
"""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import common as C

sys.path.insert(0, str(C.EXP / "dlr_unet_study_20261104" / "pkdot_deps"))

FIDELITY = {
    "B0": "DLR U-Net (Orynbaikyzy et al. 2025, Fig. 7) on Table-1 input; shared recipe",
    "T_fusion": "privileged Teacher: same U-Net, input SAR+terrain+optical (18 ch)",
    "T_optical": "optical-only Teacher (12 ch) required by Nair & Hansch 2024",
    "T_MEF": "Multi-to-Uni teacher: two U-Net streams with MEF gates (RECONSTRUCTION, paper paywalled)",
    "KD": "Hinton et al. 2015 soft-target KD per pixel; binary logits as [0,z]; T=4, lambda=1 (local choice)",
    "AT": "Zagoruyko & Komodakis ICLR17, official at()/at_loss(), beta=1000, encoder stages 2-5",
    "WKD": "Zhang et al. CVPR22 wavelet KD: L1 on Haar high-frequency bands of the OUTPUT map "
           "(probability map replaces generated image), lambda=1 (local choice)",
    "DIS2": "Kieu et al. WACV26-W author code DLKD_ver4 (comparisons/dis2/source): l2_kd_loss on pooled "
            "features x4 + spatial penultimate + logit KL, T=2, weight 5 after epoch 5. Frozen teacher -> "
            "global-average pooling and identity-initialised 1x1 student projections; diversity term "
            "omitted (needs a modality-specific branch the U-Net student does not have)",
    "ACAM": "Lan & Tian ICCV25 eq.2-9 on the predicted segmentation maps (paper: segmentation distils "
            "segmentation maps), M = number of classes = 2, alpha = lambda = 1, f_align identity 1x1",
    "MFSAD": "Tong et al. Remote Sens. 18(17):3008: bk + Gaussian-FFT + JSCAD + output KL(T=10) with "
             "the paper weights 100:80:10:0.12 recorded in train_kd_baselines.py (paper page 403 today, "
             "weights not re-verified)",
    "M2U": "Multi-to-Uni (Inf. Fusion 138:104696) FAD + CAD from the MEF teacher; lambda 1/1; CAD uses "
           "raw cosine affinity MSE (RECONSTRUCTION, paper paywalled)",
    "Nair": "Nair & Hansch CVPRW24 Int+Ft-A: optical teacher weights, label-free cosine feature "
            "reconstruction pretext (3 enc + 3 dec layers, 100 epochs, local choice), then full "
            "supervised fine-tuning",
    "PKDOT": "Aslam et al. author code PKDOT_kfold.py@4f1b6d4: cosine similarity matrix, 30 lowest teacher "
             "similarities, min-shift, geomloss Sinkhorn p=2 blur=.1, lambda=0.6; samples = 4x4 region "
             "tokens of the stride-4 decoder map",
    "ProtoKD": "Wang et al. ICASSP23 author code ProtoKD@fa96162: prototype_loss (2-D port) weight 0.1 + "
               "softmax_kl_loss(T=10).mean() weight 10 on last decoder feature",
    "OursS": "Ours: optical-defined sub-type labels (3 landslide + 2 background types) as a training-only "
             "multi-class target; deployment collapses to the binary logit",
    "OursS_SAR": "control for OursS: identical procedure, sub-types defined from SAR change instead of optical",
    "OursR": "Ours: training-only heteroscedastic regression of optical change maps from the last "
             "decoder feature",
}


def two_class(z):
    return torch.cat((torch.zeros_like(z), z), 1)


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def down(valid, like):
    return F.adaptive_avg_pool2d(valid.float(), like.shape[-2:]).eq(1).float()


def identity_1x1(c):
    conv = nn.Conv2d(c, c, 1)
    nn.init.eye_(conv.weight[:, :, 0, 0])
    nn.init.zeros_(conv.bias)
    return conv


# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
class MEF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.go, self.gs, self.fuse = (nn.Conv2d(2 * c, c, 1) for _ in range(3))

    def forward(self, o, s):
        cat = torch.cat((o, s), 1)
        o2 = o + torch.sigmoid(self.go(cat)) * s
        s2 = s + torch.sigmoid(self.gs(cat)) * o
        return self.fuse(torch.cat((o2, s2), 1))


class MEFTeacher(nn.Module):
    """Two U-Net streams fused by MEF at every decoder level (16..128 px);
    main head on the finest fused level, auxiliary heads on every level."""

    taps = (512, 256, 128, 64)

    def __init__(self):
        super().__init__()
        self.opt, self.sar = C.UNet(12), C.UNet(6)
        self.mef = nn.ModuleList(MEF(c) for c in self.taps)
        self.aux = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in self.taps)
        self.head = nn.Conv2d(64, 1, 1)
        self.fused = []

    def forward(self, optical, x):
        zo, zs = self.opt(optical), self.sar(x)
        self.fused = [m(o, s) for m, o, s in zip(self.mef, self.opt.dec, self.sar.dec)]
        size = x.shape[-2:]
        aux = [F.interpolate(h(f), size=size, mode="bilinear", align_corners=False)
               for h, f in zip(self.aux, self.fused)]
        return {"main": self.head(self.fused[-1]), "opt": zo, "sar": zs, "aux": aux}


def fusion_input(b):
    return torch.cat((b["x"], b["optical"]), 1)


# --------------------------------------------------------------------------- #
# Loss terms
# --------------------------------------------------------------------------- #
def hinton_kd(zs, zt, valid, T):
    kl = F.kl_div(F.log_softmax(two_class(zs) / T, 1), F.softmax(two_class(zt) / T, 1),
                  reduction="none").sum(1, keepdim=True)
    return T * T * masked_mean(kl, valid)


def at(x):
    """Official AT: F.normalize(x.pow(2).mean(1).view(x.size(0), -1))."""
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def haar(x):
    a, b, c, d = x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


def wavelet_kd(ps, pt, valid):
    mask = F.avg_pool2d(valid.float(), 2).eq(1).float()
    hs, ht = haar(ps * valid), haar(pt * valid)
    return sum(masked_mean((a - b).abs(), mask) for a, b in zip(hs[1:], ht[1:]))


def l2_kd_loss(a, b, spatial=False):
    """DIS2 DLKD_ver4.l2_kd_loss, verbatim."""
    a_norm = F.normalize(a, p=2, dim=1, eps=1e-6)
    b_norm = F.normalize(b, p=2, dim=1, eps=1e-6)
    if spatial:
        return (a_norm - b_norm).pow(2).mean(dim=(0, 1, 2, 3))
    return F.mse_loss(a_norm, b_norm, reduction="mean")


def dis2_logit_kl(zs, zt, T=2):
    """DIS2: F.kl_div(log_softmax(s/T), softmax(t/T), reduction='mean') * T * T."""
    return F.kl_div(F.log_softmax(two_class(zs) / T, 1), F.softmax(two_class(zt.detach()) / T, 1),
                    reduction="mean") * T * T


class ACAM(nn.Module):
    """ACAM-KD eq. 2-9 on C-channel segmentation maps (C = M = 2)."""

    def __init__(self, c=2, masks=2, chunk=2048):
        super().__init__()
        self.masks, self.chunk = masks, chunk
        cq = max(c // 2, 1)
        self.q, self.k, self.v = nn.Conv2d(c, cq, 1), nn.Conv2d(c, cq, 1), nn.Conv2d(c, c, 1)
        self.align = identity_1x1(c)
        self.mc = nn.Linear(c, masks * c)
        self.ms = nn.Conv2d(c, masks, 1)

    @staticmethod
    def _block(q, k, v, scale):
        return torch.softmax(q @ k * scale, -1) @ v

    def forward(self, fs, ft, valid):
        b, c, h, w = fs.shape
        mask = valid.float()
        s = self.align(fs) * mask
        t = ft.detach() * mask
        q = self.q(t).flatten(2).transpose(1, 2)              # B,N,Cq   (teacher query)
        k = self.k(s).flatten(2)                               # B,Cq,N   (student key)
        v = self.v(s).flatten(2).transpose(1, 2)               # B,N,C    (student value)
        scale = q.shape[-1] ** -.5
        fused = torch.cat([checkpoint(self._block, q[:, i:i + self.chunk], k, v, scale, use_reentrant=False)
                           for i in range(0, q.shape[1], self.chunk)], 1)
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        mc = torch.sigmoid(self.mc(fused.mean((2, 3)))).reshape(b, self.masks, c)    # eq.5
        ms = torch.sigmoid(self.ms(fused))                                           # B,M,H,W
        err = ((t - s) * mask).unsqueeze(1)                                          # B,1,C,H,W
        pixels = mask.sum((1, 2, 3)).clamp_min(1)
        lc = ((err * mc[..., None, None]).square().sum((2, 3, 4)) /
              (mc.sum(2) * pixels[:, None]).clamp_min(1e-6)).mean()                 # eq.6
        lsp = ((err * ms[:, :, None]).square().sum((2, 3, 4)) /
               ((ms * mask).sum((2, 3)) * c).clamp_min(1e-6)).mean()                # eq.7
        flat = (ms * mask).flatten(2)
        div = 0.
        for i in range(self.masks):
            for j in range(self.masks):
                if i != j:
                    div = div + (flat[:, i] * flat[:, j]).sum(1) / (
                        flat[:, i].square().sum(1) + flat[:, j].square().sum(1)).clamp_min(1e-6)
        div = 2 * div.mean()                                                         # eq.8
        return lc, lsp, div


def mfsad_neck(student, teacher, valid):
    """Gaussian-windowed FFT amplitude + joint spatial-channel attention MSE
    (kept identical to dlr_unet_study_20261104/distillation_losses.py)."""
    h, w = student.shape[-2:]

    def amplitude(x):
        return torch.fft.fftshift(torch.fft.fft2(x.float() * valid, norm="ortho"), dim=(-2, -1)).abs()
    y = torch.arange(h, device=student.device) - h // 2
    x = torch.arange(w, device=student.device) - w // 2
    gaussian = torch.exp(-(y[:, None].square() + x[None, :].square()) / (2 * (min(h, w) / 4) ** 2))
    freq = ((amplitude(student) - amplitude(teacher)).square() * gaussian).sum()
    freq = freq / (student.shape[0] * student.shape[1] * gaussian.sum())
    spatial = teacher.abs().mean(1, keepdim=True)
    lo = spatial.masked_fill(valid == 0, float("inf")).amin((2, 3), keepdim=True)
    hi = spatial.masked_fill(valid == 0, -float("inf")).amax((2, 3), keepdim=True)
    spatial = torch.nan_to_num((spatial - lo) / (hi - lo).clamp_min(1e-6)) * valid
    channel = (teacher.abs() * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
    lo, hi = channel.amin(1, keepdim=True), channel.amax(1, keepdim=True)
    channel = (channel - lo) / (hi - lo).clamp_min(1e-6)
    joint = ((student - teacher).square() * spatial * channel * valid).sum()
    return freq, joint / (valid.sum().clamp_min(1) * student.shape[1])


def fad(fs, ft, valid):
    """Haar high-frequency energy distillation of low-level features."""
    mask = F.avg_pool2d(valid, 2).eq(1).float()

    def energy(band):
        e = (band.square().mean(1, keepdim=True) + 1e-6).sqrt() * mask
        return e / (e.sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)).clamp_min(1e-6)

    def norm(x):
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        x = (x - (x * valid).sum((2, 3), keepdim=True) / n) * valid
        return x / ((x.square().sum((2, 3), keepdim=True) / n + 1e-6).sqrt())
    hs, ht = haar(norm(fs.float())), haar(norm(ft.float()))
    return sum(masked_mean(F.smooth_l1_loss(energy(a), energy(b), reduction="none"), mask)
               for a, b in zip(hs[1:], ht[1:])) / 3


def cad(fs, ft):
    a = F.normalize(fs.float().flatten(2), dim=1)
    b = F.normalize(ft.float().flatten(2), dim=1)
    return (a.transpose(1, 2) @ a - b.transpose(1, 2) @ b).square().mean()


def nair_cosine(s_maps, t_maps, valid):
    out = []
    for s, t in zip(s_maps, t_maps):
        m = down(valid, s)
        use = m.flatten(1).any(1).float()
        sim = F.cosine_similarity((s * m).flatten(1), (t * m).flatten(1), dim=1)
        out.append(((1 - sim) * use).sum() / use.sum().clamp_min(1))
    return torch.stack(out).mean()


def pkdot_relations(fs, ft, valid, sinkhorn):
    """PKDOT_kfold.py: cos-sim matrices, 30 lowest teacher entries per row,
    min-shift, Sinkhorn(teacher, student)."""
    cover = F.adaptive_avg_pool2d(F.interpolate(valid.float(), size=fs.shape[-2:], mode="area"), 4)
    support = cover.flatten() >= .75

    def tokens(x):
        v = F.interpolate(valid.float(), size=x.shape[-2:], mode="area")
        pooled = F.adaptive_avg_pool2d(x.float() * v, 4) / cover.clamp_min(1e-6)
        return F.normalize(pooled.permute(0, 2, 3, 1).flatten(0, 2)[support], dim=1)
    s, t = tokens(fs), tokens(ft.detach())
    if len(s) < 3:
        return fs.sum() * 0
    s, t = s @ s.T, t @ t.T
    k = min(30, len(t) - 1)
    idx = t.topk(k, largest=False, sorted=True).indices
    return sinkhorn((t - t.amin()).gather(1, idx), (s - s.amin()).gather(1, idx))


def prototype_loss_2d(feature, feature_t, label, num_cls=2):
    """ProtoKD loss.prototype_loss with the 3-D sums replaced by 2-D sums."""
    eps = 1e-5
    label = label[:, 0]
    s, t = [], []
    for i in range(num_cls):
        mask = label == i
        if (torch.sum(mask, dim=(-2, -1)) > 0).all():
            proto_s = torch.sum(feature * mask[:, None], dim=(-2, -1)) / (torch.sum(mask[:, None], dim=(-2, -1)) + eps)
            proto_t = torch.sum(feature_t * mask[:, None], dim=(-2, -1)) / (torch.sum(mask[:, None], dim=(-2, -1)) + eps)
            s.append(F.cosine_similarity(feature, proto_s[:, :, None, None], dim=1, eps=eps).unsqueeze(1))
            t.append(F.cosine_similarity(feature_t, proto_t[:, :, None, None], dim=1, eps=eps).unsqueeze(1))
    if not s:
        return feature.sum() * 0
    return torch.mean((torch.cat(s, 1) - torch.cat(t, 1)) ** 2)


def softmax_kl_loss(input_logits, target_logits):
    """ProtoKD loss.softmax_kl_loss, verbatim (reduction='none')."""
    return F.kl_div(F.log_softmax(input_logits, dim=1), F.softmax(target_logits, dim=1), reduction="none")


# --------------------------------------------------------------------------- #
# Ours: optical sub-types and optical-change regression
# --------------------------------------------------------------------------- #
def optical_change(b):
    """[dNDVI, dB04, dB08, dB11] in train-normalised units (post - pre)."""
    o = b["optical"]
    return torch.cat((b["ndvi"][:, 1:2] - b["ndvi"][:, 0:1],
                      o[:, 8:9] - o[:, 2:3], o[:, 9:10] - o[:, 3:4], o[:, 10:11] - o[:, 4:5]), 1)


def sar_change(b):
    """[dVV, dVH, POST VH] in train-normalised dB units."""
    x = b["x"]
    return torch.cat((x[:, 2:3] - x[:, 0:1], x[:, 3:4] - x[:, 1:2], x[:, 3:4]), 1)


def fit_subtypes(ds, kind, k=3, seed=C.SEED, samples=200000):
    """K-means (k landslide types) on training GT-positive pixels and a
    background split at the 90th percentile of change magnitude.  Train only."""
    feats_fg, mags_bg = [], []
    for item in ds.cache:
        b = {key: v[None] for key, v in item.items() if isinstance(v, torch.Tensor)}
        f = (optical_change(b) if kind == "optical" else sar_change(b))[0]
        ok = b["valid"][0, 0] & (b["optical_ok"][0, 0] if kind == "optical" else True)
        y = b["mask"][0, 0] > .5
        feats_fg.append(f[:, ok & y].T.numpy())
        mags_bg.append(f[:, ok & ~y].norm(dim=0).numpy())
    fg = np.concatenate(feats_fg)
    rng = np.random.default_rng(seed)
    fg = fg[rng.choice(len(fg), min(samples, len(fg)), replace=False)]
    mu, sd = fg.mean(0), fg.std(0) + 1e-6
    z = (fg - mu) / sd
    centres = z[rng.choice(len(z), k, replace=False)]
    for _ in range(100):
        assign = ((z[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([z[assign == j].mean(0) if (assign == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    threshold = float(np.percentile(np.concatenate(mags_bg), 90))
    counts = np.bincount(assign, minlength=k).tolist()
    return {"kind": kind, "k": k, "mu": mu.tolist(), "sd": sd.tolist(), "centres": centres.tolist(),
            "background_threshold": threshold, "fg_cluster_counts": counts}


def subtype_labels(b, spec):
    """0 = stable background, 1 = changed background, 2.. = landslide types,
    -100 = ignore (invalid SAR, or unusable optical for the optical kind)."""
    f = optical_change(b) if spec["kind"] == "optical" else sar_change(b)
    mu = torch.tensor(spec["mu"], device=f.device)[None, :, None, None]
    sd = torch.tensor(spec["sd"], device=f.device)[None, :, None, None]
    cen = torch.tensor(spec["centres"], device=f.device, dtype=f.dtype)
    z = (f - mu) / sd
    d = ((z[:, None] - cen[None, :, :, None, None]) ** 2).sum(2)
    fg = d.argmin(1) + 2
    bg = (f.norm(dim=1) > spec["background_threshold"]).long()
    y = b["mask"][:, 0] > .5
    lab = torch.where(y, fg, bg)
    ok = b["valid"][:, 0]
    if spec["kind"] == "optical":
        ok = ok & b["optical_ok"][:, 0]
    return torch.where(ok, lab, torch.full_like(lab, -100))


def binary_from_subtypes(z):
    return torch.logsumexp(z[:, 2:], 1, keepdim=True) - torch.logsumexp(z[:, :2], 1, keepdim=True)
