"""Optical U-Net distillation on representations used by the segmentation head."""
import torch
from torch.nn import functional as F


def output_kd(student, teacher, valid, temperature=10.):
    student = torch.cat((torch.zeros_like(student), student), dim=1) / temperature
    teacher = torch.cat((torch.zeros_like(teacher), teacher.detach()), dim=1) / temperature
    divergence = F.kl_div(student.log_softmax(1), teacher.softmax(1), reduction="none").sum(1, keepdim=True)
    return temperature**2 * (divergence * valid).sum() / valid.sum().clamp_min(1)


def attention(student, teacher, valid):
    student, teacher = student.abs() * valid, teacher.abs() * valid
    def descriptors(x):
        return x.mean(1).flatten(1), x.mean((2, 3))
    use = valid.flatten(1).any(1)
    return sum(((1 - F.cosine_similarity(s, t)) * use).sum() / use.sum().clamp_min(1)
               for s, t in zip(descriptors(student), descriptors(teacher)))


def wavelet(student, teacher, valid):
    def bands(x):
        n = valid.sum((2, 3), keepdim=True).clamp_min(1)
        x = (x - (x * valid).sum((2, 3), keepdim=True) / n) * valid
        x = x / ((x.square().sum((2, 3), keepdim=True) / n + 1e-6).sqrt())
        a, b, c, d = x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]
        return ((a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2)
    s, t = bands(student.float()), bands(teacher.float())
    mask = F.avg_pool2d(valid, 2) == 1
    low = ((1 - F.cosine_similarity(s[0], t[0], dim=1)) * mask[:, 0]).sum() / mask.sum().clamp_min(1)
    def energy(x):
        e = (x.square().mean(1, keepdim=True) + 1e-6).sqrt() * mask
        return e / (e.sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)).clamp_min(1e-6)
    detail = sum((F.smooth_l1_loss(energy(a), energy(b), reduction="none") * mask).sum()
                 for a, b in zip(s[1:], t[1:])) / (3 * mask.sum().clamp_min(1))
    return low + detail


def mfsad_neck(student, teacher, valid):
    h, w = student.shape[-2:]
    def amplitude(x):
        return torch.fft.fftshift(torch.fft.fft2(x.float() * valid, norm="ortho"), dim=(-2, -1)).abs()
    y = torch.arange(h, device=student.device) - h//2
    x = torch.arange(w, device=student.device) - w//2
    gaussian = torch.exp(-(y[:, None].square()+x[None, :].square()) / (2*(min(h,w)/4)**2))
    freq = ((amplitude(student)-amplitude(teacher)).square()*gaussian).sum()
    freq = freq / (student.shape[0] * student.shape[1] * gaussian.sum())
    spatial = teacher.abs().mean(1, keepdim=True)
    lo = spatial.masked_fill(valid == 0, float("inf")).amin((2, 3), keepdim=True)
    hi = spatial.masked_fill(valid == 0, -float("inf")).amax((2, 3), keepdim=True)
    spatial = torch.nan_to_num((spatial - lo) / (hi-lo).clamp_min(1e-6)) * valid
    channel = (teacher.abs()*valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
    lo, hi = channel.amin(1, keepdim=True), channel.amax(1, keepdim=True)
    channel = (channel-lo) / (hi-lo).clamp_min(1e-6)
    joint = ((student-teacher).square()*spatial*channel*valid).sum()
    return freq, joint / (valid.sum().clamp_min(1)*student.shape[1])
