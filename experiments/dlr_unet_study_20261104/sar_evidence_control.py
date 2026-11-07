"""Prototype: privileged optical conditioning of a SAR-only U-Net value path."""
import torch
from torch import nn
from torch.nn import functional as F

from unet import PaperUNet


class EvidenceUNet(nn.Module):
    """Gate one stride-4 decoder representation and preserve all U-Net skips.

    Auxiliary input drives bounded gains only. Both modes use the same parameter
    shapes. The SAR control pads its four inputs to the auxiliary encoder width.
    """

    def __init__(self, backbone, optical_teacher=False):
        super().__init__()
        self.backbone = backbone
        self.optical_teacher = optical_teacher
        self.context = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.controller = nn.Sequential(
            nn.Conv2d(256 + 16, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 256, 1),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, sar, optical=None):
        if self.optical_teacher:
            if optical is None or optical.shape[1] != 12:
                raise ValueError("Teacher requires six optical bands at each of two dates")
            conditioning = optical
        else:
            if optical is not None:
                raise ValueError("SAR-only mode does not accept optical input")
            conditioning = torch.cat((sar, torch.zeros_like(sar), torch.zeros_like(sar)), dim=1)
        context = self.context(conditioning)
        features = []
        x = sar
        for i, block in enumerate(self.backbone.enc):
            x = block(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        for i, (up, block, skip) in enumerate(zip(self.backbone.up, self.backbone.dec, features[-2::-1])):
            x = block(torch.cat((skip, up(x)), 1))
            if i == 1:
                x, control = self.intervene(x, context)
        return {"z": self.backbone.head(x), **control}

    def intervene(self, evidence, context):
        gain = .5 * self.controller(torch.cat((evidence, context), 1)).tanh()
        delta = evidence * gain
        return evidence + delta, {"gain": gain, "evidence": evidence, "delta": delta}


def smoke():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    backbone = PaperUNet(4).eval()
    teacher = EvidenceUNet(backbone, optical_teacher=True).eval()
    student = EvidenceUNet(backbone, optical_teacher=False).eval()
    sar = torch.randn(2, 4, 32, 32)
    optical = torch.randn(2, 12, 32, 32, requires_grad=True)
    with torch.no_grad():
        ref = backbone(sar)
        torch.testing.assert_close(teacher(sar, optical)["z"], ref, atol=0, rtol=0)
        torch.testing.assert_close(student(sar)["z"], ref, atol=0, rtol=0)
        teacher.controller[-1].weight.normal_(std=.01)
        a = teacher(sar, optical)
        b = teacher(sar, optical.flip(0))
        torch.testing.assert_close(a["evidence"], b["evidence"], atol=0, rtol=0)
        assert (a["gain"] - b["gain"]).abs().max() > 0
        assert (a["z"] - b["z"]).abs().max() > 0
    teacher(sar, optical)["z"].square().mean().backward()
    assert optical.grad is not None and optical.grad.abs().sum() > 0
    assert sum(p.numel() for p in teacher.parameters()) == sum(p.numel() for p in student.parameters())
    print("PASS: exact identity initialization; optical changes gain only; task gradient reaches optical controller")
    print("controller + conditioning parameters:", sum(p.numel() for module in (teacher.context, teacher.controller)
                                                      for p in module.parameters()))


if __name__ == "__main__":
    smoke()
