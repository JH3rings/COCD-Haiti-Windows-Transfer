from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN


CHANNELS = (96, 192, 384, 768)


def load_haiti_initialization(template: ConvNeXtTinyFPN, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items()
                 if key.startswith("backbone.")}
    template.load_state_dict(state, strict=True)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = copy.deepcopy(features)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.features[0](image)
        c2 = self.features[1](x)
        x = self.features[2](c2)
        c3 = self.features[3](x)
        x = self.features[4](c3)
        c4 = self.features[5](x)
        x = self.features[6](c4)
        c5 = self.features[7](x)
        return c2, c3, c4, c5


class FPNDecoder(nn.Module):
    """FPN/decoder copied from the project's ConvNeXtTinyFPN checkpoint."""
    def __init__(self, template: ConvNeXtTinyFPN):
        super().__init__()
        self.l2 = copy.deepcopy(template.l2)
        self.l3 = copy.deepcopy(template.l3)
        self.l4 = copy.deepcopy(template.l4)
        self.l5 = copy.deepcopy(template.l5)
        self.body = copy.deepcopy(template.head[:-1])
        old = template.head[-1]
        self.classifier = nn.Conv2d(old.in_channels, 2, kernel_size=1)
        with torch.no_grad():
            # softmax([-z/2,+z/2]) == sigmoid(z), preserving the seed head's
            # initial binary probabilities while exposing explicit two-class logits.
            self.classifier.weight[0].copy_(-0.5 * old.weight[0])
            self.classifier.weight[1].copy_(0.5 * old.weight[0])
            self.classifier.bias[0].copy_(-0.5 * old.bias[0])
            self.classifier.bias[1].copy_(0.5 * old.bias[0])

    def forward(self, features: tuple[torch.Tensor, ...], output_size=(128, 128)):
        c2, c3, c4, c5 = features
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        decoder_feature = self.body(p2)
        logits = self.classifier(decoder_feature)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits, decoder_feature


class SymmetricMFF(nn.Module):
    """A symmetric per-level fusion port: mean plus learned joint residual."""
    def __init__(self, channels: int):
        super().__init__()
        self.joint = nn.Conv2d(3 * channels, channels, kernel_size=1)
        nn.init.zeros_(self.joint.weight)
        nn.init.zeros_(self.joint.bias)

    def forward(self, asc: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        mean = 0.5 * (asc + desc)
        joint = torch.cat((mean, torch.abs(asc - desc), asc * desc), dim=1)
        return mean + self.joint(joint)


class SingleModalFeatureGeneration(nn.Module):
    """SFG port that maps a branch's own maps to pseudo-fusion maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.proj(feature)


class SingleOrbitBranch(nn.Module):
    def __init__(self, template: ConvNeXtTinyFPN):
        super().__init__()
        self.encoder = ConvNeXtEncoder(template.features)
        self.sfg = nn.ModuleList(SingleModalFeatureGeneration(c) for c in CHANNELS)
        self.decoder = FPNDecoder(template)

    def forward(self, image: torch.Tensor):
        encoded = self.encoder(image)
        pseudo = tuple(layer(feature) for layer, feature in zip(self.sfg, encoded))
        logits, decoder_feature = self.decoder(pseudo, image.shape[-2:])
        return {"logits": logits, "decoder": decoder_feature,
                "encoded": encoded, "pseudo_fusion": pseudo}


class PairedFusionTeacher(nn.Module):
    def __init__(self, template: ConvNeXtTinyFPN):
        super().__init__()
        self.encoder_asc = ConvNeXtEncoder(template.features)
        self.encoder_desc = ConvNeXtEncoder(template.features)
        self.mff = nn.ModuleList(SymmetricMFF(c) for c in CHANNELS)
        self.decoder = FPNDecoder(template)

    def forward(self, asc: torch.Tensor, desc: torch.Tensor):
        fa, fd = self.encoder_asc(asc), self.encoder_desc(desc)
        fused = tuple(layer(a, d) for layer, a, d in zip(self.mff, fa, fd))
        logits, decoder_feature = self.decoder(fused, asc.shape[-2:])
        return {"logits": logits, "decoder": decoder_feature,
                "encoded_asc": fa, "encoded_desc": fd, "fused": fused}


class MMOKDPort(nn.Module):
    """Online fusion Teacher and two target-only deployable Student branches."""
    def __init__(self, initialization: str | Path):
        super().__init__()
        template = ConvNeXtTinyFPN()
        load_haiti_initialization(template, Path(initialization))
        self.teacher = PairedFusionTeacher(template)
        self.student_asc = SingleOrbitBranch(template)
        self.student_desc = SingleOrbitBranch(template)
        self.initialization = str(initialization)

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        teacher = self.teacher(asc, desc)
        student_asc = self.student_asc(asc)
        student_desc = self.student_desc(desc)
        return {"teacher": teacher, "student_asc": student_asc,
                "student_desc": student_desc}

    def predict_asc(self, asc: torch.Tensor) -> torch.Tensor:
        return self.student_asc(asc)["logits"]

    def predict_desc(self, desc: torch.Tensor) -> torch.Tensor:
        return self.student_desc(desc)["logits"]


__all__ = ["MMOKDPort", "SingleOrbitBranch"]
