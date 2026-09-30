from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN


def load_backbone_state(model: ConvNeXtTinyFPN, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items()
                 if key.startswith("backbone.")}
    model.load_state_dict(state, strict=True)


class ConvNeXtFeatureEncoder(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features

    def forward(self, x):
        x = self.features[0](x)
        c2 = self.features[1](x)
        x = self.features[2](c2)
        c3 = self.features[3](x)
        x = self.features[4](c3)
        c4 = self.features[5](x)
        x = self.features[6](c4)
        c5 = self.features[7](x)
        return c2, c3, c4, c5


class FPNDecoderOnly(nn.Module):
    """FPN decoder isomorphic to the locked single-orbit ConvNeXtTinyFPN decoder."""
    def __init__(self, template: ConvNeXtTinyFPN):
        super().__init__()
        self.l2 = copy.deepcopy(template.l2)
        self.l3 = copy.deepcopy(template.l3)
        self.l4 = copy.deepcopy(template.l4)
        self.l5 = copy.deepcopy(template.l5)
        self.head = copy.deepcopy(template.head)

    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        decoder = self.head[:-1](p2)
        logits = self.head[-1](decoder)
        logits = F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)
        return logits, decoder


class MEFBlock(nn.Module):
    """Bidirectional mutual feature enhancement with gated residual cross-view messages."""
    def __init__(self, channels: int):
        super().__init__()
        self.message_asc = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
                                         nn.Conv2d(channels, channels, 3, padding=1))
        self.message_desc = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
                                          nn.Conv2d(channels, channels, 3, padding=1))
        self.gate_asc = nn.Conv2d(2 * channels, channels, 1)
        self.gate_desc = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, asc, desc):
        joint_a = torch.cat((asc, desc), dim=1)
        joint_d = torch.cat((desc, asc), dim=1)
        enhanced_a = asc + torch.sigmoid(self.gate_asc(joint_a)) * self.message_asc(joint_a)
        enhanced_d = desc + torch.sigmoid(self.gate_desc(joint_d)) * self.message_desc(joint_d)
        return enhanced_a, enhanced_d, 0.5 * (enhanced_a + enhanced_d)


class Multi2UniPort(nn.Module):
    """MEF teacher + single-orbit student with low-level FAD and global CAD."""
    def __init__(self, initialization: str):
        super().__init__()
        template = ConvNeXtTinyFPN()
        load_backbone_state(template, initialization)
        self.encoder_asc = ConvNeXtFeatureEncoder(copy.deepcopy(template.features))
        self.encoder_desc = ConvNeXtFeatureEncoder(copy.deepcopy(template.features))
        self.teacher_decoder = FPNDecoderOnly(template)
        self.mef = nn.ModuleList([MEFBlock(c) for c in (96, 192, 384, 768)])
        self.student = ConvNeXtTinyFPN()
        load_backbone_state(self.student, initialization)
        self.initialization = initialization

    def forward_pair(self, asc, desc):
        fa = self.encoder_asc(asc)
        fd = self.encoder_desc(desc)
        enhanced_a, enhanced_d, fused = [], [], []
        for block, xa, xd in zip(self.mef, fa, fd):
            ea, ed, f = block(xa, xd)
            enhanced_a.append(ea); enhanced_d.append(ed); fused.append(f)
        teacher_logits, teacher_decoder_feature = self.teacher_decoder(tuple(fused))
        sa, sd = self.student.encode(asc), self.student.encode(desc)
        student_logits_a = self.student.decode(*sa)
        student_logits_d = self.student.decode(*sd)
        return {"teacher_logits": teacher_logits, "teacher_decoder_feature": teacher_decoder_feature,
                "teacher_asc": tuple(enhanced_a), "teacher_desc": tuple(enhanced_d),
                "teacher_fused": tuple(fused), "student_logits_asc": student_logits_a,
                "student_logits_desc": student_logits_d, "student_asc": sa, "student_desc": sd}

    def predict_single(self, x):
        return self.student.decode(*self.student.encode(x))

