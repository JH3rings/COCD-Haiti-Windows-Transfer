from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, nn.LayerNorm) or type(module).__name__ == "LayerNorm2d"


def _tie_encoder_except_norms(a: nn.Module, b: nn.Module) -> None:
    """Share ConvNeXt encoder parameters while retaining orbit-specific LN affine terms."""
    norm_parameter_names: set[str] = set()
    for name, module in a.features.named_modules():
        if _is_norm_module(module):
            for parameter_name, _ in module.named_parameters(recurse=False):
                norm_parameter_names.add(f"{name}.{parameter_name}" if name else parameter_name)
    modules_b = dict(b.features.named_modules())
    for name, parameter in a.features.named_parameters():
        if name in norm_parameter_names:
            continue
        parent_name, _, leaf = name.rpartition(".")
        modules_b[parent_name]._parameters[leaf] = parameter


def load_backbone_state(model: ConvNeXtTinyFPN, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items()
                 if key.startswith("backbone.")}
    # The locked SO initialization has a one-logit FPN classifier. Preserve its
    # semantic direction when lifting it to the explicit background/landslide head.
    for key in ("head.2.weight", "head.2.bias"):
        if key in state and key in model.state_dict() and state[key].shape != model.state_dict()[key].shape:
            value = state[key]
            if value.shape[0] == 1 and model.state_dict()[key].shape[0] == 2:
                state[key] = torch.cat((-value, value), dim=0)
    model.load_state_dict(state, strict=True)


def binary_head_from_single(backbone: ConvNeXtTinyFPN) -> None:
    old = backbone.head[-1]
    if old.out_channels == 2:
        return
    new = nn.Conv2d(old.in_channels, 2, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[0].copy_(-old.weight[0])
        new.weight[1].copy_(old.weight[0])
        if old.bias is not None:
            new.bias[0].copy_(-old.bias[0])
            new.bias[1].copy_(old.bias[0])
    backbone.head[-1] = new


def _last_norm_weight(stage: nn.Module) -> torch.Tensor:
    norms = [module for module in stage.modules() if _is_norm_module(module)]
    if not norms:
        raise RuntimeError(f"no affine LayerNorm in stage {type(stage).__name__}")
    weight = norms[-1].weight
    if weight is None:
        raise RuntimeError("OGP channel exchange requires an affine normalization scale")
    return weight


class OGPNetPort(nn.Module):
    """DMC/DUR/GSU port; the deployed path is one orbit branch only."""
    def __init__(self, gamma_threshold: float = 1e-3):
        super().__init__()
        self.branch_asc = ConvNeXtTinyFPN()
        self.branch_desc = copy.deepcopy(self.branch_asc)
        _tie_encoder_except_norms(self.branch_asc, self.branch_desc)
        binary_head_from_single(self.branch_asc)
        binary_head_from_single(self.branch_desc)
        self.gamma_threshold = float(gamma_threshold)

        # One shared auxiliary head follows the paper's fourth-encoder-stage supervision.
        self.aux_head = nn.Conv2d(768, 2, 1)
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, 64, 1), nn.ReLU(inplace=True))
            for cin in (96, 192, 384, 768)
        ])
        self.pixel_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, 64))

        # Gated Spectral Unit operating on the two decoder feature maps.
        self.h_asc = nn.Conv2d(64, 64, 3, padding=1)
        self.h_desc = nn.Conv2d(64, 64, 3, padding=1)
        self.h_sum = nn.Conv2d(64, 64, 3, padding=1)
        self.z_asc = nn.Conv2d(192, 64, 1)
        self.z_desc = nn.Conv2d(192, 64, 1)
        self.z_sum = nn.Conv2d(192, 64, 1)
        self.fused_head = nn.Conv2d(64, 2, 1)
        self.channel_exchange_counts: dict[str, int] = {}

    def _paired_encode(self, asc: torch.Tensor, desc: torch.Tensor):
        a, d = self.branch_asc.features[0](asc), self.branch_desc.features[0](desc)
        asc_features, desc_features = [], []
        last_stage_indices = {1: 1, 3: 3, 5: 5, 7: 7}
        self.channel_exchange_counts = {}
        for index in range(1, 8):
            a = self.branch_asc.features[index](a)
            d = self.branch_desc.features[index](d)
            if index in last_stage_indices:
                ga = _last_norm_weight(self.branch_asc.features[index]).detach().abs()
                gd = _last_norm_weight(self.branch_desc.features[index]).detach().abs()
                ma = (ga < self.gamma_threshold).view(1, -1, 1, 1)
                md = (gd < self.gamma_threshold).view(1, -1, 1, 1)
                a_old, d_old = a, d
                a = torch.where(ma, d_old, a_old)
                d = torch.where(md, a_old, d_old)
                self.channel_exchange_counts[str(index)] = int((ma | md).sum().item())
                # Spatial exchange is used at the last two encoder stages: odd columns swap.
                if index in (5, 7):
                    width = a.shape[-1]
                    mask = (torch.arange(width, device=a.device) % 2 == 1).view(1, 1, 1, width)
                    a_old, d_old = a, d
                    a = torch.where(mask, d_old, a_old)
                    d = torch.where(mask, a_old, d_old)
            if index in last_stage_indices:
                asc_features.append(a)
                desc_features.append(d)
        return tuple(asc_features), tuple(desc_features)

    @staticmethod
    def _decode(backbone: ConvNeXtTinyFPN, features):
        c2, c3, c4, c5 = features
        p5 = backbone.l5(c5)
        p4 = backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        decoder = backbone.head[:-1](p2)
        logits = backbone.head[-1](decoder)
        return F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False), decoder

    def _decode_one(self, backbone: ConvNeXtTinyFPN, x: torch.Tensor):
        return self._decode(backbone, backbone.encode(x))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor):
        fa, fd = self._paired_encode(asc, desc)
        za, da = self._decode(self.branch_asc, fa)
        zd, dd = self._decode(self.branch_desc, fd)
        joint = torch.cat((da, dd, da + dd), dim=1)
        h1 = torch.tanh(self.h_asc(da))
        h2 = torch.tanh(self.h_desc(dd))
        h3 = torch.tanh(self.h_sum(da + dd))
        z1, z2, z3 = torch.sigmoid(self.z_asc(joint)), torch.sigmoid(self.z_desc(joint)), torch.sigmoid(self.z_sum(joint))
        fused = z1 * h1 + z2 * h2 + z3 * h3
        zf = F.interpolate(self.fused_head(fused), size=(128, 128), mode="bilinear", align_corners=False)
        aux_a = self.aux_head(fa[3])
        aux_d = self.aux_head(fd[3])
        def project(index, feature):
            value = self.projectors[index](feature)
            b, c, h, w = value.shape
            value = self.pixel_mlp(value.flatten(2).transpose(1, 2))
            return value.transpose(1, 2).reshape(b, c, h, w)
        contrast_features = tuple((project(i, fa[i]), project(i, fd[i])) for i in range(4))
        return {"z_asc": za, "z_desc": zd, "z_fused": zf,
                "d_asc": da, "d_desc": dd, "features_asc": fa, "features_desc": fd,
                "aux_asc": aux_a, "aux_desc": aux_d,
                "contrast_features": contrast_features}

    def forward_single(self, x: torch.Tensor, orbit: str):
        branch = self.branch_asc if orbit == "asc" else self.branch_desc
        return self._decode_one(branch, x)[0]
