"""Deployment cost of every compared network for one 128x128 patch:
parameters and conv / transposed-conv / linear multiply-accumulates (MACs).
The prototype module's assignment and read-out einsums are added explicitly."""
import torch
from torch import nn

import common as C
import direct as D
import final_method as FM
import pr as P


def macs(model, x):
    total = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose2d):
            total[0] += i[0].numel() // i[0].shape[0] * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += i[0].numel() // i[0].shape[-1] * m.in_features * m.out_features // max(i[0].shape[0], 1) \
                if i[0].dim() > 2 else m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    for mod in model.modules():
        if isinstance(mod, P.EPRM) and mod.assign is not None:
            b, k, h, w = mod.assign.shape
            total[0] += 2 * h * w * k * mod.q.out_channels      # cos(q,P) and a@V(P) per pixel
    return total[0]


rows = {}
x6, x18 = torch.zeros(1, 6, 128, 128), torch.zeros(1, 18, 128, 128)
for name, m, x in (("U-Net (B0, KD, AT, DIS2, ACAM, M2U, PKDOT)", C.UNet(6), x6),
                   ("Ours, single-scale", FM.FinalNet((1,)), x6),
                   ("Ours, multi-scale", FM.FinalNet((1, 2)), x6),
                   ("Ours, scale-shared prototypes (F_SSP)", FM.FinalNet((1, 2, 3), True), x6),
                   ("UNet++ (ResNet-34)", D.make("UNETPP"), x6),
                   ("DeepLabv3 (ResNet-34)", D.make("DLV3"), x6),
                   ("Optical teacher (training only)", C.UNet(18), x18)):
    m.eval()
    rows[name] = {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "MACs_G": round(macs(m, x) / 1e9, 2)}
    print(name, rows[name])
C.write(C.HERE / "results_rel" / "deployment_cost.json", rows)
