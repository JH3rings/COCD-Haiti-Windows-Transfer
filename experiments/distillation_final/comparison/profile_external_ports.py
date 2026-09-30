from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import warnings
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cocd"))
sys.path.insert(0, str(ROOT / "experiments/distillation_final"))

from models.landslide_cocd import ConvNeXtTinyFPN  # noqa: E402
from ogp_net_port.ogp_model import OGPNetPort, load_backbone_state  # noqa: E402


class OGPDeploy(torch.nn.Module):
    def __init__(self, model: OGPNetPort):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_single(x, "asc")


class Multi2UniDeploy(torch.nn.Module):
    def __init__(self, model: ConvNeXtTinyFPN):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.decode(*self.model.encode(x))


def analyze(model, sample):
    logger = logging.getLogger("fvcore.nn.jit_analysis")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            trace = FlopCountAnalysis(model, sample)
            total = float(trace.total())
            unsupported = {str(k): int(v) for k, v in trace.unsupported_ops().items()}
    finally:
        logger.setLevel(old_level)
    return {"supported_forward_flops": total,
            "supported_forward_gflops": total / 1e9,
            "unsupported_ops": unsupported}


def main():
    torch.set_num_threads(4)
    sample = torch.zeros(1, 5, 128, 128)
    ogp = OGPNetPort().cpu().eval()
    load_backbone_state(ogp.branch_asc,
        str(ROOT / "experiments/windows_main/raw_cver_v1_seed42/S0-Raw_seed42_best.pt"))
    multi = ConvNeXtTinyFPN().cpu().eval()
    multi_path = ROOT / "experiments/distillation_final/multi2uni_port/Multi2Uni-Port_deploy_student_seed42_best.pt"
    multi.load_state_dict(torch.load(multi_path, map_location="cpu", weights_only=True), strict=True)
    result = {
        "input": "single orbit 1x5x128x128, batch-one fvcore trace",
        "accounting": "supported operators only; unsupported elementwise/GELU operations omitted",
        "OGP-Net-Port": {
            "status": "architecture-only; no formal checkpoint because 32-sample deployment sanity failed",
            "inference_parameters": sum(p.numel() for p in ogp.branch_asc.parameters()),
            "full_stage2_parameters": sum(p.numel() for p in ogp.parameters()),
            "single_orbit": analyze(OGPDeploy(ogp), sample),
        },
        "Multi2Uni-Port": {
            "status": "trained best-checkpoint student",
            "inference_parameters": sum(p.numel() for p in multi.parameters()),
            "checkpoint": str(multi_path),
            "single_orbit": analyze(Multi2UniDeploy(multi), sample),
        },
    }
    output = Path(__file__).resolve().parent / "flop_profile_external.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
