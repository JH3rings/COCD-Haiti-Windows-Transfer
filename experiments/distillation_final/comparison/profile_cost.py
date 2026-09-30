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
sys.path.insert(0, str(ROOT / "cocd/third_party/landslide_baselines/dis2_official"))
sys.path.insert(0, str(ROOT / "experiments/distillation_final/dis2_port"))

from common import load_cg0_teacher, make_student  # noqa: E402
from dis2_port.train_dis2 import (  # noqa: E402
    DLKD_ver4, MISS_NDSM, MISS_RGB, official_loss, model_input,
)
from ours_s2_2stage.train_s2_2stage import kd_terms  # noqa: E402

OUT = Path(__file__).resolve().parent


class StudentInference(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x): return self.model(x)["z"]


class TeacherPair(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, asc, desc):
        a, d = self.model.forward_pair(asc, desc, counter_mode="counter")
        return a["z"], d["z"]


class SearchKDTrain(torch.nn.Module):
    def __init__(self, student, teacher):
        super().__init__(); self.student=student; self.teacher=teacher
    def forward(self, asc, desc):
        lp, le, oa, od = kd_terms(self.student, self.teacher, asc, desc)
        return lp, le, oa["z"], od["z"]


class DIS2Inference(torch.nn.Module):
    def __init__(self, model, asc): super().__init__(); self.model=model; self.asc=asc
    def forward(self, x):
        zeros=torch.zeros_like(x)
        if self.asc: return self.model(torch.cat((x,zeros),1),[MISS_NDSM],device="cpu")["seg_pred"]
        return self.model(torch.cat((zeros,x),1),[MISS_RGB],device="cpu")["seg_pred"]


class DIS2Train(torch.nn.Module):
    def __init__(self, model, miss): super().__init__(); self.model=model; self.miss=miss
    def forward(self, x, y):
        out=self.model(x,[self.miss],device="cpu")
        labels=torch.nn.functional.one_hot(y.long(),2).permute(0,3,1,2).float()
        total,_=official_loss(out,labels,epoch=6)
        return total


def analyze(model, inputs):
    logger=logging.getLogger("fvcore.nn.jit_analysis")
    old=logger.level; logger.setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flops=FlopCountAnalysis(model,inputs)
        total=float(flops.total())
        unsupported={str(k):int(v) for k,v in flops.unsupported_ops().items()}
    logger.setLevel(old)
    return {"supported_forward_flops":total,"supported_forward_gflops":total/1e9,
            "unsupported_ops":unsupported}


def main():
    torch.set_num_threads(4)
    sample=torch.zeros(1,5,128,128)
    student=make_student().cpu().eval()
    teacher=load_cg0_teacher().cpu().eval()
    s=analyze(StudentInference(student),sample)
    t=analyze(TeacherPair(teacher),(sample,sample))
    kd=analyze(SearchKDTrain(student,teacher),(sample,sample))
    d=DLKD_ver4(num_classes=2,image_size=(128,128),ndsm_encoder_channels=5,rgb_encoder_channels=5).cpu().eval()
    d.is_training=False
    dia=analyze(DIS2Inference(d,True),sample)
    did=analyze(DIS2Inference(d,False),sample)
    d.is_training=True
    y=torch.zeros(1,128,128,dtype=torch.long)
    dta=analyze(DIS2Train(d,torch.tensor([True,False])),(torch.zeros(1,10,128,128),y))
    dtd=analyze(DIS2Train(d,torch.tensor([False,True])),(torch.zeros(1,10,128,128),y))
    report={"input": "single 1x5x128x128 orbit; batch-one fvcore JIT trace",
            "accounting": "fvcore-supported operators only; unsupported elementwise, sampling, softmax/attention, pad ops omitted; no claim of exact total FLOPs",
            "Student_S0_Hinton_S2_inference":s,
            "CG0_Teacher_two_orbit_pair_forward":t,
            "S2_SearchKD_training_forward_pair":kd,
            "DIS2_ASC_target_inference":dia,"DIS2_DESC_target_inference":did,
            "DIS2_training_forward_one_direction_epoch6":dta,
            "DIS2_training_forward_reverse_direction_epoch6":dtd,
            "params":{"Student":sum(p.numel() for p in student.parameters()),
                      "Teacher":sum(p.numel() for p in teacher.parameters()),
                      "DIS2_full":sum(p.numel() for p in d.parameters()),
                      "DIS2_deployment_paths_both_orbits":sum(p.numel() for n in (
                          "rgb_encoder","rgb_else","miss_ndsm_fuse","miss_ndsm_classwise_attention",
                          "miss_ndsm_classwise_decoder","ndsm_encoder","ndsm_else","miss_rgb_fuse",
                          "miss_rgb_classwise_attention","miss_rgb_classwise_decoder")
                          for p in getattr(d,n).parameters())}}
    (OUT/"flop_profile.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
