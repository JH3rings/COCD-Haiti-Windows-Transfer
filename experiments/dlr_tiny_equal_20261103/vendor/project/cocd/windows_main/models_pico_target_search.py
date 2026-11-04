"""ConvNeXt V2-Pico replacement; preserve formal target-only value/KD paths."""
from pathlib import Path
import timm
import torch
from torch import nn
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2,AnchoredDeformablePolicy
from windows_main.models_da_search import SearchContext,DeformableSearch,LEVELS,POINTS

MODEL_NAME='convnextv2_pico.fcmae_ft_in1k'

class PicoFPN(nn.Module):
    def __init__(self,checkpoint=None):
        super().__init__()
        base=timm.create_model(MODEL_NAME,pretrained=checkpoint is not None,num_classes=0,
                              pretrained_cfg_overlay={'file':str(checkpoint)} if checkpoint is not None else None)
        old=base.stem[0];stem=nn.Conv2d(5,64,4,4)
        with torch.no_grad():
            stem.weight.zero_();stem.weight[:,:4].copy_(old.weight.mean(1,keepdim=True).repeat(1,4,1,1)*(3/4));stem.bias.copy_(old.bias)
        base.stem[0]=stem
        self.stem=base.stem;self.stages=base.stages
        self.l2=nn.Conv2d(64,128,1);self.l3=nn.Conv2d(128,128,1)
        self.l4=nn.Conv2d(256,128,1);self.l5=nn.Conv2d(512,128,1)
        self.head=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))

    def encode(self,x):
        x=self.stem(x);features=[]
        for stage in self.stages:x=stage(x);features.append(x)
        return tuple(features)

def build(model,dual,checkpoint):
    nn.Module.__init__(model)
    model.backbone=PicoFPN(checkpoint)
    model.context=SearchContext(128,dual=dual);model.policy=AnchoredDeformablePolicy(128)
    model.value_proj=nn.ModuleList([nn.Conv2d(128,128,1) for _ in range(LEVELS)])
    model.search=DeformableSearch(points=POINTS,offset_bound=2.)
    model.out_proj=nn.Conv2d(128,128,1)

class PicoStudent(StudentTargetSearchV2):
    def __init__(self,checkpoint=None):build(self,False,checkpoint)

class PicoTeacher(TargetSearchTeacherV2):
    def __init__(self,checkpoint=None):build(self,True,checkpoint)
