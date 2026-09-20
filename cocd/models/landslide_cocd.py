"""Minimal shared-backbone models for rapid single-orbit landslide mapping."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtTinyFPN(nn.Module):
    """Five-channel (pre VV/VH, post VV/VH, orbit) ConvNeXt-Tiny with P3 FPN."""
    def __init__(self, in_channels=5):
        super().__init__()
        base=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        old=base.features[0][0]
        stem=nn.Conv2d(in_channels,96,4,4)
        with torch.no_grad():
            # Four radiometric channels inherit the mean ImageNet response;
            # orbit is metadata, so its initial contribution is neutral.
            stem.weight.zero_()
            stem.weight[:,:4].copy_(old.weight.mean(1,keepdim=True).repeat(1,4,1,1)*(3/4))
            stem.bias.copy_(old.bias)
        base.features[0][0]=stem; self.features=base.features
        self.l2=nn.Conv2d(96,128,1);self.l3=nn.Conv2d(192,128,1)
        self.l4=nn.Conv2d(384,128,1);self.l5=nn.Conv2d(768,128,1)
        self.head=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
    def encode(self,x):
        x=self.features[0](x);c2=self.features[1](x)
        x=self.features[2](c2);c3=self.features[3](x)
        x=self.features[4](c3);c4=self.features[5](x)
        x=self.features[6](c4);c5=self.features[7](x)
        return c2,c3,c4,c5
    def decode(self,c2,c3,c4,c5):
        p5=self.l5(c5);p4=self.l4(c4)+F.interpolate(p5,size=c4.shape[-2:],mode='nearest')
        p3=self.l3(c3)+F.interpolate(p4,size=c3.shape[-2:],mode='nearest')
        p2=self.l2(c2)+F.interpolate(p3,size=c2.shape[-2:],mode='nearest')
        return F.interpolate(self.head(p2),size=(128,128),mode='bilinear',align_corners=False)

class TeacherCorrection(nn.Module):
    def __init__(self,channels=192):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(channels*2,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,1))
    def forward(self,target,counter): return self.net(torch.cat((target,counter),1))

class StudentCorrection(nn.Module):
    def __init__(self,channels=192):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,1))
    def forward(self,x): return self.net(x)

class DualOrbitTeacher(nn.Module):
    def __init__(self): super().__init__();self.backbone=ConvNeXtTinyFPN();self.correction=TeacherCorrection()
    def forward(self,target,counter):
        ft=self.backbone.encode(target);fc=self.backbone.encode(counter);r=self.correction(ft[1],fc[1])
        return self.backbone.decode(*ft),self.backbone.decode(ft[0],ft[1]+r,ft[2],ft[3]),r

class SingleOrbitStudent(nn.Module):
    def __init__(self,correction=False):
        super().__init__();self.backbone=ConvNeXtTinyFPN();self.correct=correction;self.correction=StudentCorrection() if correction else None
    def forward(self,x):
        f=self.backbone.encode(x);base=self.backbone.decode(*f)
        r=self.correction(f[1]) if self.correct else f[1].new_zeros(f[1].shape)
        return base,self.backbone.decode(f[0],f[1]+r,f[2],f[3]),r
