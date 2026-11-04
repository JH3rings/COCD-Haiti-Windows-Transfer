"""Matched SAR-only Student; change-then-fuse Teacher and A/B losses."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/sar_only_kd_literature_20261103'))
from model_kd import SARStudent,DualTemporalNet,PicoEncoder,_block,masked_mse,freq_loss,jscad_loss,logit_kd

class Teacher(DualTemporalNet):
 def __init__(self,initial):
  super().__init__(initial,6);self.sar_backbone=PicoEncoder(initial,3)
  self.sar_change=nn.ModuleList([_block(4*c,128) for c in self.backbone.out_channels])
  self.fuse=nn.ModuleList([_block(256,128) for _ in range(4)])
 def forward(self,pre,post,sar):
  sp=self.sar_backbone(torch.cat([sar[:,:2],sar[:,4:5]],1));sq=self.sar_backbone(torch.cat([sar[:,2:4],sar[:,4:5]],1))
  op=self.backbone(pre);oq=self.backbone(post);changes=[]
  for i,(a,b,c,d) in enumerate(zip(sp,sq,op,oq)):
   ds=self.sar_change[i](torch.cat([a,b,b-a,(b-a).abs()],1))
   do=self.change[i](torch.cat([c,d,d-c,(d-c).abs()],1))
   changes.append(self.fuse[i](torch.cat([do,ds],1)))
  fused=[None]*4;fused[3]=self.smooth[3](self.lateral[3](changes[3]))
  for i in (2,1,0):fused[i]=self.smooth[i](self.lateral[i](changes[i])+F.interpolate(fused[i+1],size=changes[i].shape[-2:],mode='nearest'))
  z=F.interpolate(self.decoder(fused[0]),size=sar.shape[-2:],mode='bilinear',align_corners=False)
  return {'z':z,'change':fused,'bk_change':changes}

class System(nn.Module):
 def __init__(self,initial,arm):
  super().__init__();self.student=SARStudent(initial);self.arm=arm
  if arm=='A':self.bk=nn.ModuleList([nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128),nn.SiLU()) for _ in range(4)])
  if arm=='B':self.proj=nn.ModuleList([nn.Conv2d(128,128,1,bias=False) for _ in range(4)])
 def forward(self,sar):
  o=self.student(sar)
  if self.arm=='A':o['bk_change']=[self.student.change[i](torch.cat([a,b,b-a,(b-a).abs()],1)) for i,(a,b) in enumerate(zip(o['feat_pre'],o['feat_post']))]
  return o

def band_distribution(x):
 # Full FFT counts +/- frequencies symmetrically, avoiding rFFT multiplicities.
 x=F.normalize(x,dim=1);e=torch.fft.fft2(x,norm='ortho').abs().square().sum(1)
 h,w=x.shape[-2:];fy=torch.fft.fftfreq(h,device=x.device);fx=torch.fft.fftfreq(w,device=x.device)
 r=(fy[:,None].square()+fx[None,:].square()).sqrt()/(.5*2**.5)
 masses=torch.stack([e[:,(r>=lo)&(r<hi)].sum(1) for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.001)]],1)
 return masses/masses.sum(1,keepdim=True).clamp_min(1e-8)

def relation(s,t,y,v):
 s=F.normalize(s,dim=1).flatten(2);t=F.normalize(t.detach(),dim=1).flatten(2)
 y=F.interpolate(y.float(),size=v.shape[-2:],mode='nearest').flatten(1)>.5;v=v.flatten(1)>.5
 losses=[]
 for i in range(len(s)):
  # Same geography, up to64 foreground +64 background points, no GT relation target.
  ids=[];gen=torch.Generator(device=s.device).manual_seed(42)
  for mask in (y[i]&v[i],~y[i]&v[i]):
   idx=mask.nonzero().flatten();ids.append(idx[torch.randperm(len(idx),device=s.device,generator=gen)[:64]])
  ids=torch.cat(ids)
  if len(ids)<2:continue
  a=s[i,:,ids].T;b=t[i,:,ids].T;keep=~torch.eye(len(ids),dtype=torch.bool,device=s.device)
  losses.append(F.smooth_l1_loss((a@a.T)[keep],(b@b.T)[keep]))
 return torch.stack(losses).mean() if losses else s.sum()*0

def terms(arm,system,so,to,b,lam):
 valid=b['valid'];terms={}
 if arm=='A':
  terms['bk']=sum(masked_mse(system.bk[i](so['bk_change'][i]),to['bk_change'][i].detach(),valid) for i in range(4))/4
  terms['freq']=sum(freq_loss(s,t.detach(),valid) for s,t in zip(so['change'],to['change']))/4
  terms['jscad']=sum(jscad_loss(s,t.detach(),valid) for s,t in zip(so['change'],to['change']))/4
  terms['kd']=logit_kd(so['z'],to['z'].detach(),valid)
 elif arm=='B':
  rel=[];freq=[]
  for i,(s,t) in enumerate(zip(so['change'],to['change'])):
   zs=system.proj[i](s);zt=t.detach() # Teacher projection is fixed identity, no drifting target.
   v=F.interpolate(valid.float(),size=s.shape[-2:],mode='nearest')
   rel.append(relation(zs,zt,b['mask'],v))
   # Identical numeric support, no learned frequency filter or forward filtering.
   freq.append(F.smooth_l1_loss(band_distribution(zs*v),band_distribution(zt*v).detach()))
  terms={'rel':torch.stack(rel).mean(),'freqrel':torch.stack(freq).mean()}
 return terms,sum(lam.get(k,0)*x for k,x in terms.items())
