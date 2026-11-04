from pathlib import Path
import json,sys
import torch,timm
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
HERE=Path(__file__).resolve().parent

class V2Encoder(nn.Module):
 def __init__(self,initial,in_ch):
  super().__init__();p=json.loads((HERE/'pretrained_location.json').read_text())['path']
  base=timm.create_model('convnextv2_tiny.fcmae_ft_in1k',pretrained=False,drop_path_rate=.05)
  base.load_state_dict(load_file(p),strict=True)
  old=base.stem[0];stem=nn.Conv2d(in_ch,96,4,4)
  with torch.no_grad():
   stem.weight.copy_(old.weight.mean(1,keepdim=True).repeat(1,in_ch,1,1)*(3/(2 if in_ch==3 else in_ch)))
   if in_ch==3:stem.weight[:,2].zero_()
   stem.bias.copy_(old.bias)
  base.stem[0]=stem;self.stem=base.stem;self.stages=base.stages;self.out_channels=(96,192,384,768)
 def forward(self,x):
  x=self.stem(x);out=[]
  for stage in self.stages:x=stage(x);out.append(x)
  return out

class Adapter(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())

class MFSAD(nn.Module):
 def __init__(self,student,arm):
  super().__init__();self.student=student
  if arm=='mfsad':
   self.bk=nn.ModuleList([Adapter(a,b) for a,b in zip((192,384,768),(128,256,512))])
   self.neck=nn.ModuleList([Adapter(128,128) for _ in range(3)])
 def forward(self,sar):return self.student(sar)

def mask_for(x,valid):return F.interpolate(valid.float(),size=x.shape[-2:],mode='nearest')
def mse(s,t,valid):
 assert s.shape==t.shape
 v=mask_for(s,valid);return ((s-t.detach()).square()*v).sum()/(v.sum().clamp_min(1)*s.shape[1])
def frequency(s,t,valid):
 assert s.shape==t.shape
 v=mask_for(s,valid);h,w=s.shape[-2:]
 fs=torch.fft.fftshift(torch.fft.fft2(s*v,norm='ortho'),dim=(-2,-1))
 ft=torch.fft.fftshift(torch.fft.fft2(t.detach()*v,norm='ortho'),dim=(-2,-1))
 yy=torch.arange(h,device=s.device)-h//2;xx=torch.arange(w,device=s.device)-w//2
 gaussian=torch.exp(-(yy[:,None].square()+xx[None,:].square())/(2*(min(h,w)/4)**2))
 return ((fs.abs()-ft.abs()).square()*gaussian).sum()/(s.shape[0]*s.shape[1]*(gaussian.sum()+1e-6))
def attention(s,t,valid):
 t=t.detach();v=mask_for(s,valid);sp=t.abs().mean(1,keepdim=True)
 lo=sp.masked_fill(v==0,float('inf')).amin((2,3),keepdim=True);hi=sp.masked_fill(v==0,-float('inf')).amax((2,3),keepdim=True)
 sp=torch.nan_to_num((sp-lo)/(hi-lo).clamp_min(1e-6))*v
 ch=(t.abs()*v).sum((2,3),keepdim=True)/v.sum((2,3),keepdim=True).clamp_min(1)
 ch=(ch-ch.amin(1,keepdim=True))/(ch.amax(1,keepdim=True)-ch.amin(1,keepdim=True)).clamp_min(1e-6)
 joint=sp*ch
 return (((s-t)*joint).square()*v).sum()/(v.sum().clamp_min(1)*s.shape[1])
def losses(arm,system,so,to,b,lam):
 if arm=='base':return {},so['z'].sum()*0
 import model_kd
 bk=[];freq=[];att=[]
 for j,i in enumerate((1,2,3)):
  for key in ('feat_pre','feat_post'):bk.append(mse(system.bk[j](so[key][i]),to[key][i],b['valid']))
  s=system.neck[j](so['change'][i]);t=to['change'][i].detach();assert s.shape==t.shape
  freq.append(frequency(s,t,b['valid']));att.append(attention(s,t,b['valid']))
 terms={'bk':torch.stack(bk).mean(),'freq':torch.stack(freq).mean(),'jscad':torch.stack(att).mean(),'kd':model_kd.logit_kd(so['z'],to['z'].detach(),b['valid'],2.)}
 return terms,sum(lam.get(k,0)*x for k,x in terms.items())
