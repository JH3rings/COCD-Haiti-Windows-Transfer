import torch
from torch import nn
from torch.nn import functional as F
import method
from model import Adapter,MFSAD

class ResidualTeacher(method.Teacher):
 """Pico dual-temporal branches, SAR-centered optical residual, four-scale FPN."""
 def __init__(self,initial):
  super().__init__(initial)
  del self.fuse
  self.residual=nn.ModuleList([nn.Conv2d(256,128,1) for _ in range(4)])
  self.gates=nn.ModuleList([nn.Conv2d(256,1,1) for _ in range(4)])
 def forward(self,pre,post,sar):
  sp=self.sar_backbone(torch.cat([sar[:,:2],sar[:,4:5]],1));sq=self.sar_backbone(torch.cat([sar[:,2:4],sar[:,4:5]],1))
  op=self.backbone(pre);oq=self.backbone(post);enhanced=[];sar_changes=[];opt_changes=[]
  for i,(a,b,c,d) in enumerate(zip(sp,sq,op,oq)):
   # Independent learned change projections provide the common128-D space.
   ds=self.sar_change[i](torch.cat([a,b,b-a,(b-a).abs()],1))
   do=self.change[i](torch.cat([c,d,d-c,(d-c).abs()],1));context=torch.cat([ds,do],1)
   enhanced.append(ds+self.gates[i](context).sigmoid()*self.residual[i](context));sar_changes.append(ds);opt_changes.append(do)
  fused=[None]*4;fused[3]=self.smooth[3](self.lateral[3](enhanced[3]))
  for i in [2,1,0]:fused[i]=self.smooth[i](self.lateral[i](enhanced[i])+F.interpolate(fused[i+1],size=enhanced[i].shape[-2:],mode='nearest'))
  z=F.interpolate(self.decoder(fused[0]),size=sar.shape[-2:],mode='bilinear',align_corners=False)
  return {'z':z,'E':tuple(enhanced),'change':tuple(fused),'feat_pre':sp,'feat_post':sq,'D_SAR':tuple(sar_changes),'D_OPT':tuple(opt_changes)}

class StudentSystem(MFSAD):
 def __init__(self,student,arm):
  super().__init__(student,'mfsad' if arm=='MFSAD' else 'base')
  if arm!='MFSAD':self.adapters=nn.ModuleList([Adapter(128,128) for _ in range(4)])
  self.captured={}
  for i,layer in enumerate(self.student.change):
   layer.register_forward_hook(self.hook(i))
 def hook(self,i):
  def capture(module,args,output):self.captured[i]=output
  return capture
 def forward(self,sar):
  self.captured.clear();out=self.student(sar)
  assert len(self.captured)==4
  return {**out,'D':tuple(self.captured[i] for i in range(4))}
