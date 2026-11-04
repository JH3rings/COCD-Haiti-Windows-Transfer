"""Three auxiliary objectives only. FFT/Haar are loss branches, not inference."""
import torch
from torch.nn import functional as F

EPS=1e-6
def mask(x,v):return F.interpolate(v.float(),size=x.shape[-2:],mode='nearest')
def cosine(a,b,v):
 assert a.shape==b.shape
 v=mask(a,v)[:,0];eligible=(b.detach().float().square().sum(1)>EPS**2).float()*v
 d=(1-F.cosine_similarity(a.float(),b.detach().float(),dim=1,eps=EPS)).clamp_min(0)
 return (d*eligible).sum()/eligible.sum().clamp_min(1)

def standardize(x,v):
 x=x.float();v=mask(x,v);n=v.sum((-2,-1),keepdim=True)
 mean=(x*v).sum((-2,-1),keepdim=True)/n.clamp_min(1)
 var=((x-mean).square()*v).sum((-2,-1),keepdim=True)/n.clamp_min(1)
 return ((x-mean)/torch.sqrt(var+EPS))*v,v

def tcsd(s,t,valid):
 assert s.shape==t.shape
 s,v=standardize(s,valid);t,_=standardize(t.detach(),valid)
 a=torch.log1p(torch.fft.rfft2(s,norm='ortho').abs()).flatten(2)
 b=torch.log1p(torch.fft.rfft2(t,norm='ortho').abs()).flatten(2)
 good=(b.square().sum(-1)>EPS**2)&(v.sum((-2,-1))>1)
 d=(1-F.cosine_similarity(a,b,dim=-1,eps=EPS)).clamp_min(0)
 return (d*good).sum()/good.sum().clamp_min(1)

def haar(x,valid):
 """Orthonormal 2x2 Haar. LH=low rows/high columns; HL=high rows/low columns.

 Odd trailing row/column is zero padded, but its coefficients are excluded by
 the strict all-four-valid coefficient mask. No reflection invents data.
 """
 h,w=x.shape[-2:];x=F.pad(x,(0,w%2,0,h%2));v=F.pad(mask(x[...,:h,:w],valid),(0,w%2,0,h%2))
 a=x[...,0::2,0::2];b=x[...,0::2,1::2];c=x[...,1::2,0::2];d=x[...,1::2,1::2]
 keep=(F.avg_pool2d(v,2,2)>=1-1e-7).float()
 return ((a+b+c+d)*.5,(-a+b-c+d)*.5,(-a-b+c+d)*.5,(a-b-c+d)*.5),keep

def energy(h,v):
 e=torch.sqrt(h.square().mean(1,keepdim=True)+EPS)
 mean=(e*v).sum((-2,-1),keepdim=True)/v.sum((-2,-1),keepdim=True).clamp_min(1)
 return (e/mean.clamp_min(EPS))*v

def wcsd(s,t,valid):
 assert s.shape==t.shape
 s,_=standardize(s,valid);t,_=standardize(t.detach(),valid)
 a,v=haar(s,valid);b,v2=haar(t,valid);assert torch.equal(v,v2)
 ll=cosine(a[0],b[0],v)
 detail=sum((F.smooth_l1_loss(energy(x,v),energy(y,v),reduction='none')*v).sum()/v.sum().clamp_min(1) for x,y in zip(a[1:],b[1:]))/3
 return ll+detail

def output_kd(zs,zt,valid,temperature=10.):
 # Bernoulli KL = categorical KL on [0,z]; class sum, valid-pixel mean, T^2.
 ps=F.log_softmax(torch.cat([torch.zeros_like(zs),zs],1).float()/temperature,1)
 pt=F.softmax(torch.cat([torch.zeros_like(zt),zt.detach()],1).float()/temperature,1)
 v=valid.float()[:,0];kl=F.kl_div(ps,pt,reduction='none').sum(1)*temperature**2
 return (kl*v).sum()/v.sum().clamp_min(1)

def all_terms(system,so,to,batch,frequencies=True):
 feat=[];fft=[];wave=[]
 for i in range(4):
  z=system.adapters[i](so['D'][i]);e=to['E'][i].detach();assert z.shape==e.shape
  feat.append(cosine(z,e,batch['valid']))
  if frequencies:fft.append(tcsd(z,e,batch['valid']));wave.append(wcsd(z,e,batch['valid']))
 raw={'feat':torch.stack(feat).mean(),'out':output_kd(so['z'],to['z'],batch['valid'])}
 if frequencies:raw.update(fft=torch.stack(fft).mean(),wave=torch.stack(wave).mean())
 return raw

def terms(arm,system,so,to,batch,lam):
 if arm=='MFSAD':
  # Actual corrected paper implementation, same new residual Teacher.
  import paper_mfsad
  return paper_mfsad.losses('mfsad',system,so,to,batch,lam)
 # Evaluate only the active frequency branch; do not change inference values.
 feat=[];freq=[]
 for i in range(4):
  z=system.adapters[i](so['D'][i]);e=to['E'][i].detach();assert z.shape==e.shape
  feat.append(cosine(z,e,batch['valid']))
  if arm=='E2':freq.append(tcsd(z,e,batch['valid']))
  elif arm=='E3':freq.append(wcsd(z,e,batch['valid']))
 raw={'feat':torch.stack(feat).mean(),'out':output_kd(so['z'],to['z'],batch['valid'])}
 if arm!='E1':raw['freq']=torch.stack(freq).mean()
 return raw,sum(lam.get(k,0.)*v for k,v in raw.items())
