"""M-FSAD Eq1-10 reimplementation for binary, two-temporal segmentation.

No author implementation was available. FDS is intentionally not guessed from
the abstract: exact RCL/STL remain blocked pending full source/formulas.
"""
import torch
from torch.nn import functional as F
from model import mask_for,mse,frequency

def joint_mask(t,valid):
 t=t.detach();v=mask_for(t,valid)
 spatial=t.abs().mean(1,keepdim=True)
 lo=spatial.masked_fill(v==0,float('inf')).amin((2,3),keepdim=True)
 hi=spatial.masked_fill(v==0,-float('inf')).amax((2,3),keepdim=True)
 spatial=torch.nan_to_num((spatial-lo)/(hi-lo).clamp_min(1e-6))*v
 channel=(t.abs()*v).sum((2,3),keepdim=True)/v.sum((2,3),keepdim=True).clamp_min(1)
 channel=(channel-channel.amin(1,keepdim=True))/(channel.amax(1,keepdim=True)-channel.amin(1,keepdim=True)).clamp_min(1e-6)
 return spatial*channel

def jscad(s,t,valid):
 """Eq8: M_joint * (Fs-Ft)^2, NOT (M_joint*(Fs-Ft))^2."""
 assert s.shape==t.shape
 v=mask_for(s,valid);w=joint_mask(t,valid)
 return ((s-t.detach()).square()*w*v).sum()/(v.sum().clamp_min(1)*s.shape[1])

def losses(arm,system,so,to,batch,lam):
 if arm=='base':return {},so['z'].sum()*0
 assert arm=='mfsad','FDS unavailable: never substitute unverified losses'
 import model_kd
 bk=[];freq=[];att=[]
 for j,i in enumerate((1,2,3)):
  for key in ['feat_pre','feat_post']:
   bk.append(mse(system.bk[j](so[key][i]),to[key][i],batch['valid']))
  s=system.neck[j](so['change'][i]);t=to['change'][i].detach();assert s.shape==t.shape
  freq.append(frequency(s,t,batch['valid']));att.append(jscad(s,t,batch['valid']))
 terms={'bk':torch.stack(bk).mean(),'freq':torch.stack(freq).mean(),
        'jscad':torch.stack(att).mean(),
        'kd':model_kd.logit_kd(so['z'],to['z'].detach(),batch['valid'],10.)}
 return terms,sum(lam.get(k,0.)*v for k,v in terms.items())
