import torch
from torch.nn import functional as F
EPS=1e-8
def land_loss(logit,target,pos_weight):
    z=logit[:,0];b=F.binary_cross_entropy_with_logits(z,target,pos_weight=torch.tensor(pos_weight,device=z.device))
    p=torch.sigmoid(z);d=1-(2*(p*target).sum((1,2))+EPS)/((p+target).sum((1,2))+EPS)
    return b+.2*d.mean()
def gain_gate(single,dual,target,target_geom,counter_geom):
    es=F.binary_cross_entropy_with_logits(single[:,0],target,reduction='none');ed=F.binary_cross_entropy_with_logits(dual[:,0],target,reduction='none')
    g=((target_geom>0)&(counter_geom==0)).float()*(es-ed).clamp_min(0)
    return g/(g.amax((1,2),keepdim=True)+EPS)
def correction_loss(student_r,teacher_r,gate):
    g=F.interpolate(gate[:,None],size=student_r.shape[-2:],mode='area')[:,0]
    a=F.normalize(student_r,dim=1);b=F.normalize(teacher_r.detach(),dim=1)
    return ((1-(a*b).sum(1))*g).sum()/(g.sum()+EPS)
def dis2_style_loss(student_r,teacher_r,teacher_logits,target):
 """Generic (non-geometry) classwise compensatory feature matching for DIS2 adaptation."""
 a=F.normalize(student_r,dim=1);b=F.normalize(teacher_r.detach(),dim=1);d=1-(a*b).sum(1)
 fg=torch.sigmoid(teacher_logits[:,0]).detach();w=target*fg+(1-target)*(1-fg)
 w=F.interpolate(w[:,None],size=d.shape[-2:],mode='area')[:,0]
 return (d*w).sum()/(w.sum()+EPS)
