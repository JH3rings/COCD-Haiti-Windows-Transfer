"""Correct target-only search KD: spatial policy JS and shared-memory evidence KD."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'cocd'))
from paths import DEV
from windows_main.data import HaitiPairs,split
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2,load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
OUT=ROOT/'experiments/windows_main/student_target_search_v2';TP=ROOT/'experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt';SP=ROOT/'experiments/windows_main/da_search/SO_backbone_seed42.pt'
SEED=42;BATCH=16;LR=5e-5;MAX=100;PATIENCE=2;KD_FRACTION=0.10;KD_FRACTIONS={};EPS=1e-8
ARMS={'S0':(0.0,0.0),'S1':(1.0,0.0),'S2':(0.5,0.5)}
def seed(): random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def dl(x,s=False):return DataLoader(x,batch_size=BATCH,shuffle=s,num_workers=0)
def qmask(ga,gd,h=32,w=32):
 x=torch.cat(((ga>0)&~(gd>0),(gd>0)&~(ga>0))).float().to(DEV)[:,None]
 return F.max_pool2d(x,kernel_size=4,stride=4)[:,0]>0
def wmean(x,m):
 x=x.reshape(x.shape[0],-1);m=m.reshape(m.shape[0],-1);w=1+m.float();return (x*w).sum()/(w.sum()+EPS)
def spatial_policy(p, memory):
 """Bilinearly splat dynamic LxK samples onto target P2 cells: P(u|q)."""
 off=2*torch.tanh(p['offsets']);wt=p['weights'];b,lv,k,_,h,w=off.shape;q=h*w;n=q
 yy,xx=torch.meshgrid(torch.arange(h,device=off.device),torch.arange(w,device=off.device),indexing='ij')
 by=yy.reshape(1,q,1).float();bx=xx.reshape(1,q,1).float();out=torch.zeros(b,q,n,device=off.device)
 for l,mem in enumerate(memory):
  scale=h/mem.shape[-2]; y=by+off[:,l,:,0].permute(0,2,3,1).reshape(b,q,k)*scale; x=bx+off[:,l,:,1].permute(0,2,3,1).reshape(b,q,k)*scale
  y=y.clamp(0,h-1);x=x.clamp(0,w-1);y0=y.floor().long();x0=x.floor().long();fy=y-y0;fx=x-x0; a=wt[:,l].permute(0,2,3,1).reshape(b,q,k)
  for yi,xi,c in ((y0,x0,(1-fy)*(1-fx)),(y0,(x0+1).clamp(max=w-1),(1-fy)*fx),((y0+1).clamp(max=h-1),x0,fy*(1-fx)),((y0+1).clamp(max=h-1),(x0+1).clamp(max=w-1),fy*fx)):
   out.scatter_add_(2,yi*w+xi,a*c)
 return out/(out.sum(-1,keepdim=True)+EPS)
def js(a,b):
 m=.5*(a+b);return .5*((a*(a.add(EPS).log()-m.add(EPS).log())).sum(-1)+(b*(b.add(EPS).log()-m.add(EPS).log())).sum(-1))
def terms(s,t,sa,sd,ga,gd):
 with torch.no_grad():ta,td=t.forward_pair(sa,sd,'counter'); tp=[ta['policy'],td['policy']]
 outs=[s(sa),s(sd)]; ms=[tuple(x.detach() for x in o['values']) for o in outs]; sp=[s.policy(s.context(tuple(x.detach() for x in o['T']))) for o in outs]
 qm=qmask(ga,gd); pl=[];el=[]
 for m,p,pt in zip(ms,sp,tp):
  P=spatial_policy(p,m);T=spatial_policy({k:v.detach() for k,v in pt.items() if k in ('offsets','weights')},m);pl.append(js(P,T)); rs=s.search(m,p);rt=s.search(m,{k:v.detach() for k,v in pt.items() if k in ('offsets','weights')});el.append(1-F.cosine_similarity(rs,rt,dim=1))
 return .5*(wmean(pl[0],qm[:len(sa)])+wmean(pl[1],qm[len(sa):])),.5*(wmean(el[0],qm[:len(sa)])+wmean(el[1],qm[len(sa):])),outs,qm
def pred(s,ds):
 s.eval();p=[];y=[];g=[];c=[]
 with torch.no_grad():
  for a,d,z,ga,gd,_ in dl(ds):
   p += [torch.sigmoid(s(a.to(DEV))['z'])[:,0].cpu().numpy(),torch.sigmoid(s(d.to(DEV))['z'])[:,0].cpu().numpy()];y += [z.numpy(),z.numpy()];g += [ga.numpy(),gd.numpy()];c += [gd.numpy(),ga.numpy()]
 return np.concatenate(p),np.concatenate(y),np.concatenate(g),np.concatenate(c)
def au(p,y,g,c,region):
 m=np.ones_like(y,bool) if region=='Overall' else ((g>0)&~(c>0));return float(average_precision_score(y[m].astype(bool),p[m]))
def main():
 OUT.mkdir(parents=True,exist_ok=True);tid,_,eid=split();tr=HaitiPairs(tid,True);te=HaitiPairs(eid,False,tr.stats);seed();t=TargetSearchTeacherV2().to(DEV);t.load_state_dict(torch.load(TP,map_location=DEV,weights_only=True));t.eval();[x.requires_grad_(False) for x in t.parameters()]
 for name,(usep,usee) in ARMS.items():
  final=OUT/f'{name}_seed42.pt';prog=OUT/f'{name}_progress.json'
  if final.exists():continue
  seed();s=StudentTargetSearchV2().to(DEV);load_so_backbone(s,str(SP));opt=torch.optim.Adam(s.parameters(),lr=LR);cal=None;best=-1;bad=0;hist=[]
  fraction=KD_FRACTIONS.get(name,KD_FRACTION)
  for ep in range(1,MAX+1):
   s.train();vals=[]
   for a,d,y,ga,gd,_ in dl(tr,True):
    a,d,y,ga,gd=a.to(DEV),d.to(DEV),y.to(DEV),ga.to(DEV),gd.to(DEV); lp,le,o,q=terms(s,t,a,d,ga,gd); seg=.5*(F.binary_cross_entropy_with_logits(o[0]['z'][:,0],y)+F.binary_cross_entropy_with_logits(o[1]['z'][:,0],y))
    if cal is None: cal=(float(lp.detach()),float(le.detach()),float(seg.detach()))
    kd=lp/(cal[0]+EPS)*float(usep)+le/(cal[1]+EPS)*float(usee); loss=seg+(fraction*cal[2])*kd; opt.zero_grad();loss.backward();opt.step();vals.append((float(seg.detach()),float(lp.detach()),float(le.detach())))
   p,y,g,c=pred(s,te);score=au(p,y,g,c,'Overall');rec={'epoch':ep,'seg':float(np.mean(vals,0)[0]),'policy_raw':float(np.mean(vals,0)[1]),'evidence_raw':float(np.mean(vals,0)[2]),'monitor_test_auprc':score,'calibration':cal};hist.append(rec)
   if score>best:best=score;bad=0;state={k:v.cpu().clone() for k,v in s.state_dict().items()};be=ep
   else:bad+=1
   prog.write_text(json.dumps({'arm':name,'history':hist,'best':best,'best_epoch':be,'teacher':str(TP)},indent=2));print(name,ep,score,flush=True)
   if bad>=PATIENCE:break
  s.load_state_dict(state);torch.save(s.state_dict(),final)
if __name__=='__main__':main()
