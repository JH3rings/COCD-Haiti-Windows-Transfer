#!/usr/bin/env python3
"""Stage-1 GO/NO-GO: rapid single-orbit landslide COCD (S0, T, KD, Ours)."""
import argparse,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,rasterio,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm
# --- package-relative resolution (Windows / Linux / CUDA port) --------------
# This file is a dependency of scripts/23: it owns the dataset loader, the
# frozen spatial split and the model zoo.  Nothing below is macOS specific.
COCD=Path(__file__).resolve().parents[1];sys.path.insert(0,str(COCD))
from paths import (DATASET_ROOT,LOADER_BATCH,OUT_ROOT,SPLIT_DIR,  # noqa: E402
                   DEV,describe)
ROOT=DATASET_ROOT;A=COCD;P=DATASET_ROOT/'processed';OUT=OUT_ROOT/'rapid_landslide_cocd';EPS=1e-8;SEED=42
RAW={('asc','pre'):'Pre_event/S1_ASC_20210805',('desc','pre'):'Pre_event/S1_DESC_20210803',('asc','post'):'Post_event/S1_ASC_20210817',('desc','post'):'Post_event/S1_DESC_20210815'}
from models.landslide_cocd import SingleOrbitStudent,DualOrbitTeacher
from losses.landslide_cocd import land_loss,gain_gate,correction_loss,dis2_style_loss
# DEV is imported from paths: CUDA > MPS > CPU.
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def read(p,bands=None):
 with rasterio.open(p) as f:return f.read(bands).astype('float32') if bands else f.read(1).astype('float32')
def split():
 """The frozen spatial split.

 Under v3 there is no validation partition: ``SPLIT_DIR`` holds only
 ``train_ids.csv`` (the merged 1370 locations) and ``test_ids.csv`` (343).
 ``val`` is then returned empty, and callers that still ask for it get an
 empty list rather than a silently redrawn partition.  The pre-v3 three-way
 split in ``data/splits/`` is still read if ``COCD_SPLIT_DIR`` points there.

 The id order matters: it fixes the dataset order, and therefore batch
 membership.  The original seed-42 draw below is only the fallback for a
 checkout that lacks the pinned files entirely.
 """
 pinned={n:SPLIT_DIR/f'{n}_ids.csv' for n in ('train','val','test')}
 if pinned['train'].exists() and pinned['test'].exists():
  train=pd.read_csv(pinned['train']).sample_id.astype(int).tolist()
  test=pd.read_csv(pinned['test']).sample_id.astype(int).tolist()
  val=(pd.read_csv(pinned['val']).sample_id.astype(int).tolist()
       if pinned['val'].exists() else [])
  return train,val,test
 d=pd.read_csv(A/'manifests'/'spatial_80_20_split.csv');tr=d[d.split=='train'].sample_id.astype(int).to_numpy();te=d[d.split=='test'].sample_id.astype(int).to_numpy();rng=np.random.default_rng(SEED);iv=set(rng.choice(tr,size=round(.1*len(tr)),replace=False));return [int(x) for x in tr if x not in iv],[int(x) for x in tr if x in iv],te.astype(int).tolist()
class HaitiPairs(Dataset):
 def __init__(self,ids,train_modes=False,stats=None):
  self.rows=[];self.ids=[int(x) for x in ids];self.train_modes=train_modes
  for sid in tqdm(ids,desc='cache rapid-landslide data',unit='location'):
   z={}
   for o in ('asc','desc'):
    for e in ('pre','post'):z[(o,e)]=read(ROOT/RAW[(o,e)]/f'{sid}.tif',[2,1])
   land=(read(P/f'sample_{sid:06d}'/'landslide_mask.tif')>0).astype('float32')
   ga=read(P/f'sample_{sid:06d}'/'asc_pre_geom.tif');gd=read(P/f'sample_{sid:06d}'/'desc_pre_geom.tif')
   self.rows.append((z,land,ga,gd))
  if stats is None:
   sm=np.zeros(2);ss=np.zeros(2);n=0
   for z,_,_,_ in self.rows:
    for x in z.values():sm+=x.sum((1,2));ss+=(x*x).sum((1,2));n+=x.shape[1]*x.shape[2]
   stats=sm/n,np.sqrt(ss/n-(sm/n)**2+1e-6)
  self.stats=stats;self.spec=[(i,m) for i in range(len(self.rows)) for m in ((0,1,2) if train_modes else (0,))]
 def __len__(self):return len(self.spec)
 def __getitem__(self,i):
  ix,mode=self.spec[i];z,land,ga,gd=self.rows[ix];m,s=self.stats
  def one(o):
   pre,post=z[(o,'pre')],z[(o,'post')]
   if mode==1:post=pre
   if mode==2:pre=post
   q=np.concatenate((pre,post,np.full((1,128,128),0. if o=='asc' else 1.,np.float32)))
   return torch.from_numpy(((q-np.r_[m,m,0.][:,None,None])/np.r_[s,s,1.][:,None,None]).astype('float32'))
  y=land if mode==0 else np.zeros_like(land)
  return one('asc'),one('desc'),torch.from_numpy(y),torch.from_numpy(ga),torch.from_numpy(gd),torch.tensor(mode==0)
def dl(ds,shuffle=False):return DataLoader(ds,batch_size=LOADER_BATCH,shuffle=shuffle,num_workers=0)
def posweight(ds):
 p=sum(float(x[1].sum()) for x in ds.rows);return (len(ds.rows)*128*128-p)/(p+EPS)
def metric(p,y,t=.5):
 p=np.asarray(p).ravel();y=np.asarray(y).astype(bool).ravel();h=p>=t;tp=(h&y).sum();fp=(h&~y).sum();fn=(~h&y).sum();pr=tp/(tp+fp+EPS);re=tp/(tp+fn+EPS)
 return {'iou':float(tp/(tp+fp+fn+EPS)),'f1':float(2*pr*re/(pr+re+EPS)),'precision':float(pr),'recall':float(re),'auprc':float(average_precision_score(y,p)) if y.any() else float('nan')}
@torch.no_grad()
def infer(m,ds,kind,teacher=None):
 m.eval();out={k:[] for k in ('p','y','gt','gc','orbit','self','dual')}
 for a,d,y,ga,gd,_ in dl(ds):
  a,d,y,ga,gd=[x.to(DEV) for x in (a,d,y,ga,gd)]
  if kind=='T':
   sa,da,_=m(a,d);sd,bd,_=m(d,a);p=torch.sigmoid(torch.cat((da,bd)))[:,0];selfp=torch.sigmoid(torch.cat((sa,sd)))[:,0];dual=p
  else:
   _,z,_=m(torch.cat((a,d)));p=torch.sigmoid(z)[:,0]
   if teacher:
    sa,da,_=teacher(a,d);sd,bd,_=teacher(d,a);selfp=torch.sigmoid(torch.cat((sa,sd)))[:,0];dual=torch.sigmoid(torch.cat((da,bd)))[:,0]
   else:selfp=dual=None
  out['p'].append(p.cpu().numpy());out['y'].append(torch.cat((y,y)).cpu().numpy());out['gt'].append(torch.cat((ga,gd)).cpu().numpy());out['gc'].append(torch.cat((gd,ga)).cpu().numpy());out['orbit'].append(np.array(['asc']*len(a)+['desc']*len(a)))
  if selfp is not None:out['self'].append(selfp.cpu().numpy());out['dual'].append(dual.cpu().numpy())
 return {k:(np.concatenate(v) if v else None) for k,v in out.items()}
def rows(name,q):
 ans=[]
 for part,ix in [('overall',np.ones(len(q['orbit']),bool)),('asc',q['orbit']=='asc'),('desc',q['orbit']=='desc')]:
  base={'method':name,'partition':part,**metric(q['p'][ix],q['y'][ix])};c=(q['gt'][ix]>0)&(q['gc'][ix]==0);cm=metric(q['p'][ix][c],q['y'][ix][c]);base.update({'complementary_pixels':int(c.sum()),'complementary_positive_pixels':int(q['y'][ix][c].sum()),'complementary_iou':cm['iou'],'complementary_f1':cm['f1'],'complementary_recall':cm['recall'],'complementary_auprc':cm['auprc']});ans.append(base)
 return ans
def frozen_teacher():
 t=DualOrbitTeacher().to(DEV);t.load_state_dict(torch.load(OUT/'T.pt',map_location=DEV,weights_only=True));t.eval()
 for p in t.parameters():p.requires_grad_(False)
 return t
def train(kind,tr,va,epochs=20):
 seed();m=DualOrbitTeacher().to(DEV) if kind=='T' else SingleOrbitStudent(correction=kind in ('KD','Ours','DIS2')).to(DEV);teacher=None if kind in ('S0','T') else frozen_teacher();op=torch.optim.Adam(m.parameters(),lr=5e-5,weight_decay=0.0);w=posweight(tr);start_ep=1;latest=OUT/f'{kind}_latest.pt';status=OUT/f'{kind}_progress.json'
 # NOTE: this is the legacy macOS-era training entry point and it is NOT on the
 # v4 execution path.  Formal v4 training goes through 23_train_ours_v2.py /
 # 23_train_external_baselines.py (internal arms and external seats) and
 # windows_main/main.py (the runner).  This function still runs the old
 # fixed-20-epoch, last-epoch-is-the-model schedule; it is kept only so the
 # old numbers stay traceable.  Do not use it for a v4 result.
 # A complete epoch is still an atomic recovery point.
 if latest.exists():
  state=torch.load(latest,map_location=DEV,weights_only=False)
  if not state.get('complete',False):
   m.load_state_dict(state['model']);op.load_state_dict(state['optimizer']);start_ep=state['epoch']+1
   print(f'[{kind}] resume from completed epoch {state["epoch"]}; next epoch={start_ep}',flush=True)
 for ep in range(start_ep,epochs+1):
  m.train();ls=[];bar=tqdm(dl(tr,True),desc=f'{kind} epoch {ep:02d}/{epochs}',unit='batch')
  for batch,a_d in enumerate(bar,1):
   a,d,y,ga,gd,real=a_d
   a,d,y,ga,gd,real=[x.to(DEV) for x in (a,d,y,ga,gd,real)];op.zero_grad()
   if kind=='T':
    sa,da,_=m(a,d);sd,bd,_=m(d,a);loss=(land_loss(sa,y,w)+land_loss(da,y,w)+land_loss(sd,y,w)+land_loss(bd,y,w))/4
   else:
    x=torch.cat((a,d));yy=torch.cat((y,y));_,z,rs=m(x);loss=land_loss(z,yy,w);rr=torch.cat((real,real)).bool()
    if teacher and rr.any():
     with torch.no_grad():
      sa,da,ra=teacher(a[real.bool()],d[real.bool()]);sd,bd,rd=teacher(d[real.bool()],a[real.bool()]);selfz=torch.cat((sa,sd));dualz=torch.cat((da,bd));rt=torch.cat((ra,rd))
     if kind=='KD':loss=loss+.2*F.mse_loss(torch.sigmoid(z[rr]),torch.sigmoid(dualz))
     elif kind=='DIS2':loss=loss+.1*dis2_style_loss(rs[rr],rt,dualz,yy[rr])+.1*F.mse_loss(torch.sigmoid(z[rr]),torch.sigmoid(dualz))
     else:
      gate=gain_gate(selfz,dualz,yy[rr],torch.cat((ga[real.bool()],gd[real.bool()])),torch.cat((gd[real.bool()],ga[real.bool()])))
      loss=loss+.2*correction_loss(rs[rr],rt,gate)
   loss.backward();op.step();ls.append(float(loss.detach().cpu()));bar.set_postfix(loss=f'{np.mean(ls):.3f}')
   if batch%10==0 or batch==1:status.write_text(json.dumps({'method':kind,'epoch':ep,'epochs':epochs,'batch':batch,'batches':len(bar),'mean_loss':round(float(np.mean(ls)),6),'state':'training'}))
  # legacy schedule: no held-out pass, the last epoch IS the model.  v4 replaces
  # this with best-monitored-epoch selection and is implemented in 23_* instead.
  torch.save({'model':m.state_dict(),'optimizer':op.state_dict(),'epoch':ep,'complete':False},latest);status.write_text(json.dumps({'method':kind,'epoch':ep,'epochs':epochs,'batch':len(bar),'batches':len(bar),'mean_loss':round(float(np.mean(ls)),6),'state':'epoch_complete','checkpoint':'epoch_20','early_stop':False}))
 torch.save(m.state_dict(),OUT/f'{kind}.pt');torch.save({'model':m.state_dict(),'optimizer':op.state_dict(),'epoch':ep,'complete':True},latest);status.write_text(json.dumps({'method':kind,'epoch':ep,'epochs':epochs,'state':'complete','checkpoint':'epoch_20','early_stop':False}));return m,teacher
def main():
 ap=argparse.ArgumentParser();ap.add_argument('method',choices=['S0','T','KD','DIS2','Ours','all']);ap.add_argument('--epochs',type=int,default=20);z=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);tid,vid,eid=split();tr=HaitiPairs(tid,True);te=HaitiPairs(eid,False,tr.stats);todo=['S0','T','KD','DIS2','Ours'] if z.method=='all' else [z.method];allrows=[]
 for k in todo:
  print(f'\\n===== START {k} =====',flush=True);m,t=train(k,tr,None,z.epochs);q=infer(m,te,k,t);allrows+=rows(k,q)
 frame=pd.DataFrame(allrows)
 # Keep a method-specific immutable result alongside the convenient latest table.
 # This matters when the four formal runs are launched separately.
 frame.to_csv(OUT/'metrics.csv',index=False)
 frame.to_csv(OUT/f'{z.method}_metrics.csv',index=False)
 if z.method=='all':
  d=pd.DataFrame(allrows);o=d[d.partition=='overall'].set_index('method');c=d[d.partition=='overall'].set_index('method');gap=(o.loc['Ours','iou']-o.loc['S0','iou'])/(o.loc['T','iou']-o.loc['S0','iou']+EPS);report=['# Rapid single-orbit landslide COCD — stage 1','',d.to_markdown(index=False),'',f'Overall IoU gap recovery: {gap:.3f}.','', 'GO requires T>S0 and Ours>KD, especially complementary recall/F1/AUPRC.'];(A/'reports'/'rapid_single_orbit_landslide_cocd.md').write_text('\\n'.join(report))
if __name__=='__main__':main()
