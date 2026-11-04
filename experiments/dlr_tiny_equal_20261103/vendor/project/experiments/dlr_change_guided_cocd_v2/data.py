"""Independent DLR input cache; no test-derived normalization or cloud removal."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,h5py,torch
from torch.utils.data import Dataset
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BANDS=['B02','B03','B04','B08','B11','B12']

def raw(row,post):
 with h5py.File(row.h5_path,'r') as f:
  i=int(row.h5_index)
  sar=np.stack([f[t+'_'+b][i,0] for t in ['PRE1','POST1'] for b in ['gVV','gVH']]).astype(np.float32)
  pre=np.stack([f['PRE1_'+b][i,0] for b in BANDS]).astype(np.float32)
  optical_post=np.stack([f['POST1_'+b][i,0] for b in BANDS]).astype(np.float32) if post else None
  mask=f['None_MASK'][i].astype(np.float32)
 return sar,pre,optical_post,mask

def numerical_valid(sar):return np.isfinite(sar).all(0)&(sar>0).all(0)
def sar_db(sar):return 10*np.log10(np.maximum(np.nan_to_num(sar,nan=0,posinf=0,neginf=0),1e-8))

def fit_normalization(train):
 # Per-polarization pooled pre/post Gamma0-dB; optical statistics from pre ONLY.
 ss=np.zeros(2);ss2=np.zeros(2);sn=np.zeros(2);os=np.zeros(6);os2=np.zeros(6);on=np.zeros(6)
 for row in train.itertuples():
  s,o,_,_=raw(row,False);v=numerical_valid(s);db=sar_db(s)
  for k in range(2):
   a=db[[k,k+2]][:,v].astype(np.float64);ss[k]+=a.sum();ss2[k]+=(a*a).sum();sn[k]+=a.size
  for k in range(6):
   a=o[k][np.isfinite(o[k])&(o[k]>0)].astype(np.float64)/10000
   os[k]+=a.sum();os2[k]+=(a*a).sum();on[k]+=a.size
 sm=ss/sn;om=os/on
 return {'sar_mean':sm.tolist(),'sar_std':np.sqrt(np.maximum(ss2/sn-sm*sm,1e-6)).tolist(),
         'optical_mean':om.tolist(),'optical_std':np.sqrt(np.maximum(os2/on-om*om,1e-6)).tolist(),
         'source':'training set only; optical PRE1 only; same transform applied to POST1 optical',
         'sar_unit':'10log10(linear Gamma0); zeros marked numerical-invalid, not physical radar shadow',
         'optical_unit':'distributed DN /10000; no undocumented BOA offset inferred','training_ids_sha256':hashlib.sha256('\n'.join(sorted(train.sample_id)).encode()).hexdigest()}

class DLRInputs(Dataset):
 def __init__(self,frame,stats,include_post=False,augment=False):
  self.frame=frame.reset_index(drop=True);self.include_post=include_post;self.augment=augment;self.cache=[]
  mean=np.array(stats['sar_mean']*2,dtype=np.float32)[:,None,None];std=np.array(stats['sar_std']*2,dtype=np.float32)[:,None,None]
  om=np.array(stats['optical_mean'],dtype=np.float32)[:,None,None];ostd=np.array(stats['optical_std'],dtype=np.float32)[:,None,None]
  for row in self.frame.itertuples():
   sar,pre,post,y=raw(row,include_post);valid=numerical_valid(sar)
   assert set(np.unique(y))<={0.,1.} and np.isfinite(y).all()
   assert np.isfinite(pre).mean()>.99 and np.any(pre>0),'Damaged/empty pre optical'
   if post is not None:assert np.isfinite(post).mean()>.99 and np.any(post>0),'Damaged/empty post optical'
   s=(sar_db(sar)-mean)/std;s[:,~valid]=0;s=np.clip(s,-8,8)
   assert row.orbit in ['ASC','DES']
   s=np.concatenate([s,np.full((1,128,128),float(row.orbit=='ASC'),np.float32)])
   def opt(a):return np.clip((np.nan_to_num(a,nan=0,posinf=0,neginf=0)/10000-om)/ostd,-8,8).astype(np.float32)
   item={'sar':torch.from_numpy(s.astype(np.float32)),'opt_pre':torch.from_numpy(opt(pre)),
         'mask':torch.from_numpy(y),'valid':torch.from_numpy(valid[None]),'sample_id':row.sample_id,'event_id':row.event_id}
   if include_post:item['opt_post']=torch.from_numpy(opt(post))
   self.cache.append(item)
 def __len__(self):return len(self.cache)
 def __getitem__(self,i):
  item=dict(self.cache[i])
  if self.augment:
   turns=int(torch.randint(0,4,()).item())
   if turns:
    for k,v in item.items():
     if isinstance(v,torch.Tensor):item[k]=torch.rot90(v,turns,(-2,-1))
   dims=[]
   if torch.rand(())<.5:dims.append(-1)
   if torch.rand(())<.5:dims.append(-2)
   if dims:
    for k,v in item.items():
     if isinstance(v,torch.Tensor):item[k]=v.flip(dims)
  return item
