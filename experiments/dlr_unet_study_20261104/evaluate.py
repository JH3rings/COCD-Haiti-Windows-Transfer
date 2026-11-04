"""Common DLR U-Net training and pooled scoring utilities."""
import hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from unet import PaperUNet

HERE = Path(__file__).resolve().parent

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(10):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.02)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def setup():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_available()


def frames():
    d=pd.read_csv(HERE/'split_manifest.csv')
    return {k:d[d.split==k].reset_index(drop=True) for k in ('train','val','test')}


def loader(ds,shuffle=False,generator=None):
    return DataLoader(ds,batch_size=16,shuffle=shuffle,generator=generator,num_workers=0,pin_memory=True)


def cuda(b):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}


def metrics(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p);pred=p>=.5
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'IoU':tp/max(tp+fp+fn,1),'F1':2*tp/max(2*tp+fp+fn,1),
            'Precision':tp/max(tp+fp,1),'Recall':tp/max(tp+fn,1),
            'AUPRC':float(average_precision_score(y,p)) if y.any() and (~y).any() else None,
            'foreground_fraction':float(y.mean()),'predicted_positive_fraction':float(pred.mean()),'pixels':len(y)}


@torch.no_grad()
def evaluate(m,ds,save=None):
    m.eval();all_y=[];all_p=[];events={};subsets={};records=[];saved=[];ids=[];valids=[];rawys=[]
    for b in loader(ds):
        b=cuda(b);p=m(b['x']).sigmoid().cpu().numpy()[:,0]
        y=b['mask'].cpu().numpy()[:,0]>.5;v=b['valid'].cpu().numpy()[:,0].astype(bool)
        for i in range(len(y)):
            yy=y[i][v[i]];pp=p[i][v[i]];all_y.append(yy);all_p.append(pp)
            sid,eid,subset=b['sample_id'][i],b['event_id'][i],b['official_split'][i]
            events.setdefault(eid,[]).append((yy,pp));subsets.setdefault(subset,[]).append((yy,pp))
            records.append({'sample_id':sid,'event_id':eid,'official_subset':subset,**metrics(yy,pp)})
            if save is not None:ids.append(sid);saved.append(p[i]);valids.append(v[i]);rawys.append(y[i])
    result=metrics(np.concatenate(all_y),np.concatenate(all_p));result['patches']=len(ds)
    for name,groups in [('per_event',events),('per_official_subset',subsets)]:
        result[name]={key:{**metrics(np.concatenate([a[0] for a in pairs]),np.concatenate([a[1] for a in pairs])),
                          'patches':len(pairs)} for key,pairs in groups.items()}
    result['macro_event_IoU']=float(np.mean([v['IoU'] for v in result['per_event'].values()]))
    result['macro_event_F1']=float(np.mean([v['F1'] for v in result['per_event'].values()]))
    if save is not None:
        np.savez_compressed(save,sample_ids=np.array(ids),probability=np.stack(saved),valid=np.stack(valids))
        pd.DataFrame(records).to_csv(save.with_suffix('.samples.csv'),index=False)
        result['all_pixels_secondary']=metrics(np.stack(rawys).reshape(-1),np.stack(saved).reshape(-1))
    return result


def optimizer(m):
    groups={True:[],False:[]};names={True:[],False:[]}
    for name,p in m.named_parameters():
        no_decay=p.ndim<=1 or name.endswith('.bias');groups[no_decay].append(p);names[no_decay].append(name)
    opt=torch.optim.AdamW([{'params':groups[k],'weight_decay':0 if k else .01,'lr':1e-4,
                           'group_name':'no_decay' if k else 'decay','names':names[k]} for k in (False,True)],lr=1e-4)
    return opt


def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def lr_factor(step, steps_per_epoch):
    h, t = 50 * steps_per_epoch, 100 * steps_per_epoch
    if not 0 <= step < t:
        raise ValueError(f'Optimizer step {step} outside the declared 100 epochs')
    return 1. if step < h else .01 + .99 * (1 + math.cos(math.pi * (step-h)/(t-1-h))) / 2
