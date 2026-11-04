"""DLR authors' U-Net architecture, declared 100-epoch task-adapted recipe.

Imports the preserved paper-based architecture verbatim. This is not author
source code and does not replace the historical 200-epoch paper-setting run.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, math, random, shutil, sys, time, traceback
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
OLD=ROOT/'experiments/dlr_unet_paper_reproduction';V2=ROOT/'experiments/dlr_change_guided_cocd_v2'
spec=importlib.util.spec_from_file_location('preserved_dlr_paper_unet',OLD/'baseline.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
PaperUNet=native.PaperUNet
sys.path.insert(0,str(ROOT/'cocd'))
from windows_main.pico_optimization_v2 import lr_factor
FIELDS=['PRE1_gVV','PRE1_gVH','POST1_gVV','POST1_gVH','None_DEM','None_SLOPE']

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

def raw(row):
    with h5py.File(row.h5_path,'r') as f:
        i=int(row.h5_index)
        x=np.stack([f[k][i,0] for k in FIELDS]).astype(np.float32)
        y=f['None_MASK'][i].astype(np.float32)
    assert np.isfinite(x).all() and set(np.unique(y))<={0.,1.}
    return x,y

class Inputs(Dataset):
    def __init__(self,frame,stats,augment=False):
        self.frame=frame.reset_index(drop=True);self.augment=augment;self.cache=[]
        mean=np.array(stats['sar_mean']*2,np.float32)[:,None,None]
        std=np.array(stats['sar_std']*2,np.float32)[:,None,None]
        tm=np.array(stats['terrain_mean'],np.float32)[:,None,None]
        ts=np.array(stats['terrain_std'],np.float32)[:,None,None]
        for row in self.frame.itertuples():
            x,y=raw(row);valid=(x[:4]>0).all(0)&np.isfinite(x[:4]).all(0)
            sar=(10*np.log10(np.maximum(x[:4],1e-8))-mean)/std;sar[:,~valid]=0
            terrain=(x[4:]-tm)/ts
            x=np.clip(np.concatenate([sar,terrain]),-8,8).astype(np.float32)
            self.cache.append({'x':torch.from_numpy(x),'mask':torch.from_numpy(y),
                               'valid':torch.from_numpy(valid[None]),'sample_id':row.sample_id,
                               'event_id':row.event_id,'official_split':row.official_split})
    def __len__(self):return len(self.cache)
    def __getitem__(self,i):
        b=dict(self.cache[i])
        if self.augment:
            turns=int(torch.randint(0,4,()).item());dims=[]
            if torch.rand(())<.5:dims.append(-1)
            if torch.rand(())<.5:dims.append(-2)
            for k,v in b.items():
                if isinstance(v,torch.Tensor):
                    if turns:v=torch.rot90(v,turns,(-2,-1))
                    if dims:v=v.flip(dims)
                    b[k]=v
        return b

def loader(ds,shuffle=False,generator=None):
    return DataLoader(ds,batch_size=16,shuffle=shuffle,generator=generator,num_workers=0,pin_memory=True)

def loss(z,b):
    mask=b['valid'].float();y=b['mask'];p=z.sigmoid();dims=(1,2,3)
    bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    dice=1-(2*(p*y*mask).sum(dims)+1e-6)/((p*mask).sum(dims)+(y*mask).sum(dims)+1e-6)
    return bce+.5*dice.mean()

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

def build():
    m=PaperUNet(6);m.load_state_dict(torch.load(OUT/'initial.pt',map_location='cpu',weights_only=True),strict=True)
    return m.cuda()

def prepare():
    setup();OUT.mkdir(exist_ok=True)
    if (OUT/'config.json').exists():raise FileExistsError('Config already locked')
    vc=json.loads((V2/'results/config.json').read_text());old=json.loads((OLD/'results/config.json').read_text())
    assert sha(OLD/'baseline.py')==old['source_sha256'] and sha(OLD/'results/initial.pt')==old['initial_sha256']
    assert sha(V2/'split_manifest.csv')==vc['split_sha256']
    shutil.copyfile(V2/'split_manifest.csv',HERE/'split_manifest.csv')
    d=frames();vstats=json.loads((V2/'results/normalization.json').read_text());s=np.zeros(2);s2=np.zeros(2);n=0
    for row in d['train'].itertuples():
        x,_=raw(row);valid=(x[:4]>0).all(0);a=x[4:,valid].astype(np.float64)
        s+=a.sum(1);s2+=(a*a).sum(1);n+=a.shape[1]
    mean=s/n;std=np.sqrt(np.maximum(s2/n-mean*mean,1e-6))
    stats={k:vstats[k] for k in ('sar_mean','sar_std','sar_unit','training_ids_sha256')}
    stats.update(terrain_mean=mean.tolist(),terrain_std=std.tolist(),
                 source='same locked V2 train-only SAR stats; DEM/slope mean/std fitted only on train joint-positive SAR pixels',
                 source_sar_normalization_sha256=sha(V2/'results/normalization.json'))
    write(OUT/'normalization.json',stats);shutil.copyfile(OLD/'results/initial.pt',OUT/'initial.pt')
    m=PaperUNet(6)
    assert sum(p.numel() for p in m.parameters())==31045249
    config={'method':'DLR paper U-Net, single-orbit SAR+terrain, common100 task adaptation',
            'author_exact_code':False,'paper_doi':old['paper_doi'],'architecture':'verbatim PaperUNet class imported from frozen paper reconstruction',
            'fields':FIELDS,'post_event_inputs':'one selected POST1 target orbit VV/VH, no optical',
            'pre_event_inputs':'PRE1 target VV/VH; DEM/slope as published SAR group; no pre optical',
            'orbit_id_channel':False,'counts':{k:len(v) for k,v in d.items()},'official_overlap_preserved':True,
            'epochs':100,'batch_size':16,'training_examples_per_epoch':355,'optimizer_updates_per_epoch':23,
            'augmentation':'one synchronized random rotation/flip view per patch per epoch, same policy as V2; no6x duplication',
            'optimizer':'AdamW1e-4 for every layer (all random initialized); wd.01 excluding bias/BatchNorm; betas(.9,.999),eps1e-8',
            'scheduler':'first50constant, last50cosine to1% per update; no warmup; no early stopping',
            'loss':'same V2 masked unweighted BCE+0.5 per-image foreground Dice, smooth1e-6',
            'selection':'best pooled validation SAR-numerical-valid foreground IoU every5epochs; threshold0.5',
            'test':'once after100epochs and frozen validation-selected checkpoint; official44/76 subsets',
            'metric_region':'same finite/positive four-SAR mask as V2; not physical geometry truth; all pixels secondary',
            'normalization':'same V2 train-only per-polarization SAR dB stats; train-only standardized DEM/slope; clip[-8,8]',
            'gradient_clip_norm':1.,'precision':'FP32 with TF32','seed':42,'parameters':31045249,
            'differences_from_paper_setting':['100instead200epochs','dB train standardization instead of linear minmax',
                'AdamW and declared decay schedule','BCE+0.5Dice instead of Dice only','one random view instead of six deterministic views per epoch',
                'numerical-valid main evaluation instead of all-pixel primary'],
            'differences_from_V2':['SAR+DEM/slope instead of SAR+pre optical','random native U-Net rather than pretrained Pico',
                'no OrbitID channel; no retrieval or KD; native U-Net has BatchNorm and31.05Mparameters'],
            'not_an_isolated_KD_control':'E0 versus E2 is the matched mechanism control; this U-Net is a published-architecture reference',
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OLD/'baseline.py',V2/'results/normalization.json',ROOT/'cocd/windows_main/pico_optimization_v2.py',HERE/'compare.py']},
            'split_sha256':sha(HERE/'split_manifest.csv'),'initial_sha256':sha(OUT/'initial.pt'),
            'normalization_sha256':sha(OUT/'normalization.json'),'input_h5_sha256':vc['input_h5_sha256']}
    write(OUT/'config.json',config);print('PREPARED',config['parameters'],config['counts'],flush=True)

def verify():
    c=json.loads((OUT/'config.json').read_text())
    for p,h in c['source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in c['input_h5_sha256'].items():assert sha(p)==h,p
    assert sha(HERE/'split_manifest.csv')==c['split_sha256']
    assert sha(OUT/'initial.pt')==c['initial_sha256'] and sha(OUT/'normalization.json')==c['normalization_sha256']
    return c

def smoke():
    setup();verify();seed();tr=frames()['train'];stats=json.loads((OUT/'normalization.json').read_text())
    b=cuda(next(iter(loader(Inputs(tr.iloc[:16],stats)))));m=build();m.train()
    total=loss(m(b['x']),b);total.backward();assert torch.isfinite(total)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    # Disposable train-only fit, separate from formal initial checkpoint.
    small={k:(v[:4] if isinstance(v,torch.Tensor) else v[:4]) for k,v in b.items()}
    seed();m=build();opt=optimizer(m);m.eval()
    with torch.no_grad():before=float(loss(m(small['x']),small))
    m.train()
    for _ in range(60):
        opt.zero_grad(set_to_none=True);l=loss(m(small['x']),small);l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    m.eval()
    with torch.no_grad():after=float(loss(m(small['x']),small))
    assert after<.9*before,(before,after)
    assert lr_factor(0,23)==1 and abs(lr_factor(2299,23)-.01)<1e-12
    write(OUT/'smoke_test.json',{'state':'PASS','full_batch':16,'input_shape':list(b['x'].shape),
          'train_only_fit':{'patches':4,'updates':60,'initial_loss':before,'final_loss':after},
          'test_loaded':False,'formal_checkpoint_modified':False,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()})
    print('SMOKE PASS',before,after,flush=True)

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def train():
    setup();verify();assert json.loads((OUT/'smoke_test.json').read_text())['state']=='PASS';seed()
    f=frames();stats=json.loads((OUT/'normalization.json').read_text());m=build();seed()
    opt=optimizer(m);g=torch.Generator().manual_seed(123);dl=loader(Inputs(f['train'],stats,True),True,g);vd=Inputs(f['val'],stats)
    history=[];best=-1;selected=None;start=1;step=0
    if (OUT/'latest.pt').exists():
        ck=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        history=ck['history'];best=ck['best'];selected=ck['selected'];start=ck['epoch']+1;step=ck['step']
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda']);g.set_state(ck['generator'])
    for epoch in range(start,101):
        m.train();begin=time.time();total=0;n=0
        for j,b in enumerate(dl):
            b=cuda(b)
            for group in opt.param_groups:group['lr']=1e-4*lr_factor(step,len(dl))
            opt.zero_grad(set_to_none=True);l=loss(m(b['x']),b);assert torch.isfinite(l);l.backward()
            grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
            total+=float(l.detach())*len(b['x']);n+=len(b['x'])
            write(OUT/'progress.json',{'state':'TRAINING','model':'DLR_U-Net_100ep','epoch':epoch,'epochs':100,
                  'batch':j+1,'batches':len(dl),'train_loss':float(l.detach()),'best_val_IoU':best,'selected_epoch':selected,'test_loaded':False})
        score=evaluate(m,vd) if epoch%5==0 else None
        if score is not None and score['IoU']>best:best=score['IoU'];selected=epoch;torch.save(m.state_dict(),OUT/'best.pt')
        history.append({'epoch':epoch,'train_seg_loss':total/n,'validation':score,'lr':opt.param_groups[0]['lr'],
                        'last_preclip_gradient_norm':float(grad),'seconds':time.time()-begin,'optimizer_steps':step})
        write(OUT/'history.json',history)
        torch.save({'epoch':epoch,'step':step,'model':m.state_dict(),'optimizer':opt.state_dict(),'history':history,
                    'best':best,'selected':selected,'rng':rng(),'generator':g.get_state()},OUT/'latest.tmp')
        (OUT/'latest.tmp').replace(OUT/'latest.pt')
        print('UNet100',epoch,'loss',round(total/n,5),'valIoU',None if score is None else round(score['IoU'],5),'seconds',round(time.time()-begin,1),flush=True)
    assert step==2300 and selected is not None
    write(OUT/'selection.json',{'state':'FROZEN','epochs_completed':100,'selected_epoch':selected,'validation_IoU':best,
          'checkpoint_sha256':sha(OUT/'best.pt'),'selected_by':'validation foreground IoU at scheduled epochs; no test use'})
    if not (OUT/'test_metrics.json').exists():
        if (OUT/'test_started.json').exists():raise RuntimeError('Interrupted final test; inspect before a repeat')
        write(OUT/'test_started.json',{'state':'FINAL_TEST_AFTER_FREEZE','checkpoint_sha256':sha(OUT/'best.pt')})
        m.load_state_dict(torch.load(OUT/'best.pt',weights_only=True))
        result=evaluate(m,Inputs(f['test'],stats),OUT/'test_predictions.npz');write(OUT/'test_metrics.json',result)
    write(OUT/'progress.json',{'state':'COMPLETE','epochs_completed':100,'selected_epoch':selected,'test_once':True})
    import compare
    compare.main()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);a=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'train':train}[a.action]()

if __name__=='__main__':
    try:main()
    except Exception as e:write(OUT/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
