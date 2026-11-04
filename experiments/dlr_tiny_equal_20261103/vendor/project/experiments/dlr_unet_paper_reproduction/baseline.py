"""Paper-based DLR U-Net reference, not an official-author-code rerun.
One selected pre/post VV/VH + elevation/slope, published split, no optical.
All departures/unspecified original parameters recorded before fitting.
"""
from pathlib import Path
import argparse,json,hashlib,time,math,random,sys,gc,traceback
import numpy as np,pandas as pd,h5py,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
FIELDS=['PRE1_gVV','PRE1_gVH','POST1_gVV','POST1_gVH','None_DEM','None_SLOPE']
EPOCHS=200;BATCH=16

def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 for attempt in range(10):
  try:t.replace(p);return
  except PermissionError:
   if attempt==9:raise
   time.sleep(.02)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def setup():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
 assert torch.cuda.is_available()
class DoubleConv(nn.Module):
 def __init__(self,cin,cout):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(),nn.BatchNorm2d(cout),nn.Conv2d(cout,cout,3,padding=1),nn.ReLU(),nn.BatchNorm2d(cout))
 def forward(self,x):return self.net(x)
class PaperUNet(nn.Module):
 """Figure7:64/128/256/512/1024, four pools, four transpose-conv skip decoders."""
 def __init__(self,channels=6):
  super().__init__();widths=[64,128,256,512,1024]
  self.enc=nn.ModuleList([DoubleConv(a,b) for a,b in zip([channels]+widths[:-1],widths)])
  self.up=nn.ModuleList([nn.ConvTranspose2d(a,b,2,2) for a,b in zip(widths[:0:-1],widths[-2::-1])])
  self.dec=nn.ModuleList([DoubleConv(2*b,b) for b in widths[-2::-1]]);self.head=nn.Conv2d(64,1,1)
 def forward(self,x):
  features=[]
  for i,block in enumerate(self.enc):
   if i:x=F.max_pool2d(x,2)
   x=block(x);features.append(x)
  for up,block,skip in zip(self.up,self.dec,features[-2::-1]):x=block(torch.cat([skip,up(x)],1))
  return self.head(x) # sigmoid is applied inside Soft Dice or evaluation

def read_row(row):
 with h5py.File(row.h5_path,'r') as f:
  x=np.stack([f[k][int(row.h5_index),0] for k in FIELDS]).astype(np.float32);y=f['None_MASK'][int(row.h5_index)].astype(np.float32)
 return x,y
class Inputs(Dataset):
 def __init__(self,frame,stats,aug=False):
  self.frame=frame.reset_index(drop=True);self.aug=aug;self.cache=[]
  lo=np.array(stats['min'],np.float32)[:,None,None];hi=np.array(stats['max'],np.float32)[:,None,None]
  for row in self.frame.itertuples():
   raw,y=read_row(row);assert np.isfinite(raw).all() and set(np.unique(y))<={0.,1.}
   numerical=(raw[:4]>0).all(0)[None]
   x=np.clip((raw-lo)/np.maximum(hi-lo,1e-8),0,1)
   self.cache.append((torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(numerical),row.sample_id,row.event_id,row.official_split))
 def __len__(self):return len(self.cache)*(6 if self.aug else 1)
 def __getitem__(self,i):
  base=i//6 if self.aug else i;transform=i%6 if self.aug else 0
  x,y,v,*meta=self.cache[base]
  def apply(a):
   if transform==1:return a.flip(-1)
   if transform==2:return a.flip(-2)
   if transform>=3:return a.rot90(transform-2,(-2,-1))
   return a
  return apply(x),apply(y),apply(v),*meta
def loader(ds,shuffle=False,generator=None):return DataLoader(ds,batch_size=BATCH,shuffle=shuffle,generator=generator,num_workers=0,pin_memory=True)
def dice(z,y):
 p=z.sigmoid();intersection=(p*y).sum();return 1-(2*intersection+1e-6)/(p.sum()+y.sum()+1e-6)
def metric(y,p):
 y=np.asarray(y,dtype=bool);pred=p>=.5;tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'IoU':tp/max(tp+fp+fn,1),'F1':2*tp/max(2*tp+fp+fn,1),'Precision':tp/max(tp+fp,1),'Recall':tp/max(tp+fn,1),'AUPRC':float(average_precision_score(y,p)) if y.any() and (~y).any() else None,'foreground_fraction':float(y.mean()),'predicted_positive_fraction':float(pred.mean())}
@torch.no_grad()
def evaluate(m,ds,prediction_path=None):
 m.eval();ys=[];pp=[];valids=[];events={};subsets={};records=[];saved=[];ids=[]
 for x,y,v,sids,eids,subs in loader(ds):
  p=m(x.cuda()).sigmoid().cpu().numpy()[:,0];y=y.numpy()[:,0]>.5;v=v.numpy()[:,0]
  for i in range(len(y)):
   a=y[i].reshape(-1);b=p[i].reshape(-1);ys.append(a);pp.append(b);valids.append(v[i].reshape(-1));events.setdefault(eids[i],[]).append((a,b));subsets.setdefault(subs[i],[]).append((a,b))
   records.append({'sample_id':sids[i],'event_id':eids[i],'official_subset':subs[i],**metric(a,b)})
   if prediction_path is not None:saved.append(p[i]);ids.append(sids[i])
 y=np.concatenate(ys);p=np.concatenate(pp);v=np.concatenate(valids)
 result={'all_pixels':metric(y,p),'SAR_numerical_valid_secondary':metric(y[v],p[v]),'per_event':{k:metric(np.concatenate([x[0] for x in a]),np.concatenate([x[1] for x in a])) for k,a in events.items()},'per_official_subset':{k:metric(np.concatenate([x[0] for x in a]),np.concatenate([x[1] for x in a])) for k,a in subsets.items()},'patches':len(ds)}
 result['macro_event_F1']=float(np.mean([x['F1'] for x in result['per_event'].values()]));result['macro_event_IoU']=float(np.mean([x['IoU'] for x in result['per_event'].values()]))
 if prediction_path is not None:
  np.savez_compressed(prediction_path,sample_ids=np.array(ids),probability=np.stack(saved));pd.DataFrame(records).to_csv(prediction_path.with_suffix('.samples.csv'),index=False)
 return result

def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'config.json').exists():raise FileExistsError('Already prepared')
 source=ROOT/'experiments/dlr_landslide_audit/audit/sample_manifest.csv';d=pd.read_csv(source);d.to_csv(HERE/'split_manifest.csv',index=False,encoding='utf-8-sig')
 lo=np.full(6,np.inf);hi=np.full(6,-np.inf)
 for row in d[d.official_split=='train'].itertuples():
  x,_=read_row(row);assert np.isfinite(x).all();lo=np.minimum(lo,x.min((1,2)));hi=np.maximum(hi,x.max((1,2)))
 stats={'fields':FIELDS,'min':lo.tolist(),'max':hi.tolist(),'source':'355official training patches only','transform':'linear Gamma0 + DEM/slope, channelwise train min-max, clip0..1; no dB conversion'};write(OUT/'normalization.json',stats)
 seed();m=PaperUNet();torch.save(m.state_dict(),OUT/'initial.pt')
 config={'description':'Paper-based reference reconstruction; NOT author-source-code exact rerun','paper_doi':'10.1080/15481603.2025.2502214','paper_figure':7,'setting':'Table1 gVV+gVH,1pre+1post,plus DEM/slope','fields':FIELDS,'optical_input':False,'orbits':'one supplied orbit direction per event, ASC or DES; no counter track','counts':d.official_split.value_counts().to_dict(),'augmented_training_instances':2130,'training_augmentation':['identity','horizontal_flip','vertical_flip','rot90','rot180','rot270'],'split_sha256':sha(HERE/'split_manifest.csv'),'known_overlap':'Published split kept as user explicitly requested;365train/val+2train/dependent-test pairs','parameters':sum(p.numel() for p in m.parameters()),'widths':[64,128,256,512,1024],'spatial_sizes':[128,64,32,16,8],'conv':'two3x3 conv-ReLU-BatchNorm per block; same padding','up':'four2x2 transposed-conv + skip concat blocks','seed':42,'batch':16,'epochs':200,'validation_every':5,'early_stopping':False,'loss':'foreground soft Dice on batch-pooled pixels; smooth1e-6','optimizer':'Adam,lr1e-4,betas(.9,.999),eps1e-8,wd0','scheduler':'constant','selection':'highest official validation all-pixel IoU every5epochs','threshold':.5,'primary_metric_region':'all128x128 finite binary-label pixels; numerical SAR-valid supplementary result reported separately','test':'once after200epochs and validation selection locked; report44independent and76dependent subsets separately','precision':'FP32 with TF32','unspecified_original_choices':['optimizer type and betas/weight decay are not reported; Adam chosen explicitly','scheduler not reported; constant selected','normalization fitted scope not reported; training-only minmax selected','checkpoint selection and validation cadence not reported; valIoU every5 selected','Soft Dice reduction/smoothing not specified; batch-pooled foreground Dice smooth1e-6 selected','conv normalization order follows conv-ReLU-BN wording; no official source verification possible'],'source_sha256':sha(Path(__file__)),'initial_sha256':sha(OUT/'initial.pt'),'input_h5_sha256':{p:sha(p) for p in sorted(d.h5_path.unique())}}
 write(OUT/'config.json',config);print('PREPARED',config['parameters'],config['counts'],flush=True)
def frames():
 d=pd.read_csv(HERE/'split_manifest.csv');return d[d.official_split=='train'],d[d.official_split=='val'],d[d.official_split.isin(['testind','testspt'])]
def verify():
 c=json.loads((OUT/'config.json').read_text());assert sha(Path(__file__))==c['source_sha256'];assert sha(HERE/'split_manifest.csv')==c['split_sha256'];assert sha(OUT/'initial.pt')==c['initial_sha256']
 for path,h in c['input_h5_sha256'].items():assert sha(path)==h
 return c
def smoke():
 setup();verify();tr,_,_=frames();stats=json.loads((OUT/'normalization.json').read_text());ds=Inputs(tr.iloc[:16],stats);x,y,*_=next(iter(loader(ds)))
 m=PaperUNet().cuda();m.load_state_dict(torch.load(OUT/'initial.pt',weights_only=True));m.train();z=m(x.cuda());loss=dice(z,y.cuda());loss.backward();assert z.shape==y.shape and torch.isfinite(loss)
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 aug=Inputs(tr.iloc[:1],stats,aug=True);assert len(aug)==6
 write(OUT/'smoke_test.json',{'state':'PASS','input_shape':list(x.shape),'output_shape':list(z.shape),'loss':float(loss.detach()),'peak_memory_bytes':torch.cuda.max_memory_allocated(),'official_training_samples_only':True,'test_loaded':False});print('SMOKE PASS',float(loss.detach()),flush=True)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def train():
 setup();c=verify();assert json.loads((OUT/'smoke_test.json').read_text())['state']=='PASS';tr,va,te=frames();stats=json.loads((OUT/'normalization.json').read_text());seed()
 ds=Inputs(tr,stats,True);vd=Inputs(va,stats);g=torch.Generator().manual_seed(123);dl=loader(ds,True,g);m=PaperUNet().cuda();m.load_state_dict(torch.load(OUT/'initial.pt',weights_only=True))
 opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=0);history=[];best=-1;selected=None;start=1
 if (OUT/'latest.pt').exists():
  ck=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];best=ck['best'];selected=ck['selected'];start=ck['epoch']+1
  random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda']);g.set_state(ck['generator'])
 for epoch in range(start,201):
  m.train();begin=time.time();total=0;n=0
  for j,(x,y,*_) in enumerate(dl):
   x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True);loss=dice(m(x),y);assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss.detach())*len(y);n+=len(y)
   if (j+1)%5==0 or j+1==len(dl):write(OUT/'progress.json',{'state':'TRAINING','model':'DLR_U-Net_paper_based_SAR_DEM','epoch':epoch,'epochs':200,'batch':j+1,'batches':len(dl),'train_soft_dice_loss':float(loss.detach()),'best_val_IoU':best,'selected_epoch':selected,'test_loaded':False})
  score=evaluate(m,vd) if epoch%5==0 else None
  if score is not None and score['all_pixels']['IoU']>best:best=score['all_pixels']['IoU'];selected=epoch;torch.save(m.state_dict(),OUT/'best.pt')
  history.append({'epoch':epoch,'train_soft_dice_loss':total/n,'validation':score,'seconds':time.time()-begin,'lr':opt.param_groups[0]['lr']});write(OUT/'history.json',history)
  torch.save({'epoch':epoch,'model':m.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best,'selected':selected,'rng':rng(),'generator':g.get_state()},OUT/'latest.tmp');(OUT/'latest.tmp').replace(OUT/'latest.pt')
  print('UNet',epoch,'loss',round(total/n,5),'valIoU',None if score is None else round(score['all_pixels']['IoU'],5),'seconds',round(time.time()-begin,1),flush=True)
 write(OUT/'selection.json',{'state':'FROZEN','epochs_completed':200,'selected_epoch':selected,'validation_IoU':best,'checkpoint_sha256':sha(OUT/'best.pt'),'selected_by':'validation all-pixel IoU, no test use'})
 if not (OUT/'test_metrics.json').exists():
  if (OUT/'test_started.json').exists():raise RuntimeError('Interrupted test; inspect before repeating')
  write(OUT/'test_started.json',{'state':'FINAL_TEST_AFTER_FREEZE','checkpoint_sha256':sha(OUT/'best.pt')});td=Inputs(te,stats);m.load_state_dict(torch.load(OUT/'best.pt',weights_only=True));result=evaluate(m,td,OUT/'test_predictions.npz');write(OUT/'test_metrics.json',result)
 write(OUT/'progress.json',{'state':'COMPLETE','epochs_completed':200,'selected_epoch':selected,'test_once':True});report()
def report():
 c=json.loads((OUT/'config.json').read_text());test=json.loads((OUT/'test_metrics.json').read_text()) if (OUT/'test_metrics.json').exists() else None
 s='# DLR U-Net paper-based SAR reference\n\n'+('COMPLETE' if test else 'TRAINING / NOT_YET_EVALUATED')+'\n\n'
 s+='This reconstructs the authors Table1 single-orbit gVV+gVH1pre/1post setting with DEM/slope, using the published HDF5 split. No optical is required at deployment. It is not a strictly SAR-only input budget because terrain layers are retained. No author training source was located; unspecified choices below are explicitly local choices, not attributed to the authors.\n\n'
 s+='Figure7 uses64/128/256/512/1024channels, four pools to8x8, and four transpose-convolution skip decoders. The schematic resolves the prose ambiguity about pooling after every encoder block. Training follows reported200epochs,batch16,lr1e-4,SoftDice and6augmentation instances per original training patch. This is separate from the100epoch COCD experiment and must not be ranked as an equal-recipe KD comparison.\n\n'
 s+='The released HDF5 DEM/slope are used unchanged, including the known KG0002 discrepancy from source TIFF crops. All pixels form the primary reference metric; SAR numerical validity is supplementary. No split membership, input fields or labels were changed according to scores.\n\n```json\n'+json.dumps(c,indent=2)+'\n```\n\n'
 if test:s+='## Measured final results\n\n```json\n'+json.dumps(test,indent=2)+'\n```\n'
 else:s+='Final test: NOT_RUN. No published number or current COCD number is substituted.\n'
 (ROOT/'reports/DLR_UNET_PAPER_REPRODUCTION.md').write_text(s,encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','report']);a=p.parse_args()
 {'prepare':prepare,'smoke':smoke,'train':train,'report':report}[a.action]()
if __name__=='__main__':
 try:main()
 except Exception as e:write(OUT/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
