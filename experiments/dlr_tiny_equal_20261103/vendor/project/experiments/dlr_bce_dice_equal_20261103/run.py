"""User-authorized 1:1 supervised loss rerun; old modules remain immutable."""
from pathlib import Path
import sys,importlib.util,json,shutil,torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
SOURCE=ROOT/'experiments/sar_relation_frequency_20261103'
sys.path.insert(0,str(SOURCE))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ab=load('equal_ab',SOURCE/'run.py');b=ab.b
from model_kd import OpticalTeacher
native=load('equal_native',ROOT/'experiments/dlr_unet_100ep_common_20261102/run.py')
ab.OUT=OUT;b.OUT=OUT;b.HERE=HERE
b.TEACHERS={'multimodal':'change-then-fuse','optical':'optical only'}
b.ARMS={'base':{'teacher':None},'A':{'teacher':'multimodal'},'B':{'teacher':'multimodal'}}

def seg(z,y,valid):
 v=valid.float();p=z.sigmoid();dims=(1,2,3)
 ce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*v).sum()/v.sum().clamp_min(1)
 dice=1-(2*(p*y*v).sum(dims)+1e-6)/((p*v).sum(dims)+(y*v).sum(dims)+1e-6)
 return ce+dice.mean()
b.seg_loss=seg
def teacher(kind):
 b.seed();m=OpticalTeacher(b.base_initial()) if kind=='optical' else ab.Teacher(b.base_initial())
 m.load_state_dict(torch.load(OUT/f'teacher_{kind}_initial.pt',map_location='cpu',weights_only=True));return m.cuda()
b.build_teacher=teacher

def verify():
 c=json.loads((OUT/'config.json').read_text())
 for p,h in c['sources'].items():assert b.sha(ROOT/p)==h,p
 for p,h in c['initials'].items():assert b.sha(OUT/p)==h,p
 assert b.sha(HERE/'split_manifest.csv')==c['split_sha256']
 return c

def prepare():
 b.setup();OUT.mkdir(exist_ok=True);assert not (OUT/'config.json').exists()
 shutil.copyfile(SOURCE/'split_manifest.csv',HERE/'split_manifest.csv')
 shutil.copyfile(SOURCE/'results/normalization.json',OUT/'normalization.json')
 shutil.copyfile(SOURCE/'results/student_initial.pt',OUT/'student_initial.pt')
 # Reuse clean INITIAL states, never any selected/trained model.
 shutil.copyfile(SOURCE/'results/teacher_multimodal_initial.pt',OUT/'teacher_multimodal_initial.pt')
 shutil.copyfile(ROOT/'experiments/sar_only_kd_literature_20261103/results/teacher_optical_initial.pt',OUT/'teacher_optical_initial.pt')
 nd=OUT/'unet';nd.mkdir()
 for name in ['initial.pt','normalization.json']:
  shutil.copyfile(ROOT/'experiments/dlr_unet_100ep_common_20261102/results'/name,nd/name)
 shutil.copyfile(HERE/'split_manifest.csv',nd/'split_manifest.csv')
 files=[Path(__file__),SOURCE/'run.py',SOURCE/'method.py',ROOT/'experiments/sar_only_kd_literature_20261103/run.py',ROOT/'experiments/sar_only_kd_literature_20261103/model_kd.py',ROOT/'experiments/dlr_unet_100ep_common_20261102/run.py']
 c={'date':'2026-11-03','loss':'1*BCEWithLogits +1*per-image foreground SoftDice, smooth1e-6, valid pixels, no positive reweighting',
 'queue':['T_multimodal','T_optical','base','A','B','unet'],'epochs_each':100,'validation_every':5,'early_stopping':False,'seed':42,
 'schedule':'unchanged AdamW encoder2e-5/new1e-4; native random U-Net all1e-4;50constant50cosine-to1%',
 'distillation':'A/B same new multimodal Teacher, training-only gradient recalibration after first epoch, unchanged target shares',
 'split_sha256':b.sha(HERE/'split_manifest.csv'),'baseline_reused':False,'trained_weights_reused':False,'train_normalization_reused':'same unchanged355train IDs and raw data; no new fitting to val/test',
 'initials':{str(p.relative_to(OUT)):b.sha(p) for p in [OUT/'student_initial.pt',OUT/'teacher_optical_initial.pt',OUT/'teacher_multimodal_initial.pt',nd/'initial.pt']},
 'sources':{str(p.relative_to(ROOT)):b.sha(p) for p in files}}
 b.write(OUT/'config.json',c)

def check():
 b.setup();verify();stats=json.loads((OUT/'normalization.json').read_text());f=b.frames()
 batch=b.batch_cuda(next(iter(b.loader(b.DLRInputs(f['train'].iloc[:4],stats,include_post=True)))))
 # Direct numeric verification of equal coefficients.
 z=torch.zeros_like(batch['mask']);ce=F.binary_cross_entropy_with_logits(z,batch['mask'],reduction='none');v=batch['valid'].float()
 ce=(ce*v).sum()/v.sum();p=z.sigmoid();y=batch['mask'];d=1-(2*(p*y*v).sum((1,2,3))+1e-6)/((p*v).sum((1,2,3))+(y*v).sum((1,2,3))+1e-6)
 assert torch.allclose(seg(z,y,v),ce+d.mean())
 fits={}
 for kind in ['multimodal','optical']:
  m=teacher(kind);opt=ab.optimizer(m);m.eval()
  with torch.no_grad():before=float(seg(b.teacher_forward(kind,m,batch)['z'],y,v))
  m.train()
  for _ in range(30):
   opt.zero_grad(set_to_none=True);l=seg(b.teacher_forward(kind,m,batch)['z'],y,v);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  m.eval()
  with torch.no_grad():after=float(seg(b.teacher_forward(kind,m,batch)['z'],y,v))
  assert after<.95*before;fits[kind]=[before,after];del m,opt;torch.cuda.empty_cache()
 b.write(OUT/'checks.json',{'state':'PASS','equal_coefficient_numeric_check':True,'train_only_fits':fits,'test_loaded':False,'formal_initials_changed':False})

def native_train():
 import types
 sys.modules['compare']=types.SimpleNamespace(main=lambda:None)
 native.HERE=OUT/'unet';native.OUT=OUT/'unet';native.verify=verify
 native.loss=lambda z,bb:seg(z,bb['mask'],bb['valid'])
 native.smoke();native.train()

def finish():
 import inspect,numpy as np,pandas as pd
 # Preserve old report as well as all old results; redirect only this report target.
 code=inspect.getsource(ab.final_tests).replace('SAR_RELATION_FREQUENCY_RESULTS.md','DLR_BCE_DICE_EQUAL_AB_DETAIL.md')
 exec(code,ab.__dict__);ab.final_tests()
 f=b.frames();stats=json.loads((OUT/'normalization.json').read_text())
 for kind in ['multimodal','optical']:
  dest=OUT/('T_'+kind)
  if (dest/'test_metrics.json').exists():continue
  assert not (dest/'test_started.json').exists()
  b.write(dest/'test_started.json',{'all_six_frozen':True,'checkpoint_sha256':b.sha(dest/'best.pt')})
  t=b.freeze_teacher(kind);ys=[];ps=[]
  with torch.no_grad():
   for batch in b.loader(b.DLRInputs(f['test'],stats,include_post=True)):
    batch=b.batch_cuda(batch);prob=b.teacher_forward(kind,t,batch)['z'].sigmoid();v=batch['valid'].bool()
    ys.append(batch['mask'][v].cpu().numpy()>.5);ps.append(prob[v].cpu().numpy())
  b.write(dest/'test_metrics.json',b.metrics(np.concatenate(ys),np.concatenate(ps)));del t;torch.cuda.empty_cache()
 rows=[]
 for arm in ['unet','T_multimodal','T_optical','base','A','B']:
  m=json.loads((OUT/arm/'test_metrics.json').read_text());rows.append({'model':arm,**{k:m[k] for k in ['IoU','F1','AUPRC','Precision','Recall']}})
 df=pd.DataFrame(rows);df.to_csv(OUT/'all_six_comparison.csv',index=False)
 (ROOT/'reports/DLR_BCE_DICE_EQUAL_RESULTS.md').write_text('# BCE:Dice =1:1 rerun\n\n'+df.to_markdown(index=False)+'\n\nAll six trained clean100epochs, validation-selected. U-Net terrain input differs from SAR-only Students. Prior losses/results remain separate.',encoding='utf-8')
 b.write(OUT/'progress.json',{'state':'ALL_SIX_COMPLETE','loss':'BCE+Dice','models':[r['model'] for r in rows]})

if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='check':check()
  elif action=='run':
   b.setup();verify();assert json.loads((OUT/'checks.json').read_text())['state']=='PASS'
   for kind in ['multimodal','optical']:b.train_teacher(kind)
   for arm in ['base','A','B']:b.train_arm(arm)
   # Native runner tests after all six validation selections freeze.
   native_train();finish()
 except Exception as e:
  import traceback
  b.write(OUT/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
