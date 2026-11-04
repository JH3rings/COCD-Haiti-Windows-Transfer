"""Isolated experiment; reuse audited lifecycle without rewriting old sources."""
from pathlib import Path
import sys,importlib.util,json,shutil,gc
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(HERE));sys.path.append(str(ROOT/'experiments/sar_only_kd_literature_20261103'))
spec=importlib.util.spec_from_file_location('legacy_lifecycle',ROOT/'experiments/sar_only_kd_literature_20261103/run.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
from method import Teacher,System,terms
b.HERE=HERE;b.OUT=OUT
b.ARMS={'base':{'teacher':None},'A':{'teacher':'multimodal'},'B':{'teacher':'multimodal'}}
b.TEACHERS={'multimodal':'Separate optical/SAR change then concatenate, no residual or retrieval'}
b.SHARES={'base':{},'A':{'bk':.1,'freq':.08,'jscad':.03,'kd':.04},'B':{'rel':.1,'freqrel':.01}}
b.aux_terms=terms

def build_teacher(kind):
 b.seed();m=Teacher(b.base_initial())
 if (OUT/'teacher_multimodal_initial.pt').exists():m.load_state_dict(torch.load(OUT/'teacher_multimodal_initial.pt',weights_only=True,map_location='cpu'))
 return m.cuda()
def build_system(arm):
 b.seed();m=System(b.base_initial(),arm);m.student.load_state_dict(torch.load(OUT/'student_initial.pt',weights_only=True,map_location='cpu'));return m.cuda()
b.build_teacher=build_teacher;b.build_system=build_system
old_opt=b.make_optimizer
def optimizer(module,prefix=''):
 opt=old_opt(module,prefix)
 # Both Teacher encoders are pretrained; move SAR encoder parameters into same LR/WD groups.
 if isinstance(module,Teacher):
  for group in list(opt.param_groups):
   move=[i for i,n in enumerate(group['names']) if n.startswith(('sar_backbone.stem.','sar_backbone.stages.'))]
   if not move:continue
   params=[group['params'][i] for i in move];names=[group['names'][i] for i in move]
   group['params']=[p for i,p in enumerate(group['params']) if i not in move];group['names']=[n for i,n in enumerate(group['names']) if i not in move]
   opt.add_param_group(dict(params=params,names=names,group_name='sar_encoder_'+group['group_name'],lr=2e-5,peak_lr=2e-5,weight_decay=group['weight_decay'],schedule_step=0))
 return opt
b.make_optimizer=optimizer
def calibrate(arm,system,teacher,kind,batches):
 system.eval();norms={k:0. for k in ['seg',*b.SHARES[arm]]}
 params=[p for p in system.student.parameters() if p.requires_grad]
 for batch in batches:
  o=system(batch['sar'])
  with torch.no_grad():to=b.teacher_forward(kind,teacher,batch)
  ts,_=terms(arm,system,o,to,batch,{})
  vals={'seg':b.seg_loss(o['z'],batch['mask'],batch['valid']),**ts}
  for k,l in vals.items():
   grads=torch.autograd.grad(l,params,retain_graph=True,allow_unused=True)
   norms[k]+=float(sum(g.square().sum() for g in grads if g is not None).sqrt())/len(batches)
 assert all(x>1e-12 for x in norms.values()),norms
 lam={k:min(1000.,share*norms['seg']/norms[k]) for k,share in b.SHARES[arm].items()}
 return {'gradient_norms':norms,'shares':b.SHARES[arm],'lambda':lam,'scope':'first64 TRAIN samples; shared deployed Student gradients; fixed after calibration'}
b.calibrate=calibrate

def prepare():
 OUT.mkdir(exist_ok=True);assert not (OUT/'config.json').exists()
 old=ROOT/'experiments/sar_only_kd_literature_20261103'
 for source,dest in [(old/'split_manifest.csv',HERE/'split_manifest.csv'),(old/'results/normalization.json',OUT/'normalization.json'),(old/'results/student_initial.pt',OUT/'student_initial.pt')]:shutil.copyfile(source,dest)
 # Reuse the exact matched baseline, including its already completed test. No re-evaluation.
 (OUT/'base').mkdir()
 for name in ['best.pt','selection.json','test_metrics.json','test_predictions.npz','history.json']:shutil.copyfile(old/'results/base'/name,OUT/'base'/name)
 b.seed();t=Teacher(b.base_initial());torch.save(t.state_dict(),OUT/'teacher_multimodal_initial.pt')
 config={'arms':b.ARMS,'teachers':b.TEACHERS,'shares':b.SHARES,'epochs':100,'validation_every':5,'seed':42,
 'student_initial_sha256':b.sha(OUT/'student_initial.pt'),'baseline_reused_from':str(old/'results/base'),
 'split_sha256':b.sha(HERE/'split_manifest.csv'),'normalization_sha256':b.sha(OUT/'normalization.json'),
 'sources':{str(p.relative_to(ROOT)):b.sha(p) for p in [HERE/'run.py',HERE/'method.py',old/'run.py',old/'model_kd.py']},
 'teacher_reuse_rejected':'old teachers fuse before change or use residuals, incompatible with new specification',
 'teacher_parameters':sum(p.numel() for p in t.parameters()),'student_parameters':sum(p.numel() for p in build_system('base').student.parameters())}
 b.write(OUT/'config.json',config)

def check():
 b.setup();f=b.frames();stats=json.loads((OUT/'normalization.json').read_text());batch=b.batch_cuda(next(iter(b.loader(b.DLRInputs(f['train'].iloc[:4],stats,include_post=True)))))
 t=build_teacher('multimodal').eval();t.requires_grad_(False);checks={}
 for arm in ['A','B']:
  m=build_system(arm);o=m(batch['sar']);o['feat_pre'][0].retain_grad();o['feat_post'][0].retain_grad()
  with torch.no_grad():to=b.teacher_forward('multimodal',t,batch)
  ts,l=terms(arm,m,o,to,batch,{k:1 for k in b.SHARES[arm]});l.backward()
  g=[float(o[k][0].grad.norm()) for k in ['feat_pre','feat_post']];assert min(g)>0 and torch.isfinite(l)
  checks[arm]={'losses':{k:float(v.detach()) for k,v in ts.items()},'both_SAR_path_gradients':g};del m,o,l,ts
 assert all(p.grad is None for p in t.parameters())
 # Small fitting on TRAIN only, no formal state changes.
 m=build_teacher('multimodal');opt=optimizer(m);m.eval()
 with torch.no_grad():before=float(b.seg_loss(b.teacher_forward('multimodal',m,batch)['z'],batch['mask'],batch['valid']))
 m.train()
 for _ in range(40):
  opt.zero_grad(set_to_none=True);loss=b.seg_loss(b.teacher_forward('multimodal',m,batch)['z'],batch['mask'],batch['valid']);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
 m.eval()
 with torch.no_grad():after=float(b.seg_loss(b.teacher_forward('multimodal',m,batch)['z'],batch['mask'],batch['valid']))
 assert after<.9*before,(before,after)
 b.write(OUT/'checks.json',{'state':'PASS','arms':checks,'teacher_train_fit':[before,after],'teacher_gradients_KD':'none','test_loaded':False})

def final_tests():
 import numpy as np,pandas as pd
 f=b.frames();stats=json.loads((OUT/'normalization.json').read_text())
 for arm in b.ARMS:
  s=json.loads((OUT/arm/'selection.json').read_text());assert s['epochs_completed']==100 and b.sha(OUT/arm/'best.pt')==s['checkpoint_sha256']
 for arm in ['A','B']:
  if (OUT/arm/'test_metrics.json').exists():continue
  assert not (OUT/arm/'test_started.json').exists()
  b.write(OUT/arm/'test_started.json',{'all_frozen':True,'checkpoint_sha256':b.sha(OUT/arm/'best.pt')})
  m=build_system(arm);m.load_state_dict(torch.load(OUT/arm/'best.pt',weights_only=True))
  score=b.evaluate(arm,m,b.DLRInputs(f['test'],stats,include_post=False),save=OUT/arm/'test_predictions.npz');b.write(OUT/arm/'test_metrics.json',score);del m;torch.cuda.empty_cache()
 rows=[]
 for arm in b.ARMS:
  m=json.loads((OUT/arm/'test_metrics.json').read_text());rows.append({'arm':arm,**{k:m[k] for k in ['IoU','F1','AUPRC','Precision','Recall']},'small_pixel_recall':m['component_sizes']['<=100px']['detected_pixels']/max(m['component_sizes']['<=100px']['positive_pixels'],1),'boundary_F1':m['boundary_band_F1_mean'],'roughness':m['prediction_roughness_TN_mean']})
 df=pd.DataFrame(rows);df.to_csv(OUT/'comparison.csv',index=False)
 strata=b.weak_change_strata();strata.to_csv(OUT/'test_strata.csv',index=False)
 valid=strata.dropna(subset=['contrast_db']).sort_values('contrast_db');weak=set(valid.iloc[:max(len(valid)//3,1)].sample_id)
 details=[]
 for arm in b.ARMS:
  z=np.load(OUT/arm/'test_predictions.npz');pairs=[(z['label'][i][z['valid'][i]],z['probability'][i][z['valid'][i]]) for i,sid in enumerate(z['sample_ids']) if str(sid) in weak]
  details.append({'arm':arm,**b.metrics(np.concatenate([x[0] for x in pairs]),np.concatenate([x[1] for x in pairs]))})
 pd.DataFrame(details).to_csv(OUT/'weak_change_results.csv',index=False)
 text='# A/B actual results\n\n'+df.to_markdown(index=False)+'\n\nFrequency contribution cannot be isolated without a separate removal arm. Boundary metrics are outcome diagnostics, not proof of frequency causality. Test comparisons are exploratory; models selected only on validation.\n'
 (ROOT/'reports/SAR_RELATION_FREQUENCY_RESULTS.md').write_text(text,encoding='utf-8')
 b.write(OUT/'progress.json',{'state':'COMPLETE','arms':list(b.ARMS)})

if __name__=='__main__':
 try:
  b.setup();action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='check':check()
  elif action=='run':
   c=json.loads((OUT/'config.json').read_text())
   for p,h in c['sources'].items():assert b.sha(ROOT/p)==h,p
   assert b.sha(HERE/'split_manifest.csv')==c['split_sha256'] and b.sha(OUT/'student_initial.pt')==c['student_initial_sha256']
   assert json.loads((OUT/'checks.json').read_text())['state']=='PASS'
   b.train_teacher('multimodal')
   for arm in ['A','B']:b.train_arm(arm)
   final_tests()
 except Exception as e:
  import traceback
  b.write(OUT/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
