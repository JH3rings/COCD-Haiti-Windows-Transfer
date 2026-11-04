from pathlib import Path
import sys,importlib.util,json,shutil,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
VENDOR=ROOT/'experiments/dlr_tiny_equal_20261103/vendor/project'
spec=importlib.util.spec_from_file_location('frozen_lifecycle',VENDOR/'experiments/dlr_bce_dice_equal_20261103/run.py');eq=importlib.util.module_from_spec(spec);spec.loader.exec_module(eq)
b=eq.b;ab=eq.ab;b.ROOT=ROOT;ab.ROOT=ROOT;eq.ROOT=ROOT
import model_kd,method
from model import V2Encoder,MFSAD,losses,frequency
OriginalEncoder=model_kd.PicoEncoder
model_kd.PicoEncoder=V2Encoder
b.OUT=OUT;b.HERE=HERE;ab.OUT=OUT;eq.OUT=OUT
b.ARMS={'base':{'teacher':None},'mfsad':{'teacher':'multimodal'}}
b.SHARES={'base':{},'mfsad':{'bk':.1,'freq':.08,'jscad':.03,'kd':.04}}
b.TEACHERS={};b.aux_terms=losses

def build_system(arm):
 b.seed();m=MFSAD(model_kd.SARStudent(None),arm)
 m.student.load_state_dict(torch.load(OUT/'student_initial.pt',map_location='cpu',weights_only=True))
 return m.cuda()
def frozen_teacher(kind):
 # Teacher SAR per-time backbone targets are its trained branch, no invented fusion.
 current=model_kd.PicoEncoder;model_kd.PicoEncoder=OriginalEncoder
 try:m=method.Teacher(b.base_initial())
 finally:model_kd.PicoEncoder=current
 path=ROOT/'experiments/dlr_bce_dice_equal_20261103/results/T_multimodal'
 s=json.loads((path/'selection.json').read_text());assert b.sha(path/'best.pt')==s['checkpoint_sha256']
 m.load_state_dict(torch.load(path/'best.pt',weights_only=True,map_location='cpu'));m.eval();m.requires_grad_(False)
 # Expose outputs of the two existing SAR calls; no recomputation or model change.
 temporal=[]
 m.sar_backbone.register_forward_hook(lambda module,args,output:temporal.append(output))
 original_forward=m.forward
 def forward_with_temporal(*args,**kwargs):
  temporal.clear();out=original_forward(*args,**kwargs)
  assert len(temporal)==2
  return {**out,'feat_pre':temporal[0],'feat_post':temporal[1]}
 m.forward=forward_with_temporal
 return m.cuda()
b.build_system=build_system;b.freeze_teacher=frozen_teacher

def gradient_probe(system,teacher,batch):
 state=b.rng_state();buffers={n:x.clone() for n,x in system.named_buffers()};mode=system.training
 try:
  system.train();o=system(batch['sar'])
  with torch.no_grad():to=b.teacher_forward('multimodal',teacher,batch)
  ts,_=losses('mfsad',system,o,to,batch,{})
  vals={'seg':eq.seg(o['z'],batch['mask'],batch['valid']),**ts}
  params=list(system.student.parameters());gr={};norms={}
  for k,l in vals.items():
   gr[k]=torch.autograd.grad(l,params,retain_graph=True,allow_unused=True)
   norms[k]=float(sum(g.square().sum() for g in gr[k] if g is not None).sqrt())
  cos={k:float(sum((a*g).sum() for a,g in zip(gr['seg'],gg) if a is not None and g is not None)/(norms['seg']*norms[k]+1e-12)) for k,gg in gr.items() if k!='seg'}
  return norms,cos
 finally:
  with torch.no_grad():
   for n,x in system.named_buffers():x.copy_(buffers[n])
  system.train(mode);system.zero_grad(set_to_none=True);b.set_rng(state)
def calibrate(arm,system,teacher,kind,batches):
 pairs=[gradient_probe(system,teacher,x) for x in batches]
 norms={k:sum(p[0][k] for p in pairs)/len(pairs) for k in pairs[0][0]}
 lam={k:min(1000.,share*norms['seg']/max(norms[k],1e-12)) for k,share in b.SHARES[arm].items()}
 return {'gradient_norms':norms,'lambda':lam,'shares':b.SHARES[arm],'scope':'train mode, first64TRAIN, BN/RNG restored; deployed Student gradients only'}
b.calibrate=calibrate
original_evaluate=b.evaluate
def evaluate(arm,system,ds,**kwargs):
 result=original_evaluate(arm,system,ds,**kwargs)
 if arm=='mfsad' and set(ds.frame.split)=={'val'}:
  dest=OUT/'mfsad';h=json.loads((dest/'history.json').read_text()) if (dest/'history.json').exists() else []
  epoch=len(h)+1
  if epoch in [5,25,50,75,100]:
   stats=json.loads((OUT/'normalization.json').read_text());batch=b.batch_cuda(next(iter(b.loader(b.DLRInputs(b.frames()['train'].iloc[:16],stats,include_post=True)))))
   teacher=frozen_teacher('multimodal')
   with torch.enable_grad():norms,cos=gradient_probe(system,teacher,batch)
   cal=json.loads((dest/'calibration.json').read_text());lam=cal['lambda']
   weighted={k:lam[k]*norms[k]/max(norms['seg'],1e-12) for k in lam}
   b.write(dest/f'gradient_epoch{epoch:03}.json',{'epoch':epoch,'norms':norms,'cosine_to_seg':cos,'weighted_gradient_ratio':weighted,'scope':'TRAIN16 probe, no optimizer, BN/RNG restored'})
   del teacher;torch.cuda.empty_cache()
 return result
b.evaluate=evaluate

def prepare():
 b.setup();OUT.mkdir(exist_ok=True);assert not (OUT/'config.json').exists()
 src=ROOT/'experiments/dlr_bce_dice_equal_20261103'
 shutil.copyfile(src/'split_manifest.csv',HERE/'split_manifest.csv');shutil.copyfile(src/'results/normalization.json',OUT/'normalization.json')
 b.seed();s=model_kd.SARStudent(None);torch.save(s.state_dict(),OUT/'student_initial.pt')
 t=frozen_teacher('multimodal');m=MFSAD(s,'mfsad');meta=json.loads((HERE/'pretrained_location.json').read_text());weights=Path(meta['path'])
 counts={'student_deployment':sum(p.numel() for p in s.parameters()),'teacher':sum(p.numel() for p in t.parameters()),'training_only_adapters':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('student.'))}
 files=[Path(__file__),HERE/'model.py',VENDOR/'experiments/sar_only_kd_literature_20261103/model_kd.py',VENDOR/'experiments/sar_only_kd_literature_20261103/run.py',VENDOR/'experiments/sar_relation_frequency_20261103/method.py']
 c={'student':'convnextv2_tiny.fcmae_ft_in1k','pretrained':meta,'pretrained_sha256':b.sha(weights),'parameters':counts,'arms':['base','mfsad'],'teacher_reused':str(src/'results/T_multimodal/best.pt'),'teacher_sha256':b.sha(src/'results/T_multimodal/best.pt'),
 'loss':'BCE+ForegroundSoftDice,1:1, per-imageDice smooth1e-6','epochs':100,'batch':16,'val_every':5,'selection':'validationIoU@.5','test':'all2freeze; each once, report44independent/76dependent separately',
 'optimizer':'AdamW encoder2e-5/new1e-4,wd.01 exemptions,50constant50cosine-to1%,clip1,seed42',
 'anchors':[8,16,32],'FFT':'full fft2 ortho, fftshift; sigma=min(H,W)/4 bins, channel/batch mean over gaussian mass; shared invalid support zeroed',
 'Bk':'PRE/POST Teacher trained SAR branch, task-adapted target; not fused change maps','JSCAD':'per-sample spatial/channel minmax; squared mask-modulated error',
 'initial_sha256':b.sha(OUT/'student_initial.pt'),'split_sha256':b.sha(HERE/'split_manifest.csv'),'normalization_sha256':b.sha(OUT/'normalization.json'),'sources':{str(p.relative_to(ROOT)):b.sha(p) for p in files}}
 b.write(OUT/'config.json',c);print('PREPARED',counts,flush=True)
def verify():
 c=json.loads((OUT/'config.json').read_text())
 for p,h in c['sources'].items():assert b.sha(ROOT/p)==h,p
 assert b.sha(OUT/'student_initial.pt')==c['initial_sha256']
 assert b.sha(Path(c['pretrained']['path']))==c['pretrained_sha256']
 assert b.sha(HERE/'split_manifest.csv')==c['split_sha256']
def check():
 b.setup();verify();stats=json.loads((OUT/'normalization.json').read_text());batch=b.batch_cuda(next(iter(b.loader(b.DLRInputs(b.frames()['train'].iloc[:16],stats,include_post=True)))))
 m=build_system('mfsad');t=frozen_teacher('multimodal');o=m(batch['sar'])
 with torch.no_grad():to=b.teacher_forward('multimodal',t,batch)
 o['feat_pre'][1].retain_grad();o['feat_post'][1].retain_grad()
 ts,l=losses('mfsad',m,o,to,batch,{k:1. for k in b.SHARES['mfsad']});l.backward()
 grads=[float(o[k][1].grad.norm()) for k in ['feat_pre','feat_post']];assert min(grads)>0 and all(p.grad is None for p in t.parameters())
 assert all(torch.isfinite(v) for v in ts.values())
 ident=frequency(o['change'][1].detach(),o['change'][1].detach(),batch['valid']);assert ident.abs()<1e-7
 # Full FFT Gaussian and Hermitian-correct orthonormal half-spectrum agree.
 x=torch.randn(2,3,16,16,device='cuda');y=torch.randn_like(x);valid=torch.ones(2,1,16,16,device='cuda');full=frequency(x,y,valid)
 fy=torch.fft.fftfreq(16,device='cuda')*16;fx=torch.fft.rfftfreq(16,device='cuda')*16;g=torch.exp(-(fy[:,None].square()+fx[None,:].square())/(2*4**2));mult=torch.ones(9,device='cuda');mult[1:-1]=2
 half=((torch.fft.rfft2(x,norm='ortho').abs()-torch.fft.rfft2(y,norm='ortho').abs()).square()*g*mult).sum()/(6*(g*mult).sum())
 assert torch.allclose(full,half,atol=1e-6)
 shapes={'student_temporal':[list(z.shape) for z in o['feat_pre']],'teacher_SAR_temporal':[list(z.shape) for z in to['feat_pre']],'neck':[list(z.shape) for z in o['change']]}
 del m,o,to,l,ts;torch.cuda.empty_cache()
 m=build_system('base');opt=b.make_optimizer(m,prefix='student.');small={k:v[:4] for k,v in batch.items()};m.eval()
 with torch.no_grad():before=float(eq.seg(m(small['sar'])['z'],small['mask'],small['valid']))
 m.train()
 for _ in range(30):
  opt.zero_grad(set_to_none=True);loss=eq.seg(m(small['sar'])['z'],small['mask'],small['valid']);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
 m.eval()
 with torch.no_grad():after=float(eq.seg(m(small['sar'])['z'],small['mask'],small['valid']))
 assert after<.95*before
 b.write(OUT/'checks.json',{'state':'PASS','full_batch':16,'both_temporal_gradients':grads,'shapes':shapes,'FFT_full_half_equivalence':True,'train_fit':[before,after],'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'test_loaded':False})
 print('CHECK PASS',before,after,flush=True)

def finish():
 import numpy as np,pandas as pd
 for a in b.ARMS:
  sel=json.loads((OUT/a/'selection.json').read_text());assert sel['epochs_completed']==100 and b.sha(OUT/a/'best.pt')==sel['checkpoint_sha256']
 for a in b.ARMS:
  d=OUT/a
  if (d/'test_metrics.json').exists():continue
  assert not (d/'test_started.json').exists();b.write(d/'test_started.json',{'all2frozen':True,'checkpoint_sha256':b.sha(d/'best.pt')})
  m=build_system(a);m.load_state_dict(torch.load(d/'best.pt',map_location='cpu',weights_only=True));stats=json.loads((OUT/'normalization.json').read_text());score=b.evaluate(a,m,b.DLRInputs(b.frames()['test'],stats,include_post=False),save=d/'test_predictions.npz');b.write(d/'test_metrics.json',score);del m;torch.cuda.empty_cache()
 rows=[]
 for a in b.ARMS:
  q=json.loads((OUT/a/'test_metrics.json').read_text());rows.append({'model':a,**{k:q[k] for k in ['IoU','F1','AUPRC','Recall','Precision']},'independent_IoU':q['per_official_subset']['testind']['IoU'],'dependent_IoU':q['per_official_subset']['testspt']['IoU']})
 pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
 (ROOT/'reports/MFSAD_V2TINY_RESULTS.md').write_text('# Corrected M-FSAD V2-Tiny actual results\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\nSee experiment README for native mechanism versus segmentation adaptations and per-epoch gradients. No gain assumed.',encoding='utf-8')
 b.write(OUT/'progress.json',{'state':'COMPLETE','arms':list(b.ARMS)})
if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='check':check()
  elif action=='run':
   b.setup();verify();assert json.loads((OUT/'checks.json').read_text())['state']=='PASS'
   for a in b.ARMS:b.train_arm(a)
   finish()
 except Exception as e:
  import traceback
  b.write(OUT/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
