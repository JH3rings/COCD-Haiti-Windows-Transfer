from pathlib import Path
import sys,importlib.util,json,shutil,torch,hashlib,inspect
from contextlib import contextmanager
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';OLD=ROOT/'experiments/mfsad_v2tiny_20261103'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('verified_runtime',OLD/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
import model_kd
from model import Adapter
from network import ResidualTeacher,StudentSystem
from losses import terms,all_terms
registry_spec=importlib.util.spec_from_file_location('canonical_dlr_registry',ROOT/'cocd/windows_main/dlr_baseline_registry.py');registry=importlib.util.module_from_spec(registry_spec);registry_spec.loader.exec_module(registry);resolve=registry.resolve
spec=importlib.util.spec_from_file_location('paper_mfsad',ROOT/'experiments/dlr_published_kd_20261103/methods.py');paper=importlib.util.module_from_spec(spec);sys.modules['paper_mfsad']=paper;spec.loader.exec_module(paper)
b=prior.b;eq=prior.eq;b.OUT=OUT;b.HERE=HERE;prior.OUT=OUT
b.ARMS={a:{'teacher':'residual'} for a in ['E1','E2','E3','MFSAD']}
b.TEACHERS={'residual':'SAR-centered optical residual'}
b.SHARES={'E1':{'feat':.05,'out':.05},'E2':{'feat':.05,'out':.05,'freq':.05},'E3':{'feat':.05,'out':.05,'freq':.05},'MFSAD':{k:.1 for k in ['bk','freq','jscad','kd']}}
b.aux_terms=terms
original_write=b.write
def write(path,obj):
 path=Path(path)
 if path==OUT/'progress.json' and obj.get('state')=='TRAINING':
  original_write(OUT/'progress_by_arm'/f"{obj['arm']}.json",obj);return
 original_write(path,obj)
 if path.name=='history.json' and path.parent.name=='T_residual' and obj:
  x=obj[-1];best=max((y['validation']['IoU'] for y in obj if y.get('validation')),default=None)
  original_write(OUT/'progress_by_arm/T_residual.json',{'state':'TRAINING','arm':'T_residual','epoch':x['epoch'],'epochs':100,'loss_terms':{'seg':x['train_loss']},'best_val_IoU':best,'validation':x['validation'],'test_loaded':False})
b.write=write

@contextmanager
def preserve_rng():
 state=b.rng_state()
 try:yield
 finally:b.set_rng(state)

def optimizer(module,prefix=''):
 exempt=set()
 for m in module.modules():
  if any(k in type(m).__name__.lower() for k in ['norm','grn','globalresponse']):exempt.update(id(p) for p in m.parameters(recurse=False))
 groups={};seen=set()
 for name,p in module.named_parameters():
  if not p.requires_grad:continue
  assert id(p) not in seen;seen.add(id(p));enc=any(k in name for k in ['backbone.stem.','backbone.stages.']);nd=p.ndim<=1 or name.endswith('.bias') or id(p) in exempt
  key=('encoder' if enc else 'new')+('_no_decay' if nd else '_decay');lr=2e-5 if enc else 1e-4
  if key not in groups:groups[key]={'params':[],'names':[],'group_name':key,'lr':lr,'peak_lr':lr,'weight_decay':0 if nd else .01,'schedule_step':0}
  groups[key]['params'].append(p);groups[key]['names'].append(name)
 return torch.optim.AdamW(list(groups.values()),betas=(.9,.999),eps=1e-8)
b.make_optimizer=optimizer

def fresh_teacher():
 current=model_kd.PicoEncoder;model_kd.PicoEncoder=prior.OriginalEncoder
 try:t=ResidualTeacher(b.base_initial())
 finally:model_kd.PicoEncoder=current
 return t
def build_teacher(kind):
 assert kind=='residual';b.seed();t=fresh_teacher();t.load_state_dict(torch.load(OUT/'teacher_initial.pt',weights_only=True,map_location='cpu'));return t.cuda()
def freeze_teacher(kind):
 d=OUT/'T_residual';s=json.loads((d/'selection.json').read_text());assert s['epochs_completed']==100 and b.sha(d/'best.pt')==s['checkpoint_sha256']
 if (OUT/'teacher_lock.json').exists():assert s['checkpoint_sha256']==json.loads((OUT/'teacher_lock.json').read_text())['checkpoint_sha256']
 t=build_teacher(kind);t.load_state_dict(torch.load(d/'best.pt',weights_only=True,map_location='cpu'));t.eval();t.requires_grad_(False);return t
def build_system(arm):
 assert arm in b.ARMS,'Baseline is reuse-only; no E0 training entry'
 b.seed();s=model_kd.SARStudent(None);s.load_state_dict(torch.load(OUT/'student_initial.pt',weights_only=True,map_location='cpu'));m=StudentSystem(s,arm)
 if arm!='MFSAD':m.adapters.load_state_dict(torch.load(OUT/'adapter_initial.pt',weights_only=True,map_location='cpu'))
 return m.cuda()
b.build_teacher=build_teacher;b.freeze_teacher=freeze_teacher;b.build_system=build_system
b.teacher_forward=lambda kind,t,x:t(x['opt_pre'],x['opt_post'],x['sar'])

def probe(arm,m,t,batch,all_frequencies=False):
 state=b.rng_state();buffers={n:x.clone() for n,x in m.named_buffers()};mode=m.training
 try:
  m.train();so=m(batch['sar'])
  with torch.no_grad():to=b.teacher_forward('residual',t,batch)
  if all_frequencies:raw=all_terms(m,so,to,batch)
  else:raw,_=terms(arm,m,so,to,batch,{})
  vals={'seg':eq.seg(so['z'],batch['mask'],batch['valid']),**raw};params=list(m.student.parameters());gr={};norms={}
  for k,l in vals.items():gr[k]=torch.autograd.grad(l,params,retain_graph=True,allow_unused=True);norms[k]=float(sum(x.square().sum() for x in gr[k] if x is not None).sqrt())
  cos={k:float(sum((a*g).sum() for a,g in zip(gr['seg'],gs) if a is not None and g is not None)/(norms['seg']*norms[k]+1e-12)) for k,gs in gr.items() if k!='seg'}
  return norms,cos
 finally:
  with torch.no_grad():
   for n,x in m.named_buffers():x.copy_(buffers[n])
  m.train(mode);m.zero_grad(set_to_none=True);b.set_rng(state)
def calibrate(arm,m,t,kind,batches):
 if arm!='MFSAD' and (OUT/'shared_calibration.json').exists():
  common=json.loads((OUT/'shared_calibration.json').read_text())
 else:
  assert arm in ['E1','MFSAD'],'E1 must publish shared TRAIN-only calibration first'
  pairs=[probe(arm,m,t,x,all_frequencies=arm!='MFSAD') for x in batches]
  norms={k:sum(p[0][k] for p in pairs)/len(pairs) for k in pairs[0][0]};assert all(x>1e-12 for x in norms.values())
  keys=['bk','freq','jscad','kd'] if arm=='MFSAD' else ['feat','out','fft','wave'];share=.1 if arm=='MFSAD' else .05
  weights={k:min(1000.,share*norms['seg']/norms[k]) for k in keys}
  ratios={k:weights[k]*norms[k]/norms['seg'] for k in keys}
  if arm!='MFSAD':assert all(abs(x-.05)<1e-5 for x in ratios.values()),'Frequency budget not matched within coefficient cap'
  common={'gradient_norms':norms,'lambda_all':weights,'initial_ratios':ratios,'scope':'TRAIN64 after first segmentation-only epoch, Student gradients only; BN/RNG restored; fixed thereafter'}
  if arm!='MFSAD':b.write(OUT/'shared_calibration.json',common)
  else:return {**common,'lambda':weights}
 weights=common['lambda_all'];lam={'feat':weights['feat'],'out':weights['out']}
 if arm!='E1':lam['freq']=weights['fft' if arm=='E2' else 'wave']
 return {**common,'lambda':lam,'shared_feature_output_weights':True}
b.calibrate=calibrate
# Protect the complete diagnostic, including temporary DataLoader iteration.
# The inherited loop otherwise consumes RNG before entering calibrate().
student_loop=inspect.getsource(b.train_arm)
assert '            calib = calibrate(' in student_loop and '            lam = calib["lambda"]' in student_loop
student_loop=student_loop.replace('            calib = calibrate(', '            diagnostic_rng = rng_state()\n            calib = calibrate(').replace('            lam = calib["lambda"]','            set_rng(diagnostic_rng)\n            lam = calib["lambda"]')
exec(compile(student_loop,'RNG_protected_student_loop','exec'),b.__dict__)
original_evaluate=prior.original_evaluate
def evaluate(arm,m,ds,**kwargs):
 result=original_evaluate(arm,m,ds,**kwargs)
 if set(ds.frame.split)=={'val'}:
  d=OUT/arm;h=json.loads((d/'history.json').read_text()) if (d/'history.json').exists() else [];epoch=len(h)+1
  if epoch in [5,25,50,75,100]:
   with preserve_rng():
    stats=json.loads((OUT/'normalization.json').read_text());batch=b.batch_cuda(next(iter(b.loader(b.DLRInputs(b.frames()['train'].iloc[:8],stats,include_post=True)))));t=freeze_teacher('residual')
    with torch.enable_grad():norms,cos=probe(arm,m,t,batch)
    lam=json.loads((d/'calibration.json').read_text())['lambda'];weighted={k:lam[k]*norms[k]/max(norms['seg'],1e-12) for k in lam}
    b.write(d/f'gradient_epoch{epoch:03}.json',{'norms':norms,'cosine_to_seg':cos,'weighted_gradient_ratio':weighted,'scope':'TRAIN8; outer Teacher/Loader/probe RNG fully restored; no optimizer changes'});del t;torch.cuda.empty_cache()
 return result
b.evaluate=evaluate

def verify():
 c=json.loads((OUT/'config.json').read_text())
 for p,h in c['sources'].items():assert b.sha(ROOT/p)==h,p
 for p,h in c['initials'].items():assert b.sha(OUT/p)==h,p
 assert b.sha(HERE/'split_manifest.csv')==c['split_sha256'] and b.sha(OUT/'normalization.json')==c['normalization_sha256']
 assert b.sha(Path(c['pretrained']['path']))==c['pretrained_sha256']
 return c
def prepare():
 b.setup();OUT.mkdir(exist_ok=True);assert not (OUT/'config.json').exists();reg=resolve(ROOT)
 shutil.copyfile(ROOT/reg['registry']['locked_artifacts']['split']['path'],HERE/'split_manifest.csv');shutil.copyfile(ROOT/reg['registry']['locked_artifacts']['normalization']['path'],OUT/'normalization.json');shutil.copyfile(reg['clean_initialization'],OUT/'student_initial.pt')
 b.seed(43);ad=nn.ModuleList([Adapter(128,128) for _ in range(4)]);torch.save(ad.state_dict(),OUT/'adapter_initial.pt')
 b.seed();t=fresh_teacher();torch.save(t.state_dict(),OUT/'teacher_initial.pt');counts={'Teacher':sum(p.numel() for p in t.parameters()),'Student_deployed':28809793,'E123_training_adapters':sum(p.numel() for p in ad.parameters()),'MFSAD_training_adapters':567808};del t
 # Reuse the already completed, unaffected Teacher. Never resume old Students.
 parent=ROOT/'experiments/dlr_threelevel_frequency_20261103/results'
 selection=json.loads((parent/'T_residual/selection.json').read_text());assert selection['epochs_completed']==100 and b.sha(parent/'T_residual/best.pt')==selection['checkpoint_sha256']
 shutil.copyfile(parent/'teacher_initial.pt',OUT/'teacher_initial.pt');(OUT/'T_residual').mkdir()
 for name in ['best.pt','selection.json','history.json']:shutil.copyfile(parent/'T_residual'/name,OUT/'T_residual'/name)
 shutil.copyfile(parent/'teacher_lock.json',OUT/'teacher_lock.json')
 (HERE/'student_loop.py').write_text(student_loop,encoding='utf-8')
 old=json.loads((OLD/'results/config.json').read_text());deps=old['sources'];deps[str((ROOT/'experiments/dlr_published_kd_20261103/methods.py').relative_to(ROOT))]=b.sha(ROOT/'experiments/dlr_published_kd_20261103/methods.py')
 for p,h in deps.items():assert b.sha(ROOT/p)==h,p
 files=[Path(__file__),HERE/'network.py',HERE/'losses.py',HERE/'checks.py',HERE/'student_loop.py',HERE/'USER_REQUEST.md',ROOT/'cocd/windows_main/dlr_baseline_registry.py']
 c={'method':'SAR-centered residual optical change Teacher; common feature/output KD; FFT-TCSD vs Haar-WCSD',
 'queue':['E1','E2','E3','MFSAD'],'baseline':'configs/active_dlr_baseline.json; reuse-only, never retrain/retest',
 'Teacher_reused':{'directory':str(parent/'T_residual'),'checkpoint_sha256':selection['checkpoint_sha256'],'reason':'complete100epoch Teacher unaffected by Student diagnostic RNG; old Students archived, restart clean'},
 'Student':'convnextv2_tiny.fcmae_ft_in1k; shared temporal encoder/change/FPN/head; native128; pure target VV/VHprepost+OrbitID; no search',
 'Teacher':'dual-temporal V2-Pico modality encoders; independently projected changes128D; E=SAR+sigmoid(g)*R; four levels; FPN/head; clean init, no trained legacy weights',
 'parameters':counts,'initials':{p.name:b.sha(p) for p in OUT.glob('*initial.pt')},
 'pretrained':old['pretrained'],'pretrained_sha256':old['pretrained_sha256'],
 'split_sha256':b.sha(HERE/'split_manifest.csv'),'normalization_sha256':b.sha(OUT/'normalization.json'),
 'loss':'BCEWithLogits + per-image foreground SoftDice1:1,valid,smooth1e-6',
 'epochs':100,'val_every':5,'batch':16,'early_stop':False,'selection':'validation pooled foregroundIoU@.5',
 'optimizer':'AdamW both modality encoders2e-5/new1e-4;wd.01 norm/bias/GRN exempt;clip1;50constant50cosine-to1%;no warmup;max encoderDropPath.05',
 'KD_strategy':'epoch1 segmentation-only; TRAIN64 common calibration before epoch2; E123 grad shares.05 per active term; shared feat/out coeffs; FFT and Haar each.05; maxcoef1000; MFSAD.1 each as prior adapted method; no retuning',
 'frequency':'four pre-FPN change levels4/8/16/32; per-channel valid spatial mean/std; FP32; rfft2 ortho,log1p,l2cos shape; Haar onelevel,strict2x2valid,LLchannelcos plus equal three directional energy SmoothL1,per-sample valid mean-energy normalization; odd trailing invalid padding',
 'output_KD':'T10 BernoulliKL equivalent softmax([0,z]/10),sum both classes,valid mean,T^2',
 'test':'all four Students andTeacher freeze before any new test; E0predictions reused; independent44 subset ofall120; official overlap unchanged',
 'MFSAD_comparison':'new same-residual-Teacher run; old Teacher result is historical, not matched',
 'sources':{**deps,**{str(p.relative_to(ROOT)):b.sha(p) for p in files}}}
 b.write(OUT/'config.json',c);print('PREPARED',counts,flush=True)

def finish():
 for a in ['T_residual',*b.ARMS]:
  d=OUT/a;s=json.loads((d/'selection.json').read_text());assert s['epochs_completed']==100 and b.sha(d/'best.pt')==s['checkpoint_sha256']
 for arm in [*b.ARMS,'T_residual']:
  d=OUT/arm
  if (d/'test_metrics.json').exists():continue
  assert not (d/'test_started.json').exists();b.write(d/'test_started.json',{'all5frozen':True,'checkpoint_sha256':b.sha(d/'best.pt')})
  is_teacher=arm=='T_residual';m=freeze_teacher('residual') if is_teacher else build_system(arm)
  if not is_teacher:m.load_state_dict(torch.load(d/'best.pt',weights_only=True,map_location='cpu'))
  stats=json.loads((OUT/'normalization.json').read_text());ds=b.DLRInputs(b.frames()['test'],stats,include_post=is_teacher)
  if is_teacher:
   class AdapterModel(nn.Module):
    def __init__(self):super().__init__();self.model=m;self.i=0
    def forward(self,sar):
     items=ds.cache[self.i:self.i+len(sar)];self.i+=len(sar)
     return self.model(torch.stack([x['opt_pre'] for x in items]).cuda(),torch.stack([x['opt_post'] for x in items]).cuda(),sar)
   ev=AdapterModel()
  else:ev=m
  score=original_evaluate(arm,ev,ds,save=d/'test_predictions.npz');b.write(d/'test_metrics.json',score);del m,ev,ds;torch.cuda.empty_cache()
 import deliver
 deliver.main(b,ROOT,HERE)
 b.write(OUT/'complete.json',{'state':'COMPLETE','baseline_retrained':False,'Teacher_shared':True})

if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='check':
   b.setup();verify();import checks;checks.main(sys.modules[__name__])
  elif action=='teacher':
   b.setup();verify();assert json.loads((OUT/'checks.json').read_text())['state']=='PASS';b.train_teacher('residual');s=json.loads((OUT/'T_residual/selection.json').read_text());b.write(OUT/'teacher_lock.json',s)
  elif action=='student':
   b.setup();verify();arm=sys.argv[2];assert arm in b.ARMS;b.train_arm(arm)
  elif action=='finish':b.setup();verify();finish()
 except Exception as e:
  import traceback
  a=sys.argv[2] if len(sys.argv)>2 else sys.argv[1];b.write(OUT/'errors'/f'{a}.json',{'error':str(e),'traceback':traceback.format_exc()});raise
