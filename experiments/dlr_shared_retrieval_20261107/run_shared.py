import json,sys,traceback,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model_shared import SharedRetrieval,base
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'experiments/dlr_unet_study_20261104'))
from train_baseline import common,segmentation,restore_rng
from train_distillation import REFERENCE,ROOT as OLD
from train_multiscale_3plus1 import PrivilegedInputs

def build(version,control=False):
    common.seed();return SharedRetrieval(version,control).cuda().to(memory_format=torch.channels_last)

def losses(m,o,b):
    bank,ps,pt,p,t=m.distill(o,b['x'],b['optical'])
    mask=F.adaptive_avg_pool2d(b['valid'].float(),bank[0].shape[-2:]).eq(1)[:,0]
    pol=(base.js(p,t)*mask.flatten(1)).sum()/mask.sum().clamp_min(1)
    es=m.core.retrieve(bank,ps);et=m.core.retrieve(bank,pt).detach()
    ev=((1-F.cosine_similarity(es,et,dim=1))*mask).sum()/mask.sum().clamp_min(1)
    return pol,ev

@torch.no_grad()
def evaluate(m,data,diagnostic=False):
    m.eval();ys=[];pred={k:[] for k in ['student','teacher']+(['mismatch','no_change','zero'] if diagnostic else [])}
    for raw in common.loader(data):
        b=common.cuda(raw);o=m.details(b['x'],b['optical']);v=b['valid'];ys.append(b['mask'][v].bool().cpu().numpy())
        pred['student'].append(o['zs'].sigmoid()[v].cpu().numpy());pred['teacher'].append(o['zt'].sigmoid()[v].cpu().numpy())
        if diagnostic:
            for key,opt in [('mismatch',b['optical'].roll(1,0)),('no_change',b['optical'][:,:6].repeat(1,2,1,1))]:
                _,pt=m.policies(o['bank'],b['x'],opt)
                z=m.core.decode(o['maps'],m.core.retrieve(o['bank'],pt));pred[key].append(z.sigmoid()[v].cpu().numpy())
            z=m.core.decode(o['maps'],torch.zeros_like(o['es']));pred['zero'].append(z.sigmoid()[v].cpu().numpy())
    y=np.concatenate(ys)
    return {k:common.metrics(y,np.concatenate(v)) for k,v in pred.items()}

def checks(train):
    rng=common.rng();records={}
    try:
        b=common.cuda(torch.utils.data.default_collate([train[i] for i in range(2)]))
        for version in ['V0','V2']:
            m=build(version).eval();o=m.details(b['x'],b['optical']);p,e=losses(m,o,b)
            # Direct KD teacher-target and encoder gradients must be absent; seg trains shared encoder.
            gs=torch.autograd.grad(p+e,list(m.teacher_head.parameters())+list(m.core.backbone.enc.parameters()),allow_unused=True,retain_graph=True)
            assert all(g is None or not g.any() for g in gs)
            seg=segmentation(o['zs'],b)+segmentation(o['zt'],b)
            seg.backward();assert m.core.backbone.enc[0].net[0].weight.grad.norm()>0
            with torch.no_grad():
                maps,bank=m.core.memory(b['x']);ps,pt=m.policies(bank,b['x'],b['optical'])
                fixed=m.core.decode(maps,m.core.retrieve(bank,pt))
                # Optical is never consumed by memory or decoder; only policy path can alter output.
                swapped=m.details(b['x'],b['optical'].roll(1,0))
                assert all(torch.equal(a,c) for a,c in zip(bank,swapped['bank']))
            records[version]={'shared_memory_decoder':True,'kd_teacher_encoder_direct_gradient_zero':True,'seg_encoder_gradient':float(m.core.backbone.enc[0].net[0].weight.grad.norm()),'shape':[list(x.shape) for x in bank],'parameters':sum(x.numel() for x in m.parameters()),'deploy_parameters':sum(x.numel() for x in m.core.parameters())}
            del m,o,gs,bank,maps,seg,p,e;torch.cuda.empty_cache()
    finally:restore_rng(rng)
    common.write(OUT/'checks.json',records)

def fit(version,arm,train,val):
    folder=OUT/(version+'_'+arm);folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    m=build(version,arm=='sar_dual');opt=common.optimizer(m);common.seed()
    gen=torch.Generator().manual_seed(123);dl=common.loader(train,True,gen)
    history=[];best=-1;chosen=None;step=0;enabled=False
    if (folder/'latest.pt').exists():
        s=torch.load(folder/'latest.pt',weights_only=False,map_location='cpu');m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);history=s['history'];best=s['best'];chosen=s['chosen'];step=s['step'];enabled=s['enabled'];gen.set_state(s['generator']);restore_rng(s['rng']);del s
    for epoch in range(len(history)+1,101):
        m.train();totals={}
        for index,raw in enumerate(dl,1):
            b=common.cuda(raw);opt.zero_grad(set_to_none=True)
            lr=1e-4*common.lr_factor(step,len(dl))
            for group in opt.param_groups:group['lr']=lr
            o=m.details(b['x'],b['optical']);ss=segmentation(o['zs'],b);st=segmentation(o['zt'],b)
            loss=(ss+st)/2;terms={'student_seg':ss,'teacher_seg':st}
            if arm=='kd' and enabled:
                p,e=losses(m,o,b);loss=loss+.05*(p+e);terms.update(policy=p,evidence=e)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();step+=1
            for k,v in terms.items():totals[k]=totals.get(k,0)+float(v.detach())*len(b['x'])
            common.write(folder/'progress.json',{'epoch':epoch,'batch':index,'batches':len(dl),'losses':{k:float(v.detach()) for k,v in terms.items()},'lr':lr,'best_validation_IoU':best,'KD_enabled':enabled,'gradient_norm':float(gn)})
        scores=evaluate(m,val) if epoch%5==0 else None
        if scores:
            enabled=scores['teacher']['IoU']-scores['student']['IoU']>=.002
            if scores['student']['IoU']>best:
                best=scores['student']['IoU'];chosen=epoch;torch.save(m.state_dict(),folder/'best.pt')
        history.append({'epoch':epoch,'losses':{k:v/len(train) for k,v in totals.items()},'validation':scores,'next_KD_enabled':enabled})
        common.write(folder/'history.json',history)
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),history=history,best=best,chosen=chosen,step=step,enabled=enabled,generator=gen.get_state(),rng=common.rng()),folder/'latest.tmp');(folder/'latest.tmp').replace(folder/'latest.pt')
        print(version,arm,epoch,best,flush=True)
    m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));result=evaluate(m,val,True)
    common.write(folder/'validation.json',result);record={'epochs_completed':100,'selected_epoch':chosen,'validation_IoU':best,'checkpoint_sha256':common.sha(folder/'best.pt')};common.write(folder/'selection.json',record)
    torch.save(m.core.state_dict(),folder/'deploy_student.pt');del m,opt;torch.cuda.empty_cache();return record

def main():
    common.setup();OUT.mkdir(exist_ok=True)
    common.write(OUT/'config.json',{'epochs':100,'batch':16,'val_every':5,'seed':42,'optimizer':'AdamW1e-4 wd.01 clip1, norm/bias excluded','schedule':'50constant50cosine1percent','base_loss':'BCE+Dice1:1','dual_seg':'0.5*(student_seg+teacher_seg)','KD_weights':[.05,.05],'KD_gate':'previous scheduled validation teacher IoU - student IoU >=.002; gated each next5 epochs','advantage_gate':'noKD teacher > capacity matched SARdual teacher by .002, and > own student by .002','test':'NOT_ACCESSED','split_sha256':common.sha(ROOT/'experiments/dlr_unet_study_20261104/split_manifest.csv')})
    frames=common.frames();stats=json.loads((REFERENCE/'normalization.json').read_text());optical=json.loads((OLD/'optical_normalization.json').read_text())
    train=PrivilegedInputs(frames['train'],stats,optical,augment=True);val=PrivilegedInputs(frames['val'],stats,optical)
    if not (OUT/'checks.json').exists():checks(train)
    for version in ['V0','V2']:
        for arm in ['sar_dual','no_kd']:
            common.write(OUT/'status.json',{'state':'TRAINING','version':version,'arm':arm});fit(version,arm,train,val)
        a=json.loads((OUT/(version+'_sar_dual')/'validation.json').read_text());b=json.loads((OUT/(version+'_no_kd')/'validation.json').read_text())
        passed=b['teacher']['IoU']>=a['teacher']['IoU']+.002 and b['teacher']['IoU']>=b['student']['IoU']+.002
        common.write(OUT/(version+'_gate.json'),{'passed':passed,'capacity_control':a,'optical_joint_noKD':b})
        if passed:fit(version,'kd',train,val)
        else:common.write(OUT/(version+'_kd_NOT_RUN.json'),{'reason':'Teacher shared-memory search advantage gate not passed'})
    common.write(OUT/'status.json',{'state':'STAGE_V0_V2_COMPLETE','later_versions':'NOT_RUN pending V2 gate and inspection','test':'NOT_ACCESSED'})
if __name__=='__main__':
    try:main()
    except Exception:
        common.write(OUT/'status.json',{'state':'FAILED','error':traceback.format_exc()});raise
