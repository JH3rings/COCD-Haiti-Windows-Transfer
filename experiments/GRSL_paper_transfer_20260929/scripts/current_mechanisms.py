"""Locked CG0/S2 interventions. Never writes existing checkpoints or predictions."""
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
PACK=Path(__file__).resolve().parents[1]
OUT=PACK/'new_evidence/20260930'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import make_loaders,loader,metric_rows,seed_everything,DEV
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2
from windows_main.models_da_search import pyramid

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def modify(policy,mode,generator):
    p={k:v.clone() for k,v in policy.items()}
    if mode=='center':p['offsets'].zero_()
    if mode=='random':
        # Uniform bounded feature offsets [-2,2]; retain anchored centre and learned weights.
        bounded=torch.rand(p['offsets'].shape,generator=generator)*3.9998-1.9999
        bounded[:,:,0]=0
        p['offsets']=torch.atanh(bounded.to(DEV)/2)
    return p

def run():
    OUT.mkdir(parents=True,exist_ok=True);seed_everything();torch.set_num_threads(4)
    train,test,info=make_loaders()
    np.savez_compressed(OUT/'normalization.npz',mean=train.stats[0],std=train.stats[1])
    student=StudentTargetSearchV2().to(DEV).eval()
    sp=ROOT/'experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt'
    student.load_state_dict(torch.load(sp,map_location=DEV,weights_only=True))
    teacher=TargetSearchTeacherV2().to(DEV).eval()
    tp=ROOT/'experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt'
    teacher.load_state_dict(torch.load(tp,map_location=DEV,weights_only=True))
    arms=[('S2-2Stage','learned'),('S2-2Stage','center'),('S2-2Stage','random'),('S2-2Stage','no-search'),
          ('CG0','paired'),('CG0','center'),('CG0','random'),('CG0','no-search'),('CG0','counter-shuffle'),('CG0','self-context')]
    probs={f'{a}:{m}':[] for a,m in arms};labels=[];gts=[];gcs=[];ids=[];orbits=[]
    points=[];align={};shuffle_record=[]
    # Global cyclic derangement: every complementary image belongs to a different location.
    shift=42
    def add(key,values,mask):
        vals=values[mask];s,n=align.get(key,(0.,0));align[key]=(s+float(vals.double().sum()),n+vals.numel())
    begin=time.time()
    with torch.inference_mode():
        for bi,(a,d,y,ga,gd,_real) in enumerate(loader(test)):
            a,d=a.to(DEV),d.to(DEV);start=bi*16;batch_ids=test.ids[start:start+len(a)]
            shids=[(start+j+shift)%len(test) for j in range(len(a))]
            sha=torch.stack([test[j][0] for j in shids]).to(DEV)
            shd=torch.stack([test[j][1] for j in shids]).to(DEV)
            ta,td=pyramid(teacher.backbone,a),pyramid(teacher.backbone,d)
            tsa,tsd=pyramid(teacher.backbone,sha),pyramid(teacher.backbone,shd)
            for target,counter,t,c,shc,gt,gc,orbit in ((a,d,ta,td,tsd,ga,gd,'asc'),(d,a,td,ta,tsa,gd,ga,'desc')):
                so=student(target);to=teacher.from_pyramids(t,c,(128,128))
                sho=teacher.from_pyramids(t,shc,(128,128));selfo=teacher.from_pyramids(t,t,(128,128))
                for kind,mode in arms:
                    o,model=(so,student) if kind=='S2-2Stage' else (to,teacher)
                    if mode in ('learned','paired'):z=o['z']
                    elif mode=='counter-shuffle':z=sho['z']
                    elif mode=='self-context':z=selfo['z']
                    elif mode=='no-search':
                        # Bypass aggregation: use projected target P2, retain trained decoder.
                        r=o['values'][0]
                        z=student.decode_R(r,(128,128)) if kind=='S2-2Stage' else teacher._decode_from_search(r,(128,128))
                    else:
                        gen=torch.Generator().manual_seed(42000+start+(0 if orbit=='asc' else 10000))
                        r=model.search(o['values'],modify(o['policy'],mode,gen))
                        z=model.decode_R(r,(128,128)) if kind=='S2-2Stage' else model._decode_from_search(r,(128,128))
                    probs[f'{kind}:{mode}'].append(torch.sigmoid(z[:,0]).cpu().numpy())
                labels.append(y.numpy());gts.append(gt.numpy());gcs.append(gc.numpy());ids.extend(batch_ids);orbits.extend([orbit]*len(a))
                for j,sid in enumerate(batch_ids):shuffle_record.append({'sample_id':sid,'orbit':orbit,'counter_sample_id':test.ids[shids[j]]})
                # Pixel-weighted query statistics, joint 3x4 candidate-weight JS (not spatial-splat training loss).
                wt=to['policy']['weights'].flatten(1,2).clamp_min(1e-8)
                ws=so['policy']['weights'].flatten(1,2).clamp_min(1e-8)
                wh=sho['policy']['weights'].flatten(1,2).clamp_min(1e-8)
                def js(p,q):
                    m=(p+q)/2;return .5*((p*(p.log()-m.log())).sum(1)+(q*(q.log()-m.log())).sum(1))
                rt=student.search(so['values'],to['policy']);rh=student.search(so['values'],sho['policy'])
                for region,mask in [('TargetValid',gt==0),('BothValid',(gt==0)&(gc==0)),('G10',(gt>0)&(gc==0))]:
                    qm=F.adaptive_avg_pool2d(mask[:,None].float().to(DEV),ws.shape[-2:])[:,0]>=1-1e-7
                    for name,val in [('JS_CG0_S2',js(wt,ws)),('JS_CG0_shuffle',js(wt,wh)),
                            ('cosine_distance_CG0_S2',1-F.cosine_similarity(rt,so['R'],dim=1)),
                            ('cosine_distance_shuffle_S2',1-F.cosine_similarity(rh,so['R'],dim=1))]:add((region,name),val,qm)
                # Save every query's raw policy in selected scenes later via dedicated extraction.
            print(f'batch {bi+1}/22 elapsed={time.time()-begin:.1f}s',flush=True)
    common={'y':np.concatenate(labels),'target_geometry':np.concatenate(gts),'counter_geometry':np.concatenate(gcs),'sample_id':np.array(ids),'orbit':np.array(orbits)}
    np.savez_compressed(OUT/'intervention_predictions.npz',**common,**{k:np.concatenate(v) for k,v in probs.items()})
    rows=[]
    for key,p in probs.items():
        met=metric_rows((np.concatenate(p),common['y'],common['target_geometry'],common['counter_geometry']))
        for region,values in met.items():rows.append({'arm':key,'region':region,'evidence':'inference intervention, no retraining',**values})
    save_csv(OUT/'current_search_interventions.csv',rows)
    save_csv(OUT/'counter_derangement.csv',shuffle_record)
    save_csv(OUT/'current_search_alignment.csv',[{'region':r,'metric':k,'mean':s/n if n else None,'query_count':n} for (r,k),(s,n) in align.items()])
    sources={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (sp,tp,Path(__file__),ROOT/'data/splits_v3/test_ids.csv')}
    (OUT/'intervention_protocol.json').write_text(json.dumps({'sources':sources,'regions':'Overall all; TargetValid gt=0; BothValid gt=gc=0; G10 gt>0,gc=0','threshold':.5,'split':info,'random':'uniform feature offsets [-2,2], fixed centre, learned weights; seed 42000+batch start+orbit offset','counter_shuffle':'global shift 42 in pinned sample order, no self pairs','self_context':'CG0 frozen weights with target as counter; not retrained SG0','no_search':'inference bypass via projected P2; not architecture retraining','alignment':'pixel-weighted strict full-valid 4x4 query blocks; JS on 12 candidate weights, not spatial training JS'},indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':run()
