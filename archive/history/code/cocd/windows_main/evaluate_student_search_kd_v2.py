"""Locked-checkpoint evaluation for S0/S1/S2 search-KD mechanism evidence."""
from __future__ import annotations
import csv,json,sys,os
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'cocd'))
from paths import DEV
from windows_main.data import HaitiPairs,split
from windows_main.models_da_search import TargetOnlyBaseline
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2,load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.student_search_kd_v2 import OUT,TP,SP,dl,spatial_policy,js,qmask,EPS
def metric(p,y,m):
 p=p[m];y=y[m].astype(bool);h=p>=.5;tp=(h&y).sum();fp=(h&~y).sum();fn=((~h)&y).sum();pr=tp/(tp+fp+EPS);rc=tp/(tp+fn+EPS)
 return {'pixels':int(m.sum()),'auprc':float(average_precision_score(y,p)),'iou':float(tp/(tp+fp+fn+EPS)),'f1':float(2*pr*rc/(pr+rc+EPS)),'precision':float(pr),'recall':float(rc),'accuracy':float((h==y).mean())}
@torch.no_grad()
def main():
 sweep=os.environ.get('COCD_EVAL_SWEEP')=='1';strength=os.environ.get('COCD_EVAL_STRENGTH')=='1'
 outdir=OUT/'strength_sweep' if strength else (OUT/'weight_sweep' if sweep else OUT)
 tid,_,eid=split();tr=HaitiPairs(tid,True);te=HaitiPairs(eid,False,tr.stats)
 t=TargetSearchTeacherV2().to(DEV);t.load_state_dict(torch.load(TP,map_location=DEV,weights_only=True));t.eval()
 so=TargetOnlyBaseline().to(DEV);load_so_backbone(so,str(SP));so.eval()
 paths=({'NoKD':OUT/'S0_seed42.pt','KD05':outdir/'KD05_seed42.pt','KD10':OUT/'weight_sweep'/'P50E50_seed42.pt','KD20':outdir/'KD20_seed42.pt'} if strength else ({'NoKD':OUT/'S0_seed42.pt','P0E100':outdir/'P0E100_seed42.pt','P25E75':outdir/'P25E75_seed42.pt','P50E50':outdir/'P50E50_seed42.pt','P100E0':outdir/'P100E0_seed42.pt'} if sweep else {n:OUT/(n+'_seed42.pt') for n in ('S0','S1','S2')}))
 ss={n:StudentTargetSearchV2().to(DEV).eval() for n in paths}
 for n,s in ss.items():s.load_state_dict(torch.load(paths[n],map_location=DEV,weights_only=True))
 store={n:{'p':[],'js':[],'ev':[]} for n in ss}; yy=[];gg=[];cc=[];ps=[];pt=[]
 for a,d,y,ga,gd,_ in dl(te):
  a,d=a.to(DEV),d.to(DEV);ta,td=t.forward_pair(a,d,'counter'); tpcy=[ta['policy'],td['policy']]
  ps += [torch.sigmoid(so(a)['z'])[:,0].cpu().numpy(),torch.sigmoid(so(d)['z'])[:,0].cpu().numpy()];pt += [torch.sigmoid(ta['z'])[:,0].cpu().numpy(),torch.sigmoid(td['z'])[:,0].cpu().numpy()]
  yy += [y.numpy(),y.numpy()];gg += [ga.numpy(),gd.numpy()];cc += [gd.numpy(),ga.numpy()];qm=qmask(ga,gd)
  for n,s in ss.items():
   oa,od=s(a),s(d);oo=[oa,od]; jmaps=[];emaps=[]
   for o,tc in zip(oo,tpcy):
    mem=tuple(x.detach() for x in o['values']); sp=s.policy(s.context(tuple(x.detach() for x in o['T']))); P=spatial_policy(sp,mem);T=spatial_policy({k:v for k,v in tc.items() if k in ('offsets','weights')},mem);jmaps.append(js(P,T));emaps.append(1-F.cosine_similarity(s.search(mem,sp),s.search(mem,{k:v for k,v in tc.items() if k in ('offsets','weights')}),dim=1))
   store[n]['p'] += [torch.sigmoid(oa['z'])[:,0].cpu().numpy(),torch.sigmoid(od['z'])[:,0].cpu().numpy()];store[n]['js'].append(torch.cat(jmaps).cpu().numpy());store[n]['ev'].append(torch.cat(emaps).cpu().numpy())
 y=np.concatenate(yy);g=np.concatenate(gg);c=np.concatenate(cc);so=np.concatenate(ps);teacher=np.concatenate(pt);tgd=(g>0)&~(c>0);omega=tgd&((so>=.5)!=(y>0))&((teacher>=.5)==(y>0));rows=[]
 for n,x in store.items():
  p=np.concatenate(x['p']);j=np.concatenate(x['js']).reshape(-1);e=np.concatenate(x['ev']).reshape(-1)
  for rn,m in [('Overall',np.ones_like(tgd,bool)),('TGD',tgd),('Remaining',~tgd),('OmegaTR',omega)]:
   z=metric(p,y,m);qm=F.max_pool2d(torch.from_numpy(m.astype(np.float32))[:,None],4,4)[:,0].numpy().reshape(-1)>0;z.update({'method':n,'region':rn,'policy_js':float(j[qm].mean()),'evidence_distance':float(e[qm].mean())});rows.append(z)
 with (outdir/'locked_region_mechanism_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 by={(r['method'],r['region']):r for r in rows};report={'omegaTR_definition':'TGD AND SO incorrect AND CG10 Teacher correct at threshold 0.5','omegaTR_pixels':int(omega.sum()),'metrics':rows}
 if not sweep and not strength:report['S2_minus_S1']={r:{k:by['S2',r][k]-by['S1',r][k] for k in ('auprc','iou','f1','precision','recall','accuracy')} for r in ('Overall','TGD','Remaining','OmegaTR')}
 (outdir/'region_mechanism_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'omega_pixels':int(omega.sum()),'sweep':sweep,'strength':strength},indent=2))
if __name__=='__main__':main()
