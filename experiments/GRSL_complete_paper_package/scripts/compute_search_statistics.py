"""Compute target-memory policy/evidence diagnostics for frozen CG and S2.

Evidence uses the canonical same-target-memory definition: target values come
from the Student branch, while Teacher/correct-or-shuffled policies only
choose where to retrieve them.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"cocd"))
from paths import DEV  # noqa:E402
from windows_main.data import HaitiPairs,split  # noqa:E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa:E402
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2  # noqa:E402

SEED=42; BATCH=16
TEACHER=ROOT/"experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
STUDENT=ROOT/"experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt"
OUT=ROOT/"experiments/GRSL_complete_paper_package/03_search_behavior"

def strict_qmask(pixel_mask,hq,wq):
    if pixel_mask.ndim==3: pixel_mask=pixel_mask[:,None]
    return F.adaptive_avg_pool2d(pixel_mask.float(),(hq,wq))[:,0] >= 1.0-1e-7

def js_query(p,q):
    p=p.clamp_min(1e-8);q=q.clamp_min(1e-8);m=.5*(p+q)
    return .5*((p*(p.log()-m.log())).sum(1)+(q*(q.log()-m.log())).sum(1))

def main():
    train_ids,valid_ids,test_ids=split()
    if valid_ids: raise RuntimeError("Expected the locked no-validation protocol")
    train=HaitiPairs(train_ids,True); test=HaitiPairs(test_ids,False,train.stats)
    teacher=TargetSearchTeacherV2().to(DEV).eval();teacher.load_state_dict(torch.load(TEACHER,map_location=DEV,weights_only=True))
    student=StudentTargetSearchV2().to(DEV).eval();student.load_state_dict(torch.load(STUDENT,map_location=DEV,weights_only=True))
    gen=torch.Generator().manual_seed(SEED+1701)
    acc={}
    def add(region,key,val):
        if val is not None and np.isfinite(val):acc.setdefault((region,key),[]).append(float(val))
    with torch.inference_mode():
        for batch in DataLoader(test,batch_size=BATCH,shuffle=False,num_workers=0):
            a,d,y,ga,gd=batch[:5];a=a.to(DEV);d=d.to(DEV)
            ca,cd=teacher.forward_pair(a,d,"counter")
            sa,sd=student(a),student(d)
            perm=torch.randperm(len(a),generator=gen)
            if torch.equal(perm,torch.arange(len(a))):perm=torch.roll(perm,1)
            perm=perm.to(DEV)
            sha=teacher(a,d[perm]);shd=teacher(d,a[perm])
            for gt,gc,to,so,sho in ((ga,gd,ca,sa,sha),(gd,ga,cd,sd,shd)):
                for region,pixel_mask in (("TargetValid",gt==0),("BothValid",(gt==0)&(gc==0))):
                    w_t=to["policy"]["weights"];w_s=so["policy"]["weights"];w_h=sho["policy"]["weights"]
                    hq,wq=w_t.shape[-2:];qm=strict_qmask(pixel_mask.to(DEV),hq,wq)
                    jt=js_query(w_t.flatten(1,2),w_s.flatten(1,2))
                    jh=js_query(w_t.flatten(1,2),w_h.flatten(1,2))
                    if qm.any():
                        add(region,"policy_js_CG_vs_S2",jt[qm].mean().item())
                        add(region,"policy_js_CG_vs_shuffled",jh[qm].mean().item())
                    # Compare retrievals in one fixed target-side value memory.
                    memory=tuple(x.detach() for x in so["values"])
                    r_student=student.search(memory,so["policy"])
                    r_teacher=student.search(memory,to["policy"])
                    r_shuffle=student.search(memory,sho["policy"])
                    c1=F.cosine_similarity(r_teacher,r_student,dim=1,eps=1e-8)
                    ch=F.cosine_similarity(r_shuffle,r_student,dim=1,eps=1e-8)
                    if qm.any():
                        add(region,"evidence_distance_CG_vs_S2",(1-c1[qm]).mean().item())
                        add(region,"evidence_cosine_CG_vs_S2",c1[qm].mean().item())
                        add(region,"evidence_distance_shuffled_vs_S2",(1-ch[qm]).mean().item())
                    for name,policy in (("CG",to["policy"]),("S2",so["policy"]),("Shuffle",sho["policy"])):
                        weights=policy["weights"]
                        offsets=2*torch.tanh(policy["offsets"])
                        for li in range(weights.shape[1]):
                            mass=weights[:,li].sum(1)
                            if qm.any():add(region,f"{name}_weight_mass_P{li+2}",mass[qm].mean().item())
                        mag=torch.linalg.vector_norm(offsets[:,:,1:],dim=3).mean(2).permute(0,2,3,1)
                        if qm.any():add(region,f"{name}_offset_magnitude",mag[qm].mean().item())
    rows=[{"Region":region,"Metric":key,"Mean":float(np.mean(v)),"BatchesOrOrbitRecords":len(v),
           "EvidenceDefinition":"cosine distance between CG/S2 policy retrievals over the same Student target memory" if key.startswith("evidence_") else ""}
          for (region,key),v in acc.items()]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"search_statistics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"search_statistics.json").write_text(json.dumps({f"{r}/{k}":float(np.mean(v)) for (r,k),v in acc.items()},indent=2),encoding="utf-8")
    print(f"SEARCH_STATS_OK rows={len(rows)}",flush=True)

if __name__=="__main__":main()
