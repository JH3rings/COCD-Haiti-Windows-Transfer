"""Unified evaluation of the masked v4 SGRI/POWC diagnostic runs."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"cocd"))
from paths import DEV
from windows_main.data import HaitiPairs, split
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2
from train_sgri_powc_masked_v4 import (OUT,BATCH,TEACHER,PairedVerifier,loader,metric_rows,
    evaluate, guided_test, spatial_policy, query_valid, js_distance, make_student, write_json)

TEACHER_SG=ROOT/"experiments/windows_main/geometry_transfer_restored_v1/SG_seed42_bestval.pt"
EPS=1e-8

def load_model(path,kind="student",verifier=False):
    model=make_student() if kind=="student" else TargetSearchTeacherV2().to(DEV)
    d=torch.load(path,map_location=DEV,weights_only=True)
    if kind=="teacher": state=d
    else: state=d["student"] if "student" in d else d
    model.load_state_dict(state); model.eval()
    vm=None
    if verifier:
        vm=PairedVerifier().to(DEV); vm.load_state_dict(d["verifier"]); vm.eval()
    return model,vm

@torch.no_grad()
def teacher_pred(model,ds,context):
    model.eval(); ps=[];ys=[];gas=[];gds=[]
    for a,d,y,ga,gd,_ in loader(ds):
        a,d=a.to(DEV),d.to(DEV)
        oa,od=model.forward_pair(a,d,context)
        ps.extend((torch.sigmoid(oa["z"][:,0]).cpu().numpy(),torch.sigmoid(od["z"][:,0]).cpu().numpy()))
        ys.extend((y.numpy(),y.numpy()));gas.extend((ga.numpy(),gd.numpy()));gds.extend((gd.numpy(),ga.numpy()))
    return tuple(np.concatenate(x,axis=0) for x in (ps,ys,gas,gds))

@torch.no_grad()
def teacher_pred_shuffled(model,ds):
    model.eval(); ps=[];ys=[];gas=[];gds=[];gen=torch.Generator().manual_seed(42+1701)
    for a,d,y,ga,gd,_ in loader(ds):
        a,d=a.to(DEV),d.to(DEV); b=len(a)
        perm=torch.randperm(b,generator=gen)
        if torch.equal(perm,torch.arange(b)): perm=torch.roll(perm,1)
        oa=model(a,d[perm.to(DEV)]); od=model(d,a[perm.to(DEV)])
        ps.extend((torch.sigmoid(oa["z"][:,0]).cpu().numpy(),torch.sigmoid(od["z"][:,0]).cpu().numpy()))
        ys.extend((y.numpy(),y.numpy()));gas.extend((ga.numpy(),gd.numpy()));gds.extend((gd.numpy(),ga.numpy()))
    return tuple(np.concatenate(x,axis=0) for x in (ps,ys,gas,gds))

@torch.no_grad()
def search_distances(student,teacher,ds):
    student.eval();teacher.eval(); rows={"policy_js":[],"evidence_distance":[],"query_pixels":[],"both_valid_queries":[]}
    for a,d,y,ga,gd,_ in loader(ds):
        a,d=a.to(DEV),d.to(DEV); ta,td=teacher.forward_pair(a,d,"counter"); oa,od=student(a),student(d)
        for o,tp,g,c in ((oa,ta["policy"],ga,gd),(od,td["policy"],gd,ga)):
            mem=tuple(v.detach() for v in o["values"]); sp=student.policy(student.context(tuple(v.detach() for v in o["T"])))
            p1=spatial_policy(sp,mem); p2=spatial_policy({k:v for k,v in tp.items() if k in ("offsets","weights")},mem)
            js=js_distance(p1,p2); qm=query_valid(g.to(DEV)); qmf=qm.reshape(qm.shape[0],-1)
            r1=student.search(mem,sp); r2=student.search(mem,{k:v for k,v in tp.items() if k in ("offsets","weights")})
            ev=1-F.cosine_similarity(r1,r2,dim=1)
            both=query_valid(g.to(DEV)) & query_valid(c.to(DEV))
            rows["policy_js"].append((js*qmf).sum().item()/qmf.sum().clamp_min(1).item())
            rows["evidence_distance"].append((ev*qm).sum().item()/qm.sum().clamp_min(1).item())
            rows["query_pixels"].append(qm.float().mean().item())
            rows["both_valid_queries"].append(both.float().mean().item())
    return {k:float(np.mean(v)) for k,v in rows.items()}

def best_record(name):
    d=json.loads((OUT/f"{name}_progress.json").read_text(encoding="utf-8"))
    b=d["best_test_auprc_raw_overall"]
    r=max(d["history"],key=lambda x:x["monitor_test_auprc_raw_overall"])
    return d,r

def main():
    OUT.mkdir(parents=True,exist_ok=True); tid,vid,eid=split()
    if vid: raise RuntimeError("Expected no validation partition under canonical protocol")
    train=HaitiPairs(tid,True); test=HaitiPairs(eid,False,train.stats)
    table={}; mechanisms={}; best_epochs={}
    for name in ("S0-mask","S2-mask","SGRI-A1","SGRI-A2","POWC-B1","POWC-B2"):
        d,r=best_record(name); best_epochs[name]=d["best_epoch"]
        s,v=load_model(OUT/f"{name}_seed42_best.pt",verifier=name.startswith("POWC"))
        pred=evaluate(s,test,"single",None)
        table[name]={"metrics":metric_rows(pred),"monitor_raw_overall_ap":d["best_test_auprc_raw_overall"],"best_epoch":d["best_epoch"]}
        if name.startswith("SGRI"):
            table[name]["guided_path_metrics"]=guided_test(s,load_model(TEACHER,"teacher")[0],test)
            mechanisms[name]={"best_epoch_train_diagnostics":r["mechanism"],"best_self_path_metrics":table[name]["metrics"],"guided_path_test_metrics":table[name]["guided_path_metrics"],
                              "best_alpha":r["mechanism"].get("guide_alpha"),"internalization_ap_gap_bothvalid":table[name]["metrics"]["BothValid"]["AUPRC"]-table[name]["guided_path_metrics"]["BothValid"]["AUPRC"]}
        if name.startswith("POWC"):
            paired=metric_rows(evaluate(s,test,"paired",v)); table[name]["paired_guided_metrics"]=paired
            mechanisms[name]={"best_epoch_train_diagnostics":r["mechanism"],"single_path_metrics":table[name]["metrics"],"paired_guided_metrics":paired,
                "paired_minus_single_ap_bothvalid":paired["BothValid"]["AUPRC"]-table[name]["metrics"]["BothValid"]["AUPRC"]}
        if name=="S2-mask":
            t,_=load_model(TEACHER,"teacher"); mechanisms[name]=search_distances(s,t,test)
    # Masked SG/CG are reused from the prior geometry-masked experiment. They
    # remain an explicit legacy split/validation-selection reference.
    for name,ckpt,ctx in (("SG-Teacher",TEACHER_SG,"self"),("CG-Teacher",TEACHER,"counter")):
        m,_=load_model(ckpt,"teacher"); table[name]={"metrics":metric_rows(teacher_pred(m,test,ctx)),"best_epoch":None,
            "checkpoint":str(ckpt),"context":ctx,"protocol_status":"legacy masked train=1233/validation=137; reused without retraining"}
        prog=ROOT/"experiments/windows_main/geometry_transfer_restored_v1"/("SG_progress.json" if name.startswith("SG-") else "CG_progress.json")
        if prog.exists():
            pd=json.loads(prog.read_text(encoding="utf-8")); table[name]["best_epoch"]=pd.get("best_epoch")
        if name=="CG-Teacher":
            table["CG-shuffle"]={"metrics":metric_rows(teacher_pred_shuffled(m,test)),"best_epoch":table[name]["best_epoch"],
                "checkpoint":str(ckpt),"context":"batchwise counter permutation; fixed seed 42+1701",
                "protocol_status":"diagnostic intervention; no retraining"}
    # Main wide table, metrics preserved at full precision in JSON.
    methods=("SG-Teacher","CG-Teacher","CG-shuffle","S0-mask","S2-mask","SGRI-A1","SGRI-A2","POWC-B1","POWC-B2")
    cols=["Method","BestEpoch"]+[f"{r}_{m}" for r in ("BothValid","TargetValid","OverallValid") for m in ("AUPRC","IoU","F1")]
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for name in methods:
            row={"Method":name,"BestEpoch":table[name].get("best_epoch")}
            for region in ("BothValid","TargetValid","OverallValid"):
                for met in ("AUPRC","IoU","F1"): row[f"{region}_{met}"]=table[name]["metrics"][region][met]
            w.writerow(row)
    result={"protocol":"v4 train=1370/test=343/no validation; Adam 5e-5 batch16; test raw Overall AUPRC monitor; patience2; max100; seed42; target geometry>0 excluded from all new Student segmentation losses and KD queries",
       "geometry":{"0":"valid","1":"layover","2":"shadow","3":"layover+shadow"},"teacher_guide":"masked CG reused from legacy 1233/137 split; frozen; not retrained",
       "metrics":table,"mechanisms":mechanisms,"best_epochs":best_epochs,
       "teacher_deltas":{"CG_minus_SG":{"BothValid":{k:table['CG-Teacher']['metrics']['BothValid'][k]-table['SG-Teacher']['metrics']['BothValid'][k] for k in ('AUPRC','IoU','F1')},"TargetValid":{k:table['CG-Teacher']['metrics']['TargetValid'][k]-table['SG-Teacher']['metrics']['TargetValid'][k] for k in ('AUPRC','IoU','F1')}},
          "CG_minus_shuffle":{"BothValid":{k:table['CG-Teacher']['metrics']['BothValid'][k]-table['CG-shuffle']['metrics']['BothValid'][k] for k in ('AUPRC','IoU','F1')},"TargetValid":{k:table['CG-Teacher']['metrics']['TargetValid'][k]-table['CG-shuffle']['metrics']['TargetValid'][k] for k in ('AUPRC','IoU','F1')}}},
       "tgd_role":"TGD_G10 is diagnostic only and excluded from optimization; OverallValid is target valid OR counter valid.",
       "test_selection_risk":"Project v4 selects early stopping/best epoch using test AUPRC; region metrics are therefore not strictly untouched test estimates."}
    write_json(OUT/"final_metrics.json",result)
    print(json.dumps({"comparison":str(OUT/"comparison.csv"),"best_epochs":best_epochs,"mechanisms":mechanisms},indent=2,ensure_ascii=False))

if __name__=="__main__": main()
