"""Masked geometry diagnostic: S0/S2 controls, SGRI-A1/A2 and POWC-B1/B2.

This runner follows the active Windows v4 train/test monitoring and optimizer
contract. Geometry enters loss masks only; it is never a model input.
"""
from __future__ import annotations
import json, random, hashlib, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from paths import DEV
from windows_main.data import HaitiPairs, split
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2, load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2

OUT = ROOT / "experiments/windows_main/sgri_powc_masked_v4_seed42_20260925"
INIT = ROOT / "experiments/windows_main/da_search/SO_backbone_seed42.pt"
TEACHER = ROOT / "experiments/windows_main/geometry_transfer_restored_v1/CG_seed42_bestval.pt"
SEED, BATCH, LR, MAX_EPOCHS, PATIENCE, WD = 42, 16, 5e-5, 100, 2, 0.0
ARMS = ("S0-mask", "S2-mask", "SGRI-A1", "SGRI-A2", "POWC-B1", "POWC-B2")
EPS = 1e-8

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)

def masked_bce(logits, y, valid):
    px = F.binary_cross_entropy_with_logits(logits[:, 0], y, reduction="none")
    m = valid.to(px.dtype)
    return (px * m).sum() / m.sum().clamp_min(1.0), px

def query_valid(g, size=32):
    # Every 4x4 target pixel under the 1/4-resolution query must be valid.
    invalid = (g > 0).float().unsqueeze(1)
    return F.max_pool2d(invalid, kernel_size=4, stride=4)[:, 0] == 0

def spatial_policy(p, memory):
    """Splat the existing deformable policy onto Student memory query cells."""
    off = 2 * torch.tanh(p["offsets"]); wt = p["weights"]
    b, lv, k, _, h, w = off.shape; q = h * w
    yy, xx = torch.meshgrid(torch.arange(h, device=off.device), torch.arange(w, device=off.device), indexing="ij")
    by, bx = yy.reshape(1, q, 1).float(), xx.reshape(1, q, 1).float()
    out = torch.zeros(b, q, q, device=off.device)
    for l, mem in enumerate(memory):
        scale = h / mem.shape[-2]
        oy = off[:, l, :, 0].permute(0, 2, 3, 1).reshape(b, q, k) * scale
        ox = off[:, l, :, 1].permute(0, 2, 3, 1).reshape(b, q, k) * scale
        y, x = (by + oy).clamp(0, h - 1), (bx + ox).clamp(0, w - 1)
        y0, x0 = y.floor().long(), x.floor().long(); fy, fx = y - y0, x - x0
        a = wt[:, l].permute(0, 2, 3, 1).reshape(b, q, k)
        for yi, xi, c in ((y0, x0, (1-fy)*(1-fx)), (y0, (x0+1).clamp(max=w-1), (1-fy)*fx),
                          ((y0+1).clamp(max=h-1), x0, fy*(1-fx)),
                          ((y0+1).clamp(max=h-1), (x0+1).clamp(max=w-1), fy*fx)):
            out.scatter_add_(2, yi*w + xi, a*c)
    return out / out.sum(-1, keepdim=True).clamp_min(EPS)

def js_distance(a, b):
    m = .5 * (a+b)
    return .5*((a*(a.add(EPS).log()-m.add(EPS).log())).sum(-1) +
               (b*(b.add(EPS).log()-m.add(EPS).log())).sum(-1))

def make_student():
    m = StudentTargetSearchV2().to(DEV)
    load_so_backbone(m, str(INIT))
    return m

def make_teacher():
    t = TargetSearchTeacherV2().to(DEV)
    t.load_state_dict(torch.load(TEACHER, map_location=DEV, weights_only=True))
    t.eval()
    for p in t.parameters(): p.requires_grad_(False)
    return t

class PairedVerifier(nn.Module):
    """One training-only target-memory verifier; never called at deployment."""
    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Conv2d(channels*3, channels*2, 1)
    def forward(self, a, d):
        x = torch.cat((a-d, (a-d).abs(), a*d), dim=1)
        qa, qd = self.net(x).chunk(2, dim=1)
        return qa, qd

def evaluate(s, ds, mode="single", verifier=None):
    s.eval()
    if verifier is not None: verifier.eval()
    ps, ys, gas, gds = [], [], [], []
    with torch.no_grad():
        for a,d,y,ga,gd,_ in loader(ds):
            a,d = a.to(DEV),d.to(DEV)
            oa, od = s(a), s(d)
            if mode == "paired" and verifier is not None:
                qa, qd = verifier(oa["T"][0], od["T"][0])
                oa_z = s.decode_R(oa["R"] + qa, a.shape[-2:])
                od_z = s.decode_R(od["R"] + qd, d.shape[-2:])
            else:
                oa_z, od_z = oa["z"], od["z"]
            ps.extend((torch.sigmoid(oa_z[:,0]).cpu().numpy(), torch.sigmoid(od_z[:,0]).cpu().numpy()))
            ys.extend((y.numpy(),y.numpy())); gas.extend((ga.numpy(),gd.numpy())); gds.extend((gd.numpy(),ga.numpy()))
    return tuple(np.concatenate(x,axis=0) for x in (ps,ys,gas,gds))

def region_mask(g,c,name):
    if name == "BothValid": return (g == 0) & (c == 0)
    if name == "TargetValid": return g == 0
    if name == "OverallValid": return (g == 0) | (c == 0)
    if name == "TGD_G10": return (g > 0) & (c == 0)
    if name == "Overall": return np.ones_like(g,dtype=bool)
    raise ValueError(name)

def metric_rows(pred):
    p,y,g,c = pred; out={}
    for region in ("BothValid","TargetValid","OverallValid","TGD_G10","Overall"):
        m=region_mask(g,c,region).reshape(-1); yy=y.reshape(-1)[m].astype(bool); pp=p.reshape(-1)[m]
        prob=pp>=.5; tp=int(np.logical_and(prob,yy).sum()); fp=int(np.logical_and(prob,~yy).sum()); fn=int(np.logical_and(~prob,yy).sum())
        out[region]={"AUPRC":float(average_precision_score(yy,pp)),"IoU":tp/(tp+fp+fn+EPS),"F1":2*tp/(2*tp+fp+fn+EPS),"pixels":int(m.sum())}
    return out

def ap_all(pred):
    p,y,_,_=pred
    return float(average_precision_score(y.reshape(-1).astype(bool),p.reshape(-1)))

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")

def prediction_losses(s,a,d,y,ga,gd,orbit_mode="both"):
    oa=od=None
    if orbit_mode in ("both","asc"): oa=s(a)
    if orbit_mode in ("both","desc"): od=s(d)
    terms=[]
    if oa is not None: terms.append(masked_bce(oa["z"],y,ga==0))
    if od is not None: terms.append(masked_bce(od["z"],y,gd==0))
    return oa,od,terms

def train_arm(name, train, test, teacher=None):
    out_ckpt=OUT/f"{name}_seed42_best.pt"; prog=OUT/f"{name}_progress.json"
    if out_ckpt.exists() and prog.exists() and json.loads(prog.read_text(encoding="utf-8")).get("complete"):
        print(f"SKIP complete {name}",flush=True); return
    set_seed(); s=make_student()
    verifier=PairedVerifier().to(DEV) if name.startswith("POWC") else None
    params=list(s.parameters())+(list(verifier.parameters()) if verifier is not None else [])
    opt=torch.optim.Adam(params,lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=WD)
    best=-1.; bad=0; best_state=None; best_epoch=None; history=[]; calibration=None; single_counter=0
    # Guidance is phased over a 30-epoch convergence horizon, based on the
    # existing canonical S0 (19) and CG10 (28) best epochs; no sweep/selection.
    for ep in range(1,MAX_EPOCHS+1):
        s.train()
        if verifier is not None: verifier.train()
        totals=[]; diagnostics={"self_loss":[],"guide_loss":[],"guide_benefit":[],"guide_alpha":[],"paired_loss":[],"single_loss":[],"beneficial_ratio":[]}
        for a,d,y,ga,gd,_ in loader(train,True):
            a,d,y,ga,gd=[x.to(DEV) for x in (a,d,y,ga,gd)]
            opt.zero_grad(set_to_none=True)
            if name in ("S0-mask","S2-mask"):
                oa,od,terms=prediction_losses(s,a,d,y,ga,gd,"both")
                seg=.5*(terms[0][0]+terms[1][0])
                loss=seg
                if name=="S2-mask":
                    with torch.no_grad(): ta,td=teacher.forward_pair(a,d,"counter")
                    kd=[]; raw=[]
                    for o,tp,g in ((oa,ta["policy"],ga),(od,td["policy"],gd)):
                        mem=tuple(v.detach() for v in o["values"])
                        ps=s.policy(s.context(tuple(v.detach() for v in o["T"])))
                        p_self=spatial_policy(ps,mem); p_teacher=spatial_policy({k:v.detach() for k,v in tp.items() if k in ("offsets","weights")},mem)
                        js=js_distance(p_self,p_teacher)
                        v_self=s.search(mem,ps); v_teacher=s.search(mem,{k:v.detach() for k,v in tp.items() if k in ("offsets","weights")})
                        ev=1-F.cosine_similarity(v_self,v_teacher,dim=1)
                        qm=query_valid(g); qmf=qm.reshape(qm.shape[0],-1)
                        kd.extend(((js*qmf).sum()/qmf.sum().clamp_min(1),(ev*qm).sum()/qm.sum().clamp_min(1)))
                        raw.append((float(((js*qmf).sum()/qmf.sum().clamp_min(1)).detach()),float(((ev*qm).sum()/qm.sum().clamp_min(1)).detach())))
                    if calibration is None:
                        calibration={"policy":max(sum(x[0] for x in raw)/2,EPS),"evidence":max(sum(x[1] for x in raw)/2,EPS),"seg":float(seg.detach())}
                    kd_norm=.5*(kd[0]/calibration["policy"]+kd[2]/calibration["policy"]+kd[1]/calibration["evidence"]+kd[3]/calibration["evidence"])
                    loss=seg+0.10*calibration["seg"]*kd_norm
                    diagnostics.setdefault("policy_js",[]).append(float((.5*(kd[0]+kd[2])).detach()))
                    diagnostics.setdefault("evidence_distance",[]).append(float((.5*(kd[1]+kd[3])).detach()))
                diagnostics["self_loss"].append(float(seg.detach()))
            elif name.startswith("SGRI"):
                with torch.no_grad(): ta,td=teacher.forward_pair(a,d,"counter")
                outs=(s(a),s(d)); guided=[]; self_terms=[]; guide_terms=[]; masks=[]
                for o,tp,g in zip(outs,(ta,td),(ga,gd)):
                    ls,pxs=masked_bce(o["z"],y,g==0)
                    rt=s.search(o["values"],{k:v.detach() for k,v in tp["policy"].items() if k in ("offsets","weights")})
                    zg=s.decode_R(rt,y.shape[-2:]); lg,pxg=masked_bce(zg,y,g==0)
                    m=(pxg.detach()<pxs.detach()) & (g==0)
                    self_terms.append((ls,pxs));guide_terms.append((lg,pxg));masks.append(m)
                aweight=1.0 if ep<=9 else (1.0-.75*(ep-9)/12 if ep<=21 else max(.25*(1-(ep-21)/9),0.0))
                segself=.5*(self_terms[0][0]+self_terms[1][0]); segguide=.5*(guide_terms[0][0]+guide_terms[1][0])
                if name=="SGRI-A1": guide_obj=segguide
                else:
                    per=[]
                    for (_,pxs),(_,pxg),m in zip(self_terms,guide_terms,masks):
                        per.append((pxg*m).sum()/m.sum().clamp_min(1.0))
                    guide_obj=.5*(per[0]+per[1])
                loss=segself+aweight*guide_obj
                diagnostics["self_loss"].append(float(segself.detach())); diagnostics["guide_loss"].append(float(segguide.detach()))
                diagnostics["guide_benefit"].append(float((segguide-segself).detach())); diagnostics["guide_alpha"].append(aweight)
                diagnostics["beneficial_ratio"].append(float(.5*(masks[0].float().mean()+masks[1].float().mean())))
            else:  # POWC
                paired=(random.random()<.5)
                if paired:
                    oa,od,terms=prediction_losses(s,a,d,y,ga,gd,"both")
                    qa,qd=verifier(oa["T"][0],od["T"][0])
                    za=s.decode_R(oa["R"]+qa,y.shape[-2:]); zd=s.decode_R(od["R"]+qd,y.shape[-2:])
                    pair_a,pxpa=masked_bce(za,y,(ga==0)&(gd==0)); pair_d,pxpd=masked_bce(zd,y,(gd==0)&(ga==0))
                    single_a,pxsa=masked_bce(oa["z"],y,ga==0); single_d,pxsd=masked_bce(od["z"],y,gd==0)
                    if name=="POWC-B1": pair_obj=pair_a+pair_d
                    else:
                        ma=(pxpa.detach()<pxsa.detach())&(ga==0)&(gd==0); md=(pxpd.detach()<pxsd.detach())&(gd==0)&(ga==0)
                        pair_obj=(pxpa*ma).sum()/ma.sum().clamp_min(1)+(pxpd*md).sum()/md.sum().clamp_min(1)
                        diagnostics["beneficial_ratio"].append(float(.5*(ma.float().mean()+md.float().mean())))
                    loss=single_a+single_d+pair_obj
                    diagnostics["single_loss"].append(float((.5*(single_a+single_d)).detach())); diagnostics["paired_loss"].append(float((.5*(pair_a+pair_d)).detach()))
                else:
                    single_counter+=1; use_asc=(single_counter%2==1)
                    oa,od,terms=prediction_losses(s,a,d,y,ga,gd,"asc" if use_asc else "desc")
                    loss=terms[0][0]
                    diagnostics["single_loss"].append(float(loss.detach()))
            loss.backward(); opt.step(); totals.append(float(loss.detach()))
        pred_self=evaluate(s,test)
        score=ap_all(pred_self)
        row={"epoch":ep,"train_loss":float(np.mean(totals)),"monitor_test_auprc_raw_overall":score,
             "valid_metrics":metric_rows(pred_self),"mechanism":{k:(float(np.mean(v)) if v else None) for k,v in diagnostics.items()}}
        if name.startswith("SGRI"):
            row["guide_test_metrics"]=metric_rows(evaluate(s,test,"paired")) if False else None
            # Test-time diagnostic only: apply frozen paired Teacher policy on Student target memory.
            row["guided_path_test_metrics"]=guided_test(s,teacher,test)
        elif name.startswith("POWC"):
            row["paired_guided_test_metrics"]=metric_rows(evaluate(s,test,"paired",verifier))
        history.append(row)
        if score>best:
            best,bad,best_epoch=score,0,ep
            best_state={"student":{k:v.detach().cpu().clone() for k,v in s.state_dict().items()},
                        "verifier":None if verifier is None else {k:v.detach().cpu().clone() for k,v in verifier.state_dict().items()}}
        else: bad+=1
        payload={"arm":name,"seed":SEED,"best_test_auprc_raw_overall":best,"best_epoch":best_epoch,
                 "epoch":ep,"patience":PATIENCE,"max_epochs":MAX_EPOCHS,"monitor_risk":"test AUPRC selects checkpoint under canonical project v4 rule",
                 "optimizer":{"name":"Adam","lr":LR,"betas":[.9,.999],"eps":1e-8,"weight_decay":WD},
                 "batch_size":BATCH,"init":str(INIT),"teacher":str(TEACHER) if name in ("S2-mask","SGRI-A1","SGRI-A2") else None,
                 "geometry_loss_mask":"per-orbit target geometry code == 0; all codes >0 excluded",
                 "history":history,"complete":False}
        write_json(prog,payload)
        print(f"{name} epoch={ep} monitor_AP={score:.6f} best={best:.6f}@{best_epoch}",flush=True)
        if bad>=PATIENCE: break
    if best_state is None: raise RuntimeError(f"No checkpoint selected for {name}")
    torch.save(best_state,out_ckpt)
    payload["complete"]=True;payload["checkpoint"]=str(out_ckpt)
    write_json(prog,payload)
    print(f"COMPLETE {name} best={best:.6f}@{best_epoch}",flush=True)

@torch.no_grad()
def guided_test(s,teacher,test):
    s.eval();teacher.eval();ps=[];ys=[];gas=[];gds=[]
    for a,d,y,ga,gd,_ in loader(test):
        a,d=a.to(DEV),d.to(DEV); ta,td=teacher.forward_pair(a,d,"counter")
        oa,od=s(a),s(d)
        za=s.decode_R(s.search(oa["values"],{k:v for k,v in ta["policy"].items() if k in ("offsets","weights")}),y.shape[-2:])
        zd=s.decode_R(s.search(od["values"],{k:v for k,v in td["policy"].items() if k in ("offsets","weights")}),y.shape[-2:])
        ps.extend((torch.sigmoid(za[:,0]).cpu().numpy(),torch.sigmoid(zd[:,0]).cpu().numpy()));ys.extend((y.numpy(),y.numpy()));gas.extend((ga.numpy(),gd.numpy()));gds.extend((gd.numpy(),ga.numpy()))
    return metric_rows(tuple(np.concatenate(x,axis=0) for x in (ps,ys,gas,gds)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tid,vid,eid=split()
    if len(vid)!=0: raise RuntimeError(f"Expected canonical no-validation split, got {len(vid)} val ids")
    train=HaitiPairs(tid,True); test=HaitiPairs(eid,False,train.stats)
    metadata={"protocol":"active Windows formal v4; train=1370, test=343, no validation; test AUPRC monitor; max 100; patience 2",
      "seed":SEED,"train_ids":len(tid),"test_ids":len(eid),"val_ids":len(vid),"train_modes":len(train),"init":str(INIT),"teacher":str(TEACHER),
      "geometry_codes":{"0":"valid","1":"layover","2":"shadow","3":"layover+shadow"},
      "region_rules":{"BothValid":"target==0 & counter==0","TargetValid":"target==0","OverallValid":"target==0 | counter==0","TGD_G10":"target>0 & counter==0, diagnostic only"},
      "evaluation_note":"Test AUPRC selects best epoch under project v4 protocol; geometry-region metrics are reported post hoc.",
      "guide_checkpoint_provenance":"Frozen geometry-masked CG from geometry_transfer_restored_v1; trained with target geometry >0 excluded, but on legacy train=1233/validation=137 and validation selection. Reused to honor no-Teacher-retraining; guide provenance is not v4-matched.",
      "POWC_verifier_init":"standard Conv2d Kaiming initialization (same seed and verifier initialization for B1/B2); zero-init dead-gate pilot preserved under zero_init_gate_reference.",
      "fixed30_separation":"This run uses a new v4-tagged directory; the prior fixed-30 diagnostic remains archived separately."}
    write_json(OUT/"protocol.json",metadata)
    if not TEACHER.exists(): raise FileNotFoundError(TEACHER)
    teacher=make_teacher()
    for arm in ARMS: train_arm(arm,train,test,teacher)
    print("ALL ARMS COMPLETE",flush=True)

if __name__=="__main__": main()
