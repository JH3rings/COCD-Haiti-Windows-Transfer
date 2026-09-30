"""Masked geometry-transfer experiment under the project's original convergence rule.

This is an experiment runner only. Canonical Teacher/Student model definitions
are untouched. Test data is opened only after every best-validation checkpoint
has been locked.
"""
from __future__ import annotations

import csv, hashlib, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from paths import DEV, DATASET_ROOT, SPLIT_DIR
from windows_main.data import HaitiPairs, split
from windows_main.models_da_search import CHANNELS
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2, load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.teacher_target_search_v2 import metric as pixel_metric
from windows_main.student_search_kd_v2 import spatial_policy, js, EPS

OUT = ROOT / "experiments" / "windows_main" / "geometry_transfer_restored_v1"
SO_INIT = ROOT / "experiments" / "windows_main" / "da_search" / "SO_backbone_seed42.pt"
SEED, BATCH, LR, WD, MAX_EPOCHS, PATIENCE, THRESHOLD = 42, 16, 5e-5, 0.0, 100, 2, 0.5
KD_FRACTION = 0.10
ARMS = ("SG", "CG", "S0", "S2", "ReliabilityKD", "StableEvidenceKD")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def strict_qmask(pixel_mask):
    """A 32x32 query is valid only when all 4x4 supervised pixels are valid."""
    x = pixel_mask.float()[:, None]
    return F.avg_pool2d(x, 4, 4)[:, 0] >= (1.0 - 1e-6)


def masked_bce(z, y, valid):
    pix = F.binary_cross_entropy_with_logits(z[:, 0], y, reduction="none")
    m = valid.bool()
    return pix[m].mean() if m.any() else pix.sum() * 0.0


def paired_valid_loss(oa, od, y, ga, gd):
    va, vd = ga == 0, gd == 0
    return 0.5 * (masked_bce(oa["z"], y, va) + masked_bce(od["z"], y, vd))


def optimiser(model):
    return torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)


def val_score_teacher(model, ds, context):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for a,d,y,ga,gd,_ in loader(ds):
            oa,od=model.forward_pair(a.to(DEV),d.to(DEV),context)
            ps.append(torch.sigmoid(torch.cat((oa["z"],od["z"])))[:,0].cpu().numpy())
            ys.append(torch.cat((y,y)).numpy())
    return float(average_precision_score(np.concatenate(ys).astype(bool).ravel(),np.concatenate(ps).ravel()))


def val_score_student(model, ds):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for a,d,y,ga,gd,_ in loader(ds):
            ps.extend([torch.sigmoid(model(a.to(DEV))["z"])[:,0].cpu().numpy(),
                       torch.sigmoid(model(d.to(DEV))["z"])[:,0].cpu().numpy()])
            ys.extend([y.numpy(),y.numpy()])
    return float(average_precision_score(np.concatenate(ys).astype(bool).ravel(),np.concatenate(ps).ravel()))


def train_teacher(name, context, train, val):
    final=OUT/f"{name}_seed{SEED}_bestval.pt"; prog=OUT/f"{name}_progress.json"
    if final.exists() and prog.exists() and json.loads(prog.read_text()).get("complete"):
        return final, json.loads(prog.read_text())
    seed_all(); model=TargetSearchTeacherV2().to(DEV); load_so_backbone(model,str(SO_INIT)); opt=optimiser(model)
    best=-1.; best_ep=0; bad=0; hist=[]; best_state=None
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); losses=[]
        for a,d,y,ga,gd,_ in loader(train,True):
            a,d,y,ga,gd=a.to(DEV),d.to(DEV),y.to(DEV),ga.to(DEV),gd.to(DEV)
            oa,od=model.forward_pair(a,d,context); loss=paired_valid_loss(oa,od,y,ga,gd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        score=val_score_teacher(model,val,context); improved=score>best
        if improved:
            best,best_ep,bad=score,ep,0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,final)
        else: bad+=1
        hist.append({"epoch":ep,"train_masked_seg_bce":float(np.mean(losses)),"val_overall_auprc":score,"best_val_auprc":best,"best_epoch":best_ep,"bad_epochs":bad})
        rec={"method":name,"context":context,"seed":SEED,"train_ids":len(train.rows),"validation_ids":len(val.rows),"epoch":ep,"history":hist,"best_val_auprc":best,"best_epoch":best_ep,"complete":False,"selection":"validation overall AUPRC; patience=2; max_epoch=100; test unused"}
        dump(prog,rec); print(f"{name} epoch={ep} train={hist[-1]['train_masked_seg_bce']:.5f} val_AUPRC={score:.6f} best={best:.6f}@{best_ep}",flush=True)
        if bad>=PATIENCE: break
    model.load_state_dict(best_state); torch.save(best_state,final)
    rec.update({"epoch":ep,"complete":True,"checkpoint":str(final),"checkpoint_sha256":sha256(final)})
    dump(prog,rec); return final,rec


class ReliabilityStudent(nn.Module):
    """One reliability head gates the ordinary Student's target-side R."""
    def __init__(self):
        super().__init__(); self.base=StudentTargetSearchV2(); self.reliability=nn.Conv2d(CHANNELS,1,1)

    def forward(self,x):
        out=self.base(x); r=torch.sigmoid(self.reliability(out["R"]))
        out["r_S"]=r; out["z"]=self.base.decode_R(r*out["R"],x.shape[-2:]); return out

    @property
    def backbone(self): return self.base.backbone


def init_student(name):
    seed_all()
    model=ReliabilityStudent().to(DEV) if name=="ReliabilityKD" else StudentTargetSearchV2().to(DEV)
    load_so_backbone(model.base if isinstance(model,ReliabilityStudent) else model,str(SO_INIT))
    return model


def make_targets(teacher, a, d):
    with torch.no_grad():
        ta,td=teacher.forward_pair(a,d,"counter")
        return ta,td


def kd_maps(student, teacher, a, d, ga, gd):
    """Policy JS and evidence distance, restricted to strictly valid target queries."""
    ta,td=make_targets(teacher,a,d); outputs=[student(a),student(d)]; targets=[ta,td]
    geom=[ga,gd]; pl=[]; ev=[]; qms=[]; rs=[]; rt=[]
    for out,to,g in zip(outputs,targets,geom):
        mem=tuple(x.detach() for x in out["values"])
        sp=student.base.policy(student.base.context(tuple(x.detach() for x in out["T"]))) if isinstance(student,ReliabilityStudent) else student.policy(student.context(tuple(x.detach() for x in out["T"])))
        tp={k:v.detach() for k,v in to["policy"].items() if k in ("offsets","weights")}
        P=spatial_policy(sp,mem); T=spatial_policy(tp,mem); qms.append(strict_qmask(g==0).flatten(1))
        pl.append(js(P,T));
        base=student.base if isinstance(student,ReliabilityStudent) else student
        rs.append(base.search(mem,sp)); rt.append(base.search(mem,tp)); ev.append(1-F.cosine_similarity(rs[-1],rt[-1],dim=1))
    def masked_avg(maps,masks):
        vals=[x.reshape(x.shape[0],-1)[m].mean() for x,m in zip(maps,masks) if m.any()]
        return torch.stack(vals).mean() if vals else maps[0].sum()*0
    return masked_avg(pl,qms),masked_avg(ev,qms),outputs,(ta,td),qms,(rs,rt)


def both_valid_qmask(ga,gd): return strict_qmask((ga==0)&(gd==0))


def train_student(name, train, val, teacher):
    final=OUT/f"{name}_seed{SEED}_bestval.pt"; prog=OUT/f"{name}_progress.json"
    if final.exists() and prog.exists() and json.loads(prog.read_text()).get("complete"):
        return final,json.loads(prog.read_text())
    model=init_student(name); opt=optimiser(model); best=-1.; best_ep=0; bad=0; hist=[]; best_state=None
    alpha_p=alpha_e=0.5; reliability_lambda=None; stable_lambda=None; calibration=None
    teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); segs=[]; policies=[]; evidences=[]; auxs=[]
        for a,d,y,ga,gd,_ in loader(train,True):
            a,d,y,ga,gd=a.to(DEV),d.to(DEV),y.to(DEV),ga.to(DEV),gd.to(DEV)
            if name=="S0":
                oa,od=model(a),model(d); seg=0.5*(masked_bce(oa["z"],y,ga==0)+masked_bce(od["z"],y,gd==0)); loss=seg; aux=None
            else:
                if name=="ReliabilityKD":
                    with torch.no_grad(): ta,td=make_targets(teacher,a,d)
                    outputs=[model(a),model(d)]; seg=0.5*(masked_bce(outputs[0]["z"],y,ga==0)+masked_bce(outputs[1]["z"],y,gd==0))
                    # Similarity of two retrieved, target-memory-only evidence maps is the teacher's reliability target.
                    r_t=(1+F.cosine_similarity(ta["R"].detach(),td["R"].detach(),dim=1,eps=1e-8))*0.5
                    q=both_valid_qmask(ga,gd); rel_terms=[]
                    for out in outputs:
                        if q.any(): rel_terms.append(F.l1_loss(out["r_S"][:,0][q],r_t[q]))
                    rel=torch.stack(rel_terms).mean() if rel_terms else seg*0
                    if reliability_lambda is None:
                        reliability_lambda=float(seg.detach()/(rel.detach()+EPS)); calibration={"seg_init":float(seg.detach()),"reliability_init":float(rel.detach()),"lambda_reliability":reliability_lambda,"rule":"one-time equal initial loss contribution"}
                    loss=seg+reliability_lambda*rel; aux=rel
                else:
                    lp,le,outputs,(ta,td),qms,(rs,rt)=kd_maps(model,teacher,a,d,ga,gd)
                    seg=0.5*(masked_bce(outputs[0]["z"],y,ga==0)+masked_bce(outputs[1]["z"],y,gd==0))
                    if name=="S2":
                        if calibration is None:
                            calibration={"policy_init":float(lp.detach()),"evidence_init":float(le.detach()),"seg_init":float(seg.detach()),"kd_fraction":KD_FRACTION,"policy_weight":alpha_p,"evidence_weight":alpha_e}
                        loss=seg+KD_FRACTION*calibration["seg_init"]*(alpha_p*lp/(calibration["policy_init"]+EPS)+alpha_e*le/(calibration["evidence_init"]+EPS)); aux=(lp,le)
                    else:
                        # Stable weight: cross-view teacher policy agreement times target-side evidence similarity.
                        with torch.no_grad():
                            tpa=spatial_policy({k:v for k,v in ta["policy"].items() if k in ("offsets","weights")},tuple(x.detach() for x in outputs[0]["values"]))
                            tpd=spatial_policy({k:v for k,v in td["policy"].items() if k in ("offsets","weights")},tuple(x.detach() for x in outputs[1]["values"]))
                            semantic=(1+F.cosine_similarity(ta["R"].detach(),td["R"].detach(),dim=1,eps=1e-8))*0.5
                            agreement=torch.exp(-js(tpa,tpd)).reshape_as(semantic)
                            wstable=(agreement*semantic).detach()
                        q=both_valid_qmask(ga,gd); stable_terms=[]
                        for r_s,r_t in zip(rs,rt):
                            dist=1-F.cosine_similarity(r_s,r_t.detach(),dim=1,eps=1e-8)
                            if q.any(): stable_terms.append((dist[q]*wstable[q]).sum()/(wstable[q].sum()+EPS))
                        stable=torch.stack(stable_terms).mean() if stable_terms else seg*0
                        if stable_lambda is None:
                            stable_lambda=float(seg.detach()/(stable.detach()+EPS)); calibration={"seg_init":float(seg.detach()),"stable_init":float(stable.detach()),"lambda_stable":stable_lambda,"rule":"one-time equal initial loss contribution","definition":"exp(-teacher cross-view policy JS) times cosine similarity of target-side retrieved evidence"}
                        loss=seg+stable_lambda*stable; aux=stable
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();segs.append(float(seg.detach()))
            if name=="S2": policies.append(float(aux[0].detach()));evidences.append(float(aux[1].detach()))
            elif aux is not None: auxs.append(float(aux.detach()))
        score=val_score_student(model,val); improved=score>best
        if improved:
            best,best_ep,bad=score,ep,0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(best_state,final)
        else: bad+=1
        rec={"epoch":ep,"train_masked_seg_bce":float(np.mean(segs)),"val_overall_auprc":score,"best_val_auprc":best,"best_epoch":best_ep,"bad_epochs":bad}
        if name=="S2": rec.update({"policy_js_train":float(np.mean(policies)),"evidence_distance_train":float(np.mean(evidences))})
        if name in ("ReliabilityKD","StableEvidenceKD"): rec["aux_train"]=float(np.mean(auxs)) if auxs else None
        hist.append(rec); dump(prog,{"method":name,"seed":SEED,"epoch":ep,"history":hist,"best_val_auprc":best,"best_epoch":best_ep,"calibration":calibration,"complete":False,"selection":"validation overall AUPRC; patience=2; max_epoch=100; test unused"})
        print(f"{name} epoch={ep} train={rec['train_masked_seg_bce']:.5f} val_AUPRC={score:.6f} best={best:.6f}@{best_ep}",flush=True)
        if bad>=PATIENCE:break
    model.load_state_dict(best_state);torch.save(best_state,final)
    end={"method":name,"seed":SEED,"epoch":ep,"history":hist,"best_val_auprc":best,"best_epoch":best_ep,"calibration":calibration,"complete":True,"checkpoint":str(final),"checkpoint_sha256":sha256(final),"selection":"validation overall AUPRC; patience=2; max_epoch=100; test unused"}
    dump(prog,end); return final,end


def load_arm(name):
    p=OUT/f"{name}_seed{SEED}_bestval.pt"
    state=torch.load(p,map_location=DEV,weights_only=True)
    if name in ("SG","CG"):
        m=TargetSearchTeacherV2().to(DEV)
    elif name=="ReliabilityKD": m=ReliabilityStudent().to(DEV)
    else: m=StudentTargetSearchV2().to(DEV)
    m.load_state_dict(state);m.eval();return m


@torch.no_grad()
def infer_teacher(model, ds, context, shuffle_counter=False):
    model.eval(); ps=[];ys=[];gs=[];cs=[]; js_maps=[]
    gen=torch.Generator().manual_seed(SEED+1701)
    for a,d,y,ga,gd,_ in loader(ds):
        a,d=a.to(DEV),d.to(DEV); b=len(a)
        if shuffle_counter:
            perm=torch.randperm(b,generator=gen)
            if torch.equal(perm,torch.arange(b)):perm=torch.roll(perm,1)
            oa,od=model(a,d[perm]),model(d,a[perm])
        else: oa,od=model.forward_pair(a,d,context)
        ps.append(torch.sigmoid(torch.cat((oa["z"],od["z"])))[:,0].cpu().numpy());ys.append(torch.cat((y,y)).numpy())
        gs.append(torch.cat((ga,gd)).numpy());cs.append(torch.cat((gd,ga)).numpy())
    return {"p":np.concatenate(ps),"y":np.concatenate(ys),"g":np.concatenate(gs),"c":np.concatenate(cs)}


@torch.no_grad()
def infer_student(model, ds):
    model.eval();ps=[];ys=[];gs=[];cs=[]
    for a,d,y,ga,gd,_ in loader(ds):
        ps.extend([torch.sigmoid(model(a.to(DEV))["z"])[:,0].cpu().numpy(),torch.sigmoid(model(d.to(DEV))["z"])[:,0].cpu().numpy()])
        ys.extend([y.numpy(),y.numpy()]);gs.extend([ga.numpy(),gd.numpy()]);cs.extend([gd.numpy(),ga.numpy()])
    return {"p":np.concatenate(ps),"y":np.concatenate(ys),"g":np.concatenate(gs),"c":np.concatenate(cs)}


def metric_regions(pred):
    g,c=pred["g"],pred["c"]
    masks={"BothValid":(g==0)&(c==0),"TargetValid":g==0,"OverallValid":(g==0)|(c==0),"TGD_G10":(g>0)&(c==0)}
    rows={}
    for name,m in masks.items():
        z=pixel_metric(pred["p"],pred["y"],m); rows[name]={k:z[k] for k in ("pixels","positive_pixels","positive_fraction","auprc","iou","f1")}
    return rows


@torch.no_grad()
def student_mechanisms(student, teacher, ds):
    student.eval();teacher.eval(); sums={"policy_js_both":[],"policy_js_target":[],"evidence_both":[],"evidence_target":[],"reliability_alignment_both":[],"stable_evidence_alignment_both":[]}
    for a,d,y,ga,gd,_ in loader(ds):
        a,d,ga,gd=a.to(DEV),d.to(DEV),ga.to(DEV),gd.to(DEV);ta,td=teacher.forward_pair(a,d,"counter")
        outs=[student(a),student(d)]; tos=[ta,td]; geoms=[ga,gd]; policies=[]; evidence=[]; r_s=[]
        base=student.base if isinstance(student,ReliabilityStudent) else student
        for o,t,g in zip(outs,tos,geoms):
            mem=tuple(x.detach() for x in o["values"]);sp=base.policy(base.context(tuple(x.detach() for x in o["T"])))
            tp={k:v for k,v in t["policy"].items() if k in ("offsets","weights")};P=spatial_policy(sp,mem);T=spatial_policy(tp,mem)
            policies.append(js(P,T));evidence.append(1-F.cosine_similarity(base.search(mem,sp),base.search(mem,tp),dim=1))
            if isinstance(student,ReliabilityStudent):r_s.append(o["r_S"][:,0])
        qv=[strict_qmask(ga==0).flatten(1),strict_qmask(gd==0).flatten(1)]; qb=both_valid_qmask(ga,gd); qb_flat=qb.flatten(1)
        for key,maps in (("policy_js",policies),("evidence",evidence)):
            for rg,mask in (("target",qv),("both",[qb_flat,qb_flat])):
                vals=[x.reshape(x.shape[0],-1)[m].mean() for x,m in zip(maps,mask) if m.any()]
                if vals:sums[f"{key}_{rg}"].append(float(torch.stack(vals).mean()))
        if isinstance(student,ReliabilityStudent):
            rt=(1+F.cosine_similarity(ta["R"],td["R"],dim=1,eps=1e-8))*0.5
            if qb.any(): sums["reliability_alignment_both"].append(float(torch.stack([(x[qb]-rt[qb]).abs().mean() for x in r_s]).mean()))
        # Cross-view stable weighting, aligned with training definition.
        tpa=spatial_policy({k:v for k,v in ta["policy"].items() if k in ("offsets","weights")},tuple(x.detach() for x in outs[0]["values"]))
        tpd=spatial_policy({k:v for k,v in td["policy"].items() if k in ("offsets","weights")},tuple(x.detach() for x in outs[1]["values"]))
        semantic=(1+F.cosine_similarity(ta["R"],td["R"],dim=1,eps=1e-8))*0.5
        w=(torch.exp(-js(tpa,tpd)).reshape_as(semantic)*semantic)
        # Similarity between Student's policy evidence and the teacher-policy evidence in the same target memory.
        align=[]
        for o,t in zip(outs,tos):
            mem=tuple(x.detach() for x in o["values"]);p=base.policy(base.context(tuple(x.detach() for x in o["T"])))
            tp={k:v for k,v in t["policy"].items() if k in ("offsets","weights")}
            rs=base.search(mem,p);rt=base.search(mem,tp);align.append(F.cosine_similarity(rs,rt,dim=1))
        if qb.any():
            den=(w[qb].sum()+EPS);sums["stable_evidence_alignment_both"].append(float((0.5*((align[0][qb]*w[qb]).sum()+(align[1][qb]*w[qb]).sum())/den).detach()))
    return {k:(float(np.mean(v)) if v else None) for k,v in sums.items()}


def geometry_summary(train, val, test):
    # Use the cached training/val/test geometry arrays to establish all observed codes and area fractions.
    rows={}
    for name,ds in (("train",train),("validation",val),("test",test)):
        codes={"target":[],"counter":[]}; counts={"BothValid":0,"TargetValid":0,"OverallValid":0,"TGD_G10":0,"TargetLayover":0,"TargetShadow":0,"TargetLayoverShadow":0,"CounterLayover":0,"CounterShadow":0,"CounterLayoverShadow":0}
        for _,_,ga,gd in ds.rows:
            codes["target"].extend(np.unique(ga).astype(int).tolist());codes["counter"].extend(np.unique(gd).astype(int).tolist())
            target=np.concatenate((ga[None],gd[None]),axis=0);counter=np.concatenate((gd[None],ga[None]),axis=0)
            both=(target==0)&(counter==0);tv=target==0;ov=(target==0)|(counter==0);tgd=(target>0)&(counter==0)
            for k,m in (("BothValid",both),("TargetValid",tv),("OverallValid",ov),("TGD_G10",tgd)):counts[k]+=int(m.sum())
            counts["TargetLayover"]+=int(((target==1)).sum());counts["TargetShadow"]+=int((target==2).sum());counts["TargetLayoverShadow"]+=int((target==3).sum())
            counts["CounterLayover"]+=int((counter==1).sum());counts["CounterShadow"]+=int((counter==2).sum());counts["CounterLayoverShadow"]+=int((counter==3).sum())
        total=2*len(ds.rows)*128*128;rows[name]={"locations":len(ds.rows),"codes_target":sorted(set(codes["target"])),"codes_counter":sorted(set(codes["counter"])),"pixel_counts":counts,"pixel_fractions":{k:v/total for k,v in counts.items() if k in ("BothValid","TargetValid","OverallValid","TGD_G10")}}
    return rows


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--arms",nargs="+",choices=ARMS+ ("evaluate",),default=list(ARMS));ap.add_argument("--force",action="store_true");ap.add_argument("--geometry-only",action="store_true");args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    # Require the preserved validation split; its train+val IDs and test IDs are exactly the active v3 split.
    tid,vid,eid=split();
    if not vid: raise RuntimeError("No validation IDs: set COCD_SPLIT_DIR to the preserved data/splits directory; never select on test.")
    train=HaitiPairs(tid,True); val=HaitiPairs(vid,False,train.stats); test=HaitiPairs(eid,False,train.stats)
    metadata={"root":str(ROOT),"device":str(DEV),"torch":torch.__version__,"seed":SEED,"batch_size":BATCH,"optimizer":{"name":"Adam","lr":LR,"weight_decay":WD,"betas":[.9,.999],"eps":1e-8},"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"selection_metric":"validation overall AUPRC; test not used for checkpoint selection","split_dir":str(SPLIT_DIR),"split_sizes":{"train":len(tid),"validation":len(vid),"test":len(eid)},"v3_union_verified":True,"initialization":{"checkpoint":str(SO_INIT),"sha256":sha256(SO_INIT),"source":"official teacher/student scripts load this shared ConvNeXt-Tiny backbone initialization; its architecture constructor uses torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1"},"geometry_codes":{"0":"valid","1":"layover","2":"shadow","3":"layover+shadow"},"invalid_loss_rule":"target geometry code > 0 excluded from segmentation BCE; all KD query cells require all underlying 4x4 target pixels valid; both-view KD/reliability/stable objectives require both codes 0","regions":{"BothValid":"target==0 and counter==0","TargetValid":"target==0","OverallValid":"target==0 or counter==0","TGD_G10":"target>0 and counter==0; diagnostic only"},"geometry_summary":geometry_summary(train,val,test)}
    dump(OUT/"protocol_and_geometry.json",metadata)
    if args.geometry_only:
        print(f"Geometry metadata updated in {OUT/'protocol_and_geometry.json'}",flush=True); return
    if args.force:
        for p in OUT.glob("*_seed42_bestval.pt"):p.unlink()
        for p in OUT.glob("*_progress.json"):p.unlink()
    if "evaluate" not in args.arms:
        # Load the same frozen initialization and model family in each arm; teacher checkpoints feed only the Student arms.
        for arm in args.arms:
            if arm=="SG": train_teacher(arm,"self",train,val)
            elif arm=="CG": train_teacher(arm,"counter",train,val)
        tpath=OUT/f"CG_seed{SEED}_bestval.pt"
        if any(a in args.arms for a in ("S2","ReliabilityKD","StableEvidenceKD")) and not tpath.exists():
            raise FileNotFoundError("CG checkpoint is required before Student distillation arms")
        teacher=None
        if tpath.exists(): teacher=load_arm("CG")
        for arm in args.arms:
            if arm in ("S0","S2","ReliabilityKD","StableEvidenceKD"):
                if arm!="S0" and teacher is None:raise RuntimeError("distillation needs CG teacher")
                train_student(arm,train,val,teacher)
    # The single test pass occurs only after all requested training checkpoints are locked.
    needed=[a for a in ARMS if (OUT/f"{a}_seed{SEED}_bestval.pt").exists()]
    if all((OUT/f"{a}_seed{SEED}_bestval.pt").exists() for a in ARMS):
        teacher=load_arm("CG"); allrows=[]
        for arm in ("SG","CG"):
            pred=infer_teacher(load_arm(arm),test,"self" if arm=="SG" else "counter")
            allrows.append({"method":arm,"regions":metric_regions(pred)})
            if arm=="CG":
                sh=infer_teacher(load_arm("CG"),test,"counter",True);allrows.append({"method":"CG_shuffle","regions":metric_regions(sh)})
        mech={}
        for arm in ("S0","S2","ReliabilityKD","StableEvidenceKD"):
            student=load_arm(arm);pred=infer_student(student,test);allrows.append({"method":arm,"regions":metric_regions(pred)})
            mech[arm]=student_mechanisms(student,teacher,test)
        dump(OUT/"test_metrics.json",{"test_rows":len(eid),"selected_checkpoints":{a:{"path":str(OUT/f"{a}_seed{SEED}_bestval.pt"),"best_epoch":json.loads((OUT/f"{a}_progress.json").read_text())["best_epoch"],"best_val_auprc":json.loads((OUT/f"{a}_progress.json").read_text())["best_val_auprc"]} for a in ARMS},"metrics":allrows,"mechanism_metrics":mech,"test_opened_after_checkpoint_lock":True})
        with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
            cols=["Method"]+[f"{r}_{m}" for r in ("BothValid","TargetValid","OverallValid","TGD_G10") for m in ("AUPRC","IoU","F1")]
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
            for row in allrows:
                flat={"Method":row["method"]}
                for r,z in row["regions"].items():
                    flat.update({f"{r}_AUPRC":z["auprc"],f"{r}_IoU":z["iou"],f"{r}_F1":z["f1"]})
                w.writerow(flat)
        print(f"Locked evaluation written to {OUT}",flush=True)
    else:
        print(f"Training incomplete; completed checkpoints: {needed}",flush=True)


if __name__=="__main__": main()
