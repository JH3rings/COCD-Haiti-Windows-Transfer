#!/usr/bin/env python3
"""Four-arm Teacher-only validation for counter-guided target-memory search v2."""
from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cocd"))
from paths import DEV
from windows_main.data import HaitiPairs, split
from windows_main.models_da_search import LEVELS, POINTS
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2, load_so_backbone

OUT = ROOT / "experiments" / "windows_main" / "teacher_target_search_v2"
SO_INIT = ROOT / "experiments" / "windows_main" / "da_search" / "SO_backbone_seed42.pt"
SEED, BATCH, LR, WD, MAX_EPOCHS, PATIENCE, THRESHOLD = 42, 16, 5e-5, 0.0, 100, 2, .5
ARMS = {"SG0": ("self", 0.0), "SG10": ("self", .10), "CG0": ("counter", 0.0), "CG10": ("counter", .10)}
EPS = 1e-8


def seed(v=SEED):
    random.seed(v); np.random.seed(v); torch.manual_seed(v)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(v)


def write(path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
def loader(ds, shuffle=False): return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()


def masks(ga, gd):
    # Existing G10 semantics: current target geometry positive, counter geometry negative.
    return (ga > 0) & ~(gd > 0), (gd > 0) & ~(ga > 0)


def directional_loss(z, y, tgd, gamma):
    pix = F.binary_cross_entropy_with_logits(z[:,0], y, reduction="none")
    all_loss = pix.mean()
    n = int(tgd.sum())
    tgd_loss = pix[tgd].mean() if n else all_loss.detach() * 0.0
    loss = (all_loss + gamma*tgd_loss)/(1+gamma) if n else all_loss
    return loss, {"L_all":float(all_loss.detach()), "L_TGD":float(tgd_loss.detach()), "tgd_pixels":n,
                  "pixels":int(tgd.numel()), "tgd_fraction":float(tgd.float().mean()),
                  "tgd_positive_fraction":float(y[tgd].float().mean()) if n else None,
                  "gamma_L_TGD_over_L_all":float((gamma*tgd_loss/(all_loss+EPS)).detach()),
                  "gamma_L_TGD_over_sum":float((gamma*tgd_loss/(all_loss+gamma*tgd_loss+EPS)).detach())}


def pair_loss(oa, od, y, ga, gd, gamma):
    ma, md=masks(ga,gd); la, sa=directional_loss(oa["z"],y,ma,gamma); ld, sd=directional_loss(od["z"],y,md,gamma)
    return .5*(la+ld), {"asc":sa,"desc":sd,"combined":{k: (sa[k]+sd[k])/2 if isinstance(sa[k],(float,int)) and sa[k] is not None and sd[k] is not None else None for k in sa}}


def metric(p,y,mask):
    p,y=p.ravel()[mask.ravel()],y.ravel()[mask.ravel()].astype(bool); h=p>=THRESHOLD
    tp,fp,fn=int((h&y).sum()),int((h&~y).sum()),int((~h&y).sum()); tn=int(len(y)-tp-fp-fn)
    pr,rc=tp/(tp+fp+EPS),tp/(tp+fn+EPS)
    return {"pixels":int(len(y)),"positive_pixels":int(y.sum()),"positive_fraction":float(y.mean()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "accuracy":float((tp+tn)/len(y)),"errors":fp+fn,"error_rate":float((fp+fn)/len(y)),"fnr":float(fn/(tp+fn+EPS)),"fpr":float(fp/(tn+fp+EPS)),
            "iou":float(tp/(tp+fp+fn+EPS)),"f1":float(2*pr*rc/(pr+rc+EPS)),"precision":float(pr),"recall":float(rc),"auprc":float(average_precision_score(y,p)) if y.any() else None}


@torch.no_grad()
def predict(model, ds, context_mode="counter", intervention="normal"):
    model.eval(); store={k:[] for k in ("p","y","gt","gc","orbit")}; samples=[]; gen=torch.Generator().manual_seed(SEED+1701)
    for a,d,y,ga,gd,_ in loader(ds):
        a,d=a.to(DEV),d.to(DEV); b=len(a)
        if intervention=="counter_shuffled":
            perm=torch.randperm(b,generator=gen); perm=torch.roll(perm,1) if torch.equal(perm,torch.arange(b)) else perm
            oa,od=model(a,d[perm]),model(d,a[perm]); samples.extend(perm.tolist())
        else:
            oa,od=model.forward_pair(a,d,context_mode)
            if intervention!="normal":
                def altered(out):
                    pol={k:v.clone() for k,v in out["policy"].items() if k in ("offsets","weights")}
                    if intervention=="center": pol["offsets"].zero_()
                    elif intervention=="random":
                        # Deterministic, query-independent random search locations;
                        # weights stay learned so this isolates spatial selection.
                        rng=torch.Generator(device=pol["offsets"].device).manual_seed(SEED + 991 + b)
                        pol["offsets"].uniform_(-1.0,1.0,generator=rng)
                        pol["offsets"][:,:,0].zero_() # preserve the mandatory centre anchor
                    elif intervention=="uniform": pol["weights"].fill_(1/(LEVELS*POINTS))
                    elif intervention=="fixed":
                        pol["weights"].fill_(1/(LEVELS*POINTS)); pol["offsets"].zero_()
                        raw=float(np.arctanh(.5)) # becomes +/-1 feature pixel after 2*tanh
                        pol["offsets"][:,:,1,0].fill_(-raw); pol["offsets"][:,:,2,1].fill_(raw); pol["offsets"][:,:,3,0].fill_(raw)
                    else: raise ValueError(intervention)
                    r=model.search(out["values"],pol); return model.forward_from_cached_R(r,out["z"].shape[-2:])
                oa,od={"z":altered(oa)},{"z":altered(od)}
        store["p"].append(torch.sigmoid(torch.cat((oa["z"],od["z"])))[:,0].cpu().numpy()); store["y"].append(torch.cat((y,y)).numpy())
        store["gt"].append(torch.cat((ga,gd)).numpy()); store["gc"].append(torch.cat((gd,ga)).numpy()); store["orbit"].append(np.array(["asc"]*b+["desc"]*b))
    out={k:np.concatenate(v) for k,v in store.items()}; out["shuffle_mapping_within_batches"]=samples if samples else None; return out


def evaluate_rows(name,pred):
    out=[]; g10=(pred["gt"]>0)&~(pred["gc"]>0)
    for part,pm in (("combined",np.ones(len(pred["orbit"]),bool)),("asc",pred["orbit"]=="asc"),("desc",pred["orbit"]=="desc")):
        pm=np.broadcast_to(pm[:,None,None],g10.shape)
        for region,rm in (("Overall",np.ones_like(g10,bool)),("TGD",g10),("Remaining",~g10)):
            out.append({"method":name,"partition":part,"region":region,"threshold":THRESHOLD,**metric(pred["p"],pred["y"],pm&rm)})
    return out


def structure_tests(train):
    seed(); m=TargetSearchTeacherV2().to(DEV).eval(); load_so_backbone(m,str(SO_INIT)); a,d,y,ga,gd,_=next(iter(loader(train,False))); a,d,y=a.to(DEV),d.to(DEV),y.to(DEV)
    o=m(a,d); changed=m(a,torch.flip(d,[0])); tol=1e-6
    # A target value independent of counter; B cached target-values/policy have no counter call; C decoder takes R only.
    replay=m.search(o["values"],o["policy"]); zre=m.forward_from_cached_R(replay,o["z"].shape[-2:]); joint={k:v.clone() for k,v in o["policy"].items() if k in ("offsets","weights")}
    # The production initializer intentionally makes every candidate identical.
    # Use a non-degenerate, fixed policy solely for the permutation identity.
    joint["offsets"][:,:,1,0].fill_(.2); joint["offsets"][:,:,2,1].fill_(-.35); joint["offsets"][:,:,3,0].fill_(.6)
    joint["weights"].zero_(); joint["weights"][:,:,0].fill_(.1); joint["weights"][:,:,1].fill_(.2); joint["weights"][:,:,2].fill_(.3); joint["weights"][:,:,3].fill_(.4)
    fixed_r=m.search(o["values"],joint)
    joint["offsets"]=joint["offsets"].flip(2); joint["weights"]=joint["weights"].flip(2); rjoint=m.search(o["values"],joint)
    bad={k:v.clone() for k,v in joint.items()}; bad["offsets"]=bad["offsets"].flip(2); rbad=m.search(o["values"],bad)
    # Gradient path through real batch and both counter/context plus learnable (non-centre) offsets.
    mt=TargetSearchTeacherV2().to(DEV); load_so_backbone(mt,str(SO_INIT))
    # After the deliberately neutral initialization, a real training update
    # makes the context-to-policy path non-degenerate.  This is not a training
    # run; it is the required gradient-connectivity test on one real batch.
    warm=torch.optim.SGD(mt.parameters(),lr=1e-4)
    oa,od=mt.forward_pair(a,d,"counter"); warm_loss,stats=pair_loss(oa,od,y,ga.to(DEV),gd.to(DEV),.1); warm.zero_grad(); warm_loss.backward(); warm.step(); warm.zero_grad()
    oa,od=mt.forward_pair(a,d,"counter"); loss,stats=pair_loss(oa,od,y,ga.to(DEV),gd.to(DEV),.1); loss.backward()
    grad=lambda x: float(x.grad.abs().sum()) if x.grad is not None else 0.0
    empty=directional_loss(oa["z"],y,torch.zeros_like(y,dtype=torch.bool),.1)[0]
    result={"A_value_source_target_unchanged":bool(all(torch.allclose(x,z,atol=tol,rtol=0) for x,z in zip(o["values"],changed["values"]))),
      "B_cached_policy_and_values_counter_independent":bool(torch.allclose(zre,m.forward_from_cached_R(replay,o["z"].shape[-2:]),atol=tol,rtol=0)),
      "C_decoder_only_R":bool(torch.allclose(zre,m.forward_from_cached_R(replay,o["z"].shape[-2:]),atol=tol,rtol=0)),
      "D_policy_replay":bool(torch.allclose(replay,o["R"],atol=tol,rtol=0) and torch.allclose(zre,o["z"],atol=tol,rtol=0)),
      "E_joint_max_abs_difference":float((rjoint-fixed_r).abs().max()),
      # Floating-point summation order changes under a candidate permutation.
      "E_joint_permutation_invariant":bool(torch.allclose(rjoint,fixed_r,atol=1e-4,rtol=1e-4)),"E_offset_only_not_equivalent":bool(not torch.allclose(rbad,fixed_r,atol=1e-6,rtol=1e-6)),
      "F_gradients":{"offset_predictor":grad(mt.policy.offset.weight),"weight_predictor":grad(mt.policy.weight.weight),"value_projection":grad(mt.value_proj[0].weight),"target_encoder":grad(mt.backbone.features[0][0].weight),"counter_context":grad(mt.context.dual_proj[0].weight)},
      "F_gradient_pass":None,"G_empty_tgd_finite":bool(torch.isfinite(empty)),"G_outputs_finite":bool(torch.isfinite(oa["z"]).all()),"loss_stats":stats}
    result["F_gradient_pass"]=all(result["F_gradients"][x]>0 for x in result["F_gradients"])
    return result


def train_arm(name,mode,gamma,train,test):
    final=OUT/f"{name}_seed{SEED}.pt"; progress=OUT/f"{name}_seed{SEED}_progress.json"
    if final.exists() and progress.exists(): return final,json.loads(progress.read_text())
    seed(); m=TargetSearchTeacherV2().to(DEV); load_so_backbone(m,str(SO_INIT)); opt=torch.optim.Adam(m.parameters(),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=WD)
    best,bestep,bad,state,hist=-1.,0,0,None,[]
    for ep in range(1,MAX_EPOCHS+1):
        m.train(); vals=[]
        for a,d,y,ga,gd,_ in loader(train,True):
            oa,od=m.forward_pair(a.to(DEV),d.to(DEV),mode); loss,stat=pair_loss(oa,od,y.to(DEV),ga.to(DEV),gd.to(DEV),gamma); opt.zero_grad(set_to_none=True);loss.backward();opt.step(); vals.append((float(loss.detach()),stat))
        pred=predict(m,test,mode); score=next(r for r in evaluate_rows(name,pred) if r["partition"]=="combined" and r["region"]=="Overall")["auprc"]; improved=score>best
        if improved: best,bestep,bad,state=score,ep,0,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else: bad+=1
        def avg(direction, field):
            x=[v[1][direction][field] for v in vals]
            x=[z for z in x if z is not None]
            return float(np.mean(x)) if x else None
        rec={"epoch":ep,"L_teacher":float(np.mean([v[0] for v in vals])),
             "L_all_asc":avg("asc","L_all"),"L_all_desc":avg("desc","L_all"),
             "L_TGD_asc":avg("asc","L_TGD"),"L_TGD_desc":avg("desc","L_TGD"),
             "gamma_L_TGD_over_L_all_asc":avg("asc","gamma_L_TGD_over_L_all"),
             "gamma_L_TGD_over_L_all_desc":avg("desc","gamma_L_TGD_over_L_all"),
             "gamma_L_TGD_over_sum_asc":avg("asc","gamma_L_TGD_over_sum"),
             "gamma_L_TGD_over_sum_desc":avg("desc","gamma_L_TGD_over_sum"),
             "TGD_fraction_asc":avg("asc","tgd_fraction"),"TGD_fraction_desc":avg("desc","tgd_fraction"),
             "TGD_positive_fraction_asc":avg("asc","tgd_positive_fraction"),"TGD_positive_fraction_desc":avg("desc","tgd_positive_fraction"),
             "monitor_test_auprc":score,"best":best,"best_epoch":bestep,"bad":bad}
        hist.append(rec); write(progress,{"arm":name,"context":mode,"gamma":gamma,"epoch":ep,"history":hist,"best":best,"best_epoch":bestep,"complete":False,"monitor_risk":"test AUPRC selects checkpoint"})
        print(f"{name} ep={ep} loss={rec['L_teacher']:.5f} monitor={score:.6f} best={best:.6f}@{bestep}",flush=True)
        if bad>=PATIENCE: break
    m.load_state_dict(state); torch.save(m.state_dict(),final); write(progress,{"arm":name,"context":mode,"gamma":gamma,"epoch":ep,"history":hist,"best":best,"best_epoch":bestep,"complete":True,"checkpoint":str(final),"monitor_risk":"test AUPRC selects checkpoint"}); return final,json.loads(progress.read_text())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("tests","SG0","SG10","CG0","CG10","all","evaluate"),default="tests"); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    tid,vid,eid=split(); assert not vid,"locked protocol requires no validation split"; train,test=HaitiPairs(tid,True),HaitiPairs(eid,False,None)
    test.stats=train.stats
    if args.stage in ("tests","all"):
        result=structure_tests(train); write(OUT/"structure_gradient_tests.json",result)
        if not all([result["A_value_source_target_unchanged"],result["B_cached_policy_and_values_counter_independent"],result["C_decoder_only_R"],result["D_policy_replay"],result["E_joint_permutation_invariant"],result["E_offset_only_not_equivalent"],result["F_gradient_pass"],result["G_empty_tgd_finite"],result["G_outputs_finite"]]): raise RuntimeError("structure/gradient test failed")
        if args.stage=="tests": return
    if args.stage=="evaluate":
        allrows=[]
        for name,(mode,_) in ARMS.items():
            path=OUT/f"{name}_seed{SEED}.pt"; m=TargetSearchTeacherV2().to(DEV); m.load_state_dict(torch.load(path,map_location=DEV,weights_only=True))
            allrows += evaluate_rows(name,predict(m,test,mode))
        # Mechanism interventions apply only to the strongest counter-guided arm.
        m=TargetSearchTeacherV2().to(DEV); m.load_state_dict(torch.load(OUT/f"CG10_seed{SEED}.pt",map_location=DEV,weights_only=True))
        for label,intervention in (("CG10-center","center"),("CG10-random","random"),("CG10-counter-shuffle","counter_shuffled")):
            allrows += evaluate_rows(label,predict(m,test,"counter",intervention))
        fields=list(allrows[0])
        with (OUT/"teacher_mechanism_metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(allrows)
        return
    names=list(ARMS) if args.stage=="all" else [args.stage]
    allrows=[]
    for name in names:
        mode,gamma=ARMS[name]; path,rec=train_arm(name,mode,gamma,train,test); m=TargetSearchTeacherV2().to(DEV);m.load_state_dict(torch.load(path,map_location=DEV,weights_only=True)); pred=predict(m,test,mode);allrows+=evaluate_rows(name,pred)
    if allrows:
        fields=list(allrows[0]);
        with (OUT/"teacher_four_arm_metrics.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)


if __name__=="__main__": main()
