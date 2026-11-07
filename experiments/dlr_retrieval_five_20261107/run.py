"""Independent, validation-gated M0--M4 retrieval experiment queue."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from model import RetrievalUNet, kd_losses, policy_distribution, js

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
ROOT = HERE.parents[1]
STUDY = ROOT/"experiments/dlr_unet_study_20261104"
sys.path.insert(0,str(STUDY))
from train_baseline import common, restore_rng, segmentation
from train_distillation import REFERENCE, ROOT as OLD
from train_multiscale_3plus1 import SARInputs, PrivilegedInputs
from train_evidence_controls import optical_view

METHODS = ("M0","M1","M2","M3","M4")
ARCHITECTURES = ("M0","M1","M3","M4")
B1 = STUDY/"multiscale_3plus1_results/B1_3plus1"
AUX_WEIGHT = .05
REPEAT_MARGIN = .002


def seeded(number):
    import random
    random.seed(number)
    np.random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed_all(number)


def build(method,optical=False,number=42):
    seeded(number)
    return RetrievalUNet(method,optical).cuda().to(memory_format=torch.channels_last)


def selection(folder):
    info=json.loads((folder/"selection.json").read_text())
    assert info["epochs_completed"]==100
    assert common.sha(folder/"best.pt")==info["checkpoint_sha256"]
    return info


def load(method,role,number=42):
    model=build(method,role=="teacher",number)
    folder=OUT/f"{method}_{role}_seed{number}"
    selection(folder)
    model.load_state_dict(torch.load(folder/"best.pt",weights_only=True,map_location="cpu"))
    return model.eval()


def mechanism_checks():
    """Synthetic invariants precede reading validation or test labels."""
    from windows_main.train_student import spatial_policy,js_divergence
    records={}
    rng=common.rng()
    try:
        for method in ARCHITECTURES:
            model=build(method,True).eval()
            sar=torch.randn(2,8,64,64,device="cuda")
            optical=torch.randn(2,12,64,64,device="cuda",requires_grad=True)
            with torch.no_grad():
                model.policy_head.weight.weight.normal_(std=.02)
                model.policy_head.offset.weight.normal_(std=.02)
                first=model.details(sar,optical)
                second=model.details(sar,optical.flip(-1),forced_policy=first["policy"])
                assert all(torch.equal(a,b) for a,b in zip(first["memory"],second["memory"]))
                assert all(torch.equal(a,b) for a,b in zip(first["maps"],second["maps"]))
                assert torch.equal(first["z"],second["z"])
                changed=model.details(sar,optical.flip(-1))
                delta=float((first["policy"]["weights"]-changed["policy"]["weights"]).abs().max())
                assert delta>0
                policy={k:v.clone() for k,v in first["policy"].items()}
                policy["offsets"]=policy["offsets"]+.7
                shift=float((model.retrieve(first["memory"],policy)-first["retrieved"]).abs().mean())
                assert shift>1e-5
                sizes=[list(x.shape) for x in first["memory"]]
            result=model(sar,optical).square().mean()
            result.backward()
            assert optical.grad is not None and optical.grad.abs().sum()>0
            assert model.backbone.enc[0].net[0].weight.grad.abs().sum()>0
            records[method]={"fixed_policy_optical_invariance":True,"dynamic_sampling_change":shift,
                             "optical_changes_policy":delta,"memory_shapes_64":sizes}
            del model,first,second,changed,policy,result,sar,optical
            torch.cuda.empty_cache()

        model=build("M0").eval()
        sar=torch.randn(1,8,32,32,device="cuda")
        with torch.no_grad():
            _,bank=model.memory(sar)
            p=model.policy(bank,sar)
            q={k:v.clone() for k,v in p.items()}
            q["offsets"]+=.23
            q["weights"]=q["weights"].roll(1,1)
            sparse=js(policy_distribution(p,bank),policy_distribution(q,bank))
            dense=js_divergence(spatial_policy(p,bank),spatial_policy(q,bank))
            torch.testing.assert_close(sparse,dense,atol=2e-6,rtol=2e-5)
            # All samples at the same location: scale changes vanish only in M0.
            p["offsets"].zero_();q["offsets"].zero_()
            p["weights"].fill_(1/12);q["weights"].zero_();q["weights"][:,0]=1/4
            original=js(policy_distribution(p,bank),policy_distribution(q,bank)).mean()
            retained=js(policy_distribution(p,bank,True),policy_distribution(q,bank,True)).mean()
            assert original<1e-6 and retained>.1
        records["policy_representation"]={"dense_canonical_JS_equivalent":True,
                                        "M2_retains_scale_mass":float(retained)}
        del model,sar,bank,p,q,sparse,dense
        torch.cuda.empty_cache()
    finally:
        restore_rng(rng)
    common.write(OUT/"mechanism_AB_checks.json",records)


def train(method,role,train_data,val,number=42,guide=None):
    folder=OUT/f"{method}_{role}_seed{number}"
    folder.mkdir(exist_ok=True)
    if (folder/"selection.json").exists():
        return selection(folder)
    model=build(method,role=="teacher",number)
    optimizer=common.optimizer(model)
    seeded(number)
    generator=torch.Generator().manual_seed(number+81)
    dl=common.loader(train_data,True,generator)
    latest=folder/"latest.pt"
    history,best,chosen,step=[], -1., None, 0
    if latest.exists():
        state=torch.load(latest,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        history,best,chosen,step=state["history"],state["best"],state["chosen"],state["step"]
        generator.set_state(state["generator"]);restore_rng(state["rng"])
        del state
    common.write(OUT/"status.json",{"state":"TRAINING","method":method,"role":role,"seed":number})
    for epoch in range(len(history)+1,101):
        model.train()
        totals={}
        for index,b in enumerate(dl,1):
            b=common.cuda(b)
            optimizer.zero_grad(set_to_none=True)
            lr=1e-4*common.lr_factor(step,len(dl))
            for group in optimizer.param_groups:
                group["lr"]=lr
            if role=="teacher":
                out=model.details(b["x"],b["optical"])
            else:
                out=model.details(b["x"])
            terms={"seg":segmentation(out["z"],b)}
            if role=="student" and epoch>=6:
                terms["policy"],terms["evidence"]=kd_losses(model,guide,out,b,method)
            loss=terms["seg"]+sum(AUX_WEIGHT*v for k,v in terms.items() if k!="seg")
            assert torch.isfinite(loss)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            for k,v in {**terms,"total":loss}.items():
                totals[k]=totals.get(k,0.)+float(v.detach())*len(b["x"])
            step+=1
            common.write(folder/"progress.json",{"state":"TRAINING","epoch":epoch,"batch":index,
                         "batches":len(dl),"losses":{k:float(v.detach()) for k,v in terms.items()},
                         "gradient_norm":float(grad),"lr":lr,"best_validation_IoU":best})
        scores=common.evaluate(model,optical_view(val) if role=="teacher" else val) if epoch%5==0 else None
        if scores is not None and scores["IoU"]>best:
            best,chosen=scores["IoU"],epoch
            torch.save(model.state_dict(),folder/"best.pt")
        history.append({"epoch":epoch,"losses":{k:v/len(dl.dataset) for k,v in totals.items()},
                        "validation":scores})
        common.write(folder/"history.json",history)
        temp=latest.with_suffix(".tmp")
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,
                    "best":best,"chosen":chosen,"step":step,"generator":generator.get_state(),
                    "rng":common.rng()},temp)
        temp.replace(latest)
        print(method,role,number,epoch,"best validation",best,flush=True)
    assert len(history)==100 and step==2300
    common.write(folder/"selection.json",{"state":"FROZEN","epochs_completed":100,"selected_epoch":chosen,
                 "validation_IoU":best,"checkpoint_sha256":common.sha(folder/"best.pt"),
                 "selection":"all scheduled validation epochs including segmentation warmup"})
    del model,optimizer
    torch.cuda.empty_cache()
    return selection(folder)


@torch.no_grad()
def swap_test(method,dataset):
    a,b=load(method,"sar"),load(method,"teacher")
    predictions={k:[] for k in ("AA","AB","BA","BB","native_A","native_B")}
    truth=[]
    motion={"offset":[],"scale":[],"policy_JS":[],"cosine":[]}
    for raw in common.loader(dataset):
        batch=common.cuda(raw)
        sar,opt,valid=batch["x"],batch["optical"],batch["valid"]
        ya=a.details(sar);yb=b.details(sar,opt)
        truth.append(batch["mask"][valid].cpu().numpy().astype(bool))
        predictions["native_A"].append(ya["z"].sigmoid()[valid].cpu().numpy())
        predictions["native_B"].append(yb["z"].sigmoid()[valid].cpu().numpy())
        for host,name in ((a,"A"),(b,"B")):
            maps,bank=host.memory(sar)
            pa=a.policy(bank,sar)
            pb=b.policy(bank,sar,opt)
            ra,rb=host.retrieve(bank,pa),host.retrieve(bank,pb)
            for key,r in ((name+"A",ra),(name+"B",rb)):
                predictions[key].append(host.decode(maps,r).sigmoid()[valid].cpu().numpy())
            if name=="A":
                motion["offset"].append(float((2*pb["offsets"].tanh()).norm(dim=3).mean()))
                motion["scale"].append(pb["weights"].sum(2).mean((0,2,3)).cpu().tolist())
                motion["policy_JS"].append(float(js(policy_distribution(pa,bank,True),
                                                     policy_distribution(pb,bank,True)).mean()))
                motion["cosine"].append(float(F.cosine_similarity(ra,rb,dim=1).mean()))
    y=np.concatenate(truth)
    result={k:common.metrics(y,np.concatenate(v)) for k,v in predictions.items()}
    gains=[result["AB"]["IoU"]-result["AA"]["IoU"],result["BB"]["IoU"]-result["BA"]["IoU"]]
    passed=(result["native_B"]["IoU"]>result["native_A"]["IoU"] and
            np.mean(gains)>0 and max(gains)>=.002)
    result.update(fixed_host_gains=gains,gate_pass=bool(passed),
                  retrieval_summary={k:np.mean(v,axis=0).tolist() for k,v in motion.items()})
    del a,b
    torch.cuda.empty_cache()
    return result


def real_gradient_check(train_data):
    rng=common.rng()
    output={}
    try:
        batch=common.cuda(torch.utils.data.default_collate([train_data[i] for i in range(16)]))
        for method in METHODS:
            s,t=build(method),build(method,True).eval().requires_grad_(False)
            # Perturb the guide policy only to expose the KD derivative paths.
            with torch.no_grad():
                t.policy_head.weight.weight.normal_(std=.01)
                t.policy_head.offset.weight.normal_(std=.01)
            result=s.details(batch["x"])
            pol,ev=kd_losses(s,t,result,batch,method)
            parameters=tuple(s.parameters())
            def norm(value):
                grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                return sum(g.square().sum() for g in grads if g is not None).sqrt().item()
            pn,en=norm(pol),norm(ev)
            seg=segmentation(result["z"],batch)
            (seg+AUX_WEIGHT*(pol+ev)).backward()
            assert 0<pn<float("inf") and 0<en<float("inf")
            assert s.backbone.enc[0].net[0].weight.grad.abs().sum()>0
            assert all(p.grad is None for p in t.parameters())
            output[method]={"policy":float(pol.detach()),"evidence":float(ev.detach()),
                            "policy_gradient_norm":pn,"evidence_gradient_norm":en,
                            "encoder_receives_seg_gradient":True,"teacher_frozen":True}
            del s,t,result,parameters,seg,pol,ev
            torch.cuda.empty_cache()
    finally:
        restore_rng(rng)
    common.write(OUT/"real_train16_gradients.json",output)


def main():
    OUT.mkdir(exist_ok=True)
    common.setup()
    frozen=selection(B1)
    config={"epochs":100,"val_every":5,"batch":16,"seeds":[42],
            "train_val_test":[355,88,120],"threshold":.5,
            "loss":"BCE + per-image foreground Dice (1:1)",
            "policy_weight":AUX_WEIGHT,"evidence_weight":AUX_WEIGHT,"KD_start_epoch":6,
            "optimizer":"AdamW1e-4,wd.01 exclude bias/norm,clip1,50constant50cosine",
            "B1_checkpoint_sha256":frozen["checkpoint_sha256"],
            "split_sha256":common.sha(STUDY/"split_manifest.csv"),
            "matched_model_initialization":"seed42 random common parameters; all stages trainable",
            "teacher_gate":"native val gain >0; average two fixed-host rule gains >0; at least one host >=0.2pp",
            "repeat_rule":"seed43 for validation gains >=0.2pp over same-architecture seed42 control",
            "test_gate":"finish/freeze all eligible arms and repeats before reading test images"}
    common.write(OUT/"config.json",config)
    if not (OUT/"mechanism_AB_checks.json").exists():
        mechanism_checks()
    frames=common.frames()
    assert {k:len(v) for k,v in frames.items()}=={"train":355,"val":88,"test":120}
    stats=json.loads((REFERENCE/"normalization.json").read_text())
    optical=json.loads((OLD/"optical_normalization.json").read_text())
    common.write(OUT/"status.json",{"state":"LOADING_TRAIN_AND_VALIDATION"})
    train_data=PrivilegedInputs(frames["train"],stats,optical,augment=True)
    val=PrivilegedInputs(frames["val"],stats,optical)
    if not (OUT/"real_train16_gradients.json").exists():
        real_gradient_check(train_data)
    # Mechanism C requires a trained matched pair before any KD Student.
    for role in ("sar","teacher"):
        train("M0",role,train_data,val)
    gate=swap_test("M0",val)
    common.write(OUT/"M0_teacher_search_gate.json",gate)
    if not gate["gate_pass"]:
        common.write(OUT/"status.json",{"state":"TEACHER_SEARCH_GATE_NOT_PASSED",
                     "students":"NOT_RUN","reason":"fixed-memory/decoder validation evidence insufficient"})
        return
    eligible=[]
    for method in METHODS:
        arch="M0" if method=="M2" else method
        if arch!="M0":
            for role in ("sar","teacher"):
                train(arch,role,train_data,val)
            check=swap_test(arch,val)
            common.write(OUT/f"{arch}_teacher_search_gate.json",check)
            if not check["gate_pass"]:
                continue
        guide=load(arch,"teacher").requires_grad_(False)
        train(method,"student",train_data,val,guide=guide)
        del guide
        eligible.append(method)
    repeats=[]
    for method in eligible:
        arch="M0" if method=="M2" else method
        a=selection(OUT/f"{arch}_sar_seed42")
        b=selection(OUT/f"{method}_student_seed42")
        if b["validation_IoU"]-a["validation_IoU"]>=REPEAT_MARGIN:
            train(arch,"sar",train_data,val,43)
            guide=load(arch,"teacher").requires_grad_(False)
            train(method,"student",train_data,val,43,guide)
            del guide
            repeats.append(method)
    common.write(OUT/"validation_method_selection.json",{"eligible":eligible,"seed43_repeats":repeats})
    del train_data,val
    torch.cuda.empty_cache()
    from final_evaluation import finish
    finish(eligible,repeats,frames,stats,optical)
    common.write(OUT/"status.json",{"state":"COMPLETE"})


if __name__=="__main__":
    try:
        main()
    except Exception:
        common.write(OUT/"status.json",{"state":"FAILED","error":traceback.format_exc()})
        raise
