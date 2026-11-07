"""Frozen test scoring, matched error changes and deployment cost reporting."""
import json

import numpy as np
import pandas as pd
from scipy.ndimage import label
import torch
from torch import nn
from torch.nn import functional as F

from run import (OUT, B1, common, load, selection, SARInputs, PrivilegedInputs, optical_view)
from model import policy_distribution, js


def small_landslides(y,p,valid):
    positives=recalled=components=detected=0
    for gt,prob,mask in zip(y,p,valid):
        segments,n=label(gt & mask)
        sizes=np.bincount(segments.ravel())
        for k in range(1,n+1):
            if sizes[k]>100:
                continue
            region=segments==k
            hits=int((region & (prob>=.5)).sum())
            positives+=int(sizes[k]);recalled+=hits
            components+=1;detected+=hits>=max(1,int(np.ceil(.1*sizes[k])))
    return {"definition":"4-connected within-patch components <=100 valid pixels",
            "valid_positive_pixels":positives,"pixel_recall":recalled/max(1,positives),
            "components":components,"detected_at_10_percent_overlap":detected/max(1,components),
            "limitation":"patch components can fragment a physical landslide; overlapping patches repeat objects"}


def complexity(model,optical):
    total=0
    handles=[]
    def count(m,inputs,out):
        nonlocal total
        if isinstance(m,nn.Conv2d):
            total+=out.numel()*(m.in_channels//m.groups)*int(np.prod(m.kernel_size))
        elif isinstance(m,nn.ConvTranspose2d):
            total+=inputs[0].numel()*(m.out_channels//m.groups)*int(np.prod(m.kernel_size))
        else:
            total+=out.numel()*(m.in_channels//m.groups)*int(np.prod(m.kernel_size))
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Conv1d)):
            handles.append(m.register_forward_hook(count))
    try:
        with torch.no_grad():
            x=torch.zeros(1,20 if optical else 8,128,128,device="cuda")
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {"parameters":sum(p.numel() for p in model.parameters()),
            "convolution_GMACs":float(total/1e9),
            "excluded":"grid_sample, interpolation, normalization, nonlinearities, reductions and memory traffic",
            "latency":"NOT_MEASURED with concurrent GPU work"}


@torch.no_grad()
def student_diagnostic(method,data):
    arch="M0" if method=="M2" else method
    s,t,a=load(method,"student"),load(arch,"teacher"),load(arch,"sar")
    groups={k:[] for k in ("student","teacher_rule","sar_rule")}
    labels=[];sums={"JS":0.,"cosine":0.,"offset":0.};count=0
    scales=np.zeros(3)
    for raw in common.loader(data):
        b=common.cuda(raw)
        out=s.details(b["x"])
        _,bank_t=t.memory(b["x"])
        _,bank_a=a.memory(b["x"])
        pt=t.policy(bank_t,b["x"],b["optical"])
        pa=a.policy(bank_a,b["x"])
        bank=out["memory"]
        valid=b["valid"]
        labels.append(b["mask"][valid].bool().cpu().numpy())
        groups["student"].append(out["z"].sigmoid()[valid].cpu().numpy())
        for key,p in (("teacher_rule",pt),("sar_rule",pa)):
            z=s.decode(out["maps"],s.retrieve(bank,p))
            groups[key].append(z.sigmoid()[valid].cpu().numpy())
        sums["JS"]+=float(js(policy_distribution(out["policy"],bank,True),
                             policy_distribution(pt,bank,True)).mean())*len(b["x"])
        rt=s.retrieve(bank,pt)
        sums["cosine"]+=float(F.cosine_similarity(out["retrieved"],rt,dim=1).mean())*len(b["x"])
        sums["offset"]+=float((2*out["policy"]["offsets"].tanh()).norm(dim=3).mean())*len(b["x"])
        scales+=out["policy"]["weights"].sum(2).mean((0,2,3)).cpu().numpy()*len(b["x"])
        count+=len(b["x"])
    y=np.concatenate(labels)
    report={"fixed_student_memory_and_decoder":{k:common.metrics(y,np.concatenate(v)) for k,v in groups.items()},
            "query_summary":{k:v/count for k,v in sums.items()},"scale_mass":(scales/count).tolist(),
            "note":"Teacher and control rules are generated from their own SAR memories then applied to the same frozen Student value bank"}
    del s,t,a
    torch.cuda.empty_cache()
    return report


def qualitative(method,test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    arch="M0" if method=="M2" else method
    folders=[B1,OUT/f"{arch}_sar_seed42",OUT/f"{arch}_teacher_seed42",OUT/f"{method}_student_seed42"]
    arrays=[np.load(p/"test_predictions.npz")["probability"] for p in folders]
    events=test.frame.event_id.unique()
    chosen=[]
    for event in events:
        candidates=np.flatnonzero(test.frame.event_id.to_numpy()==event)
        sizes=[int(test.cache[i]["mask"].sum()) for i in candidates]
        chosen.append(int(candidates[np.argsort(sizes)[len(sizes)//2]]))
    fig,axs=plt.subplots(len(chosen),6,figsize=(12,2*len(chosen)),layout="constrained",squeeze=False)
    for row,i in zip(axs,chosen):
        sample=test.cache[i]
        gt=sample["mask"].numpy()[0]>.5;valid=sample["valid"].numpy()[0]
        sar=sample["x"].numpy()[-1]
        lo,hi=np.percentile(sar[valid],[2,98])
        row[0].imshow(sar,cmap="gray",vmin=lo,vmax=hi)
        row[0].set_ylabel(sample["event_id"])
        row[1].imshow(gt,cmap="gray",vmin=0,vmax=1)
        for j,probs in enumerate(arrays):
            pred=probs[i]>=.5
            error=np.zeros(gt.shape,np.uint8)
            error[gt&pred]=1;error[~gt&pred]=2;error[gt&~pred]=3;error[~valid]=4
            row[j+2].imshow(error,cmap=ListedColormap(["#fafafa","#3c9967","#c34e43","#e0b44c","#a4a4a4"]),vmin=0,vmax=4)
        for ax in row:
            ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axs[0],("POST VH","GT","B1","SAR search","Teacher","Student")):
        ax.set_title(title)
    fig.suptitle(method+" | green TP, red FP, yellow FN; per-event median-GT-area examples")
    fig.savefig(OUT/f"{method}_qualitative.png",dpi=160)
    plt.close(fig)


def finish(eligible,repeats,frames,stats,optical):
    common.write(OUT/"status.json",{"state":"EVALUATING"})
    folders=sorted(p.parent for p in OUT.glob("*_seed42/selection.json"))
    assert len(folders) == 13
    locks={p.name:selection(p) for p in folders}
    common.write(OUT/"all_models_frozen_before_test.json",locks)
    test=SARInputs(frames["test"],stats)
    paired=PrivilegedInputs(frames["test"],stats,optical)
    truth=np.stack([b["mask"].numpy()[0]>.5 for b in test.cache])
    valid=np.stack([b["valid"].numpy()[0] for b in test.cache])
    summary={"B1":json.loads((B1/"test_metrics.json").read_text())}
    costs={}
    for folder in folders:
        method,role,seed=folder.name.split("_")
        model=load(method,role,int(seed.removeprefix("seed")))
        if not (folder/"test_metrics.json").exists():
            marker=folder/"test_started.json"
            if marker.exists():
                raise RuntimeError(f"Interrupted final test requires inspection: {folder}")
            common.write(marker,locks[folder.name])
            dataset=optical_view(paired) if role=="teacher" else test
            metrics=common.evaluate(model,dataset,folder/"test_predictions.npz")
            common.write(folder/"test_metrics.json",metrics)
        summary[folder.name]=json.loads((folder/"test_metrics.json").read_text())
        predictions=np.load(folder/"test_predictions.npz")
        assert np.array_equal(predictions["sample_ids"],test.frame.sample_id.to_numpy())
        p=predictions["probability"]
        small=small_landslides(truth,p,valid)
        common.write(folder/"small_landslides.json",small)
        costs[folder.name]=complexity(model,role=="teacher")
        del model
        torch.cuda.empty_cache()
    for method in eligible:
        arch="M0" if method=="M2" else method
        a=np.load(OUT/f"{arch}_sar_seed42/test_predictions.npz")["probability"]>=.5
        b=np.load(OUT/f"{method}_student_seed42/test_predictions.npz")["probability"]>=.5
        common.write(OUT/f"{method}_error_changes.json",{
            "FN_fixed":int((valid&truth&~a&b).sum()),"FN_added":int((valid&truth&a&~b).sum()),
            "FP_fixed":int((valid&~truth&a&~b).sum()),"FP_added":int((valid&~truth&~a&b).sum())})
        common.write(OUT/f"{method}_retrieval_diagnostic.json",student_diagnostic(method,paired))
        qualitative(method,test)
    table=[]
    for name,m in summary.items():
        for scope,metrics in (("all120",m),("ind44",m["per_official_subset"]["testind"])):
            table.append({"model":name,"scope":scope,**{k:metrics[k] for k in
                         ("IoU","F1","AUPRC","Precision","Recall")}})
    pd.DataFrame(table).to_csv(OUT/"metrics.csv",index=False)
    common.write(OUT/"complexity.json",costs)
    common.write(OUT/"comparison.json",summary)
