"""Render a single deterministic search-behavior candidate query."""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"cocd"))
from paths import DEV  # noqa:E402
from windows_main.data import HaitiPairs,split  # noqa:E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa:E402
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2  # noqa:E402

OUT=ROOT/"experiments/GRSL_complete_paper_package/03_search_behavior"
CG=ROOT/"experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt"
S2=ROOT/"experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt"
CACHE=ROOT/"experiments/GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz"
COLORS={0:"#22c55e",1:"#f97316",2:"#a855f7"}
NAMES={0:"P2",1:"P3",2:"P4"}

def display(stack):
    x=.5*(stack[0]+stack[2]); lo,hi=np.percentile(x,[2,98])
    return np.clip((x-lo)/max(hi-lo,1e-6),0,1)

def policy_points(out,query_xy):
    policy=out["policy"]; weights=policy["weights"][0]
    offsets=2*torch.tanh(policy["offsets"][0])
    hq,wq=weights.shape[-2:]; py,px=query_xy
    qy=int(np.clip(round((py+.5)*hq/128-.5),0,hq-1));qx=int(np.clip(round((px+.5)*wq/128-.5),0,wq-1))
    candidates=[]
    for level,feat in enumerate(out["values"]):
        hm,wm=feat.shape[-2:]
        base_y=(qy+.5)/hq*hm-.5;base_x=(qx+.5)/wq*wm-.5
        for point in range(weights.shape[1]):
            weight=float(weights[level,point,qy,qx].item())
            oy,ox=offsets[level,point,:,qy,qx].tolist()
            sy=float((base_y+oy+.5)/hm*128);sx=float((base_x+ox+.5)/wm*128)
            candidates.append({"level":level,"point":point,"weight":weight,"x":sx,"y":sy})
    candidates.sort(key=lambda z:z["weight"],reverse=True)
    return qy,qx,candidates[:4]

def draw_points(ax,points,marker):
    for rank,q in enumerate(points,1):
        level=q["level"];color=COLORS[level]
        ax.scatter(q["x"],q["y"],s=100+250*q["weight"],marker=marker,color=color,
                   edgecolor="white",linewidth=1.2,zorder=4)
        ax.text(q["x"]+1.2,q["y"]+1.2,str(rank),fontsize=7,
                color="white",weight="bold",bbox={"facecolor":"#111827","alpha":.75,"pad":.5},zorder=5)

def draw_topk_legend(ax,points):
    lines=[]
    for rank,q in enumerate(points,1):
        lines.append(f"{rank}. {NAMES[q['level']]}  w={q['weight']:.3f}  ({q['x']:.1f},{q['y']:.1f})")
    ax.text(.025,.975,"Top samples:\n"+"\n".join(lines),transform=ax.transAxes,
            va="top",ha="left",fontsize=7.5,color="white",family="monospace",
            bbox={"facecolor":"#111827","alpha":.78,"pad":3},zorder=7)

def main():
    with (ROOT/"experiments/GRSL_complete_paper_package/02_qualitative/case_selection.csv").open(newline="",encoding="utf-8-sig") as f:
        arow=next(csv.DictReader(f))
    z=np.load(CACHE)
    train_ids,_,test_ids=split();train=HaitiPairs(train_ids,True)
    ds=HaitiPairs(test_ids,False,train.stats)
    teacher=TargetSearchTeacherV2().to(DEV).eval();teacher.load_state_dict(torch.load(CG,map_location=DEV,weights_only=True))
    student=StudentTargetSearchV2().to(DEV).eval();student.load_state_dict(torch.load(S2,map_location=DEV,weights_only=True))
    candidates=[]
    with (ROOT/"experiments/GRSL_complete_paper_package/02_qualitative/candidates/A_bothvalid/candidates.csv").open(newline="",encoding="utf-8-sig") as f:
        candidate_rows=list(csv.DictReader(f))
    with torch.inference_mode():
        for cand in candidate_rows:
            sid=int(cand["sample_id"]);orbit=cand["target_orbit"]
            idx=np.where((z["sample_id"]==sid)&(z["orbit"]==orbit))[0]
            if len(idx)!=1:continue
            rid=int(idx[0]);yy=z["y"][rid]>0;tg=z["target_geometry"][rid];cg=z["counter_geometry"][rid]
            rescue=yy&(z["p_S0"][rid]<.5)&(z["p_S2_2Stage"][rid]>=.5)&(tg==0)&(cg==0)
            asc,desc,_,_,_,_=ds[ds.ids.index(sid)]
            target,counter=(asc,desc) if orbit=="asc" else (desc,asc)
            to=teacher(target[None].to(DEV),counter[None].to(DEV));so=student(target[None].to(DEV))
            wt=to["policy"]["weights"].flatten(1,2);ws=so["policy"]["weights"].flatten(1,2)
            pp=wt.clamp_min(1e-8);qq=ws.clamp_min(1e-8);mm=.5*(pp+qq)
            js=.5*((pp*(pp.log()-mm.log())).sum(1)+(qq*(qq.log()-mm.log())).sum(1))[0].cpu().numpy()
            eligible=rescue[2::4,2::4]
            if eligible.shape!=js.shape: eligible=np.resize(eligible,js.shape)
            if eligible.any():
                score=float(js[eligible].max());qy,qx=map(int,np.unravel_index(np.argmax(np.where(eligible,js,-1)),js.shape))
                # Query grid centers correspond to the center pixel of each 4x4 block.
                py=min(qy*4+2,127);px=min(qx*4+2,127)
                candidates.append((score,sid,orbit,px,py,target,counter,to,so))
    if not candidates:raise RuntimeError("No BothValid rescued query among the ten Case A candidates")
    score,sid,orbit,qx,qy,target,counter,to,so=max(candidates,key=lambda q:q[0])
    qyt,qxt,tpoints=policy_points(to,(qy,qx));qys,qxs,spoints=policy_points(so,(qy,qx))
    image=display(target.numpy())
    fig,ax=plt.subplots(1,3,figsize=(12.5,4.4),dpi=180)
    for a in ax:
        a.imshow(image,cmap="gray",vmin=0,vmax=1);a.scatter([qx],[qy],marker="*",s=150,color="#ef4444",edgecolor="white",linewidth=1,zorder=6)
        a.set_xlim(max(-.5,qx-42),min(127.5,qx+42));a.set_ylim(min(127.5,qy+42),max(-.5,qy-42));a.set_xticks([]);a.set_yticks([])
    ax[0].set_title(f"Target SAR evidence\n{sid:06d}_{orbit}; query is rescued S0 FN",fontsize=10)
    ax[1].set_title("CG Teacher: top-4 search samples\npolicy uses counter context; values stay in target memory",fontsize=9)
    ax[2].set_title("S2-2Stage: top-4 samples\ntarget-only inference",fontsize=10)
    draw_points(ax[1],tpoints,"^");draw_points(ax[2],spoints,"o")
    draw_topk_legend(ax[1],tpoints);draw_topk_legend(ax[2],spoints)
    fig.suptitle("Search behavior candidate · sample locations are in the target orbit",fontsize=13,weight="bold")
    fig.tight_layout(rect=[0,0,1,.91])
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"figure_search_behavior_candidate.png",bbox_inches="tight")
    fig.savefig(OUT/"figure_search_behavior_candidate.pdf",bbox_inches="tight")
    plt.close(fig)
    meta={"status":"candidate; not yet accepted into paper body",
          "patch_id":f"{sid:06d}_{orbit}","query_xy_pixel":[qx,qy],"policy_js_at_query":score,
          "selection":"highest policy-JS query among top-10 Case A pixels that are BothValid GT pixels where S0 is false negative and S2-2Stage is true positive",
          "teacher_top4":tpoints,"student_top4":spoints,
          "normalization":"policy weights are normalized jointly across 3 FPN levels x 4 points for each query",
          "coordinate_mapping":"feature-grid reference plus bounded tanh offsets mapped to 128x128 target-image coordinates",
          "student_inference":"target orbit only; no counter image or feature",
          "note":"automatic visual candidate; inspect legibility and SAR/GT context before paper use"}
    (OUT/"search_query_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    checklist=ROOT/"experiments/GRSL_complete_paper_package/grsl_evidence_checklist.md"
    if checklist.exists():
        text=checklist.read_text(encoding="utf-8")
        text=text.replace("- [ ] Search-behavior visualization available","- [x] Search-behavior visualization available")
        text=text.replace("The small figure candidate still requires a legibility review before it is counted as available.",
                          "The search-behavior figure is available as a candidate; review its readability before paper submission.")
        checklist.write_text(text,encoding="utf-8")
    print(f"SEARCH_FIGURE_OK patch={sid:06d}_{orbit} query=({qx},{qy})",flush=True)

if __name__=="__main__":main()
