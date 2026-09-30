"""All twelve actual CG0/S2 samples; coordinates follow DeformableSearch._grid."""
from pathlib import Path
import sys,csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1]
OUT=PACK/'new_evidence/20260930';FIG=PACK/'manuscript/figures'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import DEV
from windows_main.data import HaitiPairs
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2
sys.path.insert(0,str(Path(__file__).parent));from build_figures import read_sar

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    meta=json.loads((OUT/'figure_sources.json').read_text());case=meta['cases'][-1];sid=case['sample_id'];orbit=case['orbit']
    norm=np.load(OUT/'normalization.npz');ds=HaitiPairs([sid],stats=(norm['mean'],norm['std']))
    a,d,y,ga,gd,_=ds[0];target,counter=(a,d) if orbit=='asc' else (d,a);gt,gc=(ga,gd) if orbit=='asc' else (gd,ga)
    teacher=TargetSearchTeacherV2().to(DEV).eval();teacher.load_state_dict(torch.load(ROOT/'experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt',map_location=DEV,weights_only=True))
    student=StudentTargetSearchV2().to(DEV).eval();student.load_state_dict(torch.load(ROOT/'experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt',map_location=DEV,weights_only=True))
    with torch.inference_mode():to=teacher(target[None].to(DEV),counter[None].to(DEV));so=student(target[None].to(DEV))
    eligible=((gt.numpy()>0)&(gc.numpy()==0)&(y.numpy()>0)).reshape(32,4,32,4).mean((1,3))
    ys,xs=np.where(eligible>=.75);cy,cx=ys.mean(),xs.mean();pick=np.argmin((ys-cy)**2+(xs-cx)**2);qy,qx=int(ys[pick]),int(xs[pick])
    query_x=(qx+.5)*4-.5;query_y=(qy+.5)*4-.5
    sar,_=read_sar(sid,orbit);lo,hi=case['sar_display_range'];rows=[];layers={}
    fig,axes=plt.subplots(3,3,figsize=(9.6,7.8),gridspec_kw={'height_ratios':[1,1,.64],'hspace':.28,'wspace':.13})
    colors=['#24ad54','#e07717','#8362cb'];maxmag=0
    for ri,(name,o) in enumerate([('CG0 Teacher',to),('S2-2Stage Student',so)]):
        raw=o['policy']['offsets'][0].cpu().numpy();w=o['policy']['weights'][0].cpu().numpy();bounded=2*np.tanh(raw)
        layers[name+'_raw_offsets']=raw;layers[name+'_weights']=w
        for level in range(3):
            ax=axes[ri,level];ax.imshow(sar[2],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
            hm,wm=o['values'][level].shape[-2:]
            for k in range(4):
                oy,ox=bounded[level,k,:,qy,qx];px=((qx+.5)/32*wm+ox)/wm*128-.5;py=((qy+.5)/32*hm+oy)/hm*128-.5
                weight=float(w[level,k,qy,qx]);mag=float(np.hypot(px-query_x,py-query_y));maxmag=max(maxmag,mag)
                rows.append({'model':name,'sample_id':sid,'orbit':orbit,'query_grid_y':qy,'query_grid_x':qx,'reference_x':query_x,'reference_y':query_y,
                             'level':'P'+str(level+2),'point':k,'x':float(px),'y':float(py),'weight':weight,'offset_y_feature':float(oy),'offset_x_feature':float(ox),'distance_image_pixels':mag})
                ax.scatter([px],[py],s=70+200*weight,facecolors='none',edgecolors=colors[level],marker='o' if ri else '^',linewidths=1.2,zorder=4)
            ax.scatter([query_x],[query_y],s=35,color='#ef3030',marker='+',linewidth=1,zorder=5)
            ax.set_xlim(query_x-20,query_x+20);ax.set_ylim(query_y+20,query_y-20);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{name}: P{level+2}',fontsize=9)
    for level in range(3):
        ax=axes[2,level];p=[r['weight'] for r in rows if r['model']=='CG0 Teacher' and r['level']==f'P{level+2}'];s=[r['weight'] for r in rows if r['model']=='S2-2Stage Student' and r['level']==f'P{level+2}']
        xx=np.arange(4);ax.bar(xx-.18,p,.36,label='CG0',color='#3673ac');ax.bar(xx+.18,s,.36,label='Student',color='#df8841')
        ax.set_xticks(xx,['fixed','1','2','3']);ax.set_ylim(0,max(r['weight'] for r in rows)*1.15);ax.set_title(f'P{level+2}: mass CG0={sum(p):.3f}, S={sum(s):.3f}',fontsize=9)
        if level==0:ax.set_ylabel('Joint candidate weight');ax.legend(fontsize=7)
    fig.text(.5,.01,f'{sid:06d} {orbit.upper()}; query=({query_x:.1f},{query_y:.1f}); red + = reference; all 12 points shown; max displacement={maxmag:.2f} px',ha='center',fontsize=9)
    fig.savefig(FIG/'fig_search_current_samples.pdf',bbox_inches='tight');fig.savefig(FIG/'fig_search_current_samples.png',dpi=300,bbox_inches='tight');plt.close(fig)
    with (OUT/'current_query_samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'current_query_policy_layers.npz',target_sar=sar,query_xy=[query_x,query_y],**layers)
    (OUT/'current_query_metadata.json').write_text(json.dumps({'case':case,'query_grid_yx':[qy,qx],'query_xy':[query_x,query_y],'query_selection':'closest eligible G10/GT query to G10/GT support centroid; no prediction-based ranking','max_displacement_pixels':maxmag,'coordinate_rule':'pixel centers; bounded offsets in feature pixels; align_corners=False; no display jitter','scope':'single illustrative query, not segmentation causal proof'},indent=2))
    # Correct current Teacher statistics, replacing the former CG10 plot.
    with (OUT/'current_search_alignment.csv').open() as f:stats={(r['region'],r['metric']):float(r['mean']) for r in csv.DictReader(f)}
    fig,axs=plt.subplots(1,2,figsize=(6,2.4))
    for ax,keys,title in zip(axs,[('JS_CG0_S2','JS_CG0_shuffle'),('cosine_distance_CG0_S2','cosine_distance_shuffle_S2')],['Candidate-weight JS','Shared-memory cosine distance']):
        vals=[stats['TargetValid',k] for k in keys];ax.bar(['Student','Counter shuffle'],vals,color=['#3673ac','#df8841']);ax.set_title(title,fontsize=9)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(vals)*1.3)
    fig.tight_layout();fig.savefig(FIG/'fig_search_alignment_current.pdf');fig.savefig(FIG/'fig_search_alignment_current.png',dpi=300);plt.close(fig)
    print('CURRENT_SEARCH_FIGURE_COMPLETE',maxmag,flush=True)

if __name__=='__main__':main()
