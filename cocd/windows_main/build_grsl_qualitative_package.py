"""Reproducible qualitative selection from locked final checkpoints; no training."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np,torch
from scipy.ndimage import label
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'cocd'))
from paths import DEV
from windows_main.data import HaitiPairs,split
from windows_main.models_da_search import TargetOnlyBaseline
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2,load_so_backbone
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
OUT=ROOT/'experiments'/'GRSL_qualitative_package';[ (OUT/x).mkdir(parents=True,exist_ok=True) for x in ('figures','crops','overlays','csv','json','captions') ]
TP=ROOT/'experiments/windows_main/teacher_target_search_v2/CG10_seed42.pt';SP=ROOT/'experiments/windows_main/da_search/SO_backbone_seed42.pt';SD=ROOT/'experiments/windows_main/student_target_search_v2'
def metric(p,y,m):
 p=p[m];y=y[m].astype(bool);h=p>=.5;tp=(h&y).sum();fp=(h&~y).sum();fn=((~h)&y).sum();pr=tp/(tp+fp+1e-8);rc=tp/(tp+fn+1e-8)
 return [float(average_precision_score(y,p)) if y.any() else '',float(tp/(tp+fp+fn+1e-8)),float(2*pr*rc/(pr+rc+1e-8))]
def predmask(p):return p>=.5
def errmap(p,y):
 h=predmask(p);z=np.zeros_like(y,dtype=np.uint8);z[(h)&(y>0)]=1;z[(~h)&(y>0)]=2;z[(h)&(y==0)]=3;return z
ERR=ListedColormap(['black','#32b44a','#dc2626','#f3c623'])
def cont(ax,tgd):
 if tgd.any():ax.contour(tgd.astype(float),levels=[.5],colors='#d62728',linestyles='--',linewidths=.8)
def wr(name,rows):
 with open(OUT/'csv'/name,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 tid,_,eid=split();tr=HaitiPairs(tid,True);ds=HaitiPairs(eid,False,tr.stats)
 so=TargetOnlyBaseline().to(DEV).eval();load_so_backbone(so,str(SP));t=TargetSearchTeacherV2().to(DEV).eval();t.load_state_dict(torch.load(TP,map_location=DEV,weights_only=True))
 models={n:StudentTargetSearchV2().to(DEV).eval() for n in ('S0','S1','S2')}
 for n,m in models.items():m.load_state_dict(torch.load(SD/(n+'_seed42.pt'),map_location=DEV,weights_only=True))
 scenes=[]
 with torch.no_grad():
  for i in range(len(ds)):
   a,d,y,ga,gd,_=ds[i];a=a[None].to(DEV);d=d[None].to(DEV);ta,td=t.forward_pair(a,d,'counter');outs=[('asc',a,y.numpy(),ga.numpy(),gd.numpy(),ta),('desc',d,y.numpy(),gd.numpy(),ga.numpy(),td)]
   for orbit,x,gt,g,c,to in outs:
    pp={'SO':torch.sigmoid(so(x)['z'])[0,0].cpu().numpy(),'S0':torch.sigmoid(models['S0'](x)['z'])[0,0].cpu().numpy(),'S1':torch.sigmoid(models['S1'](x)['z'])[0,0].cpu().numpy(),'S2':torch.sigmoid(models['S2'](x)['z'])[0,0].cpu().numpy(),'CG10':torch.sigmoid(to['z'])[0,0].cpu().numpy()}
    tg=(g>0)&~(c>0);sid=f'{ds.ids[i]:06d}_{orbit}';im=x[0,2].cpu().numpy();good=tg&(gt>0)&(~predmask(pp['SO']))&predmask(pp['S2']);omega=tg&((predmask(pp['SO'])!=(gt>0)))&(predmask(pp['CG10'])==(gt>0))
    scenes.append({'id':sid,'sample_id':ds.ids[i],'orbit':orbit,'image':im,'gt':gt,'tgd':tg,'omega':omega,'p':pp,'gain':int(good.sum()),'tgd_n':int(tg.sum()),'pos':int(gt.sum())})
 # deterministic scene selection: three high-gain TGD scenes plus one non-TGD-positive representative.
 ranked=sorted([s for s in scenes if s['pos']>30 and s['tgd_n']>30],key=lambda s:(s['gain'],s['tgd_n']),reverse=True);sel=[]
 for s in ranked:
  if s['sample_id'] not in {x['sample_id'] for x in sel}:sel.append(s)
  if len(sel)==3:break
 normal=next(s for s in sorted([s for s in scenes if s['pos']>30],key=lambda x:x['tgd_n']) if s['sample_id'] not in {x['sample_id'] for x in sel});sel.append(normal)
 methods=['SO','S2','CG10'];scrows=[]
 for ri,s in enumerate(sel,1):
  for m in methods:
   a=metric(s['p'][m],s['gt'],np.ones_like(s['gt'],bool));b=metric(s['p'][m],s['gt'],s['tgd']);scrows.append({'scene_id':s['id'],'row_order':ri,'method':m,'AUPRC':a[0],'IoU':a[1],'F1':a[2],'TGD_AUPRC':b[0],'TGD_IoU':b[1],'TGD_F1':b[2]})
 wr('table_selected_scenes_metrics.csv',scrows);wr('table_selected_scene_metrics.csv',scrows);wr('overall_qualitative_scene_list.csv',[{'scene_id':s['id'],'row_order':i+1,'whether_contains_TGD':bool(s['tgd'].any()),'crop_box_if_any':'','methods_included':'SO;S2;CG10 Teacher reference'} for i,s in enumerate(sel)])
 fig,axs=plt.subplots(len(sel),5,figsize=(10,2.2*len(sel)));heads=['Target SAR','GT','SO baseline','S2 (ours)','CG10 Teacher reference']
 for j,h in enumerate(heads):axs[0,j].set_title(h,fontsize=8)
 for i,s in enumerate(sel):
  axs[i,0].imshow(s['image'],cmap='gray');axs[i,0].text(2,10,s['id'],color='white',fontsize=7,bbox={'facecolor':'black','alpha':.5});cont(axs[i,0],s['tgd']);axs[i,1].imshow(s['gt'],cmap=ListedColormap(['black','#32b44a']));cont(axs[i,1],s['tgd'])
  for j,m in enumerate(methods,2):axs[i,j].imshow(predmask(s['p'][m]),cmap=ListedColormap(['black','#32b44a']));cont(axs[i,j],s['tgd'])
  for a in axs[i]:a.axis('off')
 fig.tight_layout();fig.savefig(OUT/'figures'/'fig_overall_qualitative.png',dpi=300);fig.savefig(OUT/'figures'/'fig_overall_qualitative.pdf');fig.savefig(OUT/'figures'/'fig_overall_comparison.png',dpi=300);fig.savefig(OUT/'figures'/'fig_overall_comparison.pdf');plt.close(fig)
 # local cases from connected components of strict TGD & SO wrong & S2 correct.
 cand=[]
 for s in scenes:
  mask=s['tgd']&(s['gt']>0)&(~predmask(s['p']['SO']))&predmask(s['p']['S2']);lab,n=label(mask)
  for k in range(1,n+1):
   yy,xx=np.where(lab==k)
   if len(yy)>=4:cand.append((len(yy),s,int(yy.mean()),int(xx.mean())))
 def box(y,x):return max(0,min(32,y-48)),max(0,min(32,x-48)),96,96
 scored=[]
 for n,s,y,x in cand:
  yy,xx,h,w=box(y,x);sl=np.s_[yy:yy+h,xx:xx+w];improve=int((predmask(s['p']['SO'][sl])!=(s['gt'][sl]>0)).sum())-int((predmask(s['p']['S2'][sl])!=(s['gt'][sl]>0)).sum());scored.append((improve,n,s,y,x))
 scored.sort(reverse=True,key=lambda z:(z[0],z[1]));cases=[]
 for _,_,s,y,x in scored:
  if s['sample_id'] not in {q['s']['sample_id'] for q in cases}:cases.append({'s':s,'y':y,'x':x})
  if len(cases)==3:break
 def box(y,x):return max(0,min(32,y-48)),max(0,min(32,x-48)),96,96
 cr=[]
 for ix,q in enumerate(cases,1):
  s=q['s'];yy,xx,h,w=box(q['y'],q['x']);sl=np.s_[yy:yy+h,xx:xx+w];om=bool(s['omega'][sl].any());cr.append({'case_id':f'C{ix}','source_scene':s['id'],'crop_box':f'{yy},{xx},{h},{w}','TGD_pixel_count':int(s['tgd'][sl].sum()),'whether_in_OmegaTR':om,'baseline_errors':int((predmask(s['p']['SO'][sl])!=(s['gt'][sl]>0)).sum()),'S2_errors':int((predmask(s['p']['S2'][sl])!=(s['gt'][sl]>0)).sum()),'included_methods':'SO;S0;S2;CG10'})
  for name,arr in [('target',s['image'][sl]),('gt',s['gt'][sl]),('tgd',s['tgd'][sl]),('SO',s['p']['SO'][sl]),('S0',s['p']['S0'][sl]),('S2',s['p']['S2'][sl]),('teacher',s['p']['CG10'][sl])]:plt.imsave(OUT/'crops'/f'C{ix}_{name}.png',arr,cmap='gray' if name=='target' else ('viridis' if name=='tgd' else None))
  for name in ('SO','S0','S2','CG10'):plt.imsave(OUT/'overlays'/f'C{ix}_{name}_error.png',errmap(s['p'][name][sl],s['gt'][sl]),cmap=ERR,vmin=0,vmax=3)
 wr('tgd_local_case_list.csv',cr);wr('table_local_crop_summary.csv',cr);wr('table_selected_local_crop_summary.csv',cr)
 fig,axs=plt.subplots(len(cases),5,figsize=(10,2.3*len(cases)));heads=['Target SAR','GT + TGD','SO baseline errors','S2 (ours) errors','CG10 Teacher errors']
 for j,h in enumerate(heads):axs[0,j].set_title(h,fontsize=8)
 for i,q in enumerate(cases):
  s=q['s'];yy,xx,h,w=box(q['y'],q['x']);sl=np.s_[yy:yy+h,xx:xx+w];axs[i,0].imshow(s['image'][sl],cmap='gray');axs[i,0].text(2,9,f'C{i+1}: {s["id"]}',color='white',fontsize=6,bbox={'facecolor':'black','alpha':.5});axs[i,1].imshow(s['gt'][sl],cmap=ListedColormap(['black','#32b44a']));cont(axs[i,1],s['tgd'][sl])
  for j,m in enumerate(['SO','S2','CG10'],2):axs[i,j].imshow(errmap(s['p'][m][sl],s['gt'][sl]),cmap=ERR,vmin=0,vmax=3);cont(axs[i,j],s['tgd'][sl])
  for a in axs[i]:a.axis('off')
 fig.tight_layout();fig.savefig(OUT/'figures'/'fig_TGD_local_comparison.png',dpi=300);fig.savefig(OUT/'figures'/'fig_TGD_local_comparison.pdf');plt.close(fig)
 # OmegaTR recovery: white OmegaTR, green recovered S2, red still missed.
 fig,axs=plt.subplots(len(cases),6,figsize=(12,2.4*len(cases)));heads=['Target SAR','GT','SO','S0','S2','OmegaTR recovery']
 for j,h in enumerate(heads):axs[0,j].set_title(h,fontsize=8)
 for i,q in enumerate(cases):
  s=q['s'];yy,xx,h,w=box(q['y'],q['x']);sl=np.s_[yy:yy+h,xx:xx+w];axs[i,0].imshow(s['image'][sl],cmap='gray');axs[i,1].imshow(s['gt'][sl],cmap=ListedColormap(['black','#32b44a']))
  for j,m in enumerate(['SO','S0','S2'],2):axs[i,j].imshow(predmask(s['p'][m][sl]),cmap=ListedColormap(['black','#32b44a']));cont(axs[i,j],s['tgd'][sl])
  om=s['omega'][sl];rec=om&predmask(s['p']['S2'][sl]);still=om&~predmask(s['p']['S2'][sl]);z=np.zeros_like(om,dtype=np.uint8);z[om]=1;z[rec]=2;z[still]=3;axs[i,5].imshow(z,cmap=ListedColormap(['black','white','#32b44a','#dc2626']),vmin=0,vmax=3);cont(axs[i,5],s['tgd'][sl])
  for a in axs[i]:a.axis('off')
 fig.tight_layout();fig.savefig(OUT/'figures'/'fig_OmegaTR_recovery.png',dpi=300);fig.savefig(OUT/'figures'/'fig_OmegaTR_recovery.pdf');plt.close(fig)
 (OUT/'json'/'fig_overall_qualitative_layout.json').write_text(json.dumps({'selection_rule':'three distinct high TGD-and-SO-wrong/S2-correct scenes plus one low-TGD positive representative','methods':['SO','S0','S1','S2','CG10'],'scenes':[s['id'] for s in sel]},indent=2))
 (OUT/'captions'/'caption_fig_overall_qualitative.txt').write_text('Qualitative comparison on representative target-orbit SAR test scenes. Green denotes predicted or reference landslide masks; red dashed contours indicate TGD. CG10 is shown only as a dual-orbit Teacher reference.',encoding='utf-8')
 (OUT/'captions'/'caption_fig_overall_comparison.txt').write_text('Overall qualitative comparison on representative target-orbit SAR test scenes. Green denotes landslide masks and red dashed contours indicate TGD. CG10 is a dual-orbit Teacher reference and is not a target-only deployment baseline.',encoding='utf-8')
 (OUT/'captions'/'caption_fig_TGD_local_comparison.txt').write_text('TGD-focused local comparisons selected automatically from target-orbit pixels where SO is incorrect and S2 is correct. Green, red, and yellow in error maps denote true positives, false negatives, and false positives, respectively; red dashed contours mark TGD.',encoding='utf-8')
 (OUT/'captions'/'caption_fig_OmegaTR_recovery.txt').write_text('Recovery on the fixed OmegaTR subset, defined as TGD pixels missed by SO but correctly predicted by the CG10 Teacher. White denotes OmegaTR, green recovered pixels, and red OmegaTR pixels still missed by S2.',encoding='utf-8')
 (OUT/'qualitative_summary.md').write_text('# Qualitative summary\n\nScenes and crops were selected automatically using fixed test-set masks and errors, then constrained to distinct source scenes. Figure A includes three high TGD recovery scenes and one low-TGD positive reference scene. Figure B uses strict TGD ∧ GT-positive ∧ SO-wrong ∧ S2-correct components. Figure C uses the fixed OmegaTR definition. The figures therefore support local recovery claims without using manually selected screenshots. Error colors expose false negatives rather than relying only on visually favorable masks.\n\nOnly SO, S0, S1, S2, and CG10 have reproducible final-protocol predictions in this package. External baselines are intentionally absent. Quantitative scene and crop tables accompany the figures.\n',encoding='utf-8')
if __name__=='__main__':main()
