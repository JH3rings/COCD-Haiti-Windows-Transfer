"""Create cross-method qualitative figures from verified external prediction caches."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'cocd'))
from paths import DEV
from windows_main.data import HaitiPairs,split
from windows_main.models_da_search import TargetOnlyBaseline
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from windows_main.models_teacher_target_search_v2 import load_so_backbone
OUT=ROOT/'experiments/GRSL_complete_paper_package/qualitative';OLD=ROOT/'experiments/windows_main/da_search/grsl_assets_v1/figure_data/predictions/predictions_SO.npz';EXT=ROOT/'experiments/single_orbit_baselines_v3';SP=ROOT/'experiments/windows_main/da_search/SO_backbone_seed42.pt';S2=ROOT/'experiments/windows_main/student_target_search_v2/S2_seed42.pt'
DIS2=ROOT/'experiments/tgd_robustness_analysis/dis2_checkpoint_retest_predictions.npz'
VKD=ROOT/'experiments/tgd_robustness_analysis/vanilla_kd_retest_predictions.npz'
GREEN=ListedColormap(['black','#32b44a']);ERR=ListedColormap(['black','#32b44a','#dc2626','#f3c623'])
def key(y,g,c,o):return hashlib.sha1(np.ascontiguousarray(y).astype(np.uint8).tobytes()+np.ascontiguousarray(g).astype(np.uint8).tobytes()+np.ascontiguousarray(c).astype(np.uint8).tobytes()+bytes([int(o)])).hexdigest()
def emap(p,y):
 h=p>=.5;z=np.zeros_like(y,dtype=np.uint8);z[h&(y>0)]=1;z[(~h)&(y>0)]=2;z[h&(y==0)]=3;return z
def contour(ax,m):
 if m.any():ax.contour(m.astype(float),levels=[.5],colors='#d62728',linestyles='--',linewidths=.8)
def main():
 old=np.load(OLD);order=[(int(s),int(o)) for s,o in zip(old['sample_id'],old['orbit'])]
 scenes=list(csv.DictReader(open(OUT/'csv'/'overall_qualitative_scene_list.csv',encoding='utf-8')));cases=list(csv.DictReader(open(OUT/'csv'/'tgd_local_case_list.csv',encoding='utf-8')))
 required={(int(r['scene_id'].split('_')[0]),0 if r['scene_id'].endswith('_asc') else 1) for r in scenes}
 required|={(int(r['source_scene'].split('_')[0]),0 if r['source_scene'].endswith('_asc') else 1) for r in cases}
 ext={};availability=[]
 for folder,name in [('boehm','Boehm SAR U-Net++'),('cdnette','CDNetE Early Fusion'),('mfewf','MFEWF adapted')]:
  z=np.load(EXT/folder/'test_predictions.npz');
  if len(z['p'])!=len(order) or not np.array_equal(z['orbit'],old['orbit']):raise RuntimeError(f'locked order mismatch: {name}')
  ext[name]={order[i]:z['p'][i].copy() for i in range(len(order)) if order[i] in required};del z
  availability.append({'method':name,'aligned_predictions':len(ext[name]),'expected_selected_scenes':len(required),'status':'verified_locked_order' if len(ext[name])==len(required) else 'alignment_failed'})
 z=np.load(DIS2)
 dis2_orbit=np.where(z['orbit']=='asc',0,1).astype(old['orbit'].dtype)
 if len(z['p'])!=len(order) or not np.array_equal(dis2_orbit,old['orbit']):raise RuntimeError('locked order mismatch: DIS2')
 ext['DIS2-port']={order[i]:z['p'][i].copy() for i in range(len(order)) if order[i] in required};del z
 availability.append({'method':'DIS2-port','aligned_predictions':len(ext['DIS2-port']),'expected_selected_scenes':len(required),'status':'verified_locked_order' if len(ext['DIS2-port'])==len(required) else 'alignment_failed'})
 z=np.load(VKD)
 vkd_orbit=np.where(z['orbit']=='asc',0,1).astype(old['orbit'].dtype)
 if len(z['p'])!=len(order) or not np.array_equal(vkd_orbit,old['orbit']):raise RuntimeError('locked order mismatch: Vanilla KD')
 ext['Vanilla KD']={order[i]:z['p'][i].copy() for i in range(len(order)) if order[i] in required};del z
 availability.append({'method':'Vanilla KD','aligned_predictions':len(ext['Vanilla KD']),'expected_selected_scenes':len(required),'status':'verified_locked_order' if len(ext['Vanilla KD'])==len(required) else 'alignment_failed'})
 with (OUT/'csv'/'external_method_prediction_availability.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=availability[0]);w.writeheader();w.writerows(availability)
 if any(x['aligned_predictions']!=x['expected_selected_scenes'] for x in availability):raise RuntimeError(availability)
 tid,_,eid=split();tr=HaitiPairs(tid,True);ds=HaitiPairs(eid,False,tr.stats);idx={int(x):i for i,x in enumerate(ds.ids)}
 so=TargetOnlyBaseline().to(DEV).eval();load_so_backbone(so,str(SP));s2=StudentTargetSearchV2().to(DEV).eval();s2.load_state_dict(torch.load(S2,map_location=DEV,weights_only=True))
 @torch.no_grad()
 def get(sid):
  sample,orbit=sid.split('_');i=idx[int(sample)];a,d,y,ga,gd,_=ds[i];x=a if orbit=='asc' else d;g=ga.numpy() if orbit=='asc' else gd.numpy();c=gd.numpy() if orbit=='asc' else ga.numpy();o=0 if orbit=='asc' else 1
  return x[2].numpy(),y.numpy(),(g>0)&~(c>0),{'Boehm SAR U-Net++':ext['Boehm SAR U-Net++'][int(sample),o],'CDNetE Early Fusion':ext['CDNetE Early Fusion'][int(sample),o],'MFEWF adapted':ext['MFEWF adapted'][int(sample),o],'DIS2-port':ext['DIS2-port'][int(sample),o],'Vanilla KD':ext['Vanilla KD'][int(sample),o],'SO':torch.sigmoid(so(x[None].to(DEV))['z'])[0,0].cpu().numpy(),'S2':torch.sigmoid(s2(x[None].to(DEV))['z'])[0,0].cpu().numpy()}
 fig,axs=plt.subplots(len(scenes),9,figsize=(18,2.1*len(scenes)));heads=['Target SAR','GT','Boehm','CDNetE','MFEWF','DIS2','Vanilla KD','SO','S2 (ours)']
 for j,h in enumerate(heads):axs[0,j].set_title(h,fontsize=8)
 for i,row in enumerate(scenes):
  im,y,t,p=get(row['scene_id']);axs[i,0].imshow(im,cmap='gray');axs[i,0].text(2,9,row['scene_id'],color='white',fontsize=6,bbox={'facecolor':'black','alpha':.5});axs[i,1].imshow(y,cmap=GREEN)
  for j,m in enumerate(['Boehm SAR U-Net++','CDNetE Early Fusion','MFEWF adapted','DIS2-port','Vanilla KD','SO','S2'],2):axs[i,j].imshow(p[m]>=.5,cmap=GREEN)
  for a in axs[i]:a.axis('off')
 fig.tight_layout();fig.savefig(OUT/'figures'/'fig_overall_external_comparison.png',dpi=300);fig.savefig(OUT/'figures'/'fig_overall_external_comparison.pdf');plt.close(fig)
 crop_rows=[];fig,axs=plt.subplots(len(cases),8,figsize=(16,2.25*len(cases)));heads=['Target SAR','GT + TGD','Boehm errors','MFEWF errors','DIS2 errors','Vanilla KD errors','SO errors','S2 (ours) errors']
 for j,h in enumerate(heads):axs[0,j].set_title(h,fontsize=8)
 for i,row in enumerate(cases):
  im,y,t,p=get(row['source_scene']);yy,xx,h,w=map(int,row['crop_box'].split(','));sl=np.s_[yy:yy+h,xx:xx+w];axs[i,0].imshow(im[sl],cmap='gray');axs[i,0].text(2,9,row['case_id'],color='white',fontsize=7,bbox={'facecolor':'black','alpha':.5});axs[i,1].imshow(y[sl],cmap=GREEN);contour(axs[i,1],t[sl])
  for j,m in enumerate(['Boehm SAR U-Net++','MFEWF adapted','DIS2-port','Vanilla KD','SO','S2'],2):axs[i,j].imshow(emap(p[m][sl],y[sl]),cmap=ERR,vmin=0,vmax=3);contour(axs[i,j],t[sl]);crop_rows.append({'case_id':row['case_id'],'method':m,'error_count':int(((p[m][sl]>=.5)!=(y[sl]>0)).sum())})
  for a in axs[i]:a.axis('off')
 fig.tight_layout();fig.savefig(OUT/'figures'/'fig_TGD_external_comparison.png',dpi=300);fig.savefig(OUT/'figures'/'fig_TGD_external_comparison.pdf');plt.close(fig)
 with (OUT/'csv'/'table_external_local_crop_errors.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=crop_rows[0]);w.writeheader();w.writerows(crop_rows)
 (OUT/'captions'/'caption_fig_overall_external_comparison.txt').write_text('Overall qualitative comparison with verified local prediction caches for external baselines. Green denotes landslide masks. External results are shown for visual comparison only; their historical protocol status is documented in the accompanying availability table.',encoding='utf-8')
 (OUT/'captions'/'caption_fig_TGD_external_comparison.txt').write_text('TGD-focused local comparison with verified external prediction caches. Green, red, and yellow in error maps denote true positives, false negatives, and false positives, respectively; red dashed contours mark TGD.',encoding='utf-8')
 (OUT/'json'/'external_method_qualitative_manifest.json').write_text(json.dumps({'external_prediction_caches':availability,'not_figure_eligible':[],'alignment':'All cache rows were aligned by the identical locked test-order and verified orbit vector; only selected scene predictions were retained while building the figure.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
