"""Supplement final audit: integrity, joint-orbit controls, shift tests, QC and consolidation.
Run after rerun_quantitative.py. Read-only inputs, no legacy deletion.
"""
from pathlib import Path
import json,hashlib,shutil,importlib.util
import numpy as np,pandas as pd,rasterio
from scipy import ndimage,stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
R=Path(__file__).resolve().parents[2];F=R/'experiments/dataset_audit_final';S=F/'01_statistics';T=F/'02_tables';Q=F/'03_manual_qc';C=F/'99_cleanup_manifest';SEED=20260927
spec=importlib.util.spec_from_file_location('engine',C/'legacy_scripts/run_extended_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=R;m.DATA=R/'data/haiti';m.DEM_DIR=R/'experiments/dataset_audit_final/05_existing_results_reused/experiments/windows_main/physics_audit_20260924/dem'
def csv(rows,path):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
def mean(a,k):return float(np.mean(a[k])) if np.any(k) else np.nan
def ci(x):
 x=np.asarray(x);x=x[np.isfinite(x)];rng=np.random.default_rng(SEED)
 return np.quantile([rng.choice(x,len(x)).mean() for _ in range(2000)],[.025,.975]).tolist() if len(x) else [None,None]
def read(sid):
 z={}
 for o,e in m.RAW:
  with rasterio.open(m.DATA/m.RAW[o,e]/f'{sid}.tif') as src:z[o,e]=src.read();tf=src.transform
 with rasterio.open(m.DATA/f'processed/sample_{sid:06d}/landslide_mask.tif') as src:gt=src.read(1)>0
 return z,gt,tf

def supplement():
 dem,sl0,asp0,dt,dp=m.load_dem_terrain();integrity=[];rows=[];reg=[];cross=[];gv=[];terrain=[]
 shiftpool={(o,dy,dx):[] for o in m.ORBITS for dy in range(-3,4) for dx in range(-3,4)}
 for sid in range(m.N):
  z,gt,tf=read(sid);geom={o:z[o,'pre'][2] for o in m.ORBITS};bv=(geom['asc']==0)&(geom['desc']==0)
  sl=m.resample_dem(sl0,dt,tf);asp=m.resample_dem(asp0,dt,tf,resampling=m.Resampling.nearest);el=m.resample_dem(dem,dt,tf)
  paths=[(o+'_'+e,m.DATA/sub/f'{sid}.tif') for (o,e),sub in m.RAW.items()]+[(k,m.DATA/f'processed/sample_{sid:06d}/{k}.tif') for k in ['landslide_mask','asc_pre_geom','desc_pre_geom']]
  for role,p in paths:
   with rasterio.open(p) as src:
    a=src.read();cat=[2] if src.count==3 else [0];constant=[b+1 for b in range(src.count) if np.ptp(a[b])==0]
    integrity.append(dict(sample_id=sid,role=role,path=p.relative_to(R).as_posix(),exists=True,shape=str(a.shape),dtype=str(src.dtypes),nodata=src.nodata,unique_labels=json.dumps({str(b+1):np.unique(a[b]).tolist() for b in cat}),transform=str(tuple(src.transform)),crs=str(src.crs),pixel_resolution=str(src.res),strict_grid=src.transform==tf and str(src.crs)=='EPSG:4326' and a.shape[-2:]==(128,128),nonfinite_pixels=int((~np.isfinite(a)).sum()),all_zero=bool(np.all(a==0)),constant_bands=str(constant),constant_sar=bool(src.count==3 and any(b in constant for b in [1,2]))))
  boundary=gt&~ndimage.binary_erosion(gt);bg=bv&~gt;g=bv&gt
  changes={};textures={};contrasts={}
  for o in m.ORBITS:
   changes[o]=np.abs(z[o,'post'][:2]-z[o,'pre'][:2]);textures[o]={};contrasts[o]={}
   for b in [0,1]:textures[o][b],contrasts[o][b]=m.calc_local_change(z[o,'pre'][b],z[o,'post'][b])
  joint=.5*(np.hypot(*changes['asc'])+np.hypot(*changes['desc']))
  rr=dict(sample_id=sid,mean_slope=float(sl.mean()),mean_elevation=float(el.mean()),gt_low_slope_under3_fraction=mean((sl<3).astype(float),gt),gt_boundary_change_association=mean(joint,boundary&bv)-mean(joint,bg),gt_border_fraction=float(gt[np.r_[0,127],:].sum()+gt[:,np.r_[0,127]].sum())/max(1,gt.sum()),gt_rectangle_fill=np.nan)
  ys,xs=np.where(gt)
  if len(ys):rr['gt_rectangle_fill']=gt.sum()/((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1))
  for label in [False,True]:
   k=gt==label
   terrain.append(dict(sample_id=sid,gt=int(label),n=int(k.sum()),slope_mean=mean(sl,k),elevation_mean=mean(el,k),aspect_sin_mean=mean(np.sin(np.deg2rad(asp)),k),aspect_cos_mean=mean(np.cos(np.deg2rad(asp)),k)))
  for o in m.ORBITS:
   base=mean(changes[o][1],g)-mean(changes[o][1],bg);best=(-1e9,0,0);improve=[]
   for dy in range(-3,4):
    for dx in range(-3,4):
     shifted=m.shift_without_wrap(gt,dy,dx);common=np.ones(gt.shape,bool)
     if dy>0:common[:dy]=False
     if dy<0:common[dy:]=False
     if dx>0:common[:,:dx]=False
     if dx<0:common[:,dx:]=False
     k=bv&common;cur=mean(changes[o][1],shifted&k)-mean(changes[o][1],~shifted&k)
     ref=mean(changes[o][1],gt&k)-mean(changes[o][1],~gt&k);gain=cur-ref
     if np.isfinite(gain):shiftpool[o,dy,dx].append(gain)
     reg.append(dict(sample_id=sid,orbit=o,dy=dy,dx=dx,association=cur,baseline_same_support=ref,gain=gain))
     if np.isfinite(gain) and gain>best[0]:best=(gain,dy,dx)
   rr[o+'_registration_shift_score']=best[0] if best[0]>-1e8 else np.nan;rr[o+'_best_shift_dy']=best[1];rr[o+'_best_shift_dx']=best[2]
  for region,valid,invalid in [('G10','desc','asc'),('G01','asc','desc')]:
   k=gt&(geom[valid]==0)&(geom[invalid]>0)
   r=dict(sample_id=sid,region=region,gt_pixels=int(k.sum()))
   for b,pol in [(1,'VV'),(0,'VH')]:
    r[pol+'_valid_wins_fraction']=mean((changes[valid][b]>changes[invalid][b]).astype(float),k)
    for name,arr in [('abs_change',changes),('std_change',textures),('contrast_change',contrasts)]:
     r[pol+'_'+name+'_valid']=mean(arr[valid][b],k);r[pol+'_'+name+'_invalid']=mean(arr[invalid][b],k)
    for o in [valid,invalid]:
     pre=z[o,'pre'][b];std=np.sqrt(np.maximum(ndimage.uniform_filter(pre*pre,5)-ndimage.uniform_filter(pre,5)**2,0))
     r[pol+'_local_std_'+o]=mean(std,k);r[pol+'_pre_mean_'+o]=mean(pre,k)
     r[pol+'_dark_below_minus25_'+o]=mean((pre< -25).astype(float),k);r[pol+'_bright_above0_'+o]=mean((pre>0).astype(float),k)
   gv.append(r)
  # Pixel ranks and terrain-stratified matched background ranks, deterministic.
  st=np.digitize(sl,[10,20,30,40])*100+np.where(sl<3,8,np.floor((asp+22.5)/45).astype(int)%8)*10+np.digitize(el,m.ELEV_EDGES)
  eligible=np.zeros(gt.shape,bool);mb=np.zeros(gt.shape,bool)
  for s in np.unique(st[g]):
   if np.any(bg&(st==s)):eligible|=g&(st==s);mb|=bg&(st==s)
  for cls,k in [('GT',g),('matched_GT',eligible),('matched_background',mb)]:
   for metric,aa,dd in [('abs_DVV',changes['asc'][1],changes['desc'][1]),('abs_DVH',changes['asc'][0],changes['desc'][0]),('VV_std_change',textures['asc'][1],textures['desc'][1]),('VH_std_change',textures['asc'][0],textures['desc'][0])]:
    ix=np.flatnonzero(k);rng=np.random.default_rng(SEED+sid)
    if len(ix)>1024:ix=rng.choice(ix,1024,replace=False)
    corr=stats.spearmanr(aa.ravel()[ix],dd.ravel()[ix]).statistic if len(ix)>5 else np.nan
    cross.append(dict(sample_id=sid,region=cls,metric=metric,n_sampled=len(ix),spearman=corr))
  rows.append(rr)
  if sid%200==0:print('Supplement',sid,flush=True)
 csv(integrity,S/'file_integrity.csv');csv(rows,S/'supplement_patch.csv');csv(reg,S/'registration_shift_per_patch.csv');csv(cross,S/'cross_orbit_pixel_correlations.csv');csv(gv,S/'g10_g01_patch_statistics.csv');csv(terrain,S/'gt_terrain_patch_statistics.csv')
 out=[]
 for (o,dy,dx),v in shiftpool.items():
  lo,hi=ci(v);out.append(dict(orbit=o,dy=dy,dx=dx,n_patches=len(v),mean_gain=np.mean(v) if v else np.nan,ci_low=lo,ci_high=hi,fraction_gain_positive=np.mean(np.array(v)>0) if v else np.nan))
 csv(out,S/'registration_shift_statistics.csv')
 df=pd.read_csv(S/'dataset_patch_audit.csv');df=df[[c for c in df if not c.startswith('legacy_') and c not in ['slope_mean','slope_p90','elev_mean']]].merge(pd.DataFrame(rows),on='sample_id',validate='one_to_one');df['slope_mean']=df.mean_slope
 df['registration_shift_score']=df[['asc_registration_shift_score','desc_registration_shift_score']].max(axis=1)
 df['matched_background_change_ratio']=df[['asc_dvv_abs_matched_ratio','desc_dvv_abs_matched_ratio']].mean(axis=1)
 csv(df,T/'dataset_patch_audit.csv')
 # No missing metric is filled with zero for scoring.
 scores={'A':1-df[['asc_dvv_abs_matched_ratio','desc_dvv_abs_matched_ratio']].max(axis=1).rank(pct=True),'B':df[['asc_gt_in_bad','desc_gt_in_bad']].max(axis=1),'C':df.gt_slope_under10_frac,'D':1-df[['asc_slope_mask_spearman','desc_slope_mask_spearman']].mean(axis=1).rank(pct=True),'E':df.registration_shift_score.rank(pct=True)}
 ar=[]
 for typ,score in scores.items():
  eligible=(df.gt_frac>=.02)&score.notna()
  if typ=='A':eligible&=df.gt_bothvalid_pixels>=100
  sub=df.loc[eligible].assign(score=score[eligible]).nlargest(20,'score')
  dest=F/f'04_anomaly_cases/type_{typ}';dest.mkdir(exist_ok=True)
  records=[dict(type=typ,rank=i+1,sample_id=int(r.sample_id),score=r.score,status='needs_manual_review',rule={'A':'GT>=2%; BothValid GT>=100; lowest max orbit terrain-matched VV ratio','B':'GT>=2%; highest fraction GT in either invalid mask','C':'GT>=2%; highest GT slope<10deg fraction','D':'GT>=2%; weakest slope-mask association proxy; aspect physics unresolved','E':'GT>=2%; highest clipped same-support shift gain'}[typ]) for i,(_,r) in enumerate(sub.iterrows())]
  csv(records,dest/'ranking.csv');ar+=records
 csv(ar,T/'anomaly_summary.csv');print('Supplement done',flush=True)


def export_tables():
 s=json.loads((S/'audit_summary.json').read_text());g=s['geometry']
 csv([dict(orbit=o,code=k,meaning=['valid','layover','shadow','layover+shadow'][k],n_pixels=n,fraction=n/sum(v)) for o,v in g['class_counts_by_code'].items() for k,n in enumerate(v)],S/'geometry_summary.csv')
 joint=g['binary_joint_counts_rows_ASC_invalid_0_1_cols_DESC_invalid_0_1'];csv([dict(ASC_invalid=a,DESC_invalid=d,n_pixels=joint[a][d],fraction=joint[a][d]/s['patch_pixels_with_repetition']) for a in [0,1] for d in [0,1]],S/'joint_geometry_summary.csv')
 csv([dict(orbit=o,**r) for o,rs in g['slope_geometry_rates'].items() for r in rs]+[dict(orbit=o,**r) for o,rs in g['aspect_geometry_rates'].items() for r in rs],S/'slope_aspect_statistics.csv')
 cont=[]
 for o,rs in g['slope_geometry_rates'].items():
  for r in rs:
   a,c=r['invalid_GT_bg_counts'];b,d=r['valid_GT_bg_counts'];den=lambda x,y:x/y if y else np.nan
   cont.append(dict(orbit=o,slope_bin=r['slope_bin_deg'],P_invalid_given_GT=den(a,a+b),P_invalid_given_background=den(c,c+d),P_GT_given_invalid=den(a,a+c),P_GT_given_valid=den(b,b+d),IoU=den(a,a+b+c),Dice=den(2*a,2*a+b+c),odds_ratio=den(a*d,b*c),relative_risk=den(den(a,a+b),den(c,c+d)),invalid_GT=a,valid_GT=b,invalid_BG=c,valid_BG=d))
 csv([dict(orbit=o,slope_bin='all',**r) for o,r in g['overlap_gt'].items()]+cont,S/'gt_geometry_statistics.csv')
 csv([dict(orbit=o,metric=k,region=c,**v) for o,metrics in s['bothvalid_sar_change'].items() for k,regions in metrics.items() for c,v in regions.items()],S/'sar_change_statistics.csv')
 csv([dict(orbit=o,metric=k,**v) for o,metrics in s['terrain_matched_change']['matched_background_results_by_orbit_and_metric'].items() for k,v in metrics.items()],S/'matched_background_statistics.csv')
 df=pd.read_csv(T/'dataset_patch_audit.csv');csv(df[df.gt_slope_under10_frac>=.5].sort_values('gt_slope_under10_frac',ascending=False),T/'abnormal_low_slope_gt_patches.csv')
 v=pd.read_csv(S/'g10_g01_patch_statistics.csv');r=[]
 for region in ['G10','G01']:
  for pol in ['VV','VH']:
   x=v.loc[v.region==region,pol+'_valid_wins_fraction'].dropna();lo,hi=ci(x)
   r.append(dict(region=region,metric=pol+' valid stronger pixel fraction',n_patches=len(x),mean_patch_fraction=x.mean(),ci_low=lo,ci_high=hi,interpretation='numeric amplitude proxy; human interpretability is not measured'))
 csv(r,S/'g10_g01_statistics.csv')
 figs=S/'figures';figs.mkdir(exist_ok=True)
 fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 gt=np.array(g['global_gt_slope_hist_counts']);bg=np.array(g['global_bg_slope_hist_counts']);x=np.arange(5)
 ax[0].bar(x-.18,gt/gt.sum(),.36,label='GT');ax[0].bar(x+.18,bg/bg.sum(),.36,label='BG');ax[0].set_xticks(x,g['gt_slope_bins']);ax[0].set_title('Slope normalized frequency');ax[0].legend()
 ax[1].bar(x,gt/(gt+bg));ax[1].set_xticks(x,g['gt_slope_bins']);ax[1].set_title('P(GT | slope bin)')
 for o in m.ORBITS:ax[2].plot(x,[v['invalid_fraction'] for v in g['slope_geometry_rates'][o]],marker='o',label=o)
 ax[2].set_xticks(x,g['gt_slope_bins']);ax[2].set_title('P(invalid | slope bin)');ax[2].legend();fig.savefig(figs/'slope_gt_geometry.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(12,6),layout='constrained')
 for i,o in enumerate(m.ORBITS):
  data=np.array([[r[k] for r in g['aspect_geometry_rates'][o]] for k in ['valid_pct','layover_pct','shadow_pct','both_pct']]);im=ax[i].imshow(data,aspect='auto',vmin=0,vmax=100);ax[i].set_xticks(range(9),g['gt_aspect_bins']);ax[i].set_yticks(range(4),['valid','layover','shadow','both']);ax[i].set_title(o.upper()+' geometry % within aspect sector');fig.colorbar(im,ax=ax[i])
 fig.savefig(figs/'aspect_geometry_heatmap.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');tp=pd.read_csv(S/'gt_terrain_patch_statistics.csv')
 for label in [0,1]:
  sub=tp[tp['gt']==label];ax[0].hist(sub.elevation_mean,bins=30,histtype='step',density=True,label=['BG','GT'][label]);ang=np.mod(np.rad2deg(np.arctan2(sub.aspect_sin_mean,sub.aspect_cos_mean)),360);ax[1].hist(ang,bins=np.arange(0,361,45),histtype='step',density=True,label=['BG','GT'][label])
 ax[0].set_title('Patch-class mean elevation');ax[1].set_title('Patch-class circular mean aspect');ax[0].legend();ax[1].legend();fig.savefig(figs/'gt_elevation_aspect.png',dpi=200);plt.close(fig)
 return s

def qc():
 df=pd.read_csv(T/'dataset_patch_audit.csv').set_index('sample_id',drop=False);rng=np.random.default_rng(SEED)
 moderate=df[(df.gt_frac>=.02)&(df.gt_frac<=df.gt_frac.quantile(.8))&(df.gt_bothvalid_frac>=.8)]
 groups=[('1',moderate.sample(n=min(10,len(moderate)),random_state=SEED),'random seed; GT>=2%, <=P80 area; >=80% GT BothValid'),('2',df[(df.gt_frac>=.02)&(df.asc_gt_in_bad>.3)&(df.desc_gt_in_bad<.3)].sample(n=8,random_state=SEED),'GT>=2%; ASC overlap>30%, DESC<30%; random'),('3',df[(df.gt_frac>=.02)&(df.desc_gt_in_bad>.3)&(df.asc_gt_in_bad<.3)].sample(n=8,random_state=SEED),'GT>=2%; DESC overlap>30%, ASC<30%; random'),('4',df[(df.gt_frac>=.02)&(df.gt_bothvalid_pixels>=100)].assign(weak=df[['asc_dvv_abs_gt_bothvalid','desc_dvv_abs_gt_bothvalid']].max(axis=1)).nsmallest(10,'weak'),'GT>=2%; >=100 BothValid GT; lowest max orbit absolute VV change'),('5',df[df.gt_frac>=.02].assign(overlap=df[['asc_gt_in_bad','desc_gt_in_bad']].max(axis=1)).nlargest(10,'overlap'),'GT>=2%; greatest GT-in-either-invalid fraction')]
 dem,sl0,asp0,dt,_=m.load_dem_terrain();index=[];key=[];limits=[]
 gcmap=ListedColormap(['#eeeeee','#ef9038','#4c83c4','#b949ad']);gnorm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
 blindids={int(sid):f'B{i+1:03d}' for i,sid in enumerate(rng.permutation(sorted(set(int(sid) for g,sub,_ in groups if g in ['1','4'] for sid in sub.index))))}
 def plot(sid,r,path,blind=False):
  z,gt,tf=read(sid);sl=m.resample_dem(sl0,dt,tf);asp=m.resample_dem(asp0,dt,tf,resampling=m.Resampling.nearest)
  ranges={(o,b):np.percentile(np.r_[z[o,'pre'][b].ravel(),z[o,'post'][b].ravel()],[2,98]) for o in m.ORBITS for b in [0,1]}
  d={o:z[o,'post'][:2]-z[o,'pre'][:2] for o in m.ORBITS};vmax=max(np.percentile(np.abs(d[o]),98) for o in m.ORBITS)
  fig,axes=plt.subplots(5,4,figsize=(16,20));fig.subplots_adjust(top=.945,bottom=.035,left=.035,right=.97,hspace=.29,wspace=.22)
  title=f"Blind {blindids[sid]}" if blind else f"Patch {sid} | GT {r.gt_frac:.1%} | ASC invalid {r.asc_bad_frac:.1%} | DESC invalid {r.desc_bad_frac:.1%} | slope {r.mean_slope:.1f} deg"
  fig.suptitle(title,fontsize=16)
  def show(i,a,title,cmap='gray',vmin=None,vmax=None,norm=None,contour=False):
   ax=axes.flat[i];im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,norm=norm);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
   if contour and gt.any():ax.contour(gt,levels=[.5],colors='red',linewidths=.6)
   cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
   if norm is not None:cb.set_ticks([0,1,2,3],labels=['valid','layover','shadow','both']);cb.ax.tick_params(labelsize=7)
  for oi,o in enumerate(m.ORBITS):
   lo,hi=ranges[o,1];i=oi*4;show(i,z[o,'pre'][1],o.upper()+' pre VV [dB]',vmin=lo,vmax=hi);show(i+1,z[o,'post'][1],o.upper()+' post VV [dB]',vmin=lo,vmax=hi);show(i+2,d[o][1],o.upper()+' signed delta VV [dB]','RdBu_r',-vmax,vmax);show(i+3,z[o,'pre'][2],o.upper()+' geometry',gcmap,norm=gnorm)
  for oi,o in enumerate(m.ORBITS):
   lo,hi=ranges[o,0];show(8+oi*2,z[o,'pre'][0],o.upper()+' pre VH [dB]',vmin=lo,vmax=hi);show(9+oi*2,z[o,'post'][0],o.upper()+' post VH [dB]',vmin=lo,vmax=hi)
  show(12,sl if blind else .5*(z['asc','pre'][1]+z['desc','pre'][1]),'Slope [deg]' if blind else 'GT contour on mean pre VV', 'terrain' if blind else 'gray',0 if blind else None,60 if blind else None,contour=not blind)
  show(13,sl,'Slope [deg]','terrain',0,60);show(14,asp,'Downhill aspect [deg]','twilight',0,360)
  show(15,.5*(np.abs(d['asc'][1])+np.abs(d['desc'][1])),'Mean |delta VV|'+(' + GT contour' if not blind else ''),'magma',0,vmax,contour=not blind)
  for oi,o in enumerate(m.ORBITS):show(16+oi,np.abs(d[o][1]),o.upper()+' |delta VV|','magma',0,vmax);show(18+oi,d[o][0],o.upper()+' signed delta VH [dB]','RdBu_r',-vmax,vmax)
  fig.text(.04,.012,'Paired SAR uses shared 2-98% limits; delta common symmetric limits across orbits/pols. No optical locally. Terrain from 30m DEM.',fontsize=10)
  fig.savefig(path,dpi=150);plt.close(fig)
  limits.append(dict(sample_id=sid,file=path.relative_to(F).as_posix(),delta_symmetric_limit=vmax,paired_ranges=json.dumps({str(k):v.tolist() for k,v in ranges.items()})))
 for group,sub,rule in groups:
  for sid,r in sub.iterrows():
   sid=int(sid);name=f'group{group}_patch{sid:04d}.png';plot(sid,r,Q/'contact_sheets'/name)
   index.append(dict(sample_id=sid,group=group,selection_rule=rule,seed=SEED,GT_fraction=r.gt_frac,slope=r.mean_slope,ASC_invalid_fraction=r.asc_bad_frac,DESC_invalid_fraction=r.desc_bad_frac,anomaly_score=r.registration_shift_score,contact_sheet='contact_sheets/'+name))
   print('QC',group,sid,flush=True)
 for sid,bid in blindids.items():
  plot(sid,df.loc[sid],Q/'blind'/f'{bid}.png',True);key.append(dict(blind_id=bid,sample_id=sid,groups=','.join(g for g,sub,_ in groups if sid in sub.index),GT_fraction=df.loc[sid].gt_frac,answer_sheet=';'.join(x['contact_sheet'] for x in index if x['sample_id']==sid)))
 csv(index,Q/'manual_qc_index.csv');csv(key,Q/'blind_ground_truth_key.csv');csv(limits,Q/'display_ranges.csv')
 fields=['landslide_shape_plausible','on_real_hillslope','asc_change_visible','desc_change_visible','at_least_one_orbit_supports_gt','geometry_mask_matches_visual_distortion','asc_desc_geometry_complementary','possible_registration_shift','possible_annotation_problem','notes']
 csv([dict(sample_id=sid,**{k:'' for k in fields}) for sid in sorted(set(x['sample_id'] for x in index))],Q/'manual_qc_form.csv')
 (Q/'manual_qc_instructions.md').write_text('人工填写，AI 未填写。yes/uncertain/no；change 字段 clear/weak/none。先盲检再查看答案；blind 文件名随机化，无 GT、group、异常标签。5 组共 46 张，可能跨组重复。所有 panel 保留，SAR 同 orbit/pol pre/post 同范围；变化使用公共对称范围，absolute panel 使用其正半轴。contact sheet 的 contour 不遮挡 SAR。Type D 仅坡度代理，缺原始轨道方位不能裁决方向物理。',encoding='utf-8')
 # Every top-20 anomaly gets a full sheet; reuse QC files when possible.
 ar=pd.read_csv(T/'anomaly_summary.csv');existing={x['sample_id']:Q/x['contact_sheet'] for x in index}
 for sid in sorted(set(ar.sample_id)):
  if sid not in existing:
   p=F/'04_anomaly_cases'/f'patch{sid:04d}.png';plot(int(sid),df.loc[sid],p);existing[sid]=p
 for _,r in ar.iterrows():shutil.copy2(existing[r.sample_id],F/f'04_anomaly_cases/type_{r.type}/patch{int(r.sample_id):04d}.png')
 print('QC done',len(index),len(blindids),flush=True)

if __name__=='__main__':
 import sys
 stage=sys.argv[1] if len(sys.argv)>1 else 'all'
 if stage in ['all','supplement']:supplement()
 if stage in ['all','tables']:export_tables()
 if stage in ['all','qc']:qc()
