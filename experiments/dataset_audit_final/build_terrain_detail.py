"""Joint slope/aspect geometry consistency; empirical review score, not orbit physics proof."""
import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,rasterio
R=Path(__file__).resolve().parents[2];F=R/'experiments/dataset_audit_final';S=F/'01_statistics';T=F/'02_tables'
sp=importlib.util.spec_from_file_location('b',F/'build_final_audit.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
dem,sl,asp,tf,_=b.m.load_dem_terrain();hist=[];eh=np.zeros((2,16),np.int64)
for sid in range(1713):
 z,gt,dt=b.read(sid);s=b.m.resample_dem(sl,tf,dt);a=b.m.resample_dem(asp,tf,dt,resampling=b.m.Resampling.nearest);e=b.m.resample_dem(dem,tf,dt)
 sb=np.digitize(s,[10,20,30,40]);ab=np.where(s<3,8,np.floor((a+22.5)/45).astype(int)%8)
 for o in ['asc','desc']:
  codes=z[o,'pre'][2].astype(int);h=np.bincount(((sb*9+ab)*4+codes).ravel(),minlength=180).reshape(5,9,4);hist.append((sid,o,h))
 for cls in [0,1]:eh[cls]+=np.histogram(e[(gt==bool(cls))],bins=np.arange(-100,1501,100))[0]
 if sid%400==0:print('Terrain detail',sid,flush=True)
pooled={o:sum(h for _,orb,h in hist if orb==o) for o in ['asc','desc']};rows=[];scores=[]
for o,h in pooled.items():
 for si in range(5):
  for ai in range(9):
   for code in range(4):rows.append(dict(orbit=o,slope_bin=si,aspect_bin=ai,geometry_code=code,n_pixels=int(h[si,ai,code]),fraction_within_terrain_cell=h[si,ai,code]/max(1,h[si,ai].sum())))
for sid,o,h in hist:
 other=pooled[o]-h;n=h.sum(axis=2);bad=h[:,:,1:].sum(axis=2);expected=other[:,:,1:].sum(axis=2)/np.maximum(other.sum(axis=2),1);valid=(n>=20)&(other.sum(axis=2)>=100)
 score=np.average((bad[valid]/n[valid]-expected[valid])**2,weights=n[valid])**.5 if valid.any() else np.nan
 scores.append(dict(sample_id=sid,orbit=o,slope_aspect_invalid_residual_rmse=score,covered_pixels=int(n[valid].sum()),interpretation='leave-one-patch-out empirical terrain-cell inconsistency; shared overlapping ground is not removed'))
b.csv(rows,S/'slope_aspect_joint_geometry.csv');b.csv(scores,S/'slope_aspect_patch_consistency.csv');b.csv([dict(gt=cls,elevation_bin=f'{lo} to {lo+100} m',n_pixels=int(n)) for cls in [0,1] for lo,n in zip(range(-100,1500,100),eh[cls])],S/'gt_elevation_histogram.csv')
df=pd.read_csv(T/'dataset_patch_audit.csv');wide=pd.DataFrame(scores).pivot(index='sample_id',columns='orbit',values='slope_aspect_invalid_residual_rmse');wide.columns=[o+'_slope_aspect_inconsistency' for o in wide.columns];df=df.merge(wide,on='sample_id',validate='one_to_one');b.csv(df,T/'dataset_patch_audit.csv')
# Keep original top20 D score as review history; use joint terrain residual for final D ranking.
ar=pd.read_csv(T/'anomaly_summary.csv');b.csv(ar[ar.type=='D'],F/'99_cleanup_manifest/type_D_slope_only_prior_ranking.csv');score=df[['asc_slope_aspect_inconsistency','desc_slope_aspect_inconsistency']].max(axis=1);sub=df[(df.gt_frac>=.02)&score.notna()].assign(score=score).nlargest(20,'score')
new=[dict(type='D',rank=i+1,sample_id=int(r.sample_id),score=r.score,status='needs_manual_review',rule='GT>=2%; greatest leave-one-patch-out slope x aspect invalid-rate residual RMSE; empirical consistency proxy, not physical proof') for i,(_,r) in enumerate(sub.iterrows())]
ar=pd.concat([ar[ar.type!='D'],pd.DataFrame(new)],ignore_index=True);b.csv(ar,T/'anomaly_summary.csv');b.csv(new,F/'04_anomaly_cases/type_D/ranking.csv')
print('Terrain detail done',flush=True)

