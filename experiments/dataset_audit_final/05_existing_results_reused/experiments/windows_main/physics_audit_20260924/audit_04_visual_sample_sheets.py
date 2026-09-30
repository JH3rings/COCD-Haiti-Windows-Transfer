"""Deterministic contact sheets for geometry/GT visual sampling (no optical in package)."""
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:/deeplearning/COCD_Haiti_Windows_Transfer')
D=ROOT/'data/haiti'; O=ROOT/'experiments/windows_main/physics_audit_20260924'
N=1713
# Derive 30 m DEM slope once; visualize its nearest-sampled value on each S1 patch grid.
dems=[]
for fn in ('N18W075.tif','N18W074.tif'):
    with rasterio.open(O/'dem'/fn) as ds: dems.append(ds.read(1).astype(np.float32))
dem=np.concatenate(dems,axis=1); ddeg=1/3600; top=19.00013888888889; lon0=-75.00013888888889
lats=top-ddeg*(np.arange(dem.shape[0])+.5); dy=111320*ddeg; dx=111320*np.cos(np.deg2rad(lats))*ddeg
z=np.pad(dem,1,mode='edge'); dzdy=(z[2:,1:-1]-z[:-2,1:-1])/(2*dy); dzdx=(z[1:-1,2:]-z[1:-1,:-2])/(2*dx[:,None])
slope=np.degrees(np.arctan(np.sqrt(dzdx**2+dzdy**2))).astype(np.float32)
def patch_slope(bounds):
    x0=(bounds[0]-lon0)/ddeg; x1=(bounds[2]-lon0)/ddeg
    y0=(top-bounds[3])/ddeg; y1=(top-bounds[1])/ddeg
    xs=np.clip(np.linspace(x0+.5,x1-.5,128).astype(int),0,slope.shape[1]-1)
    ys=np.clip(np.linspace(y0+.5,y1-.5,128).astype(int),0,slope.shape[0]-1)
    return slope[np.ix_(ys,xs)]
RAW={'ap':'Pre_event/S1_ASC_20210805','aq':'Post_event/S1_ASC_20210817',
     'dp':'Pre_event/S1_DESC_20210803','dq':'Post_event/S1_DESC_20210815'}
groups={k:[] for k in ['GT_BothValid','GT_ASC_layover','GT_ASC_shadow','BG_ASC_layover',
                       'BG_ASC_shadow','ASCbad_DESCvalid','ASCvalid_DESCbad','BothValid']}
for sid in range(N):
    p=D/'processed'/f'sample_{sid:06d}'
    gt=rasterio.open(p/'landslide_mask.tif').read(1)>0
    ga=rasterio.open(p/'asc_pre_geom.tif').read(1); gd=rasterio.open(p/'desc_pre_geom.tif').read(1)
    av=ga==0; dv=gd==0
    counts={'GT_BothValid':int((gt&av&dv).sum()),'GT_ASC_layover':int((gt&(ga==1)).sum()),
      'GT_ASC_shadow':int((gt&(ga==2)).sum()),'BG_ASC_layover':int((~gt&(ga==1)).sum()),
      'BG_ASC_shadow':int((~gt&(ga==2)).sum()),'ASCbad_DESCvalid':int(((~av)&dv).sum()),
      'ASCvalid_DESCbad':int((av&(~dv)).sum()),'BothValid':int((av&dv).sum())}
    for k,c in counts.items():
        minimum={'GT_BothValid':20,'GT_ASC_layover':5,'GT_ASC_shadow':1,'BG_ASC_layover':100,
                 'BG_ASC_shadow':5,'ASCbad_DESCvalid':500,'ASCvalid_DESCbad':500,'BothValid':3000}[k]
        if c>=minimum: groups[k].append((sid,c))

rng=np.random.default_rng(20260924); manifest=[]
for name,pool in groups.items():
    if len(pool)<10: print('WARN',name,'pool',len(pool))
    # sample across the eligible pool, fixed seed; sort by id for readable sheets
    take=rng.choice(len(pool),size=min(10,len(pool)),replace=False)
    ids=sorted([pool[i][0] for i in take]); manifest.append({'group':name,'pool_n':len(pool),'sample_ids':ids})
    fig,axs=plt.subplots(len(ids),9,figsize=(18,2.05*len(ids)),squeeze=False)
    for row,sid in enumerate(ids):
        p=D/'processed'/f'sample_{sid:06d}'
        gt=rasterio.open(p/'landslide_mask.tif').read(1)>0
        ga=rasterio.open(p/'asc_pre_geom.tif').read(1); gd=rasterio.open(p/'desc_pre_geom.tif').read(1)
        ims={}
        for key,rel in RAW.items():
            with rasterio.open(D/rel/f'{sid}.tif') as ds:
                ims[key]=ds.read()
        with rasterio.open(D/RAW['ap']/f'{sid}.tif') as ds: sl=patch_slope(ds.bounds)
        panels=[(ims['ap'][1],'ASC pre VV'),(ims['aq'][1],'ASC post VV'),
                (np.abs(ims['aq'][1]-ims['ap'][1]),'ASC |ΔVV|'),
                (ims['dp'][1],'DESC pre VV'),(ims['dq'][1],'DESC post VV'),
                (np.abs(ims['dq'][1]-ims['dp'][1]),'DESC |ΔVV|'),
                (None,'GT + ASC geom'),(None,'GT + DESC geom'),(sl,'GLO-30 slope (deg)')]
        for col,(im,title) in enumerate(panels):
            ax=axs[row,col]
            if im is not None:
                if 'Δ' in title: ax.imshow(im,cmap='magma',vmin=0,vmax=np.percentile(im,99))
                elif 'slope' in title: ax.imshow(im,cmap='terrain',vmin=0,vmax=60)
                else:
                    lo,hi=np.percentile(im,[2,98]); ax.imshow(np.clip((im-lo)/(hi-lo+1e-6),0,1),cmap='gray')
            else:
                g=ga if col==6 else gd
                ax.imshow(np.zeros_like(gt),cmap='gray',vmin=0,vmax=1)
                for cls,clr,ls in [(1,'orange','-'),(2,'deepskyblue',':'),(3,'magenta','--')]:
                    if np.any(g==cls): ax.contour(g==cls,levels=[.5],colors=clr,linewidths=.5,linestyles=ls)
                if gt.any(): ax.contour(gt,levels=[.5],colors='red',linewidths=.65)
            if row==0: ax.set_title(title,fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
            if col==0: ax.set_ylabel(f'id={sid}',rotation=0,labelpad=26,fontsize=8)
    fig.suptitle(f'{name} — deterministic random sample (seed 20260924); red=GT, orange=layover, blue=shadow, magenta=both',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(O/f'visual_{name}.png',dpi=110); plt.close(fig)
pd.DataFrame(manifest).to_json(O/'visual_sample_manifest.json',orient='records',indent=2)
print(pd.DataFrame(manifest).to_string(index=False))
