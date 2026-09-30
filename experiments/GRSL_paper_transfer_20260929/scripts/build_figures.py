"""Native-patch scientific figures from recorded unmasked probabilities."""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import rasterio

ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1]
OUT=PACK/'new_evidence/20260930';FIG=PACK/'manuscript/figures'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import metric_rows
CACHE=ROOT/'experiments/GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
COL=np.array([[0.09,0.09,0.09],[0.12,.73,.28],[.90,.23,.16],[.16,.47,.95]])

def read_sar(sid,orbit):
    raw={'asc':('20210805','20210817','ASC'),'desc':('20210803','20210815','DESC')}[orbit]
    images=[];sources=[]
    for event,date in zip(('Pre_event','Post_event'),raw[:2]):
        path=ROOT/f'data/haiti/{event}/S1_{raw[2]}_{date}/{sid}.tif';sources.append(str(path.relative_to(ROOT)))
        with rasterio.open(path) as d:images.append(d.read([2,1]))
    return np.concatenate(images),sources

def err(p,y):
    hard=p>=.5;land=y>0
    return np.where(hard&land,1,np.where(hard&~land,2,np.where(~hard&land,3,0)))

def sarshow(ax,sar,lo,hi):
    ax.imshow(sar[2],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')

def clean(ax):ax.set_xticks([]);ax.set_yticks([])

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    z=np.load(CACHE);y=z['y'];gt=z['target_geometry'];gc=z['counter_geometry']
    rows=[]
    for name in ('S0','S2_2Stage','Multi2Uni_Port'):
        for r,v in metric_rows((z['p_'+name],y,gt,gc)).items():rows.append({'method':name,'region':r,**v})
    with (OUT/'cached_metrics_recomputed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Retain two already documented scenes and add a distortion-support scene.
    selection=[int(np.where((z['sample_id']==1061)&(z['orbit']=='desc'))[0][0]),
               int(np.where((z['sample_id']==697)&(z['orbit']=='asc'))[0][0])]
    g10=(gt>0)&(gc==0);support=(g10&(y>0)).sum((1,2))
    rid=int(np.argmax(support));selection.append(rid)
    records=[]
    fig,axes=plt.subplots(3,7,figsize=(14,6.7),gridspec_kw={'wspace':.045,'hspace':.19})
    titles=['Target post-event VV','Counter post-event VV','Target geometry','Ground truth','S0 errors','Multi2Uni errors','Ours errors']
    for row,i in enumerate(selection):
        sid=int(z['sample_id'][i]);orbit=str(z['orbit'][i]);other='desc' if orbit=='asc' else 'asc'
        sar,sp=read_sar(sid,orbit);counter,cp=read_sar(sid,other)
        lo,hi=np.percentile(np.concatenate((sar[[0,2]].ravel(),counter[[0,2]].ravel())),[2,98])
        sarshow(axes[row,0],sar,lo,hi);sarshow(axes[row,1],counter,lo,hi)
        geom_cmap=ListedColormap(['#eff4f7','#b377cc','#eeaa44','#995c83'])
        axes[row,2].imshow(gt[i],cmap=geom_cmap,vmin=0,vmax=3,interpolation='nearest')
        axes[row,3].imshow(y[i],cmap=ListedColormap(['#171717','#20ba47']),vmin=0,vmax=1,interpolation='nearest')
        for j,key in enumerate(('p_S0','p_Multi2Uni_Port','p_S2_2Stage'),4):
            axes[row,j].imshow(COL[err(z[key][i],y[i])],interpolation='nearest')
        for j,ax in enumerate(axes[row]):
            clean(ax)
            if row==0:ax.set_title(titles[j],fontsize=9)
            if j==0:ax.set_ylabel(f'{sid:06d} {orbit.upper()}',fontsize=9)
        layer={'target_sar':sar,'counter_sar':counter,**{k:z[k][i] for k in ('y','target_geometry','counter_geometry','p_S0','p_Multi2Uni_Port','p_S2_2Stage')}}
        np.savez_compressed(OUT/f'layers_{sid:06d}_{orbit}.npz',**layer)
        records.append({'sample_id':sid,'orbit':orbit,'crop_xyxy':[0,0,128,128],'sar_display_range':[float(lo),float(hi)],'sources':sp+cp,
                        'selection':'existing documented candidate' if row<2 else 'largest G10-positive pixel support, independent of method performance',
                        'G10_positive_pixels':int(support[i]),'manual_selected':None,'needs_manual_review':True})
    fig.text(.5,.005,'Errors at threshold 0.5: green = TP, red = FP, blue = FN; dark = TN. Predictions remain visible at every pixel.',ha='center',fontsize=10)
    save(fig,'fig_qualitative_current_raw')
    sid=int(z['sample_id'][rid]);orbit=str(z['orbit'][rid]);sar,_=read_sar(sid,orbit);counter,_=read_sar(sid,'desc' if orbit=='asc' else 'asc')
    # Choose 64x64 crop with most positive G10 support; performance is never used to locate it.
    best=max(((int((g10[rid]& (y[rid]>0))[yy:yy+64,xx:xx+64].sum()),xx,yy) for yy in range(0,65,8) for xx in range(0,65,8)))
    _,x0,y0=best;slice2=np.s_[y0:y0+64,x0:x0+64];mask=g10[rid][slice2]
    lo,hi=np.percentile(np.concatenate((sar[[0,2]].ravel(),counter[[0,2]].ravel())),[2,98])
    fig,axs=plt.subplots(2,4,figsize=(10,5.6),gridspec_kw={'wspace':.05,'hspace':.19})
    panels=[sar[2][slice2],counter[2][slice2],gt[rid][slice2],y[rid][slice2],
            COL[err(z['p_S0'][rid],y[rid])][slice2],COL[err(z['p_Multi2Uni_Port'][rid],y[rid])][slice2],
            COL[err(z['p_S2_2Stage'][rid],y[rid])][slice2],mask]
    names=['Target VV','Counter VV (explanation only)','Target geometry (0 / 1 / 2 / 3)','Landslide ground truth','S0: TP / FP / FN','Multi2Uni: TP / FP / FN','Ours: TP / FP / FN','G10 evaluation support']
    for j,(ax,im,title) in enumerate(zip(axs.flat,panels,names)):
        if j<2:ax.imshow(im,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        elif j==2:ax.imshow(im,cmap=geom_cmap,vmin=0,vmax=3,interpolation='nearest')
        elif j in (3,7):ax.imshow(im,cmap=ListedColormap(['#171717','#20ba47' if j==3 else '#f5d442']),vmin=0,vmax=1,interpolation='nearest')
        else:ax.imshow(im,interpolation='nearest')
        if j!=7 and mask.any() and not mask.all():ax.contour(mask.astype(float),levels=[.5],colors=['#ffe45c'],linewidths=.55)
        ax.set_title(title,fontsize=9);clean(ax)
    fig.text(.5,.012,f'{sid:06d} {orbit.upper()}; crop x=[{x0},{x0+64}), y=[{y0},{y0+64}); yellow contour = G10; predictions are unmasked',ha='center',fontsize=10)
    save(fig,'fig_g10_raw_zoom')
    records[-1]['G10_zoom_xyxy']=[x0,y0,x0+64,y0+64]
    (OUT/'figure_sources.json').write_text(json.dumps({'prediction_cache':str(CACHE.relative_to(ROOT)),'sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),'cases':records,'errors':'TP green, FP red, FN blue, TN dark; threshold .5; no geometry graying','display':'shared pre/post VV range for both orbits per scene; raw layers retain VV and VH','pdf':'vector text/contours with native raster data embedded'},indent=2))
    print('FIGURES_COMPLETE',sid,orbit,x0,y0,flush=True)

if __name__=='__main__':main()
