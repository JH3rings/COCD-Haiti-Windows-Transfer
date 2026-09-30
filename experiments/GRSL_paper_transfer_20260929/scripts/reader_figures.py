"""Publication comparisons: geometry-excluded and full-pixel predictions."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score
from build_figures import COL,err
PACK=Path(__file__).resolve().parents[1];ROOT=PACK.parents[1];OUT=PACK/'new_evidence/20260930';FIG=PACK/'manuscript/figures'

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def display(ax,im,title,kind='errors',rng=None,exclude=None):
    if kind=='sar':ax.imshow(im,cmap='gray',vmin=rng[0],vmax=rng[1],interpolation='nearest')
    elif kind=='distortion':ax.imshow(im>0,cmap=ListedColormap(['#edf1f4','#b377cc']),vmin=0,vmax=1,interpolation='nearest')
    elif kind=='reference':ax.imshow(im,cmap=ListedColormap(['#171717','#20ba47']),vmin=0,vmax=1,interpolation='nearest')
    elif kind=='support':ax.imshow(im,cmap=ListedColormap(['#d0d0d0','#ffffff']),vmin=0,vmax=1,interpolation='nearest')
    else:
        rgb=im.copy()
        if exclude is not None:rgb[exclude]=[.78,.78,.78]
        ax.imshow(rgb,interpolation='nearest')
    ax.set_title(title,fontsize=6);ax.set_xticks([]);ax.set_yticks([])

def main():
    cache_path=ROOT/'experiments/GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz';z=np.load(cache_path)
    ids=z['sample_id'];orbits=z['orbit'];y=z['y'];gt=z['target_geometry'];gc=z['counter_geometry']
    h=np.load(OUT/'Hinton_current_predictions.npz');index={(int(s),str(o)):i for i,(s,o) in enumerate(zip(h['sample_id'],h['orbit']))}
    join=np.array([index[(int(s),str(o))] for s,o in zip(ids,orbits)])
    assert np.array_equal(y,h['y'][join]);assert np.array_equal(gt,h['target_geometry'][join]);assert np.array_equal(gc,h['counter_geometry'][join])
    p_online=z['p_MMOKD_Port'];recorded=list(csv.DictReader((PACK/'evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv').open()));row=next(r for r in recorded if r['Method']=='MMOKD-Port')
    ap=average_precision_score(y.ravel(),p_online.ravel());valid=gt==0;vap=average_precision_score(y[valid],p_online[valid]);verification=json.loads((OUT/'online_prediction_verification.json').read_text());assert verification['cached_probability_max_abs_difference']==0;assert verification['threshold_prediction_disagreements']==0
    assert f'{100*ap:.2f}'==f"{100*float(row['Overall_AUPRC']):.2f}";assert f'{100*vap:.2f}'==f"{100*float(row['TargetValid_AUPRC']):.2f}"
    probs={'Baseline':z['p_S0'],'Logit\ndistillation':h['p'][join],'Online\ndistillation':p_online,'Multi-to-Uni':z['p_Multi2Uni_Port'],'Ours':z['p_S2_2Stage']}
    source=json.loads((OUT/'figure_sources.json').read_text());records=source['cases'];selected=records[:2]
    plt.rcParams.update({'font.size':6.5,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    fig,axs=plt.subplots(6,4,figsize=(3.5,5.6),layout='constrained',gridspec_kw={'wspace':.02,'hspace':.02})
    metadata=[]
    for ci,rec in enumerate(selected):
        sid=rec['sample_id'];orbit=rec['orbit'];i=int(np.where((ids==sid)&(orbits==orbit))[0][0]);layer=np.load(OUT/f'layers_{sid:06d}_{orbit}.npz');rng=rec['sar_display_range'];ascending=layer['target_sar'] if orbit=='asc' else layer['counter_sar'];descending=layer['target_sar'] if orbit=='desc' else layer['counter_sar'];ga=gt[i] if orbit=='asc' else gc[i];gd=gt[i] if orbit=='desc' else gc[i]
        panels=[(ascending[2],'Ascending\nbackscatter','sar'),(descending[2],'Descending\nbackscatter','sar'),(ga,'Ascending\ndistortion','distortion'),(gd,'Descending\ndistortion','distortion'),(y[i],'Landslide\nreference','reference'),(gt[i]==0,'Valid target\npixels','support')]
        panels += [(COL[err(p[i],y[i])],name,'errors') for name,p in probs.items()]
        for j,(im,title,kind) in enumerate(panels):display(axs[ci*3+j//4,j%4],im,title,kind,rng,gt[i]>0 if kind=='errors' else None)
        ax=axs[ci*3+2,3];ax.axis('off');ax.text(.5,.5,f"({'ab'[ci]})\n{'Ascending' if orbit=='asc' else 'Descending'}\ninput",ha='center',va='center',fontsize=6.5)
        metadata.append({'panel':'masked-'+str(ci+1),'sample_id':sid,'orbit':orbit,'xyxy':[0,0,128,128],'excluded_pixels':'target_geometry > 0','methods':list(probs),'selection':rec['selection'],'needs_manual_review':True})
    save(fig,'fig_comparison_valid_reader')
    rec=records[-1];sid=rec['sample_id'];orbit=rec['orbit'];i=int(np.where((ids==sid)&(orbits==orbit))[0][0]);layer=np.load(OUT/f'layers_{sid:06d}_{orbit}.npz');x0,y0,x1,y1=rec['G10_zoom_xyxy'];sl=np.s_[y0:y1,x0:x1];rng=rec['sar_display_range'];ascending=layer['target_sar'] if orbit=='asc' else layer['counter_sar'];descending=layer['target_sar'] if orbit=='desc' else layer['counter_sar']
    component={};canonical_i=index[(sid,orbit)]
    for arm,name in [('Seg','Segmentation\nonly'),('SegP','Policy\ndistillation'),('SegE','Evidence\ndistillation')]:
        c=np.load(OUT/'loss_retraining'/arm/'best_predictions.npz');assert np.array_equal(c['y'],h['y']);assert np.array_equal(c['target_geometry'],h['target_geometry']);component[name]=c['p'][canonical_i]
    ga=gt[i] if orbit=='asc' else gc[i];gd=gt[i] if orbit=='desc' else gc[i]
    panels=[(ascending[2][sl],'Ascending\nbackscatter','sar'),(descending[2][sl],'Descending\nbackscatter','sar'),(ga[sl],'Ascending\ndistortion','distortion'),(gd[sl],'Descending\ndistortion','distortion'),(y[i][sl],'Landslide\nreference','reference')]
    panels += [(COL[err(p[i],y[i])][sl],name,'errors') for name,p in probs.items()]
    panels += [(COL[err(p,y[i])][sl],name,'errors') for name,p in component.items()]
    panels.append((COL[err(probs['Ours'][i],y[i])][sl],'Policy +\nevidence','errors'))
    fig,axs=plt.subplots(3,5,figsize=(3.5,2.75),layout='constrained',gridspec_kw={'wspace':.02,'hspace':.02})
    for ax,(im,title,kind) in zip(axs.flat,panels):display(ax,im,title,kind,rng)
    axs[-1,-1].axis('off');axs[-1,-1].text(.5,.5,'Correct: green\nFalse alarm: red\nMissed: blue',ha='center',va='center',fontsize=5)
    save(fig,'fig_comparison_full_reader')
    metadata.append({'panel':'full-pixel comparison','sample_id':sid,'orbit':orbit,'xyxy':[x0,y0,x1,y1],'excluded_pixels':None,'methods':['Baseline','Logit distillation','Online multimodal distillation','Multi-to-Uni','Segmentation control','Policy component','Evidence component','Both components (ours)'],'selection':'annotated distortion-support crop, independent of comparative predictions','needs_manual_review':True})
    # Teacher/Student allocation from recorded query; no display jitter.
    samples=list(csv.DictReader((OUT/'current_query_samples.csv').open()));meta=json.loads((OUT/'current_query_metadata.json').read_text());case=meta['case'];layer=np.load(OUT/f"layers_{case['sample_id']:06d}_{case['orbit']}.npz");qx,qy=meta['query_xy'];lo,hi=case['sar_display_range']
    fig,axs=plt.subplots(3,2,figsize=(3.5,2.8),gridspec_kw={'wspace':.35,'hspace':.5})
    for li,scale in enumerate(['Fine scale','Middle scale','Coarse scale']):
        ax,bar=axs[li];ax.imshow(layer['target_sar'][2],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');ax.set_xlim(qx-12,qx+12);ax.set_ylim(qy+12,qy-12);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(scale,fontsize=6.5)
        for shift,name,color,mark in [(-.18,'CG0 Teacher','#1970b4','^'),(.18,'S2-2Stage Student','#e07b27','o')]:
            rs=[r for r in samples if r['model']==name and r['level']==f'P{li+2}'];vals=[float(r['weight']) for r in rs]
            for rr in rs:ax.scatter(float(rr['x']),float(rr['y']),s=18+70*float(rr['weight']),facecolors='none',edgecolors=color,marker=mark,linewidths=.8)
            bar.bar(np.arange(4)+shift,vals,.36,color=color)
        ax.scatter(qx,qy,marker='+',s=20,color='#d32b27');bar.set_xticks(np.arange(4),['Center','1','2','3']);bar.set_ylim(0,.45);bar.tick_params(labelsize=5.5);bar.set_ylabel('Weight',fontsize=6)
        if li==0:bar.set_title('Teacher / Student',fontsize=6.5)
    save(fig,'fig_search_reader')
    (OUT/'reader_figure_sources.json').write_text(json.dumps({'cache':str(cache_path),'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),'Hinton_index_join_verified':True,'online_distillation_metrics_verified':{'Overall_AUPRC':float(ap),'TargetValid_AUPRC':float(vap)},'error_colors':'true positive green; false positive red; false negative blue; true negative dark; excluded gray only in masked figure','case_metadata':metadata,'query_metadata_source':'current_query_metadata.json','scientific_manual_review':'pending'},indent=2))
    print('READER_COMPARISONS_COMPLETE: 5 methods, masked and full-pixel, matched components')
if __name__=='__main__':main()
