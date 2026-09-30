"""Preserve scenario; single-column results; bibliography after all floats."""
from pathlib import Path
import re,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
PACK=Path(__file__).resolve().parents[1];MS=PACK/'manuscript';OUT=PACK/'new_evidence/20260930'

def tables():
    short={'Boehm SAR U-Net++':'Boehm U-Net++','S0 target-only':'S0','CG0 paired (frozen)':'CG0 paired','CG0 center (intervention)':'CG0 center','CG0 random (intervention)':'CG0 random','CG0 no search (intervention)':'CG0 no search','CG0 counter shuffle (intervention)':'CG0 shuffled','CG0 self context (intervention)':'CG0 self context','SG0 self guided (trained)':'SG0 trained','S2 learned (frozen)':'S2 learned','S2 center (intervention)':'S2 center','S2 random (intervention)':'S2 random','S2 no search (intervention)':'S2 no search','Segmentation only':'Seg. only','Segmentation + P':'Seg. + P','Segmentation + E':'Seg. + E','Segmentation + P + E (current)':'Seg. + P + E'}
    for name in ('primary','search','loss'):
        p=MS/f'sections/current_{name}_table.tex';s=p.read_text()
        caption=re.search(r'\\caption\{(.*?)\}\n\\label',s,re.S).group(1)
        if name=='primary':caption+=' AP denotes AUPRC.'
        label=re.search(r'\\label\{(.*?)\}',s).group(1)
        rows=[]
        for line in s.splitlines():
            if ' & ' not in line or 'multicolumn' in line or 'AUPRC' in line:continue
            cells=line.removesuffix('\\\\').split(' & ')
            if len(cells)!=13:continue
            display=cells[0]
            for a,b in short.items():display=display.replace(a,b)
            rows.append((display,cells[1:]))
        assert rows
        lines=[r'\begin{table}[!t]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{2.1pt}',r'\renewcommand{\arraystretch}{1.03}',r'\begin{tabular}{lrrrrrr}',r'\toprule']
        for group,(a,b) in enumerate((('TargetValid','Overall'),('BothValid',r'$G_{10}$'))):
            if group:lines.append(r'\midrule')
            lines.extend([r'Condition & \multicolumn{3}{c}{'+a+r'} & \multicolumn{3}{c}{'+b+r'}\\',r' & AP & IoU & F1 & AP & IoU & F1\\',r'\midrule'])
            for display,v in rows:lines.append(display+' & '+' & '.join(v[group*6:group*6+6])+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
        p.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def figures():
    from build_figures import COL,err
    records=json.loads((OUT/'figure_sources.json').read_text())['cases']
    plt.rcParams.update({'font.size':7,'pdf.fonttype':42})
    fig,axs=plt.subplots(6,4,figsize=(3.5,5.5),gridspec_kw={'wspace':.06,'hspace':.32})
    for case,rec in enumerate(records):
        z=np.load(OUT/f"layers_{rec['sample_id']:06d}_{rec['orbit']}.npz");lo,hi=rec['sar_display_range']
        images=[z['target_sar'][2],z['counter_sar'][2],z['target_geometry'],z['y'],COL[err(z['p_S0'],z['y'])],COL[err(z['p_Multi2Uni_Port'],z['y'])],COL[err(z['p_S2_2Stage'],z['y'])]]
        names=['Target VV','Counter VV','Geometry','GT','S0','Multi2Uni','Ours']
        for j,(im,title) in enumerate(zip(images,names)):
            ax=axs[case*2+j//4,j%4]
            if j<2:ax.imshow(im,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
            elif j==2:ax.imshow(im,cmap=ListedColormap(['#eff4f7','#b377cc','#eeaa44','#995c83']),vmin=0,vmax=3,interpolation='nearest')
            elif j==3:ax.imshow(im,cmap=ListedColormap(['#171717','#20ba47']),vmin=0,vmax=1,interpolation='nearest')
            else:ax.imshow(im,interpolation='nearest')
            ax.set_title(title,fontsize=6);ax.set_xticks([]);ax.set_yticks([])
        ax=axs[case*2+1,3];ax.axis('off');ax.text(.5,.7,f"{rec['sample_id']:06d}\n{rec['orbit'].upper()}",ha='center',va='center',fontsize=7)
        ax.text(.5,.2,'TP green\nFP red\nFN blue',ha='center',va='center',fontsize=6)
    fig.savefig(MS/'figures/fig_qualitative_single.pdf',bbox_inches='tight');fig.savefig(MS/'figures/fig_qualitative_single.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Native G10 crop, with labels sized for the actual single-column width.
    rec=records[-1];z=np.load(OUT/f"layers_{rec['sample_id']:06d}_{rec['orbit']}.npz");lo,hi=rec['sar_display_range'];x0,y0,x1,y1=rec['G10_zoom_xyxy'];sl=np.s_[y0:y1,x0:x1];mask=((z['target_geometry']>0)&(z['counter_geometry']==0))[sl]
    panels=[z['target_sar'][2][sl],z['counter_sar'][2][sl],z['target_geometry'][sl],z['y'][sl],COL[err(z['p_S0'],z['y'])][sl],COL[err(z['p_Multi2Uni_Port'],z['y'])][sl],COL[err(z['p_S2_2Stage'],z['y'])][sl],mask]
    fig,axs=plt.subplots(2,4,figsize=(3.5,2.0),gridspec_kw={'wspace':.07,'hspace':.28})
    for j,(ax,im,title) in enumerate(zip(axs.flat,panels,['Target VV','Counter VV','Geometry','GT','S0','Multi2Uni','Ours',r'$G_{10}$ mask'])):
        if j<2:ax.imshow(im,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        elif j==2:ax.imshow(im,cmap=ListedColormap(['#eff4f7','#b377cc','#eeaa44','#995c83']),vmin=0,vmax=3,interpolation='nearest')
        elif j in (3,7):ax.imshow(im,cmap=ListedColormap(['#171717','#20ba47' if j==3 else '#f5d442']),vmin=0,vmax=1,interpolation='nearest')
        else:ax.imshow(im,interpolation='nearest')
        if j!=7:ax.contour(mask.astype(float),levels=[.5],colors=['#ffe45c'],linewidths=.4)
        ax.set_title(title,fontsize=6.5);ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(MS/'figures/fig_g10_single.pdf',bbox_inches='tight');fig.savefig(MS/'figures/fig_g10_single.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Same measured policy data, reflowed as one scale per row.
    rows=list(csv.DictReader((OUT/'current_query_samples.csv').open()));meta=json.loads((OUT/'current_query_metadata.json').read_text());case=meta['case'];z=np.load(OUT/f"layers_{case['sample_id']:06d}_{case['orbit']}.npz");qx,qy=meta['query_xy'];lo,hi=case['sar_display_range']
    fig,axs=plt.subplots(3,2,figsize=(3.5,3.4),gridspec_kw={'wspace':.35,'hspace':.5})
    for li in range(3):
        ax,bar=axs[li];ax.imshow(z['target_sar'][2],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');ax.set_xlim(qx-12,qx+12);ax.set_ylim(qy+12,qy-12);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'P{li+2}: T triangle / S circle',fontsize=6)
        mass=[]
        for shift,name,color,mark in [(-.18,'CG0 Teacher','#1970b4','^'),(.18,'S2-2Stage Student','#e07b27','o')]:
            rs=[r for r in rows if r['model']==name and r['level']==f'P{li+2}'];values=[float(r['weight']) for r in rs];mass.append(sum(values))
            for r in rs:ax.scatter(float(r['x']),float(r['y']),s=20+70*float(r['weight']),facecolors='none',edgecolors=color,marker=mark,linewidths=.8)
            bar.bar(np.arange(4)+shift,values,.36,color=color)
        ax.scatter(qx,qy,marker='+',s=20,color='#d32b27');bar.set_xticks(np.arange(4),['fixed','1','2','3']);bar.set_ylim(0,.45);bar.tick_params(labelsize=6);bar.set_title(f'Mass T={mass[0]:.3f}, S={mass[1]:.3f}',fontsize=6);bar.set_ylabel('Weight',fontsize=6)
    fig.savefig(MS/'figures/fig_search_single.pdf',bbox_inches='tight');fig.savefig(MS/'figures/fig_search_single.png',dpi=300,bbox_inches='tight');plt.close(fig)

def main():
    tables();figures()
    p=MS/'sections/method.tex';s=p.read_text()
    s=re.sub(r'\\begin\{figure\}\[[^\]]*\].*?\\label\{fig:scenario\}.*?\\end\{figure\}\s*','',s,flags=re.S)
    scenario=r'''\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_task_scenario.png}
\caption{Task scenario: both orbits observe the slope before the event; rapid response may have only the target post-event orbit. Paired-orbit training supports target-only deployment.}
\label{fig:scenario}
\end{figure}

'''
    s=s.replace(r'\begin{figure*}[!t]',scenario+r'\begin{figure*}[!t]',1)
    p.write_text(s,encoding='utf-8')
    p=MS/'sections/experiments.tex';s=p.read_text().replace(r'\begin{figure*}[!t]',r'\begin{figure}[!t]').replace(r'\end{figure*}',r'\end{figure}');s=re.sub(r'width=(?:\.\d+)?\\textwidth',lambda m:r'width=\columnwidth',s)
    s=s.replace('fig_qualitative_current_raw.pdf','fig_qualitative_single.pdf').replace('fig_search_current_compact.pdf','fig_search_single.pdf').replace('fig_g10_raw_zoom.pdf','fig_g10_single.pdf').replace('Bottom: joint candidate weights and scale mass.','Right: joint candidate weights and scale mass.');p.write_text(s,encoding='utf-8')
    p=MS/'sections/introduction.tex';s=p.read_text().replace('We therefore use paired orbits during training and deploy with only the available target orbit.','We therefore use paired orbits during training and deploy with only the available target orbit (Fig.~\\ref{fig:scenario}).');p.write_text(s,encoding='utf-8')
    p=MS/'main.tex';s=p.read_text();s=s.replace('{\\scriptsize\n\\bibliographystyle', '\\FloatBarrier\n\n{\\scriptsize\n\\bibliographystyle') if '\\FloatBarrier' not in s else s
    if r'\usepackage{balance}' not in s:s=s.replace(r'\usepackage{placeins}',r'\usepackage{placeins}'+'\n'+r'\usepackage{balance}')
    if r'\balance' not in s:s=s.replace(r'\FloatBarrier',r'\FloatBarrier'+'\n'+r'\balance')
    p.write_text(s,encoding='utf-8')
    print('USER_FORMAT: scenario retained; only architecture spans columns; references after floats')
if __name__=='__main__':main()
