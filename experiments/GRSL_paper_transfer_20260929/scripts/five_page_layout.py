"""Five-page GRSL layout: complete current tables, compact real search figure."""
from pathlib import Path
import json,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930';MS=PACK/'manuscript'

def main():
    with (OUT/'current_query_samples.csv').open() as f:rows=list(csv.DictReader(f))
    meta=json.loads((OUT/'current_query_metadata.json').read_text());sid=meta['case']['sample_id'];orbit=meta['case']['orbit']
    layer=np.load(OUT/f'layers_{sid:06d}_{orbit}.npz');sar=layer['target_sar'];lo,hi=meta['case']['sar_display_range'];qx,qy=meta['query_xy']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(9.6,3.5),gridspec_kw={'height_ratios':[1.6,1],'hspace':.45,'wspace':.22})
    for li in range(3):
        ax=axs[0,li];ax.imshow(sar[2],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        ax.set_xlim(qx-12,qx+12);ax.set_ylim(qy+12,qy-12);ax.set_xticks([]);ax.set_yticks([])
        for name,color,mark in [('CG0 Teacher','#1970b4','^'),('S2-2Stage Student','#e07b27','o')]:
            subset=[r for r in rows if r['model']==name and r['level']==f'P{li+2}']
            for r in subset:ax.scatter([float(r['x'])],[float(r['y'])],s=45+160*float(r['weight']),facecolors='none',edgecolors=color,marker=mark,linewidths=1.2)
        ax.scatter([qx],[qy],marker='+',s=35,color='#d32b27',linewidths=1.1)
        ax.set_title(f'P{li+2}: Teacher (triangle) / Student (circle)',fontsize=9)
        bar=axs[1,li];xx=np.arange(4)
        masses=[]
        for shift,name,color in [(-.18,'CG0 Teacher','#1970b4'),(.18,'S2-2Stage Student','#e07b27')]:
            vals=[float(r['weight']) for r in rows if r['model']==name and r['level']==f'P{li+2}'];masses.append(sum(vals));bar.bar(xx+shift,vals,.36,color=color)
        bar.set_xticks(xx,['fixed','1','2','3']);bar.set_ylim(0,.45);bar.tick_params(labelsize=8)
        bar.set_title(f'Scale mass: T={masses[0]:.3f}, S={masses[1]:.3f}',fontsize=9)
        if li==0:bar.set_ylabel('Joint weight',fontsize=9)
    fig.savefig(MS/'figures/fig_search_current_compact.pdf',bbox_inches='tight');fig.savefig(MS/'figures/fig_search_current_compact.png',dpi=300,bbox_inches='tight');plt.close(fig)
    # Formal results narrative and full four-region comparison.
    from update_manuscript import table,read
    current=read(PACK/'evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv')
    names=[('S0 target-only','S0-Raw'),('Boehm SAR U-Net++ \\cite{Boehm2022}','Boehm SAR U-Net++'),('Hinton-KD \\cite{Hinton2015}','Hinton-KD-Port'),('Multi2Uni','Multi2Uni-Port')]
    mainrows=[]
    for display,key in names+[('Ours','S2-2Stage')]:
        row=next(r for r in current if r['Method']==key)
        metrics={region:{m:float(row[region+'_'+m]) for m in ('AUPRC','IoU','F1')} for region in ('TargetValid','Overall','BothValid','G10')}
        mainrows.append((display,metrics))
    mainrows.insert(4,('DIS2 (archived) \\cite{Kieu2026DIS2}',{'Overall':{'AUPRC':.8733278861052558,'IoU':.6176884604656225,'F1':.7636680010378782}}))
    table('Primary comparison (\\%). TargetValid is primary; all current rows use the same predictions for every region. Archived DIS2 has only verified Overall values; missing regional scores remain blank.',
          'tab:primary',mainrows,MS/'sections/current_primary_table.tex')
    exp=MS/'sections/experiments.tex';s=exp.read_text();start=s.index(r'\subsection{Qualitative Results}')
    prefix=r'''\section{Experiments}
\label{sec:experiments}
\subsection{Data and Regional Evaluation}
We use 1,713 paired Sentinel-1 locations from the Haiti landslide-change dataset \cite{Bralet2024DataPort}. Each orbit provides pre/post VV and VH plus an orbit identifier in a $128\times128$ patch. The Student uses ConvNeXt-Tiny/FPN, Adam with learning rate $5\times10^{-5}$, and batch size 16. Training includes all labelled pixels without geometry input or geometry weighting. One inference output is scored on four masks: TargetValid ($g_t=0$), Overall (all pixels), BothValid ($g_t=g_c=0$), and $G_{10}$ ($g_t>0,g_c=0$), where zero is valid geometry. AUPRC uses scores; IoU/F1 use threshold 0.5.

\subsection{Mapping Comparison}
Table~\ref{tab:primary} compares five representative baselines with the current Student. Ours improves TargetValid AUPRC/IoU over S0 and Multi2Uni and retains gains in Overall, BothValid, and $G_{10}$. External rows are target-input adaptations. Archived DIS2 retains its recorded Overall 87.33\% AUPRC and 61.77\% IoU; original probabilities and mask/sample lineage are unavailable, so its regional cells remain blank. Its historical TGD scores are not merged with current $G_{10}$.
\input{sections/current_primary_table}

'''
    s=prefix+s[start:]
    s=re.sub(r'\n\\begin\{figure\}\[t\].*?\\label\{fig:search_alignment\}.*?\\end\{figure\}\n','\n',s,flags=re.S)
    s=re.sub(r'\n\\begin\{figure\}\[!b\].*?\\label\{fig:kd_sensitivity\}.*?\\end\{figure\}\n','\n',s,flags=re.S)
    s=s.replace('figures/fig_search_current_samples.pdf','figures/fig_search_current_compact.pdf').replace(r'width=.74\textwidth',r'width=.90\textwidth')
    s=re.sub(r'Fig\.~\\ref\{fig:qualitative\} shows three native.*?\n\n',
             lambda _: r'Fig.~\ref{fig:qualitative} provides native-patch examples; Fig.~\ref{fig:g10_zoom} retains predictions on distorted pixels. Scene/crop selection uses annotation support, independently of comparative performance.'+'\n\n',s,flags=re.S)
    s=re.sub(r'\\caption\{Native-patch comparison.*?\}\n\\label\{fig:qualitative\}',
        r'\\caption{Scenes 001061 DESC, 000697 ASC, and 000925 ASC: target/counter VV, target geometry, annotation, and unmasked predictions. Green: TP; red: FP; blue: FN; dark: TN. Both orbits share one pre/post VV display range per scene; counter SAR is explanatory only.}'+ '\n'+r'\\label{fig:qualitative}',s,flags=re.S)
    s=re.sub(r'\\caption\{Unmasked \$G_\{10\}\$ detail.*?\}\n\\label\{fig:g10_zoom\}',
        r'\\caption{Unmasked $G_{10}$ detail, 000925 ASC, crop $x\\in[24,88)$, $y\\in[8,72)$. Yellow contours and the mask identify target-distorted/counter-valid pixels. The same TP/FP/FN colors apply; no prediction is replaced by gray.}'+ '\n'+r'\\label{fig:g10_zoom}',s,flags=re.S)
    s=re.sub(r'\\caption\{Actual CG0 and S2-2Stage samples.*?\}\n\\label\{fig:search_samples\}',
        r'\\caption{Current query $(57.5,49.5)$, 000925 ASC. Teacher triangles and Student circles show all four samples at each FPN level without jitter; red crosses mark the reference. Bottom: joint candidate weights and scale mass. The three learned points coincide in both checkpoints; parameter inspection confirms identical offset heads. Effective support is one learned location plus the fixed center per level.}'+ '\n'+r'\\label{fig:search_samples}',s,flags=re.S)
    a=s.index(r'Table~\ref{tab:search_intervention} evaluates');b=s.index(r'\input{sections/current_search_table}',a)
    s=s[:a]+r'''Table~\ref{tab:search_intervention} uses frozen current checkpoints. Center sets offsets to zero; random draws bounded offsets while retaining the anchored center and learned weights; no-search feeds projected P2 to the trained decoder. Counter shuffle globally deranges sample locations. Frozen CG0 self-context differs from independently trained SG0. Counter mismatch and random sampling reduce accuracy, but Student center sampling is close to learned sampling. These are inference sensitivities, not proof that a retrained fixed sampler would be inferior.

'''+s[b:]
    exp.write_text(s,encoding='utf-8')
    # Remove redundant plot captions and version-history discussion from formal text.
    method=MS/'sections/method.tex';s=method.read_text();s=s.replace('Detailed Teacher--Student search architecture from the original manuscript. The counter orbit guides the Teacher search policy, while retrieved values remain target-side. TGD is marked optional in the editable source and disabled in CG0 and the principal Student. Both distillation retrievals use the same Student target memory; the Teacher contributes only its policy. The two Teacher encoder drawings use shared weights.',
        'Complete Teacher--Student paths. Counter context guides policy; values remain target-side. TGD is off in CG0. Both distillation retrievals use Student target memory. The Teacher encoder drawings share weights.')
    method.write_text(s,encoding='utf-8')
    # Current comprehensive rows remain; long implementation definitions live in saved protocols.
    p=MS/'sections/current_search_table.tex';s=p.read_text();s=re.sub(r'\\caption\{.*?\}\n\\label',
        r'\\caption{Current search controls (\\%). All changes except trained SG0 are frozen-weight inference interventions. Center retains weights; random retains the center candidate; no-search bypasses aggregation with projected P2. Full probabilities and intervention definitions are saved.}'+ '\n'+r'\\label',s,flags=re.S);p.write_text(s,encoding='utf-8')
    p=MS/'sections/current_loss_table.tex';s=p.read_text();s=re.sub(r'\\caption\{.*?\}\n\\label',
        r'\\caption{Current two-stage loss controls (\\%). P: spatial-policy JS; E: shared-memory cosine. Each surviving loss retains its canonical coefficient. Combined: frozen principal model; dashes: unfinished retraining.}'+ '\n'+r'\\label',s,flags=re.S);p.write_text(s,encoding='utf-8')
    print('FIVE_PAGE_LAYOUT_PREPARED; historical/standalone alignment plots retained in figures and evidence, not duplicated in main')

if __name__=='__main__':main()
