"""Populate only completed recorded results; historical rows stay segregated."""
from pathlib import Path
import csv,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];MS=PACK/'manuscript';OUT=PACK/'new_evidence/20260930'
REGIONS=['TargetValid','Overall','BothValid','G10'];METRICS=['AUPRC','IoU','F1']
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(caption,label,rows,path):
    lines=[r'\begin{table*}[!t]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+('r'*12)+'}',r'\toprule',
           'Condition & '+ ' & '.join(r'\multicolumn{3}{c}{'+r+'}' for r in REGIONS)+r'\\',
           ' & '+' & '.join(METRICS*4)+r'\\',r'\midrule']
    for name,metrics in rows:
        vals=[f'{100*float(metrics[r][k]):.2f}' if r in metrics and metrics[r].get(k) is not None else '--' for r in REGIONS for k in METRICS]
        lines.append(name+' & '+' & '.join(vals)+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    backup=OUT/'original_source';backup.mkdir(exist_ok=True)
    for src in [MS/'main.tex',*sorted((MS/'sections').glob('*.tex'))]:
        dest=backup/src.name
        if not dest.exists():shutil.copy2(src,dest)
    inter=read(OUT/'current_search_interventions.csv');sg=read(OUT/'current_sg0_metrics.csv')
    d={}
    for row in inter+sg:d.setdefault(row['arm'],{})[row['region']]=row
    searchrows=[('CG0 paired (frozen)',d['CG0:paired']),('CG0 center (intervention)',d['CG0:center']),
                ('CG0 random (intervention)',d['CG0:random']),('CG0 no search (intervention)',d['CG0:no-search']),
                ('CG0 counter shuffle (intervention)',d['CG0:counter-shuffle']),('CG0 self context (intervention)',d['CG0:self-context']),
                ('SG0 self guided (trained)',d['SG0:retrained-self-guidance']),('S2 learned (frozen)',d['S2-2Stage:learned']),
                ('S2 center (intervention)',d['S2-2Stage:center']),('S2 random (intervention)',d['S2-2Stage:random']),('S2 no search (intervention)',d['S2-2Stage:no-search'])]
    table('Current geometry-free search controls (\%). AUPRC uses scores; IoU/F1 use threshold 0.5. Center, random, no-search, shuffle, and CG0 self-context are inference interventions on frozen weights. SG0 is an independently trained self-guided Teacher. No-search feeds projected P2 directly to the trained decoder. Teachers use paired context only where specified; deployed Students are target-only.',
          'tab:search_intervention',searchrows,MS/'sections/current_search_table.tex')
    lossrows=[];flat=[];complete=[]
    for arm,name in [('Seg','Segmentation only'),('SegP','Segmentation + P'),('SegE','Segmentation + E')]:
        p=OUT/'loss_retraining'/arm/'summary.json'
        metrics={}
        if p.exists():
            summary=json.loads(p.read_text());metrics=summary['metrics'];complete.append(arm)
        lossrows.append((name,metrics))
        for region in REGIONS:
            flat.append({'arm':arm,'region':region,'status':'COMPLETE' if metrics else 'PENDING','AUPRC':metrics.get(region,{}).get('AUPRC'),'IoU':metrics.get(region,{}).get('IoU'),'F1':metrics.get(region,{}).get('F1')})
    lossrows.append(('Segmentation + P + E (current)',d['S2-2Stage:learned']))
    for r in REGIONS:flat.append({'arm':'SegPE','region':r,'status':'CURRENT_FROZEN_MAIN','AUPRC':d['S2-2Stage:learned'][r]['AUPRC'],'IoU':d['S2-2Stage:learned'][r]['IoU'],'F1':d['S2-2Stage:learned'][r]['F1']})
    table('Current two-stage loss composition (\%). P: spatial-policy JS; E: shared-memory cosine loss. New single-component arms retain their canonical coefficient rather than reallocating the removed component weight. The combined row is the frozen principal model. Dashes denote unfinished retraining, never interim results.',
          'tab:loss_parts',lossrows,MS/'sections/current_loss_table.tex')
    with (OUT/'current_loss_composition.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Redraw only the recorded historical sweep; no current weight sweep exists.
    old=read(PACK/'evidence/earlier_package/quantitative/csv/table_loss_strength.csv')
    xx=[float(r['KD_over_Seg'].strip('%')) for r in old]
    fig,axs=plt.subplots(1,2,figsize=(6,2.25));plt.rcParams['pdf.fonttype']=42
    for ax,key,title in zip(axs,['Overall_AUPRC','OmegaTR_F1'],['Historical Overall AUPRC','Historical OmegaTR F1']):
        ax.plot(xx,[100*float(r[key]) for r in old],'o-',color='#3673ac',linewidth=1.2);ax.set_title(title,fontsize=9);ax.set_xlabel('Initial KD / segmentation (%)',fontsize=8);ax.set_ylabel('%',fontsize=8);ax.set_xticks(xx);ax.tick_params(labelsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(MS/'figures/fig_kd_sensitivity_historical.pdf');fig.savefig(MS/'figures/fig_kd_sensitivity_historical.png',dpi=300);plt.close(fig)
    p=MS/'sections/method.tex';s=p.read_text()
    s=s.replace('figures/fig_architecture_grsl.png','figures/fig_architecture_current.pdf').replace('figures/fig_architecture_optional_tgd.pdf','figures/fig_architecture_current.pdf')
    s=s.replace('The auxiliary TGD branch shown in the diagram is optional and is not used for the principal geometry-free Student comparison.',
        'TGD is marked optional in the editable source and disabled in CG0 and the principal Student. Both distillation retrievals use the same Student target memory; the Teacher contributes only its policy. The two Teacher encoder drawings use shared weights.')
    s=s.replace('At each reference location $p$ and scale, the Teacher policy head predicts $K$ offsets and normalized weights from both inputs,',
        'At each reference location $p$, the Teacher predicts a joint policy over three FPN levels (P2--P4), with four candidates per level: one fixed center and three learned points. Weights are normalized across all twelve candidates. Offsets in feature-pixel units are bounded by $2\\tanh(\\cdot)$. Suppressing the level index for clarity,')
    s=s.replace('The weights are calibrated from training losses and decay during the second stage.',
        'Five initial epochs optimize segmentation alone. The fifth epoch also calibrates the distillation magnitudes without applying their gradients; the calibrated components are added from epoch six with a cosine decay. Teacher outputs, Student target memory, and backbone features used to recompute the distilled Student policy are detached. Consequently, the distillation terms update the context/policy branch, while segmentation trains the complete Student. This stop-gradient construction does not directly distill the backbone or value projection.')
    p.write_text(s,encoding='utf-8')
    # Rebuild experimental narrative around verified current evidence.
    original=(backup/'experiments.tex').read_text()
    cut=original.index(r'\subsection{Qualitative Results and Counter Guidance}')
    prefix=original[:cut]
    prefix=prefix.replace('but its exact region definition is not recorded as $G_{10}$ in the current package; we therefore keep it outside this table until the region masks can be matched.',
        'and the retained re-test code defines TGD algebraically as $g_t>0,g_c=0$. However, the original probability arrays and their exact geometry/sample lineage are missing, and the recovered re-test gives different scores. We therefore retain this historical record separately rather than treating it as a current matched $G_{10}$ result.')
    narrative=r'''
\subsection{Qualitative Results}
Fig.~\ref{fig:qualitative} shows three native $128\times128$ scenes with raw, unmasked predictions. The first two are documented candidates; the third is selected for the largest number of annotated $G_{10}$ pixels, independently of model performance. Green, red, and blue denote TP, FP, and FN at threshold 0.5. The complementary SAR is explanatory context and is never a Student inference input. Pre/post VV values from both orbits share one display range within each scene.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig_qualitative_current_raw.pdf}
\caption{Native-patch comparison for scenes 001061 DESC, 000697 ASC, and 000925 ASC. Target and complementary post-event VV, target geometry, annotation, S0, Multi2Uni, and Ours are shown. Predictions remain visible on geometrically distorted pixels. Green: TP; red: FP; blue: FN; dark: TN. Geometry codes 0/1/2/3 have distinct colors. These illustrative cases do not replace aggregate regional evaluation.}
\label{fig:qualitative}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=.84\textwidth]{figures/fig_g10_raw_zoom.pdf}
\caption{Unmasked $G_{10}$ detail from scene 000925 ASC, crop $x\in[24,88)$, $y\in[8,72)$. Columns show target VV, counter VV (explanatory only), target geometry, and annotation, followed by S0, Multi2Uni, Ours, and the evaluation mask. Yellow contours identify target-distorted/counter-valid pixels. TP/FP/FN use the same colors as Fig.~\ref{fig:qualitative}; no invalid pixel is replaced by gray. The crop maximizes annotated $G_{10}$ support within this scene, independently of predictions.}
\label{fig:g10_zoom}
\end{figure*}

\subsection{Current Search Controls}
Table~\ref{tab:search_intervention} evaluates the current CG0 Teacher and S2-2Stage Student on the same four region definitions. Center intervention sets every offset to zero and retains learned weights. Random intervention draws bounded feature offsets uniformly in $[-2,2]$, retains the fixed-center candidate, and retains learned weights. No-search bypasses aggregation with projected target P2; it is an inference intervention, not a retrained architecture. Counter shuffle uses a global cyclic derangement of locations, preventing any correctly paired counter image. Replacing counter context by the target in frozen CG0 is distinct from the independently trained SG0 self-guided Teacher.

CG0 is sensitive to counter mismatch and random sampling. For the Student, fixed-center sampling remains close to learned sampling, whereas random offsets reduce accuracy. These interventions demonstrate sensitivity of the trained computation to incorrect guidance and sampling, but do not establish that nonzero offsets are necessary or that a retrained fixed sampler would perform worse. Historical CG10 intervention rows remain in the evidence package and are not relabeled as current results.

\input{sections/current_search_table}

\subsection{Policy and Shared-Memory Evidence}
We recompute the alignment diagnostics using CG0, correcting the earlier plot whose script loaded CG10. Over fully TargetValid query blocks, candidate-weight JS is 0.0219 for Student versus CG0 and 0.0782 for shuffled versus CG0. Shared-memory cosine distance to the Student retrieval is 0.0066 for paired CG0 and 0.0595 for shuffled context. These statistics weight every eligible query equally. Candidate-weight JS is a diagnostic over twelve weights, distinct from the spatially splatted policy distribution used in the training loss.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_search_alignment_current.pdf}
\caption{Current CG0/S2-2Stage alignment on fully TargetValid query blocks. Both evidence distances compare policy retrievals from the same Student target memory. Counter shuffle is a global derangement; lower distance means closer agreement, not better segmentation by itself.}
\label{fig:search_alignment}
\end{figure}

\begin{figure*}[!t]
\centering
\includegraphics[width=.82\textwidth]{figures/fig_search_current_samples.pdf}
\caption{Actual CG0 and S2-2Stage samples for a $G_{10}$ query at $(57.5,49.5)$ in scene 000925 ASC. Top: Teacher; middle: Student; columns: P2, P3, and P4. All four candidates per level are drawn without jitter, so overlapping candidates remain overlapping. Red crosses mark the reference location. The largest displacement is 4.50 image pixels for this query. Bottom: jointly normalized candidate weights and per-scale mass. The example emphasizes scale/weight allocation rather than a large spatial displacement.}
\label{fig:search_samples}
\end{figure*}

\input{sections/current_loss_table}
The matched loss-component controls follow the current segmentation warm-up, CG0 guidance, stop-gradient construction, and two-stage schedule. Disabling one component keeps the surviving component's canonical coefficient; it does not transfer the removed component's weight to the survivor. Completed results are reported in Table~\ref{tab:loss_parts}; unfinished arms remain blank. The earlier loss decomposition used another Student configuration and is retained only as historical evidence.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_kd_sensitivity_historical.pdf}
\caption{Historical Student weight sweep redrawn from the recorded per-setting CSV: Overall AUPRC and the historical $\Omega_{TR}$ F1 diagnostic. This is not a current S2-2Stage weight scan. Lines connect completed settings for readability; no setting is promoted as the method's central conclusion.}
\label{fig:kd_sensitivity}
\end{figure}
\FloatBarrier
'''
    (MS/'sections/experiments.tex').write_text(prefix+narrative,encoding='utf-8')
    p=MS/'sections/conclusion.tex';p.write_text(r'''\section{Conclusion}
\label{sec:conclusion}
Counter-guided target evidence search uses paired orbits during training while retaining target-only Student deployment. Geometry-free training and post-hoc evaluation separate mapping gains from pixel exclusion. Current inference controls support sensitivity to correct complementary guidance and reject arbitrary random sampling. However, center sampling remains competitive, and overlapping samples favor an interpretation involving scale/weight allocation rather than large spatial displacement. The principal Student improves regional mapping, while matched component retraining is needed to determine how much each distillation objective contributes.
''',encoding='utf-8')
    mainp=MS/'main.tex';s=mainp.read_text();s=s.replace('These results indicate that counter-guided target-side search improves single-orbit mapping across both valid and distorted evaluation regions.',
        'These results show improved single-orbit mapping across valid and distorted evaluation regions. Current inference interventions show sensitivity to counter pairing and random sampling, while fixed-center sampling remains competitive.')
    mainp.write_text(s,encoding='utf-8')
    intro=MS/'sections/introduction.tex';s=intro.read_text();s=s.replace('In short, we do not transfer unavailable counter-orbit information; we transfer how to search useful evidence from the target orbit.',
        'The transferred policy is learned with privileged counter context, but its searched values and deployed execution remain target-side.')
    intro.write_text(s,encoding='utf-8')
    print('MANUSCRIPT_UPDATED completed_loss_arms='+','.join(complete),flush=True)

if __name__=='__main__':main()
