"""Scientific manuscript terminology, literature, and concise publication tables."""
from pathlib import Path
import csv,json,re
PACK=Path(__file__).resolve().parents[1];MS=PACK/'manuscript';OUT=PACK/'new_evidence/20260930';BASE=OUT/'before_reader_revision'

def read(p):return list(csv.DictReader(p.open()))
def table(path,caption,label,rows,f1=False,groups=None):
    k=3 if f1 else 2;cols=r'l'+'r'*(k*2)
    s=[r'\begin{table}[!t]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{2.5pt}',r'\begin{tabular}{'+cols+'}',r'\toprule',r'Method & \multicolumn{'+str(k)+r'}{c}{Valid pixels} & \multicolumn{'+str(k)+r'}{c}{All pixels}\\',r' & AUPRC & IoU'+(' & F1' if f1 else '')+r' & AUPRC & IoU'+(' & F1' if f1 else '')+r'\\',r'\midrule']
    previous=None
    for name,m,g in rows:
        if groups and g!=previous:
            if previous is not None:s.append(r'\midrule')
            s.append(r'\multicolumn{'+str(1+2*k)+r'}{l}{\emph{'+g+r'}}\\');previous=g
        vals=[]
        for region in ('TargetValid','Overall'):
            for key in (('AUPRC','IoU','F1') if f1 else ('AUPRC','IoU')):vals.append(f"{100*m[region][key]:.2f}" if region in m else '--')
        s.append(name+' & '+' & '.join(vals)+r'\\')
    s += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];path.write_text('\n'.join(s)+'\n',encoding='utf-8')

def main():
    introduction=r'''\section{Introduction}
\label{sec:introduction}
Earthquake-triggered landslides can affect inaccessible terrain before field surveys or cloud-free optical images become available. Synthetic aperture radar (SAR) supports rapid response through cloud cover and independent of daylight \cite{Mondini2021,Handwerger2022,Amatya2023}. Convolutional and attention-based networks have advanced SAR landslide mapping \cite{Nava2022DL,Nava2022Attention,Boehm2022,Chen2024MFEWF}. Nevertheless, foreshortening, layover, and shadow alter the terrain evidence visible from each radar look direction \cite{Shi2024}; backscatter and speckle also complicate local landslide signatures \cite{Santangelo2022}. Ascending and descending observations therefore provide complementary, rather than interchangeable, views. Requiring both post-event passes can delay the first response map. We instead use paired orbits during training and only the available target orbit during deployment (Fig.~\ref{fig:scenario}).

Learning with privileged information \cite{LopezPaz2016} and knowledge distillation (KD) \cite{Hinton2015} provide a basis for transferring training-only observations. Remote-sensing approaches transfer grouped features \cite{Li2022}, multimodal online knowledge \cite{Liu2024}, or frequency and cross-modal affinity information \cite{Sun2027}. Other methods preserve modality-specific information through contrastive distillation \cite{Sikdar2025} or compensate missing features using disentanglement and classwise attention \cite{Kieu2026DIS2}. These approaches motivate a different question for view-dependent SAR: how can an unavailable complementary observation teach a model to exploit evidence that remains observable in the target view?

Deformable convolution and attention separate sampling locations and aggregation weights from the sampled values \cite{Dai2017,Zhu2021}. Adaptive aggregation has been applied to remote-sensing segmentation and road extraction \cite{Zuo2022,Dai2023}, while deformable cross-attention addresses weak alignment across imaging modalities \cite{Hu2025}. We use complementary context to guide a target-side search policy; the searched values always come from the target orbit. This prevents a dual-orbit value-fusion path that the deployed Student could not reproduce.

The key distinction is the information being transferred. A fused Teacher representation may contain measurements visible only from the complementary orbit. Matching that representation can therefore ask the Student to reproduce evidence absent from its input. Our Teacher instead determines a policy for reading target features. Applying Teacher and Student policies to the same target memory makes their discrepancy concern the retrieval operation rather than a difference between modality feature spaces. Complementary supervision can change how available context is used, but does not restore a radar measurement that was never observed.

Our contributions are a paired-orbit training framework with target-only deployment; a counter-guided search Teacher that separates policy generation from target evidence; and policy and shared-memory evidence distillation for a target-only Student. Experiments compare several distillation methods on valid and full image domains and examine complementary guidance, sampling, and loss objectives. Geometric masks are used after prediction to distinguish improvements on visible terrain from behavior under distortion; training never discards the latter pixels.
'''
    (MS/'sections/introduction.tex').write_text(introduction,encoding='utf-8')
    method=(BASE/'method.tex').read_text()
    method=re.sub(r'\\begin\{figure\}\[[^\]]*\].*?\\end\{figure\}\s*','',method,flags=re.S)
    method=re.sub(r'\\begin\{figure\*\}\[[^\]]*\].*?\\end\{figure\*\}\s*','',method,flags=re.S)
    scenario=r'''\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{figures/fig_task_scenario.png}
\caption{Paired ascending and descending observations are available for training; rapid post-event response may have only one orbit. The Student therefore uses the available orbit alone.}
\label{fig:scenario}
\end{figure}

'''
    architecture=r'''\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig_architecture_reader.pdf}
\caption{Counter-guided search and target-only distillation. Complementary context determines the Teacher policy, but retrieved values remain target-side. Both evidence retrievals use the same Student target memory. Teacher encoder weights are shared; the auxiliary geometric loss is not used.}
\label{fig:architecture}
\end{figure*}

'''
    method=scenario+method.replace(r'\subsection{Problem Formulation}',architecture+r'\subsection{Problem Formulation}',1)
    method=method.replace('Geometry codes are used to define diagnostic evaluation regions.','Geometric-distortion masks define the evaluation regions.').replace('The primary Student does not receive geometry as an input or use it to weight its loss.','The Student receives no geometric mask and trains on every labelled pixel.')
    method=method.replace('At each reference location $p$, the Teacher predicts a joint policy over $L=3$ FPN levels (P2--P4),','A feature pyramid network (FPN) provides fine, middle, and coarse feature levels. At each reference location $p$, the Teacher predicts a joint policy over these $L=3$ levels,')
    (MS/'sections/method.tex').write_text(method,encoding='utf-8')
    current=read(PACK/'evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv')
    primary=[]
    for display,key in [('Baseline','S0-Raw'),(r'SAR U-Net++ \cite{Boehm2022}','Boehm SAR U-Net++'),(r'Logit distillation \cite{Hinton2015}','Hinton-KD-Port'),(r'Online distillation \cite{Liu2024}','MMOKD-Port'),(r'Multi-to-Uni \cite{Sun2027}','Multi2Uni-Port'),('Ours','S2-2Stage')]:
        rr=next(r for r in current if r['Method']==key);m={region:{k:float(rr[region+'_'+k]) for k in ('AUPRC','IoU','F1')} for region in ('TargetValid','Overall')};primary.append((display,m,None))
    primary.insert(-1,(r'DIS2 \cite{Kieu2026DIS2}',{'Overall':{'AUPRC':.8733278861052558,'IoU':.6176884604656225,'F1':.7636680010378782}},None))
    table(MS/'sections/current_primary_table.tex',r'Landslide segmentation (\%). Valid pixels exclude geometric distortion in the input orbit; all pixels retain it. DIS2 has only the available historical full-image record.','tab:primary',primary,True)
    controls={}
    for rr in read(OUT/'current_search_interventions.csv'):
        key=rr['arm'];region=rr['region'];controls.setdefault(key,{})[region]={k:float(rr[k]) for k in ('AUPRC','IoU','F1')}
    selfguided={rr['region']:{k:float(rr[k]) for k in ('AUPRC','IoU','F1')} for rr in read(OUT/'current_sg0_metrics.csv')}
    search=[('Self-guided',selfguided,'Teacher guidance'),('Counter-guided',controls['CG0:paired'],'Teacher guidance'),('Shuffled counter',controls['CG0:counter-shuffle'],'Teacher guidance'),('Learned sampling',controls['S2-2Stage:learned'],'Student sampling'),('Fixed center',controls['S2-2Stage:center'],'Student sampling'),('Random offsets',controls['S2-2Stage:random'],'Student sampling'),('Without search',controls['S2-2Stage:no-search'],'Student sampling')]
    table(MS/'sections/current_search_table.tex',r'Guidance and sampling comparison (\%). Teacher rows compare guidance; Student rows alter sampling during inference.','tab:search_intervention',search,groups=True)
    losses=[]
    for arm,name in [('Seg','Segmentation only'),('SegP','Policy distillation'),('SegE','Evidence distillation')]:
        ss=json.loads((OUT/'loss_retraining'/arm/'summary.json').read_text());assert ss['status']=='COMPLETE';losses.append((name,ss['metrics'],None))
    losses.append(('Policy + evidence',controls['S2-2Stage:learned'],None))
    table(MS/'sections/current_loss_table.tex',r'Distillation objectives under matched training (\%). The surviving objective retains its original coefficient.','tab:loss_parts',losses)
    experiments=r'''\section{Experiments}
\label{sec:experiments}
\subsection{Data and Evaluation}
We use 1,713 paired Sentinel-1 locations from the Haiti landslide-change dataset \cite{Bralet2024DataPort,Bralet2024}. Each $128\times128$ input stacks pre- and post-event co-polarized (VV) and cross-polarized (VH) backscatter with a constant orbit-indicator channel. The Student uses ConvNeXt-Tiny with an FPN, trained with Adam, learning rate $5\times10^{-5}$, and batch size 16. All labelled pixels contribute to training. At evaluation, we report geometrically valid input-orbit pixels and all pixels from the same predictions. Additional regional analysis considers pixels valid in both views and pixels distorted only in the input view. The area under the precision--recall curve (AUPRC) measures ranking quality; intersection over union (IoU) and F1 use a probability threshold of 0.5.

\subsection{Comparison With Existing Methods}
Table~\ref{tab:primary} compares a target-only baseline, SAR U-Net++, logit distillation, online multimodal distillation, Multi-to-Uni, DIS2, and our Student. The distillation approaches are adapted to paired SAR segmentation and use only the target orbit at deployment; these are task-specific implementations rather than the original published benchmark results. DIS2 retains the historical 87.33\% AUPRC and 61.77\% IoU; its predictions are unavailable for valid-pixel re-evaluation.
\input{sections/current_primary_table}
Ours improves the baseline and other current distillation methods in both valid-pixel and full-image evaluation. On pixels valid in both views, its AUPRC/IoU/F1 are 88.15/63.72/77.84\%, compared with 86.91/61.34/76.04\% for the baseline. Where the input view is distorted but the opposite view is valid, the scores are 89.96/65.64/79.26\%, compared with 88.82/64.14/78.15\%. Distortion therefore affects the evidence available to the model, but does not imply that every such pixel is unclassifiable.

\subsection{Qualitative Comparison}
Fig.~\ref{fig:qualitative} compares five methods after excluding distorted input pixels. Both radar views, their distortion regions, and the landslide reference reveal whether improvements concern boundaries, missed landslides, or false alarms. Fig.~\ref{fig:full_comparison} retains predictions throughout a distorted slope and compares both competing distillation methods and our loss objectives. The opposite orbit is shown to explain complementary visibility; it is not supplied to the deployed Student.
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_comparison_valid_reader.pdf}
\caption{Valid-pixel comparison for descending-input and ascending-input examples. Rows show both post-event radar views, distortion regions (purple), the landslide reference, evaluation support, and five methods. Distorted input pixels are excluded identically for every prediction (gray). Correct landslides are green; false alarms are red; missed landslides are blue.}
\label{fig:qualitative}
\end{figure}
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_comparison_full_reader.pdf}
\caption{Full-pixel comparison on an ascending-input slope. Top: both radar views, their distortion regions, and reference. Middle: five competing methods. Bottom: segmentation, policy, evidence, and both objectives under matched training. No distorted prediction is hidden. Colors follow Fig.~\ref{fig:qualitative}.}
\label{fig:full_comparison}
\end{figure}

\subsection{Complementary Guidance and Search}
Table~\ref{tab:search_intervention} separates Teacher guidance from Student sampling. The self-guided Teacher is independently trained; counter shuffling mismatches paired locations at inference. Fixed-center and random-offset tests retain learned aggregation weights; removing search feeds the fine-scale feature directly to the trained decoder. Correct pairing is beneficial, and arbitrary offsets reduce performance. Fixed-center Student sampling remains competitive, so these inference tests do not establish that learned spatial displacement is necessary.
\input{sections/current_search_table}
Fig.~\ref{fig:search_samples} shows the actual sampling and weight allocation across three feature scales. Learned points overlap, supporting an interpretation involving scale and weight allocation rather than diverse spatial displacements. Across valid query regions, Student-to-Teacher candidate-weight Jensen--Shannon divergence is 0.0219 versus 0.0782 with shuffled guidance; shared-memory cosine distance is 0.0066 versus 0.0595. Candidate-weight divergence summarizes the twelve weights; training instead aligns the spatial sampling distributions.
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_search_reader.pdf}
\caption{Teacher (blue triangles) and Student (orange circles) sampling at fine, middle, and coarse scales. The red cross marks the query; bars show candidate weights. Overlapping learned samples are displayed at their actual locations. Each scale has one fixed center and three coincident learned candidates.}
\label{fig:search_samples}
\end{figure}

\subsection{Distillation Objectives}
Table~\ref{tab:loss_parts} compares matched segmentation, policy, evidence, and combined objectives. Policy distillation improves the segmentation control, whereas evidence alone gives essentially unchanged accuracy. The combined Student obtains the highest AUPRC and IoU in both reported domains. The matched segmentation control is stronger than the primary baseline because it is separately retrained under the current schedule; the two controls are not interchangeable. These single-run comparisons do not establish statistical superiority.
\input{sections/current_loss_table}
'''
    (MS/'sections/experiments.tex').write_text(experiments,encoding='utf-8')
    (MS/'sections/conclusion.tex').write_text(r'''\section{Conclusion}
\label{sec:conclusion}
Counter-guided search uses complementary radar context to teach a target-only Student how to retrieve target-side evidence. Combined distillation improves mapping on both valid and full image domains, without excluding distortion during training. Matched controls favor the combined objectives, while evidence alone provides no clear gain. Fixed-center sampling and overlapping learned points limit claims about spatial displacement; the present evidence instead supports complementary guidance and adaptive scale/weight allocation.
''',encoding='utf-8')
    main=(BASE/'main.tex').read_text().replace(r'\usepackage{placeins}',r'\usepackage{placeins}'+'\n'+r'\usepackage{float}'+'\n'+r'\usepackage{cite}')
    main=main.replace(r'\input{sections/method}',r'\clearpage'+'\n'+r'\input{sections/method}')
    main=main.replace('Current inference interventions show sensitivity to counter pairing and random sampling, while fixed-center sampling remains competitive.','Complementary guidance is beneficial, although fixed-center sampling remains competitive.')
    main=main.replace('Student improves AUPRC from','Student improves the area under the precision--recall curve (AUPRC) from').replace('and IoU from','and intersection over union (IoU) from',1)
    (MS/'main.tex').write_text(main,encoding='utf-8')
    texts='\n'.join(p.read_text() for p in (MS/'sections').glob('*.tex'));keys=set()
    for group in re.findall(r'\\cite\{([^}]+)\}',texts):keys.update(group.split(','))
    bib=set(re.findall(r'@\w+\{([^,]+),',(MS/'references.bib').read_text()));assert keys==bib,(bib-keys,keys-bib)
    (OUT/'reader_revision_manifest.json').write_text(json.dumps({'bibliography_entries':len(bib),'cited_entries':len(keys),'publication_names':{'S0':'Baseline','Hinton-KD-Port':'Logit distillation','MMOKD-Port':'Online distillation','Multi2Uni-Port':'Multi-to-Uni','S2-2Stage':'Ours','CG0':'Counter-guided Teacher','SG0':'Self-guided Teacher','TargetValid':'Valid pixels','Overall':'All pixels','G10':'input-view-distorted, opposite-view-valid pixels'},'main_search_rows':7,'all_measured_controls_preserved_in':'current_search_interventions.csv','all_four_region_metrics_preserved_in':'existing source CSVs','scenario_placement':'explicit page break before scenario at method opening','quantitative_values':'all taken from recorded current or locked historical metrics'},indent=2))
    print('READER_REVISION: 23 cited references; concise tables; descriptive terminology')
if __name__=='__main__':main()
