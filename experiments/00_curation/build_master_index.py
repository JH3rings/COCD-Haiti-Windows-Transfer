from __future__ import annotations
import csv, json, re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments'
OUT = EXP / 'summary'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

metric_rows = []
def add(run_id, method, regime, protocol, best_epoch, metrics, source, selection='', train_ids='', val_ids='', test_ids='', mask=''):
    for region, m in metrics.items():
        if not isinstance(m, dict):
            continue
        ap = m.get('AUPRC', m.get('auprc'))
        iou = m.get('IoU', m.get('iou'))
        f1 = m.get('F1', m.get('f1'))
        if ap is None or iou is None or f1 is None:
            continue
        metric_rows.append({
            'RunID': run_id, 'Method': method, 'TrainingRegime': regime,
            'Protocol': protocol, 'BestEpoch': best_epoch, 'Region': region,
            'AUPRC': ap, 'IoU': iou, 'F1': f1, 'Pixels': m.get('pixels', ''),
            'TrainingGeometryMask': mask, 'TrainIDs': train_ids, 'ValidationIDs': val_ids,
            'TestIDs': test_ids, 'CheckpointSelection': selection,
            'SourceMetrics': source.replace('\\', '/')
        })

def readj(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))

# Current raw main benchmark: the paper package's full_metrics.csv is the source
# for the five published regions; OverallValid is kept from its separate exact
# region re-evaluation when available.
region_csv = EXP / 'GRSL_complete_paper_package/04_tables/reevaluated_region_metrics.csv'
full_csv = EXP / 'GRSL_complete_paper_package/04_tables/full_metrics.csv'
full = {r['Method']: r for r in csv.DictReader(full_csv.open(encoding='utf-8-sig', newline=''))}
runmap = {
    'S0': 'windows_main/raw_cver_v1_seed42',
    'S2-Raw-native': 'windows_main/raw_cver_v1_seed42',
    'S2-2Stage': 'distillation_final/ours_s2_2stage',
    'Hinton-KD-Port': 'distillation_final/hinton_kd',
    'DIS2-Port': 'distillation_final/dis2_port',
    'Multi2Uni-Port': 'distillation_final/multi2uni_port',
    'MMOKD-Port': 'GRSL_complete_paper_package/baselines/mmokd_port',
    'SG Teacher': 'windows_main/teacher_target_search_v2',
    'CG Teacher': 'windows_main/teacher_target_search_v2',
    'CG shuffle': 'windows_main/teacher_target_search_v2',
}
by_method = defaultdict(dict)
for r in csv.DictReader(region_csv.open(encoding='utf-8-sig', newline='')):
    by_method[r['Method']][r['Region']] = {'AUPRC':float(r['AUPRC']), 'IoU':float(r['IoU']), 'F1':float(r['F1']), 'pixels':r.get('pixels','')}
for source_method, meta in full.items():
    method = 'S0' if source_method == 'S0-Raw' else source_method
    run = runmap.get(method, 'GRSL_complete_paper_package')
    regions = {}
    for region,prefix in [('Overall','Overall'),('TargetValid','TargetValid'),('BothValid','BothValid'),('G10','G10'),('G01','G01')]:
        regions[region] = {
            'AUPRC':meta.get(f'{prefix}_AUPRC'),
            'IoU':meta.get(f'{prefix}_IoU'),
            'F1':meta.get(f'{prefix}_F1'),
        }
    add(run, method, 'Raw/no geometry mask', 'Raw target-only deployment; full final test protocol as recorded in source package', meta.get('best_epoch',''), regions,
        'experiments/GRSL_complete_paper_package/04_tables/full_metrics.csv',
        selection='source full_metrics.csv; test-monitoring protocol noted in package', mask='No')

for original, regions in by_method.items():
    if original in {'S0','S2-2Stage','Multi2Uni-Port','MMOKD-Port'}:
        meta=full.get('S0-Raw' if original=='S0' else original,{})
        run=runmap.get(original,'GRSL_complete_paper_package')
        add(run,original,'Raw/no geometry mask','Region re-evaluation from the current paper package',meta.get('best_epoch',''),
            {'OverallValid':regions['OverallValid']} if 'OverallValid' in regions else {},
            'experiments/GRSL_complete_paper_package/04_tables/reevaluated_region_metrics.csv',
            selection='source full_metrics.csv; test-monitoring protocol noted in package',mask='No')
    elif original in {'SG Teacher','CG Teacher','CG shuffle'}:
        add(runmap[original],original,'Teacher diagnostic (source protocol)','Teacher diagnostic retained in paper package', '',regions,
            'experiments/GRSL_complete_paper_package/04_tables/reevaluated_region_metrics.csv',
            selection='source package metric export',mask='No')

# Full-train masked v5 (test monitored; no validation set).
p='experiments/windows_main/mask_aware_search_kd_v5_seed42/final_metrics.json'; d=readj(p)
for method, obj in d['results'].items():
    if not isinstance(obj,dict) or not isinstance(obj.get('metrics'),dict): continue
    add('windows_main/mask_aware_search_kd_v5_seed42', method, 'Masked geometry', 'train=1370/val=0/test=343; test used by canonical per-epoch monitor; raw inference', obj.get('best_epoch',''), obj['metrics'], p,
        selection='best test Overall AUPRC (monitor, not independent test)', train_ids=1370, val_ids=0, test_ids=343, mask='target geometry code > 0 excluded from segmentation BCE; strict valid KD queries')

# Full-train masked v4 diagnostic arms; teachers are reused legacy references.
p='experiments/windows_main/sgri_powc_masked_v4_seed42_20260925/final_metrics.json'; d=readj(p)
for method, obj in d['metrics'].items():
    add('windows_main/sgri_powc_masked_v4_seed42_20260925', method, 'Masked geometry', str(d.get('protocol','')), obj.get('best_epoch',''), obj.get('metrics',{}), p,
        selection='test raw Overall AUPRC monitor, patience=2; teacher rows reused legacy checkpoints', train_ids=1370, val_ids=0, test_ids=343, mask='target code > 0 excluded from new Student segmentation loss; KD query validity restricted')

# Earlier matched train/validation split geometry-transfer study; kept separate, not pooled with full-train v4.
p='experiments/windows_main/geometry_transfer_restored_v1/test_metrics.json'; d=readj(p)
for item in d['metrics']:
    m=item['method']; ep=d.get('selected_checkpoints',{}).get(m,{}).get('best_epoch','')
    add('windows_main/geometry_transfer_restored_v1', m, 'Masked geometry; legacy validation split', 'train=1233/val=137/test=343; best checkpoint selected by validation AUPRC', ep, item['regions'], p,
        selection='validation Overall AUPRC; test held out', train_ids=1233, val_ids=137, test_ids=343, mask='target code > 0 excluded from segmentation BCE; KD queries strict valid')

# Keep every finished equal-budget controller protocol as a separate run.
for folder, protocol in [
    ('multitoone_kd_10pct_20260927','initial EMA controller'),
    ('multitoone_kd_10pct_batchratio_20260928','EMA controller targeting mean per-batch KD share'),
    ('multitoone_kd_10pct_lossratio_ema_20260928','EMA controller targeting pooled loss contribution'),
]:
    base=f'experiments/windows_main/{folder}'
    for method in ('S2-10%','RKD-10%','DIS2-RS-10%'):
        p=f'{base}/{method}_summary.json'; d=readj(p)
        add(f'windows_main/{folder}', method, 'Raw/no geometry mask; matched KD-budget diagnostic', protocol, d.get('best_epoch',''), d.get('metrics',{}), p,
            selection='source runner best test Overall AUPRC (project monitor)', train_ids=1370, val_ids=0, test_ids=343, mask='No; raw training')

metrics_path=OUT/'MASTER_METRICS.csv'
with metrics_path.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(metric_rows[0].keys())); w.writeheader();w.writerows(metric_rows)

# Run catalog: all dirs with protocol/result artefacts, including separate repeats and quarantined history.
interesting={'protocol.json','final_report.md','final_metrics.json','test_metrics.json','comparison.csv','summary.json','metrics.csv'}
run_dirs=set()
for path in EXP.rglob('*'):
    if path.is_file() and (path.name in interesting or path.name.endswith('_summary.json')):
        if any(part.startswith('paper_submission_package_') for part in path.relative_to(EXP).parts):
            continue
        run_dirs.add(path.parent)
cat=[]
for folder in sorted(run_dirs, key=lambda x:str(x).lower()):
    if folder == EXP: continue
    files=[x for x in folder.rglob('*') if x.is_file()]
    checkpoints=[x for x in files if x.suffix.lower() in {'.pt','.pth','.ckpt'}]
    figs=[x for x in files if x.suffix.lower() in {'.png','.jpg','.jpeg','.svg','.pdf'}]
    protocol_path=folder/'protocol.json'
    protocol_summary=''
    if protocol_path.exists():
        try:
            obj=json.loads(protocol_path.read_text(encoding='utf-8-sig'))
            parts=[]
            for k in ('method','seed','train_ids','validation_ids','test_ids','train_ids_count','max_epochs','patience','monitor','checkpoint_selection'):
                if k in obj: parts.append(f'{k}={obj[k]}')
            protocol_summary='; '.join(parts)[:600]
        except Exception as e: protocol_summary=f'protocol parse issue: {type(e).__name__}'
    rel=folder.relative_to(EXP).as_posix()
    status='SUPERSEDED_QUARANTINE' if '00_curation/superseded' in rel else 'HISTORICAL' if ('99_historical' in rel or 'archive' in rel.lower()) else 'RESULTS_PRESENT'
    metrics=[x.name for x in files if any(t in x.name.lower() for t in ('metric','summary','report','result')) and x.suffix.lower() in {'.csv','.json','.md','.txt'}]
    cat.append({'RunOrPackage':rel,'Status':status,'ProtocolSummary':protocol_summary,'FilesRecursive':len(files),'BytesRecursive':sum(x.stat().st_size for x in files),'CheckpointCount':len(checkpoints),'FigureCount':len(figs),'MetricReportArtifacts':'; '.join(sorted(set(metrics))[:30])})
with (OUT/'MASTER_RUN_CATALOG.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(cat[0].keys()));w.writeheader();w.writerows(cat)

# Full visual artefact inventory, with scope labels so historical figures are not mistaken for current ones.
visual=[]
for p in EXP.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in {'.png','.jpg','.jpeg','.svg','.pdf','.pptx'}: continue
    rel=p.relative_to(EXP).as_posix()
    scope='superseded/quarantined' if '00_curation/superseded' in rel else 'historical' if ('99_historical' in rel or 'archive/' in rel) else 'current experiment/package'
    visual.append({'Path':rel,'Format':p.suffix.lower().lstrip('.'),'Bytes':p.stat().st_size,'Scope':scope})
with (OUT/'ALL_FIGURE_INVENTORY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['Path','Format','Bytes','Scope']);w.writeheader();w.writerows(visual)

# Plot helpers: saved PNG for quick viewing and vector PDF for paper editing.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
def savefig(fig, name):
    fig.tight_layout(rect=(0,0.02,1,0.90))
    fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

# Current raw final method comparison (the source package's one-run final benchmark).
raw_methods=['S0-Raw','Hinton-KD-Port','DIS2-Port','Multi2Uni-Port','MMOKD-Port','S2-Raw-native','S2-2Stage']
regions=['Overall','TargetValid','BothValid']
labels={'S0-Raw':'S0','Hinton-KD-Port':'Hinton','DIS2-Port':'DIS2','Multi2Uni-Port':'Multi2Uni','MMOKD-Port':'MMOKD','S2-Raw-native':'S2-native','S2-2Stage':'S2-2Stage'}
raw_lookup={}
for method in raw_methods:
    src=full.get(method)
    if not src: continue
    for region,prefix in [('Overall','Overall'),('TargetValid','TargetValid'),('BothValid','BothValid')]:
        raw_lookup[(method,region)]={
            'AUPRC':src.get(f'{prefix}_AUPRC'),
            'IoU':src.get(f'{prefix}_IoU'),
            'F1':src.get(f'{prefix}_F1'),
        }
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,metric,title in zip(axes,['AUPRC','IoU','F1'],['AUPRC','IoU','F1']):
    x=np.arange(len(raw_methods));width=.24
    for j,reg in enumerate(regions):
        vals=[float(raw_lookup.get((m,reg),{}).get(metric,0)) for m in raw_methods]
        ax.bar(x+(j-1)*width,vals,width,label=reg)
    ax.set_title(title);ax.set_xticks(x);ax.set_xticklabels([labels[m] for m in raw_methods],rotation=35,ha='right');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False,fontsize=8)
fig.suptitle('Raw final method comparison',y=1.10)
savefig(fig,'raw_final_method_comparison')

# Raw vs masked S0/S2: show protocol families separately, with actual recorded region metrics.
points=[
 ('Raw S0','raw_cver_v1_seed42','S0','Raw/no geometry mask'),
 ('Raw S2-2Stage','ours_s2_2stage','S2-2Stage','Raw/no geometry mask'),
 ('Masked v5 S0','mask_aware_search_kd_v5_seed42','S0-clean','Masked geometry'),
 ('Masked v5 S2','mask_aware_search_kd_v5_seed42','S2-MaskAware','Masked geometry'),
 ('Masked v4 S0','sgri_powc_masked_v4_seed42_20260925','S0-mask','Masked geometry'),
 ('Masked v4 S2','sgri_powc_masked_v4_seed42_20260925','S2-mask','Masked geometry'),
]
# match by terminal run component and method, while explicitly labeling same-protocol differences.
selected=[]
for label, run_tail, method, regime in points:
    candidates=[r for r in metric_rows if r['Method']==method and r['TrainingRegime']==regime and r['RunID'].endswith(run_tail)]
    if candidates: selected.append((label,{r['Region']:r for r in candidates}))
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
colors=['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2']
for ax,metric in zip(axes,['AUPRC','IoU','F1']):
    x=np.arange(len(regions));w=.12
    for j,(label,vals) in enumerate(selected):
        ax.bar(x+(j-(len(selected)-1)/2)*w,[float(vals.get(reg,{}).get(metric,0)) for reg in regions],w,label=label,color=colors[j%len(colors)])
    ax.set_title(metric);ax.set_xticks(x);ax.set_xticklabels(regions);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False,fontsize=7)
fig.suptitle('Raw vs geometry-masked results (protocols differ; compare within each run family)',y=1.12)
savefig(fig,'raw_vs_masked_s0_s2')

# Masked teacher counter-complementarity: legacy validation-split experiment, distinguish from newer full-train rows.
geom_lookup={(r['Method'],r['Region']):r for r in metric_rows if r['RunID']=='windows_main/geometry_transfer_restored_v1'}
fig,axes=plt.subplots(1,3,figsize=(12.5,4.2))
teacher_regions=['BothValid','TargetValid','OverallValid']
for ax,metric in zip(axes,['AUPRC','IoU','F1']):
    x=np.arange(3);w=.24
    for j,method in enumerate(['SG','CG','CG_shuffle']):
        vals=[float(geom_lookup.get((method,reg),{}).get(metric,0)) for reg in teacher_regions]
        ax.bar(x+(j-1)*w,vals,w,label=method)
    ax.set_title(metric);ax.set_xticks(x);ax.set_xticklabels(teacher_regions,rotation=15);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False)
fig.suptitle('Geometry-masked Teacher counter-view test (train 1233 / val 137 / test 343)',y=1.12)
savefig(fig,'masked_teacher_counter_view')

# Geometry-masked full-train v4 comparison, kept distinct from the older
# 1233/137 validation-split experiment above.
v4_run='windows_main/sgri_powc_masked_v4_seed42_20260925'
v4_methods=['SG-Teacher','CG-Teacher','CG-shuffle','S0-mask','S2-mask','SGRI-A1','SGRI-A2','POWC-B1','POWC-B2']
v4_labels=['SG T','CG T','CG shuf','S0','S2','SGRI-A1','SGRI-A2','POWC-B1','POWC-B2']
v4_lookup={(r['Method'],r['Region']):r for r in metric_rows if r['RunID']==v4_run}
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,metric in zip(axes,['AUPRC','IoU','F1']):
    x=np.arange(len(v4_methods));w=.34
    for j,region in enumerate(['BothValid','TargetValid']):
        vals=[float(v4_lookup.get((m,region),{}).get(metric,0)) for m in v4_methods]
        ax.bar(x+(j-.5)*w,vals,w,label=region)
    ax.set_title(metric);ax.set_xticks(x);ax.set_xticklabels(v4_labels,rotation=40,ha='right');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=2,frameon=False)
fig.suptitle('Geometry-masked full-train v4 methods (train 1370 / val 0 / test 343)',y=1.12)
savefig(fig,'masked_v4_all_methods')

# Legacy masked search-KD mechanism variants on the held-out test under the
# archived validation-selected split; never merge with the full-train v4 panel.
legacy_run='windows_main/geometry_transfer_restored_v1'
legacy_methods=['S0','S2','ReliabilityKD','StableEvidenceKD']
legacy_labels=['S0','S2 Search-KD','Reliability-KD','Stable-Evidence-KD']
legacy_lookup={(r['Method'],r['Region']):r for r in metric_rows if r['RunID']==legacy_run}
fig,axes=plt.subplots(1,3,figsize=(12.5,4.8))
for ax,metric in zip(axes,['AUPRC','IoU','F1']):
    x=np.arange(3);w=.18
    for j,method in enumerate(legacy_methods):
        vals=[float(legacy_lookup.get((method,reg),{}).get(metric,0)) for reg in ['BothValid','TargetValid','OverallValid']]
        ax.bar(x+(j-1.5)*w,vals,w,label=legacy_labels[j])
    ax.set_title(metric);ax.set_xticks(x);ax.set_xticklabels(['BothValid','TargetValid','OverallValid'],rotation=15);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=4,frameon=False,fontsize=8)
fig.suptitle('Masked single-orbit student mechanisms (train 1233 / val 137 / test 343)',y=1.12)
savefig(fig,'masked_legacy_student_mechanisms')

# Equal-budget diagnostic compare completed controller implementations separately.
variants=['multitoone_kd_10pct_20260927','multitoone_kd_10pct_batchratio_20260928','multitoone_kd_10pct_lossratio_ema_20260928']
methods=['S2-10%','RKD-10%','DIS2-RS-10%']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,reg,metric in zip(axes,['BothValid','TargetValid'],['AUPRC','IoU']):
    x=np.arange(len(variants));w=.23
    for j,method in enumerate(methods):
        vals=[]
        for variant in variants:
            r=next((q for q in metric_rows if q['RunID'].endswith(variant) and q['Method']==method and q['Region']==reg),None)
            vals.append(float(r[metric]) if r else 0)
        ax.bar(x+(j-1)*w,vals,w,label=method)
    ax.set_title(f'{reg} {metric}');ax.set_xticks(x);ax.set_xticklabels(['Initial EMA','Batch-ratio EMA','Pooled-loss EMA'],rotation=20,ha='right');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False,fontsize=8)
fig.suptitle('Separate completed equal-KD controller runs (not pooled)',y=1.12)
savefig(fig,'equal_kd_completed_runs')

# Exact saved epoch-level KD contribution curves, not recomputed from rounded reports.
curve_names=['Initial EMA','Batch-ratio EMA','Pooled-loss EMA']
fig,axes=plt.subplots(2,3,figsize=(14.5,7),sharey=True)
for col,(variant,label) in enumerate(zip(variants,curve_names)):
    path=EXP/f'windows_main/{variant}/kd_budget_curve.csv'
    data=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    for method in methods:
        rows=[r for r in data if r['method']==method]
        if not rows: continue
        epochs=[int(r['epoch']) for r in rows]
        axes[0,col].plot(epochs,[100*float(r['kd_share_aggregate']) for r in rows],marker='.',linewidth=1,label=method)
        axes[1,col].plot(epochs,[100*float(r['kd_share_batch_mean']) for r in rows],marker='.',linewidth=1,label=method)
    for row in (0,1):
        axes[row,col].axhspan(8,12,color='#72B7B2',alpha=.14)
        axes[row,col].axhline(10,color='black',linewidth=.8,linestyle='--')
        axes[row,col].set_title(label if row==0 else '')
        axes[row,col].grid(alpha=.2)
        axes[row,col].set_xlabel('Epoch')
        axes[row,col].set_ylim(5,17)
axes[0,0].set_ylabel('Pooled KD contribution (%)')
axes[1,0].set_ylabel('Mean batch KD contribution (%)')
handles,legend_labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False)
fig.suptitle('Saved equal-KD epoch curves (target band 8–12%; protocols kept separate)',y=1.10)
savefig(fig,'equal_kd_epoch_curves')

# Refresh this inventory after generating the consolidated figures above.
visual=[]
for p in EXP.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in {'.png','.jpg','.jpeg','.svg','.pdf','.pptx'}: continue
    rel=p.relative_to(EXP).as_posix()
    scope='superseded/quarantined' if '00_curation/superseded' in rel else 'historical' if ('99_historical' in rel or 'archive/' in rel) else 'current experiment/package'
    visual.append({'Path':rel,'Format':p.suffix.lower().lstrip('.'),'Bytes':p.stat().st_size,'Scope':scope})
with (OUT/'ALL_FIGURE_INVENTORY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['Path','Format','Bytes','Scope']);w.writeheader();w.writerows(visual)

# Simple curated index text for the one-folder transfer workflow.
readme=f'''# Experiment data hub\n\nThis folder is the single portable home for Haiti COCD experiment data. Copy `experiments/` to move metrics, checkpoints, logs, figures, packages, and dataset-audit evidence together. Source code and the raw `data/` tree remain at the project root.\n\n## Start here\n\n- Final paper-ready package: [GRSL complete package](GRSL_complete_paper_package/README.md)\n- Master run/protocol catalog: [MASTER_RUN_CATALOG.csv](summary/MASTER_RUN_CATALOG.csv)\n- Unified long-form metrics (one row per run × method × region): [MASTER_METRICS.csv](summary/MASTER_METRICS.csv)\n- Complete image/PDF/PPTX inventory: [ALL_FIGURE_INVENTORY.csv](summary/ALL_FIGURE_INVENTORY.csv)\n- Comparison charts: [summary/figures](summary/figures/)\n- Dataset audit package: [dataset_audit_final/README.md](dataset_audit_final/README.md)\n- External KD implementations, runs and source notes: [distillation_final](distillation_final/)\n- Current and historical checkpoints: [checkpoints](checkpoints/) and [legacy_checkpoints](legacy_checkpoints/)\n- Cleanup/integrity records: [curation README](00_curation/README.md)\n\n## How to read comparisons\n\n`MASTER_METRICS.csv` keeps Raw/no-mask and geometry-masked runs in separate `TrainingRegime` and `RunID` fields. `Overall`, `OverallValid`, `TargetValid`, `BothValid`, `G10`, and `G01` remain distinct; do not infer one from another. Protocol variants and repeated completed runs are separate rows and were not averaged. The historical 1233/137/343 geometry-transfer split, the active full-train 1370/0/343 masked runs, and raw final benchmarks use different selection/training conditions; compare within the matching protocol family.\n\n## Key comparison charts\n\n- [Raw final method comparison](summary/figures/raw_final_method_comparison.png)\n- [Raw vs masked S0/S2](summary/figures/raw_vs_masked_s0_s2.png)\n- [Masked Teacher counter-view test](summary/figures/masked_teacher_counter_view.png)\n- [All masked full-train v4 methods](summary/figures/masked_v4_all_methods.png)\n- [Masked legacy Student mechanisms](summary/figures/masked_legacy_student_mechanisms.png)\n- [Completed equal-KD controller runs](summary/figures/equal_kd_completed_runs.png)\n- [Equal-KD epoch contribution curves](summary/figures/equal_kd_epoch_curves.png)\n\nEach chart is saved as PNG and vector PDF under `summary/figures/`. Existing method, qualitative, search-behavior, training-curve, and audit figures remain in their owning run/package folders and are indexed in `ALL_FIGURE_INVENTORY.csv`.\n\n## Retention and cleanup\n\nCompleted runs with distinct protocols and repeated valid runs are retained. Fixed-30 diagnostics, superseded loss-controller pilots, aborted calibration artefacts, duplicate files, and rendering caches are isolated under `00_curation/superseded/`; they are not part of the current comparison package. Their hashes and paths are recorded in the curation manifests. Files isolated there were moved rather than deleted because the execution policy rejected file deletion.\n'''
(EXP/'README.md').write_text(readme,encoding='utf-8')
print(f'wrote {metrics_path} rows={len(metric_rows)}')
print(f'wrote {OUT/"MASTER_RUN_CATALOG.csv"} rows={len(cat)}')
print(f'wrote {OUT/"ALL_FIGURE_INVENTORY.csv"} rows={len(visual)}')
print('figures:', ', '.join(p.name for p in sorted(FIG.glob('*.png'))))
