"""File/model/table correspondence and explicit non-equivalence checks."""
from pathlib import Path
import csv,json,hashlib,re
ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930';MS=PACK/'manuscript'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    def add(item,version,role,path,note=''):
        p=ROOT/path;rows.append({'paper_item':item,'model_version':version,'role':role,'source_path':path,'exists':p.exists(),'bytes':p.stat().st_size if p.is_file() else None,'sha256':sha(p) if p.is_file() else None,'note':note})
    base=str(PACK.relative_to(ROOT)).replace('\\','/')
    current=base+'/evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv'
    old=base+'/evidence/earlier_package/'
    cache='experiments/GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz'
    versions={
        'S0':('experiments/windows_main/student_target_search_v2/S0_seed42.pt','cocd/windows_main/student_search_kd_v2.py',cache),
        'Ours':('experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt','experiments/distillation_final/ours_s2_2stage/train_s2_2stage.py',cache),
        'Multi2Uni':('experiments/distillation_final/multi2uni_port/Multi2Uni-Port_deploy_student_seed42_best.pt','experiments/distillation_final/multi2uni_port/train_multi2uni.py',cache),
        'Hinton-KD':('experiments/distillation_final/hinton_kd/Hinton-KD-Port_seed42_best.pt','experiments/distillation_final/hinton_kd/train_hinton.py','experiments/distillation_final/hinton_kd/summary.json'),
        'Boehm':('experiments/single_orbit_baselines_v3/boehm/best.pt','cocd/scripts/23_train_external_baselines.py','experiments/single_orbit_baselines_v3/boehm/test_predictions.npz')}
    for model,paths in versions.items():
        add('primary comparison / G10',model,'recorded regional metrics',current)
        for role,path in zip(('checkpoint','training code','pixel cache or summary'),paths):add('primary comparison / G10',model,role,path)
    add('primary DIS2 row','historical DIS2','archived recorded metrics',old+'qualitative/csv/table_historical_external_comparison.csv','Retain 87.33/61.77; region cells null. No original DIS2 probabilities recovered.')
    add('DIS2 diagnostic only','recovered DIS2 port','different-score re-test report',old+'quantitative/json/DIS2_recovered_inference_report.json','Not the historical 87.33 row. Algebraic TGD=G10; original masks and checkpoint lineage unverified.')
    for version in ('CG0','SG0','CG10'):
        add('current versus historical Teacher',version,'checkpoint',f'experiments/windows_main/teacher_target_search_v2/{version}_seed42.pt')
    for label,path,model in [('qualitative / G10',cache,'S0, Multi2Uni, S2-2Stage'),
                             ('current search controls',base+'/new_evidence/20260930/intervention_predictions.npz','CG0/S2-2Stage'),
                             ('current search alignment',base+'/new_evidence/20260930/current_search_alignment.csv','CG0/S2-2Stage'),
                             ('current query samples',base+'/new_evidence/20260930/current_query_policy_layers.npz','CG0/S2-2Stage'),
                             ('detailed architecture',base+'/editable/architecture_current_shared_memory_checked.pptx','full native original paths; TGD off in CG0; Student shared memory'),
                             ('historical weight plot',old+'quantitative/csv/table_loss_strength.csv','historical Student'),
                             ('historical search table',old+'quantitative/csv/table_search_mechanism.csv','CG10'),
                             ('historical composition table',old+'quantitative/csv/table_loss_composition.csv','earlier Student')]:add(label,model,'original file',path)
    for label,name in [('Hinton probability verification','Hinton_current_predictions.npz'),('Hinton source metric verification','primary_probability_verification.json'),('Boehm cache order','Boehm_verified_prediction_index.csv'),('sampling-head symmetry','learned_point_symmetry.json')]:
        add(label,'current','verified evidence',base+'/new_evidence/20260930/'+name)
    add('compact search figure','CG0/S2-2Stage','main paper figure',base+'/manuscript/figures/fig_search_current_compact.pdf')
    for label,path in [('masked multi-method comparison','manuscript/figures/fig_comparison_valid_reader.pdf'),('full-pixel method and objective comparison','manuscript/figures/fig_comparison_full_reader.pdf'),('scientific sampling figure','manuscript/figures/fig_search_reader.pdf'),('publication network source','editable/architecture_paper_reader.pptx'),('publication network figure','manuscript/figures/fig_architecture_reader.pdf'),('reader figure provenance','new_evidence/20260930/reader_figure_sources.json'),('online prediction verification','new_evidence/20260930/online_prediction_verification.json'),('online fresh probabilities','new_evidence/20260930/Online_current_predictions.npz')]:
        add(label,'current publication','figure or verified input',base+'/'+path)
    for arm in ('Seg','SegP','SegE'):
        for name in ('summary.json','best.pt','best_predictions.npz'):
            add('current loss controls',arm,'new retraining '+name,base+'/new_evidence/20260930/loss_retraining/'+arm+'/'+name,'Only summary COMPLETE authorizes paper numbers.')
    for name in ('models_da_search.py','models_teacher_target_search_v2.py','models_student_target_search_v2.py','data.py'):
        add('method equations / all current inference','current v2','source implementation','cocd/windows_main/'+name)
    for src in sorted((PACK/'scripts').glob('*.py')):add('reproduction','20260930','saved calculation script',str(src.relative_to(ROOT)).replace('\\','/'))
    with (OUT/'paper_source_correspondence.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Index for canonical interleaved-by-batch probability export, distinct from cache's ASC-all then DESC-all order.
    with (ROOT/'data/splits_v3/test_ids.csv').open() as f:ids=[int(r['sample_id']) for r in csv.DictReader(f)]
    index=[]
    for start in range(0,len(ids),16):
        for orbit in ('asc','desc'):
            for sid in ids[start:start+16]:index.append({'row':len(index),'sample_id':sid,'orbit':orbit})
    dest=OUT/'loss_retraining';dest.mkdir(exist_ok=True)
    with (dest/'matched_prediction_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    # Check every LaTeX include and citation reference. Compilation is a separate check.
    paths=[];citations=set();labels=[];refs=[]
    for p in [MS/'main.tex',*sorted((MS/'sections').glob('*.tex'))]:
        s=p.read_text()
        for kind,value in re.findall(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',s):
            target=MS/value
            if kind=='input':target=target.with_suffix('.tex')
            paths.append({'source':str(p.relative_to(MS)),'reference':value,'exists':target.exists()})
        for group in re.findall(r'\\cite\{([^}]+)\}',s):citations.update(group.split(','))
        labels+=re.findall(r'\\label\{([^}]+)\}',s);refs+=re.findall(r'\\ref\{([^}]+)\}',s)
    bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(MS/'references.bib').read_text()))
    report={'includes':paths,'missing_citation_keys':sorted(citations-bibkeys),'missing_labels':sorted(set(refs)-set(labels)),
            'duplicate_labels':sorted({x for x in labels if labels.count(x)>1}),'native_editor_status':'FAILED environment: Unable to find standard directories for platform',
            'project_pdf_exists':(MS/'main.pdf').exists(),
            'scope':'Static resource checks do not verify typesetting or bibliographic accuracy.'}
    (OUT/'latex_source_checks.json').write_text(json.dumps(report,indent=2))
    (OUT/'version_boundaries.md').write_text('''# Version and claim boundaries

- Current main: frozen S2-2Stage, CG0 training guidance, all labeled pixels, target-only deployment. Current search controls load CG0 explicitly.
- Current intervention base reproduces the main probability metrics exactly. Shuffle is a global cyclic derangement; frozen self-context and trained SG0 are different controls.
- The former search-alignment CSV (0.028/0.052) was generated by a script loading CG10. It is retained as historical, replaced by explicitly CG0 results. New statistics also use query weighting rather than equally weighted batch/orbit means. Do not attribute the numerical difference only to model version.
- Historical loss and strength sweeps are not current two-stage ablations. No current per-setting weight scan has been found. The historical figure uses real per-setting values and no best-setting highlight.
- Canonical current KD detaches Student target values and the backbone features used to recompute policy. Distillation updates context/policy rather than directly updating backbone/value projection.
- Nonzero offsets are not established as necessary: fixed-center intervention is close to learned Student, several actual candidates overlap, and the no-search intervention is not architecture retraining.
- Original DIS2 probabilities and the historical checkpoint were not recovered. The re-test report is numerically different; it is not used to fill the 87.33 historical row. TGD has the same algebraic mask expression in the retained re-test code, but historical mask identity is not proved.
- Source/cache index order differs: original qualitative cache stores all ASC then all DESC; new canonical inference exports ASC then DESC within each batch. Join by sample_id+orbit, never row position alone.
- New single-component training retains coefficient 0.5 for each surviving loss, matching its contribution in the combined model. This is a component-removal study, not a re-normalized single-component weight sweep.
- All new visual cases retain blank human selection and needs_manual_review=true. The G10 scene/crop is chosen by annotation support, not comparative performance.
''',encoding='utf-8')
    print('SOURCE_AUDIT_COMPLETE',len(rows),'missing includes',sum(not x['exists'] for x in paths),flush=True)

if __name__=='__main__':main()

