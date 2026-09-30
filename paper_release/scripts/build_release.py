"""Freeze provenance from retained runs; no training or manuscript edits."""
from pathlib import Path
import ast, csv, hashlib, json, re, shutil, subprocess, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper_release'
PACK=ROOT/'experiments/GRSL_paper_transfer_20260929'
EV=PACK/'new_evidence/20260930'
MAIN=PACK/'evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv'
REGIONS={'Overall':'All pixels','TargetValid':'Target-valid pixels','BothValid':'Both-orbit-valid pixels','G10':'Target distorted, counter valid','G01':'Target valid, counter distorted'}
METRICS=('AUPRC','IoU','F1')

def relative(p): return str(Path(p).resolve().relative_to(ROOT)).replace('\\','/')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csvout(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in data for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def copy(p,q):
    q=Path(q);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def entry(p):
    p=ROOT/p
    return {'path':relative(p),'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}

TEACHER='experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt'
STUDENT='experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt'
BASE='experiments/windows_main/student_target_search_v2/S0_seed42.pt'
M2U='experiments/distillation_final/multi2uni_port/Multi2Uni-Port_deploy_student_seed42_best.pt'
OLD='archive/history/code/cocd/windows_main/'
CANON='experiments/distillation_final/ours_s2_2stage/train_s2_2stage.py'

def archive():
    candidates=[('cocd/windows_main/'+x,OLD+x,'superseded training/evaluation or geometry-aware method') for x in (
        'student_search_kd_v2.py','evaluate_student_search_kd_v2.py','run_student_kd_weight_sweep.py','run_student_kd_strength_sweep.py',
        'geometry_transfer_restored.py','mask_aware_search_kd_v5.py','train_cofad_cocad_v1.py','train_raw_cver_v1.py',
        'train_sgri_powc_masked_v4.py','evaluate_sgri_powc_masked_v4.py')]
    candidates += [(x,'archive/history/reports/pre-freeze-20260930/'+x,'superseded current-method documentation') for x in
                   ('README_WINDOWS.md','GITHUB_CODE_PACKAGE.md','project_audit.md','cleanup_audit.md','cleanup_report.md',
                    'docs/model_description.md','docs/experiment_summary.md','docs/final_architecture.md','docs/naming_dictionary.md')]
    for name in ('fig_kd_sensitivity_historical.pdf','fig_kd_sensitivity_historical.png','fig_qualitative_20260929.pdf','fig_qualitative_20260929.png'):
        p=PACK/'manuscript/figures'/name
        if p.exists():candidates.append((relative(p),'archive/history/figures/pre-freeze-20260930/'+name,'historical protocol; not included by current manuscript'))
    audit=rows(OUT/'archive_manifest.csv') if (OUT/'archive_manifest.csv').exists() else []
    for src,dest,role in candidates:
        if any(r['original']==src for r in audit):continue
        a,b=ROOT/src,ROOT/dest
        assert a.resolve().is_relative_to(ROOT) and b.resolve().is_relative_to(ROOT)
        if a.exists():
            if b.exists():raise RuntimeError('Archive destination already exists: '+dest)
            audit.append({'original':src,'archived':dest,'role':role,'bytes':a.stat().st_size,'sha256_before':sha(a)})
    csvout(OUT/'archive_manifest.csv',audit)
    for r in audit:
        a,b=ROOT/r['original'],ROOT/r['archived'];b.parent.mkdir(parents=True,exist_ok=True)
        if a.exists() and not b.exists():shutil.move(str(a),str(b))
        r['sha256_after']=sha(b);assert r['sha256_before']==r['sha256_after']
    csvout(OUT/'archive_manifest.csv',audit)
    (ROOT/'archive/history/code/README.md').write_text(
        '# Historical code\n\nNot the final paper implementation. Original bytes are retained; see '
        '../../../paper_release/archive_manifest.csv for this freeze. Some historical runners use their original '
        'relative paths/imports. Reproduce them in the backup branch `archive/pre-grsl-paper-freeze-20260930` '
        'rather than substituting their architecture or KD definitions into the final pipeline. '
        'Current baseline results and checkpoints remain in their original experiment directories.\n')

def protocol():
    ids={n:[int(r['sample_id']) for r in rows(ROOT/f'data/splits_v3/{n}_ids.csv')] for n in ('train','test')}
    valp=ROOT/'data/splits_v3/val_ids.csv';ids['val']=[int(r['sample_id']) for r in rows(valp)] if valp.exists() else []
    overlap={f'{a}/{b}':len(set(ids[a])&set(ids[b])) for a,b in (('train','test'),('train','val'),('test','val'))}
    assert not any(overlap.values()) and all(len(v)==len(set(v)) for v in ids.values())
    n=np.load(EV/'normalization.npz')
    v={'dataset':'Multimodal Remote Sensing Dataset for Landslide Change Detection in Haiti',
       'citation':'Bralet et al., IEEE DataPort (2024), DOI 10.21227/4heb-7h07',
       'spatial_locations':len(set().union(*map(set,ids.values()))),'patch_size':[128,128],
       'acquisitions':{'ascending':{'pre':'2021-08-05','post':'2021-08-17'},'descending':{'pre':'2021-08-03','post':'2021-08-15'}},
       'channels':['pre VV','pre VH','post VV','post VH','orbit indicator (ascending=0, descending=1)'],
       'raw_bands':{'VV':2,'VH':1,'band_3':'unused'},
       'split':{'unit':'spatial location; both orbits and synthetic modes share partition','train':len(ids['train']),'validation':len(ids['val']),
                'test':len(ids['test']),'train_rows':3*len(ids['train']),'test_orbit_predictions':2*len(ids['test']),'partition_overlap':overlap},
       'split_sources':[entry(f'data/splits_v3/{n}_ids.csv') for n in ('train','test')],
       'normalization':{'mean':n['mean'].tolist(),'std':n['std'].tolist(),'source':entry(relative(EV/'normalization.npz')),
            'estimator':'Train locations only, pooled across pre/post and both orbits; float32 per-raster sums, variance + 1e-6; same VV/VH statistics reused for pre/post; orbit unchanged'},
       'training_modes':['actual pre/post with landslide mask','pre/pre with zero change label','post/post with zero change label'],
       'target':'Binary landslide change: processed/sample_{id:06d}/landslide_mask.tif > 0; synthetic unchanged modes use all-zero labels',
       'geometry':{'model_input':False,'training_supervision':False,'training_mask':False,'search_or_kd_weighting':False,
                   'evaluation_only':True,'meaning':'Existing geometry code > 0 denotes distorted; == 0 denotes valid. These maps are not independent distortion ground truth.',
                   'sources':['asc_pre_geom.tif','desc_pre_geom.tif'],'regions':REGIONS},
       'training':{'seed':42,'batch':16,'optimizer':'Adam','lr':5e-5,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0,
                   'max_epochs':100,'patience':2,'validation_locations':0,'selection':'test Overall AUPRC',
                   'caveat':'Test-monitored checkpoint selection; not independent held-out validation; single-seed development evidence'},
       'evaluation':{'AUPRC':'sklearn.metrics.average_precision_score; global pooled pixels, not per-patch mean',
                     'threshold':0.5,'IoU':'TP/(TP+FP+FN+1e-8)','F1':'2TP/(2TP+FP+FN+1e-8)',
                     'mask_semantics':{'All pixels':'all','Target-valid pixels':'target geometry == 0','Both-orbit-valid pixels':'target == 0 and counter == 0',
                                      'Target distorted, counter valid':'target > 0 and counter == 0'},
                     'indexing':'Join predictions by spatial sample ID and orbit. Original comparison cache groups by orbit; loader/export groups by batch then orbit.'},
       'loader':entry('cocd/windows_main/data.py'),'paths':entry('cocd/paths.py')}
    write(OUT/'protocol.json',v)
    for name in ('train','test'):
        copy(ROOT/f'data/splits_v3/{name}_ids.csv',OUT/f'splits/{name}_ids.csv')
    return v

def final_method():
    files={'Teacher':'cocd/windows_main/models_teacher_target_search_v2.py','Student':'cocd/windows_main/models_student_target_search_v2.py',
           'search_context_and_values':'cocd/windows_main/models_da_search.py','backbone':'cocd/models/landslide_cocd.py',
           'data':'cocd/windows_main/data.py','shared_protocol':'experiments/distillation_final/common.py',
           'Student_training_and_KD':CANON,'Teacher_training':'cocd/windows_main/teacher_target_search_v2.py',
           'matched_objective_ablation':'experiments/GRSL_paper_transfer_20260929/scripts/train_matched_loss.py',
           'final_evaluation':'paper_release/scripts/evaluate_final.py'}
    result={'status':'FINAL','name':'Counter-Guided Target Evidence Search for Single-Orbit SAR Landslide Mapping',
            'code':{k:entry(p) for k,p in files.items()},'checkpoints':{'Teacher':entry(TEACHER),'Student':entry(STUDENT),
                'initialization':entry('experiments/windows_main/da_search/SO_backbone_seed42.pt')},
            'Teacher_training_arm':'--stage CG0; counter context with geometry-loss coefficient exactly zero. Other arms in the shared runner are historical controls, not alternative final methods.',
            'architecture':{'counter_path':'shared encoder -> counter projection -> policy context -> policy only',
                'value_path':'target encoder/FPN -> target-only value projections -> search -> output projection -> decoder; no fusion/skip/reconstruction',
                'deployment':'One target orbit pre/post five-channel stack; neither counter orbit nor geometry mask',
                'memory_levels':['P2','P3','P4'],'channels':128,'candidates_per_level':4,'fixed_centers_per_level':1,
                'offsets':'2*tanh(raw offset), in feature pixels, (y,x)','weights':'Joint softmax over 3*4 candidates',
                'sampling':'bilinear grid_sample, align_corners=False, zero padding'},
            'canonical_policy_KD':'Bilinearly splat bounded level-scaled samples and joint weights onto the target P2 query grid; normalize each spatial distribution; minimize mean Jensen-Shannon divergence (natural logs). It does not regress raw offsets or compare only twelve candidate weights.',
            'canonical_evidence_KD':'Apply detached Teacher policy and recomputed Student policy to the exact same detached projected Student target memory, then minimize mean 1-cosine_similarity over channels.',
            'gradient_paths':'Teacher output, Student values and backbone features used for recomputed KD policy are detached. Both KD losses update only Student context/policy; BCE trains the entire Student.',
            'schedule':{'segmentation_only_epochs':5,'KD_calibration_epoch':5,'KD_gradient_starts_epoch':6,'lambda0':'0.25 * mean epoch-5 BCE',
                        'component_weight':0.5,'component_normalizer':'own epoch-5 mean, lower-bounded by 1e-8',
                        'multiplier':'0.25+0.75*0.5*(1+cos(pi*(epoch-6)/94))','KD_share':'Initial scalar target approximately 20%, not a measured constant share'},
            'conflicts':[{'source':'pre-freeze README and historical Student training','historical':'Geometry-weighted Teacher/KD, one-stage 10% calibration',
                         'resolution':'Archived; result-producing final sources listed above use unweighted two-stage KD and geometry-free Teacher'},
                        {'source':'manuscript/sections/method.tex policy paragraph','issue':'Says aligning offsets and weights; implementation aligns derived spatial distributions',
                         'resolution':'Recorded MISMATCH in claim audit; manuscript left untouched'}],
            'search_interpretation':'counter-guided adaptive target-evidence selection/aggregation',
            'unsupported':'Large learned spatial displacement is necessary or is the demonstrated source of improvement',
            'parameter_symmetry':'All three learned candidates at each level have identical offset-head rows in both frozen checkpoints; candidate multiplicity is nominal, not demonstrated spatial diversity.'}
    write(OUT/'final_method.json',result)
    return result

def metrics_dict(row):
    return {r:{m:float(row[f'{r}_{m}']) for m in METRICS} for r in REGIONS}

def results():
    original={r['Method']:r for r in rows(MAIN)}
    online=read(EV/'online_prediction_verification.json')['metrics']
    hinton=read(EV/'primary_probability_verification.json')['Hinton_recomputed_metrics']
    baseline=read(OUT/'local_predictions/baseline.json')['metrics']
    student=read(OUT/'local_predictions/student.json')['metrics']
    m2u=read(OUT/'local_predictions/multi-to-uni.json')['metrics']
    info=[('Target-only baseline','S0-Raw',baseline,BASE,OLD+'student_search_kd_v2.py','ours reference','baseline.json'),
          ('Logit distillation (adapted)','Hinton-KD-Port',hinton,'experiments/distillation_final/hinton_kd/Hinton-KD-Port_seed42_best.pt',
           'experiments/distillation_final/hinton_kd/train_hinton.py','Hinton2015',''),
          ('Online multimodal distillation (adapted)','MMOKD-Port',online,'experiments/GRSL_complete_paper_package/baselines/mmokd_port/student_asc_best.pt;experiments/GRSL_complete_paper_package/baselines/mmokd_port/student_desc_best.pt',
           'experiments/GRSL_complete_paper_package/baselines/mmokd_port/train_mmokd.py','Liu2024',''),
          ('Multi-to-Uni (adapted)','Multi2Uni-Port',m2u,M2U,'experiments/distillation_final/multi2uni_port/train_multi2uni.py','Sun2026Online','multi-to-uni.json'),
          ('Counter-guided search Student','S2-2Stage',student,STUDENT,CANON,'proposed','student.json')]
    main=[];baseaudit=[];conflicts=[];lookup={};sources=[]
    def resultrow(name,metrics,checkpoint,source,status):
        rr={'Method':name,'Protocol status':status,'Inference input':'Single target orbit, pre/post VV/VH + orbit indicator',
            'single_seed':True,'seed':42,'Prediction availability':True,'Source checkpoint/result file':checkpoint,'Metrics source':source}
        for r in REGIONS:
            for m in METRICS:rr[REGIONS[r]+' '+m]=metrics[r][m]
        return rr
    for name,key,metric,checkpoint,implementation,citation,newjson in info:
        source=(f'paper_release/provenance/{newjson}' if newjson else
                relative(EV/('online_prediction_verification.json' if key=='MMOKD-Port' else 'primary_probability_verification.json')))
        if newjson:copy(OUT/'local_predictions'/newjson,OUT/'provenance'/newjson)
        main.append(resultrow(name,metric,checkpoint,source,'reproduced_same_protocol'))
        lookup[name]=metric
        psource=(f'paper_release/local_predictions/{newjson.replace(".json",".npz")}' if newjson else
                 relative(EV/('Online_current_predictions.npz' if key=='MMOKD-Port' else 'Hinton_current_predictions.npz')))
        nota=('Original exact operators/coefficients not independently full-text verified; adaptation only. Physical microbatch 4, effective batch 16; separate ascending/descending deployment branches.' if key=='MMOKD-Port' else
              'Segmentation-only primary baseline has the same search Student architecture; historical trainer produced this arm with both KD coefficients zero.' if key=='S0-Raw' else
              'Task/backbone/loss adaptations; not published benchmark numerical reproduction' if citation not in ('proposed','ours reference') else 'Final geometry-free two-stage search distillation')
        baseaudit.append({'Method':name,'status':'reproduced_same_protocol','implementation':implementation,'checkpoint':checkpoint,'test_split':'data/splits_v3/test_ids.csv',
                          'inference_inputs':main[-1]['Inference input'],'metrics_source':source,'prediction_raster':psource,'can_recompute_current_masks':True,
                          'citation':citation,'notes':nota,'single_seed':True})
        orig=metrics_dict(original[key])
        for r in REGIONS:
            for m in METRICS:
                if abs(metric[r][m]-orig[r][m])>1e-10:
                    conflicts.append({'Method':name,'region':REGIONS[r],'metric':m,'recorded_summary':orig[r][m],'frozen_recomputed':metric[r][m],
                                      'difference':metric[r][m]-orig[r][m],'recorded_source':relative(MAIN),'authority_source':source,
                                      'decision':'Use exact frozen prediction recomputation. Preserve recorded result; do not mutate run summaries or manuscript.'})
    csvout(OUT/'results/main_comparison.csv',main)
    different=[]
    ext=['Boehm SAR U-Net++','CDNetE Early Fusion','DeepLabV3','MFEWF adapted','Standard U-Net']
    seatmap={'Boehm SAR U-Net++':'boehm','CDNetE Early Fusion':'cdnette','DeepLabV3':'deeplabv3','MFEWF adapted':'mfewf','Standard U-Net':'unet'}
    cites={'boehm':'Boehm2022','mfewf':'Chen2024MFEWF','deeplabv3':'Chen2017DeepLab','unet':'Ronneberger2015UNet','cdnette':'bibliographic mapping unverified; local early-fusion U-Net seat'}
    for name in ext:
        seat=seatmap[name];cp=f'experiments/single_orbit_baselines_v3/{seat}/best.pt'
        source=f'experiments/single_orbit_baselines_v3/{seat}/test_predictions.npz'
        metric=metrics_dict(original[name]);different.append(resultrow(name,metric,cp,relative(MAIN),'reproduced_different_protocol'))
        baseaudit.append({'Method':name,'status':'reproduced_different_protocol','implementation':'cocd/scripts/23_train_external_baselines.py','checkpoint':cp,
            'test_split':'data/splits_v3/test_ids.csv','inference_inputs':'Single target orbit, five channels','metrics_source':relative(MAIN),
            'prediction_raster':source,'can_recompute_current_masks':(ROOT/source).exists(),'citation':cites[seat],
            'notes':'Same split/masks/optimizer/selection; ceiling 200 vs final 100. Recorded stopping before 100 means ceiling was not reached; still not labelled an exact protocol match. Architectures/initialization differ; Boehm uses ResNet-34 instead of source ResNet-50.', 'single_seed':True})
    csvout(OUT/'results/protocol_different_comparison.csv',different)
    histpath=PACK/'evidence/earlier_package/qualitative/csv/table_historical_external_comparison.csv'
    historical=rows(histpath);dis=next(r for r in historical if 'DIS2' in str(r))
    write(OUT/'provenance/DIS2_historical_row.json',{'row':dis,'source':relative(histpath),'not_same_protocol':True,'current_region_metrics':None})
    csvout(OUT/'results/historical_reported_only.csv',[{'Method':'DIS2 (historical report only)','Protocol status':'historical_report_only',
        'All pixels AUPRC':0.8733278861052558,'All pixels IoU':0.6176884604656225,'All pixels F1':0.7636680010378782,
        'Target distorted, counter valid AUPRC':None,'Prediction availability':False,'Source checkpoint/result file':relative(histpath),
        'single_seed':None,'notes':'Original metric-producing checkpoint/probability lineage unavailable; no ranking against same-protocol reproduced results.'}])
    baseaudit.append({'Method':'DIS2 (historical report only)','status':'historical_report_only','implementation':'cocd/third_party/landslide_baselines/dis2_official',
        'checkpoint':None,'test_split':'Original row lineage not verified','inference_inputs':'historically reported single-orbit; exact artifact contract unverified',
        'metrics_source':relative(histpath),'prediction_raster':None,'can_recompute_current_masks':False,'citation':'Kieu2026DIS2',
        'notes':'Never substitute the low-scoring port or a different recovered DIS2 checkpoint for this historical row.'})
    for key,name,status in [('DIS2-Port','DIS2 local port (excluded)','reproduced_same_protocol'),('S2-Raw-native','Historical geometry-weighted Student (excluded)','obsolete')]:
        r=original[key]
        baseaudit.append({'Method':name,'status':status,'implementation':'experiments/distillation_final/dis2_port' if key=='DIS2-Port' else OLD+'student_search_kd_v2.py',
                         'metrics_source':relative(MAIN),'notes':'Not eligible for final main comparison; does not generate the manuscript historical DIS2 row' if key=='DIS2-Port' else 'Teacher and KD use geometry weighting; superseded protocol',
                         'can_recompute_current_masks':None,'citation':'Kieu2026DIS2' if key=='DIS2-Port' else 'proposed historical'})
    # Catalogue all remaining comparison rows in retained result tables; do not infer their comparability.
    known={r['Method'] for r in rows(MAIN)}
    extras=[]
    for p in sorted((ROOT/'experiments').rglob('*.csv')):
        if p.stat().st_size>2_000_000 or any(s in p.parts for s in ('data','dataset_audit_final','00_curation')):continue
        try:table=rows(p)
        except (UnicodeError,csv.Error):continue
        for r in table:
            key=next((r[k] for k in ('Method','method','Model','model') if r.get(k)),None)
            if key and key not in known and len(key)<100:
                known.add(key);extras.append({'Method':key,'status':'historical_report_only','metrics_source':relative(p),
                                             'notes':'Additional retained result-table entry, not final manuscript eligible; exact protocol/checkpoint lineage unverified.',
                                             'can_recompute_current_masks':None})
    csvout(OUT/'additional_historical_results_audit.csv',extras)
    baseaudit.append({'Method':'OGP-Net local adaptation (excluded)','status':'incomplete','implementation':'experiments/distillation_final/ogp_net_port',
                      'checkpoint':None,'metrics_source':'experiments/distillation_final/ogp_net_port/sanity_32_report.json',
                      'prediction_raster':None,'can_recompute_current_masks':False,'citation':'Sikdar2025',
                      'notes':'Only 32-pair sanity gate found; no completed benchmark summary/checkpoint. No paper metric inferred.'})
    csvout(OUT/'baseline_audit.csv',baseaudit)
    csvout(OUT/'result_source_conflicts.csv',conflicts or [{'Method':'none','decision':'No full-precision conflicts'}])
    guidance=[]
    armnames={'S2-2Stage:learned':'Student: learned sampling','S2-2Stage:center':'Student: fixed center','S2-2Stage:random':'Student: random offsets',
              'S2-2Stage:no-search':'Student: search bypass','CG0:paired':'Teacher: counter guidance','CG0:center':'Teacher: fixed center',
              'CG0:random':'Teacher: random offsets','CG0:no-search':'Teacher: search bypass','CG0:counter-shuffle':'Teacher: shuffled counter',
              'CG0:self-context':'Teacher: self context with frozen counter-trained weights','SG0:retrained-self-guidance':'Teacher: independently trained self guidance'}
    for p in (EV/'current_search_interventions.csv',EV/'current_sg0_metrics.csv'):
        for r in rows(p):
            guidance.append({'Method':armnames[r['arm']],'Region':REGIONS[r['region']],**{m:float(r[m]) for m in METRICS},
                             'pixels':int(r['pixels']),'single_seed':True,'seed':42,'evidence':r['evidence'],'source':relative(p),
                             'checkpoint':STUDENT if r['arm'].startswith('S2') else 'experiments/windows_main/teacher_target_search_v2/SG0_seed42.pt' if r['arm'].startswith('SG0') else TEACHER,
                             'interpretation':'Frozen inference intervention, not architecture retraining' if 'retrained' not in r['arm'] else 'Independently trained geometry-free self-guidance control'})
    csvout(OUT/'results/mechanism_guidance.csv',guidance)
    loss=[]
    for arm,name in [('Seg','Segmentation only'),('SegP','Policy distillation'),('SegE','Evidence distillation'),('both','Policy + Evidence')]:
        source=EV/f'loss_retraining/{arm}/summary.json' if arm!='both' else ROOT/'experiments/distillation_final/ours_s2_2stage/summary.json'
        metric=read(source)['metrics'] if arm!='both' else student
        checkpoint=relative(EV/f'loss_retraining/{arm}/best.pt') if arm!='both' else STUDENT
        loss.append(resultrow(name,metric,checkpoint,relative(source),'reproduced_same_protocol'))
        loss[-1].update({'policy_coefficient':0.5 if arm in ('SegP','both') else 0,'evidence_coefficient':0.5 if arm in ('SegE','both') else 0,
                         'control':'Same initialization/schedule; disabled loss weight not reallocated; combined row is original frozen Student, not a new retraining'})
    csvout(OUT/'results/distillation_ablation.csv',loss)
    return main,different,guidance,loss,lookup

def claim_audit(main,different,guidance,loss,lookup):
    audit=[]
    def add(claim,loc,metric,stated,verified,source,status=None):
        if status is None:
            status='PASS' if isinstance(verified,(int,float)) and abs(float(stated)-float(verified))<0.00501 else 'MISMATCH'
        audit.append({'claim':claim,'manuscript_location':loc,'metric':metric,'stated_value':stated,'authoritative_source':source,'verified_value':verified,'status':status})
    mappings={
      'current_primary_table.tex':{'Baseline':main[0],'SAR U-Net++':different[0],'Logit distillation':main[1],'Online distillation':main[2],
                                   'Multi-to-Uni':main[3],'DIS2':{'All pixels AUPRC':.8733278861052558,'All pixels IoU':.6176884604656225,'All pixels F1':.7636680010378782},'Ours':main[4]},
      'current_search_table.tex':{a:next(r for r in guidance if r['Method']==b and r['Region']=='Target-valid pixels') for a,b in
          [('Self-guided','Teacher: independently trained self guidance'),('Counter-guided','Teacher: counter guidance'),('Shuffled counter','Teacher: shuffled counter'),
           ('Learned sampling','Student: learned sampling'),('Fixed center','Student: fixed center'),('Random offsets','Student: random offsets'),('Without search','Student: search bypass')]},
      'current_loss_table.tex':{r['Method'].replace('Evidence','evidence') if r['Method']=='Policy + Evidence' else r['Method']:r for r in loss}}
    for filename,mapping in mappings.items():
        p=PACK/'manuscript/sections'/filename
        for line_no,line in enumerate(p.read_text().splitlines(),1):
            parts=line.split('&');label=re.sub(r'\\cite\{[^}]+\}','',parts[0]).strip()
            if label not in mapping:continue
            r=mapping[label];nums=re.findall(r'(?<![A-Za-z])\d+\.\d+', '&'.join(parts[1:]))
            regions=['Target-valid pixels']*(3 if filename=='current_primary_table.tex' else 2)+['All pixels']*(3 if filename=='current_primary_table.tex' else 2)
            ms=list(METRICS)*2 if filename=='current_primary_table.tex' else ['AUPRC','IoU']*2
            if label=='DIS2':regions=['All pixels']*3;ms=list(METRICS)
            for stated,reg,m in zip(nums,regions,ms):
                if filename=='current_search_table.tex':
                    rr=next(x for x in guidance if x['Method']==r['Method'] and x['Region']==reg);value=rr[m]
                else:value=r[reg+' '+m]
                add(label,relative(p)+':'+str(line_no),reg+' '+m,float(stated),value*100,
                    r.get('Metrics source',r.get('source','paper_release/results/historical_reported_only.csv')))
    # Every abstract percentage, in its textual order.
    p=PACK/'manuscript/main.tex';abstract=p.read_text().split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    nums=re.findall(r'(\d+\.\d+)\\%',abstract)
    expected=[('baseline','TargetValid','AUPRC'),('student','TargetValid','AUPRC'),('baseline','TargetValid','IoU'),('student','TargetValid','IoU'),
              ('baseline','Overall','AUPRC'),('student','Overall','AUPRC'),('baseline','Overall','IoU'),('student','Overall','IoU'),('baseline','G10','IoU'),('student','G10','IoU')]
    assert len(nums)==len(expected)
    metrics={n:read(OUT/f'provenance/{n}.json')['metrics'] for n in ('baseline','student')}
    for number,(n,r,m) in zip(nums,expected):add('Abstract '+n,relative(p)+':abstract',REGIONS[r]+' '+m,float(number),metrics[n][r][m]*100,f'paper_release/provenance/{n}.json')
    # Remaining prose percentages, including region triplets and historical values.
    p=PACK/'manuscript/sections/experiments.tex';text=p.read_text()
    for reg,r in [('BothValid','BothValid'),('G10','G10')]:
        phrase='88.15/63.72/77.84' if reg=='BothValid' else '89.96/65.64/79.26'
        base='86.91/61.34/76.04' if reg=='BothValid' else '88.82/64.14/78.15'
        for n,s in [('student',phrase),('baseline',base)]:
            assert s in text
            for m,v in zip(METRICS,s.split('/')):add('Regional prose '+n,relative(p),REGIONS[r]+' '+m,float(v),metrics[n][r][m]*100,f'paper_release/provenance/{n}.json')
    for m,v,a in [('AUPRC',87.33,.8733278861052558),('IoU',61.77,.6176884604656225)]:add('Historical DIS2 prose',relative(p),m,v,a*100,'paper_release/provenance/DIS2_historical_row.json')
    align=rows(EV/'current_search_alignment.csv')
    for a,v in zip([r for r in align if r['region']=='TargetValid'],[.0219,.0782,.0066,.0595]):
        add('Alignment diagnostic (candidate weights; not training spatial JS)',relative(p),a['metric'],v,float(a['mean']),relative(EV/'current_search_alignment.csv'),
            'PASS' if abs(v-float(a['mean']))<.0000501 else 'MISMATCH')
    for claim,stated,verified in [('spatial locations',1713,1713),('patch side',128,128),('batch size',16,16),('learning rate',5e-5,5e-5),('probability threshold',.5,.5)]:
        add(claim,relative(p),claim,stated,verified,'paper_release/protocol.json')
    mp=relative(PACK/'manuscript/sections/method.tex')
    for name,stated,actual in [('feature levels',3,3),('candidates per level',4,4),('joint candidates',12,12),('offset bound',2,2),('segmentation-only epochs',5,5),('KD starts epoch',6,6)]:
        add(name,mp,name,stated,actual,'paper_release/final_method.json')
    descriptions=[
      ('Policy distillation definition',mp,'align offsets and weights','align derived spatial sampling distributions by JS','MISMATCH'),
      ('Policy distance',mp,'Jensen-Shannon distance','Jensen-Shannon divergence; no square root','MISMATCH'),
      ('Evidence memory',mp,'same Student target memory','same detached projected Student target memory','PASS'),
      ('Evidence distance',mp,'cosine distance','1-cosine_similarity across channels, mean over queries and orbits','PASS'),
      ('KD gradient path',mp,'context/policy only','detached backbone features and target values; no direct KD to backbone/value projections','PASS'),
      ('Counter improves guidance',relative(p),'beneficial','counter/self Teacher and correct/shuffled frozen comparison support conditional benefit; single seed','PASS'),
      ('Random offsets reduce performance',relative(p),'reduce performance','frozen random-offset intervention reduces AUPRC/IoU','PASS'),
      ('Learned spatial displacement necessary',relative(p),'not established','fixed-center Student AP/IoU essentially unchanged; necessity unsupported','PASS'),
      ('Scale/weight allocation important',relative(p),'interpretation involving scale and weight allocation','observed allocation supports interpretation; causal importance not established without matched controls','UNVERIFIED'),
      ('Policy-only improves segmentation control',relative(p),'improves','higher AUPRC/IoU, one seed; no significance claim','PASS'),
      ('Evidence-only improves segmentation control',relative(p),'essentially unchanged','slightly lower AUPRC/IoU, no independent gain','PASS'),
      ('Combined highest in matched objective table',relative(p),'highest','highest AUPRC/IoU in both reported regions, single seed','PASS'),
      ('Primary table same-protocol eligibility',relative(p),'adapted comparisons in one table','SAR U-Net++ ceiling differs; historical DIS2 has no matching predictions. Keep separate from exact-match ranks.','MISMATCH'),
      ('Three learned points coincide',relative(p),'three coincident learned candidates','checkpoint rows identical at each scale in Teacher/Student','PASS'),
      ('Qualitative valid-pixel caption',relative(p)+':fig:qualitative','identical gray exclusions','target geometry > 0 excluded for every method; paired ID/orbit joins audited','PASS'),
      ('Qualitative full-pixel caption',relative(p)+':fig:full_comparison','no distorted prediction hidden','full crop without prediction masks; actual matched-loss probabilities','PASS'),
      ('Qualitative scientific case selection',relative(p)+':figures','examples','illustrative candidates; human selection/annotation QC pending','UNVERIFIED'),
      ('Geometry masks constitute distortion ground truth',mp,'geometric-distortion masks','existing static geometry codes; independent distortion validation absent','UNVERIFIED')]
    for claim,loc,s,v,status in descriptions:add(claim,loc,'mechanism/provenance',s,v,'paper_release/final_method.json;paper_release/figures/figure_audit.csv',status)
    csvout(OUT/'claim_audit.csv',audit)
    # Literal coverage index makes omissions visible; numbers in equations/citations are not benchmark claims.
    coverage=[]
    for source in [PACK/'manuscript/main.tex',*sorted((PACK/'manuscript/sections').glob('*.tex'))]:
        active=source.name in {'main.tex','introduction.tex','method.tex','experiments.tex','conclusion.tex','current_primary_table.tex','current_search_table.tex','current_loss_table.tex'}
        for ln,line in enumerate(source.read_text().splitlines(),1):
            if re.search(r'\d',line):coverage.append({'source':relative(source),'line':ln,'text':line,'current_included_source':active,
                'review':'Table cells/abstract/prose/settings audited in claim_audit; equation/citation/figure filenames are identifiers' if active else 'Unincluded draft source; not a current manuscript claim'})
    csvout(OUT/'manuscript_numeric_coverage.csv',coverage)
    return audit

def figures():
    assets=[('Figure 1: task scenario','fig_task_scenario.png','current retained scenario; original existing asset',None,'training paired; deployment single orbit','FINAL'),
            ('Figure 2: final architecture','fig_architecture_reader.pdf','editable/architecture_paper_reader.pptx',None,'target-side values; shared Student memory; geometry auxiliary disabled','FINAL'),
            ('Figure 3 candidate: valid-pixel comparison','fig_comparison_valid_reader.pdf','scripts/reader_figures.py','1061 descending;697 ascending','target geometry == 0','CURRENT_CANDIDATE'),
            ('Alternative qualitative candidate: full pixels','fig_comparison_full_reader.pdf','scripts/reader_figures.py','925 ascending;crop x[24,88),y[8,72)','all crop pixels; no exclusion','CURRENT_OPTIONAL'),
            ('Optional search diagnostic','fig_search_reader.pdf','scripts/reader_figures.py','925 ascending;query (57.5,49.5)','actual candidate positions and weights','CURRENT_OPTIONAL')]
    output=[]
    for label,name,script,ids,mask,status in assets:
        p=PACK/'manuscript/figures'/name;q=OUT/'figures'/name;copy(p,q)
        if p.suffix=='.pdf' and p.with_suffix('.png').exists():copy(p.with_suffix('.png'),q.with_suffix('.png'))
        output.append({'figure':label,'status':status,'source':relative(p),'script_or_editable_source':relative(PACK/script) if '/' in script else script,
                       'sample_ids_and_orbits':ids,'checkpoints':TEACHER+';'+STUDENT if 'architecture' in label or 'search' in label else
                       'paper_release/baseline_audit.csv;paper_release/results/distillation_ablation.csv' if ids else None,
                       'evaluation_mask':mask,'generated_file':relative(q),'matches_final_result_path':True,'sha256':sha(q),
                       'manual_case_review_pending':bool(ids),'note':'Candidate preservation only; final paper should choose one compact qualitative figure; no new diagram generated'})
    csvout(OUT/'figures/figure_audit.csv',output)
    copy(PACK/'editable/architecture_paper_reader.pptx',OUT/'figures/architecture_editable.pptx')
    for name in ('reader_figure_sources.json','figure_sources.json','intervention_protocol.json','learned_point_symmetry.json','reader_delivery_checks.json'):
        copy(EV/name,OUT/'provenance'/name)
    inventory=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or not p.resolve().is_relative_to(ROOT) or p.suffix.lower() not in ('.png','.pdf','.svg','.pptx') or any(x in p.parts for x in ('.git','.venv','data','paper_release','node_modules')):continue
        role='historical/not final paper eligible'
        if p.parent==PACK/'manuscript/figures' and p.name in {a[1] for a in assets}:role='current final/candidate; see figure_audit'
        elif 'dataset_audit_final' in p.parts:role='dataset audit/QC, not manuscript comparison evidence'
        inventory.append({'path':relative(p),'bytes':p.stat().st_size,'sha256':sha(p),'role':role})
    csvout(OUT/'figure_inventory.csv',inventory)

def source_inventory():
    final={v['path'] for v in read(OUT/'final_method.json')['code'].values()}
    inventory=[];deps=[]
    # Include all repository Python source, including retained experiment generators.
    paths=[p for p in ROOT.rglob('*.py') if p.resolve().is_relative_to(ROOT) and not any(x in p.parts for x in ('.git','__pycache__','.venv','data','node_modules'))]
    names={p.stem for p in paths}
    for p in sorted(paths):
        rel=relative(p);role='FINAL dependency' if rel in final else 'historical' if rel.startswith('archive/') else 'baseline' if any(x in rel for x in ('baselines','hinton_kd','multi2uni','dis2_port','mmokd')) else 'support/retained experiment generator'
        inventory.append({'path':rel,'sha256':sha(p),'bytes':p.stat().st_size,'role':role})
        try:tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        except (UnicodeError,SyntaxError) as e:
            deps.append({'source':rel,'import':None,'classification':'parse issue','note':str(e)});continue
        for node in ast.walk(tree):
            imp=node.module if isinstance(node,ast.ImportFrom) else None
            mods=[imp] if imp else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            for mod in mods:
                if mod:
                    deps.append({'source':rel,'import':mod,'classification':'local or dynamic local' if mod.split('.')[-1] in names or mod.startswith(('windows_main','models','common','paths','archive')) else 'external/stdlib',
                                 'source_role':role})
    csvout(OUT/'source_inventory.csv',inventory);csvout(OUT/'dependency_audit.csv',deps)
    write(OUT/'audit_scope.json',{'python_sources_scanned':len(paths),'figure_inventory':'figure_inventory.csv','additional_historical_result_entries':'additional_historical_results_audit.csv',
        'boundary':'All repository Python and figure paths inventoried, final closure source-reviewed and smoke-tested. Historical results with unresolved lineage are explicitly ineligible; no claim of full historical reproduction.',
        'protected':'No raw data/checkpoint/prediction overwrite; archive changes are byte-identical moves; manuscript source/PDF untouched.'})

def main():
    archive();p=protocol();method=final_method();main,diff,guidance,loss,lookup=results()
    claims=claim_audit(main,diff,guidance,loss,lookup);figures();source_inventory()
    print('release generated;',len(claims),'claim rows;',sum(r['status']=='MISMATCH' for r in claims),'mismatches',flush=True)

if __name__=='__main__':main()
