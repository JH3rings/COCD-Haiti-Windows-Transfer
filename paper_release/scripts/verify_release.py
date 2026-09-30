"""Smoke and integrity checks; no optimizer step, training, or manuscript edit."""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys
import numpy as np
import torch
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper_release'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import DEV,metric_rows
from windows_main.data import HaitiPairs,split
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);checks={}
    p=json.loads((OUT/'protocol.json').read_text())
    tr,va,te=split();assert len(tr)==1370 and not va and len(te)==343 and not set(tr)&set(te)
    checks['split_counts_disjoint']='PASS'
    ds=HaitiPairs(te[:1],stats=(np.array(p['normalization']['mean']),np.array(p['normalization']['std'])))
    a,d,y,ga,gd,real=ds[0];assert a.shape==d.shape==(5,128,128) and y.shape==(128,128)
    assert torch.all(a[4]==0) and torch.all(d[4]==1)
    checks['dataset_loader_real_location']='PASS'
    a,d=a[None].to(DEV),d[None].to(DEV)
    teacher=TargetSearchTeacherV2().to(DEV).eval();student=StudentTargetSearchV2().to(DEV).eval()
    method=json.loads((OUT/'final_method.json').read_text())
    for name,model in [('Teacher',teacher),('Student',student)]:
        ck=ROOT/method['checkpoints'][name]['path'];assert sha(ck)==method['checkpoints'][name]['sha256']
        model.load_state_dict(torch.load(ck,map_location=DEV,weights_only=True),strict=True)
    with torch.no_grad():
        to=teacher(a,d);so=student(a)
        assert to['z'].shape==so['z'].shape==(1,1,128,128)
        assert torch.isfinite(to['z']).all() and torch.isfinite(so['z']).all()
        alternate=teacher(a,torch.zeros_like(d))
        assert all(torch.equal(x,v) for x,v in zip(to['values'],alternate['values']))
        assert not torch.equal(to['policy']['weights'],alternate['policy']['weights'])
        assert torch.equal(to['z'],teacher.forward_from_cached_R(to['R'],(128,128)))
    checks.update({'final_Teacher_forward':'PASS','final_Student_forward':'PASS','counter_affects_policy_not_values':'PASS','decoder_uses_retrieved_values_only':'PASS'})
    source=ROOT/method['code']['Student_training_and_KD']['path']
    spec=importlib.util.spec_from_file_location('frozen_kd',source);kd=importlib.util.module_from_spec(spec);spec.loader.exec_module(kd)
    for parm in teacher.parameters():parm.requires_grad_(False)
    lp,le,oa,od=kd.kd_terms(student,teacher,a,d)
    assert torch.isfinite(lp) and torch.isfinite(le)
    grads=torch.autograd.grad(lp+le,list(student.parameters()),allow_unused=True)
    active=[n for (n,_),g in zip(student.named_parameters(),grads) if g is not None]
    assert active and all(n.startswith(('context.','policy.')) for n in active),active
    checks['KD_gradient_restricted_to_context_policy']='PASS'
    # Confirm the implemented shared-memory evidence expression exactly.
    with torch.no_grad():
        ta,td=teacher.forward_pair(a,d,'counter');terms=[]
        for o,t in ((oa,ta),(od,td)):
            mem=tuple(v.detach() for v in o['values'])
            pol=student.policy(student.context(tuple(v.detach() for v in o['T'])))
            terms.append((1-torch.nn.functional.cosine_similarity(student.search(mem,pol),student.search(mem,t['policy']),dim=1)).mean())
        assert torch.allclose(le,.5*(terms[0]+terms[1]),atol=1e-7)
    checks['same_Student_memory_cosine_evidence']='PASS'
    pred=(torch.sigmoid(so['z'][:,0]).detach().cpu().numpy(),y.numpy()[None],ga.numpy()[None],gd.numpy()[None])
    report=metric_rows(pred);assert set(report)=={'Overall','BothValid','TargetValid','G10','G01'}
    checks['evaluation_metrics_smoke']='PASS'
    for name in ('baseline','student','multi-to-uni'):
        r=json.loads((OUT/f'provenance/{name}.json').read_text());assert r['locations']==343 and not r['smoke_only']
    checks['complete_frozen_checkpoint_evaluation_343_locations']='PASS'
    with (OUT/'archive_manifest.csv').open() as f:
        archived=list(csv.DictReader(f))
    assert all(sha(ROOT/r['archived'])==r['sha256_before'] for r in archived)
    checks['archive_moves_byte_identical']='PASS'
    manuscript=ROOT/'experiments/GRSL_paper_transfer_20260929/manuscript'
    before='archive/pre-grsl-paper-freeze-20260930'
    import subprocess
    for source in [manuscript/'main.tex',manuscript/'references.bib',*list((manuscript/'sections').glob('*.tex'))]:
        rel=str(source.relative_to(ROOT)).replace('\\','/')
        original=subprocess.check_output(['git','show',before+':'+rel])
        assert original.replace(b'\r\n',b'\n')==source.read_bytes().replace(b'\r\n',b'\n'),rel
    checks['manuscript_text_and_bibliography_unchanged']='PASS'
    pdf=ROOT/'experiments/GRSL_paper_transfer_20260929/preview/GRSL_updated_20260930.pdf'
    doc=pymupdf.open(pdf);assert len(doc)==5
    checks['existing_PDF_five_pages']='PASS'
    original=subprocess.check_output(['git','show',before+':'+str(pdf.relative_to(ROOT)).replace('\\','/')])
    assert hashlib.sha256(original).hexdigest()==sha(pdf)
    checks['PDF_byte_unchanged']='PASS'
    assert (OUT/'context_references.bib').read_text().count('@')==23
    checks['all_23_context_references_retained']='PASS'
    result={'checks':checks,'all_passed':True,'archive_files':len(archived),'scientific_manual_review':'pending; not inferred from smoke tests',
            'no_training':True,'no_optimizer_step':True,'device':str(DEV),'torch':torch.__version__,'paper_pdf_sha256':sha(pdf)}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
