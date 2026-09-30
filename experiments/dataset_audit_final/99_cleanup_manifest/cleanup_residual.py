"""Consolidate remaining old dataset-audit debris after manifest and hash checks."""
from pathlib import Path
import csv,hashlib,json,shutil
R=Path(__file__).resolve().parents[3];F=R/'experiments/dataset_audit_final';M=F/'99_cleanup_manifest'
oldrun=R/'experiments/windows_main/dataset_forensic_audit_20260927';oldphys=R/'experiments/windows_main/physics_audit_20260924'
assert oldrun.resolve().is_relative_to(R.resolve()) and oldphys.resolve().is_relative_to(R.resolve())
assert (oldphys/'dem').is_dir() and not list((oldphys/'dem').iterdir())
files=[oldrun/'run.log',*sorted((oldrun/'__pycache__').glob('*.pyc')),
 R/'reports/final_haiti_dataset_audit.md',R/'reports/directory_structure.md',
 R/'archive/history/reports/haiti_dataset_origin_and_convergence.md']
assert len(files)==7,files
record=[]
for p in files:
 assert p.exists() and p.resolve().is_relative_to(R.resolve())
 rel=p.relative_to(R).as_posix()
 if rel.startswith('reports/'):
  dest=F/'05_existing_results_reused/reports'/p.name
 elif rel.startswith('archive/'):
  dest=F/'05_existing_results_reused'/rel
 else:dest=None
 if dest is not None:
  assert dest.is_file() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest(),rel
 record.append(dict(path=rel,size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='duplicate historical report already preserved in final' if dest else 'failed-run log or regenerable Python bytecode; no scientific content',replaced_by=dest.relative_to(R).as_posix() if dest else '',role='old dataset audit residue',references='links updated; source scripts preserved in final',process_status='no Python command uses this old audit path',backup=dest.relative_to(R).as_posix() if dest else '',deleted=True))
# Verify the exact target set before writing the deletion manifest.
assert {p.name for p in oldrun.iterdir()}=={'run.log','__pycache__'}
assert set((oldrun/'__pycache__').iterdir())==set(files[1:4])
with (M/'residual_cleanup_manifest.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.DictWriter(h,fieldnames=record[0]);w.writeheader();w.writerows(record)
for p in files:
 assert hashlib.sha256(p.read_bytes()).hexdigest()==next(x['sha256'] for x in record if x['path']==p.relative_to(R).as_posix())
 p.unlink()
(oldrun/'__pycache__').rmdir();oldrun.rmdir();(oldphys/'dem').rmdir();oldphys.rmdir()
# Incorporate into the previous deletion record and inventory.
p=M/'deleted_files.csv'
with p.open('a',newline='',encoding='utf-8') as h:
 w=csv.DictWriter(h,fieldnames=record[0]);w.writerows(record)
inv=M/'all_existing_dataset_analysis.csv'
import pandas as pd
d=pd.read_csv(inv);newpaths={r['path'] for r in record}
d.loc[d.path.isin(newpaths),'action']='delete';d.loc[d.path.isin(newpaths),'whether_superseded']='yes; duplicate or failed-run residue';d.loc[d.path.isin(newpaths),'reason']='removed after preservation/verification; see residual_cleanup_manifest.csv';d.to_csv(inv,index=False,encoding='utf-8-sig')
d[d.action=='keep'].to_csv(M/'kept_files.csv',index=False,encoding='utf-8-sig')
d[d.action=='uncertain'].to_csv(M/'uncertain_files.csv',index=False,encoding='utf-8-sig')
state=json.loads((M/'cleanup_execution.json').read_text());state['residual_files_deleted']=len(record);state['total_deleted_files']=state['deleted_files']+len(record);state['old_audit_directories_removed']=2;(M/'cleanup_execution.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
print('Deleted',len(record),'residual files and 2 empty old audit directories')
