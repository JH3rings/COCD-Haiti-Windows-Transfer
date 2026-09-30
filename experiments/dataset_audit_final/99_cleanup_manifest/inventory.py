from pathlib import Path
import os,re,csv,hashlib,datetime
R=Path(__file__).resolve().parents[3]; O=R/'experiments/dataset_audit_final/99_cleanup_manifest'
kw=re.compile(r'audit|dataset|statistic|stats|mask|geometry|landslide|slope|aspect|overlap|g10|g01|change|analysis|distribution|quality|qc|visual',re.I)
ext={'.md','.txt','.json','.csv','.png','.jpg','.npy','.npz','.pkl','.py','.tif','.shp','.gpkg','.geojson'}
rows=[]
for base,dirs,files in os.walk(R):
 dirs[:]=[d for d in dirs if d not in {'.git','node_modules','__pycache__','dataset_audit_final'}]
 for name in files:
  p=Path(base)/name; rel=p.relative_to(R).as_posix()
  if p.suffix.lower() not in ext: continue
  content=False
  if p.suffix.lower() in {'.md','.txt','.py','.json','.csv'} and p.stat().st_size<2000000:
   try: content=bool(kw.search(p.read_text(encoding='utf-8',errors='ignore')))
   except OSError: pass
  if not kw.search(rel) and not content: continue
  protected=rel.startswith(('data/','checkpoints/','legacy_checkpoints/')) or any(x in rel.lower() for x in ['metrics','progress','train_log','training/','paper_package','grsl_','curve','validation'])
  dedicated=('/physics_audit_20260924/' in rel or '/dataset_forensic_audit_20260927/' in rel or rel in ['reports/final_haiti_dataset_audit.md','reports/directory_structure.md'] or rel.startswith('cocd/metadata/'))
  action='keep' if protected else ('merge' if dedicated else 'uncertain')
  st=p.stat(); h=hashlib.sha256(p.read_bytes()).hexdigest() if dedicated or (not protected and st.st_size<10000000) else ''
  rows.append(dict(path=rel,file_type=p.suffix,size_bytes=st.st_size,modified_time=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),inferred_purpose='protected data/model experiment' if protected else ('dataset audit/provenance' if dedicated else 'keyword candidate; semantic relevance unconfirmed'),sha256=h,duplicate_group='',whether_superseded='pending comparison',action=action,reason='protected boundary' if protected else 'retain until reviewed',protected=protected))
hashes={}
for x in rows:
 if x['sha256']:hashes.setdefault(x['sha256'],[]).append(x)
for h,g in hashes.items():
 if len(g)>1:
  for x in g:x['duplicate_group']=h[:16]
with (O/'all_existing_dataset_analysis.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Inventory',len(rows),'files; no deletion')
