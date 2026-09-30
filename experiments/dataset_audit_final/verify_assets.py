from pathlib import Path
import pandas as pd,hashlib,json
R=Path(__file__).resolve().parents[2];O=R/'experiments/dataset_audit_final/99_cleanup_manifest'
d=pd.read_csv(R/'data/haiti/DATA_MANIFEST.csv');bad=[]
for i,r in d.iterrows():
 p=R/'data/haiti'/r.relative_path
 if not p.exists():bad.append({'path':r.relative_path,'problem':'missing'});continue
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=r.sha256:bad.append({'path':r.relative_path,'problem':'sha256 mismatch'})
 if i%3000==0:print('Hash verify',i,flush=True)
files=[]
for base in ['data/splits','data/splits_v3','cocd/manifests','checkpoints','legacy_checkpoints']:
 for p in (R/base).rglob('*'):
  if p.is_file():files.append(dict(path=p.relative_to(R).as_posix(),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix in ['.csv','.json'] else None))
pd.DataFrame(files).to_csv(O/'protected_assets_before.csv',index=False)
(O/'data_hash_verification.json').write_text(json.dumps(dict(n_files=len(d),n_mismatch=len(bad),mismatches=bad),indent=2))
print('Hash complete',len(d),len(bad),flush=True)
