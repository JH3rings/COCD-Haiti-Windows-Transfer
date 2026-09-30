from pathlib import Path
import pandas as pd,json,hashlib
from PIL import Image
R=Path(__file__).resolve().parents[2];F=R/'experiments/dataset_audit_final';C=F/'99_cleanup_manifest'
idx=pd.read_csv(F/'03_manual_qc/manual_qc_index.csv');key=pd.read_csv(F/'03_manual_qc/blind_ground_truth_key.csv');form=pd.read_csv(F/'03_manual_qc/manual_qc_form.csv');ar=pd.read_csv(F/'02_tables/anomaly_summary.csv');df=pd.read_csv(F/'02_tables/dataset_patch_audit.csv')
assert len(idx)==46 and len(key)==20 and len(df)==1713 and df.sample_id.nunique()==1713
assert form.drop(columns='sample_id').isna().all().all()
for _,r in idx.iterrows():assert (F/'03_manual_qc'/r.contact_sheet).exists()
for _,r in key.iterrows():assert (F/'03_manual_qc/blind'/f'{r.blind_id}.png').exists()
for _,r in ar.iterrows():assert (F/f'04_anomaly_cases/type_{r.type}/patch{int(r.sample_id):04d}.png').exists()
images=list(F.rglob('*.png'))
for p in images:
 with Image.open(p) as im:im.verify()
moves=pd.read_csv(C/'move_manifest.csv')
for _,r in moves.iterrows():
 assert not (R/r.source).exists()
 assert hashlib.sha256((R/r.destination).read_bytes()).hexdigest()==r.sha256
for _,r in pd.read_csv(C/'deleted_files.csv').iterrows():assert not (R/r.path).exists()
new=pd.read_csv(C/'all_existing_dataset_analysis.csv');counts=new.action.value_counts().to_dict()
summary=dict(patches=len(df),contact_sheets=len(idx),unique_manual_patches=len(form),blind_sheets=len(key),anomaly_rankings=ar.groupby('type').size().to_dict(),pngs_decoded=len(images),manual_fields_all_blank=True,archived_files=len(moves),deleted_files=len(pd.read_csv(C/'deleted_files.csv')),inventory_dispositions=counts)
(C/'final_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(pd.read_csv(F/'01_statistics/cross_orbit_support_statistics.csv').to_string(index=False))
print('package MB',sum(p.stat().st_size for p in F.rglob('*') if p.is_file())/1024**2)
