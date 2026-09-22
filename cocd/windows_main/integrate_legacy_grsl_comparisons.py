"""Copy historical external comparison records with explicit comparability labels."""
from __future__ import annotations
import csv,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];old=ROOT/'experiments/windows_main/da_search/grsl_assets_v1';src=old/'tables/table1_deployment.csv'
targets=[ROOT/'experiments/GRSL_final_experiment_package',ROOT/'experiments/GRSL_qualitative_package']
with src.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
status={'SO-reference':'historical protocol-matched reference; final-v2 SO row remains MISSING','DIS2':'historical CSV only; current checkpoint and pixel cache missing','Vanilla KD':'historical CSV only; current checkpoint and pixel cache missing','Boehm SAR U-Net++':'historical local prediction cache; not re-evaluated under final-v2 protocol','CDNetE Early Fusion':'historical local prediction cache; not re-evaluated under final-v2 protocol','MFEWF adapted':'historical local prediction cache; not re-evaluated under final-v2 protocol','S0':'legacy raw-feature-bypass Student; excluded from final-v2 claims'}
for r in rows:r['evidence_status']=status[r['Method']];r['source_package']='windows_main/da_search/grsl_assets_v1';r['eligible_for_final_v2_main_ranking']='no'
for out in targets:
 (out/'csv').mkdir(exist_ok=True);p=out/'csv'/'table_historical_external_comparison.csv'
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 shutil.copy2(old/'baseline_comparability.md',out/'historical_baseline_comparability.md')
 (out/'previous_packages_index.md').write_text('# Prior package index\n\n- `experiments/windows_main/da_search/grsl_assets_v1`: historical GRSL assets, external-method metric table and legacy DA Student materials.\n- `experiments/GRSL_final_experiment_package`: final-v2 quantitative Teacher/Student package.\n- `experiments/GRSL_qualitative_package`: final-v2 reproducible qualitative figures.\n\nThe historical external CSV is copied into `csv/table_historical_external_comparison.csv` with per-method eligibility labels. Do not merge historical and final-v2 rows into a single statistical ranking.\n',encoding='utf-8')
 (out/'json').mkdir(exist_ok=True);(out/'json'/'historical_comparison_manifest.json').write_text(json.dumps({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'policy':'Historical values retained verbatim with comparability status; they are not combined with final-v2 ranking or qualitative figures.','methods':status},indent=2),encoding='utf-8')
