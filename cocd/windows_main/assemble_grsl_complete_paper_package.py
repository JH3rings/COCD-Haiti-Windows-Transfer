"""Non-destructively assemble the final quantitative and qualitative GRSL packages."""
from __future__ import annotations
import json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];exp=ROOT/'experiments';q=exp/'GRSL_final_experiment_package';v=exp/'GRSL_qualitative_package';out=exp/'GRSL_complete_paper_package'
for source,name in ((q,'quantitative'),(v,'qualitative')):shutil.copytree(source,out/name,dirs_exist_ok=True)
(out/'historical_external').mkdir(parents=True,exist_ok=True)
for source in (q/'csv/table_historical_external_comparison.csv',q/'historical_baseline_comparability.md',q/'json/historical_comparison_manifest.json'):
 shutil.copy2(source,out/'historical_external'/source.name)
(out/'README.md').write_text('# GRSL complete paper package\n\nThis is the single entry point for the completed Haiti SAR GRSL materials.\n\n- `quantitative/`: final-v2 Teacher/Student tables, loss ablations, figure data, source hashes, and final-protocol checklist.\n- `qualitative/`: reproducible Overall, TGD-local, and OmegaTR figures, crops, overlays, captions, and scene/crop metrics.\n- `historical_external/`: retained external-method metric records with explicit comparability status. These rows must not be pooled into the final-v2 main ranking.\n\nUse `quantitative/paper_summary.md` and `qualitative/qualitative_summary.md` when drafting. The original two packages remain unchanged as provenance-preserving source packages.\n',encoding='utf-8')
(out/'package_index.json').write_text(json.dumps({'quantitative_source':str(q),'qualitative_source':str(v),'historical_source':str(exp/'windows_main/da_search/grsl_assets_v1'),'policy':'Final-v2 results are primary. Historical external metrics are retained separately with eligibility labels.'},indent=2),encoding='utf-8')
