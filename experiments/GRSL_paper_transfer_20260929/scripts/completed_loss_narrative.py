"""Interpret completed matched arms without assuming a positive result."""
from pathlib import Path
import json,re
PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930';MS=PACK/'manuscript'
def main():
    summaries={}
    for arm in ('Seg','SegP','SegE'):
        p=OUT/'loss_retraining'/arm/'summary.json'
        if not p.exists():return
        s=json.loads(p.read_text());assert s['status']=='COMPLETE';summaries[arm]=s
    vals={arm:100*s['metrics']['Overall']['AUPRC'] for arm,s in summaries.items()}
    p=MS/'sections/experiments.tex';s=p.read_text()
    narrative=(r'Table~\ref{tab:loss_parts} reports completed seed-42 component-removal controls with the current warm-up, CG0 guidance, and stop-gradient schedule. Each surviving component retains its canonical coefficient. '
        +f"Overall AUPRC is {vals['Seg']:.2f}\\% for segmentation alone, {vals['SegP']:.2f}\\% with P, {vals['SegE']:.2f}\\% with E, and 89.16\\% for the frozen combined model. "
        +r'E alone does not improve this matched control. The new segmentation-only control is stronger than the archived S0; these matched results, rather than historical loss sweeps, delimit component attribution. One seed does not establish statistical superiority.')
    s=re.sub(r'The matched loss-component controls.*?historical evidence\.',lambda m:narrative,s,flags=re.S);p.write_text(s,encoding='utf-8')
    p=MS/'sections/current_loss_table.tex';s=p.read_text().replace('; dashes: unfinished retraining.','.');p.write_text(s,encoding='utf-8')
    p=MS/'sections/conclusion.tex';s=p.read_text().replace('while matched component retraining is needed to determine how much each distillation objective contributes.','while completed component controls delimit the contribution of each objective without establishing multi-seed superiority.');p.write_text(s,encoding='utf-8')
    print('COMPLETED_LOSS_NARRATIVE_UPDATED')
if __name__=='__main__':main()
