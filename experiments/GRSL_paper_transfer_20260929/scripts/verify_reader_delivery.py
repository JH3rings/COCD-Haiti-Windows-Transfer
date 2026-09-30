"""Verify published terminology, literature count, placement, and retained method."""
from pathlib import Path
import json,re,pymupdf
PACK=Path(__file__).resolve().parents[1];MS=PACK/'manuscript';OUT=PACK/'new_evidence/20260930'
doc=pymupdf.open(PACK/'preview/GRSL_updated_20260930.pdf')
assert len(doc)==5
text='\n'.join(p.get_text() for p in doc)
codes=['G10','G_{10}','CG0','SG0','S2-2Stage','001061','000697','000925','MMOKD-Port','Multi2Uni-Port']
remaining=[w for w in codes if w in text];assert not remaining
assert 'Fig. 1.' in doc[1].get_text() and 'Paired ascending and descending' in doc[1].get_text()
assert 'References' in doc[-1].get_text()
assert all('References' not in p.get_text() for p in list(doc)[:-1])
assert len(re.findall(r'\\bibitem',(MS/'main.bbl').read_text()))==23
method=(MS/'sections/method.tex').read_text()
assert method.count(r'\begin{figure*}')==1 and method.count(r'\label{fig:scenario}')==1
labels=re.findall(r'\\label\{(eq:[^}]+)\}',method)
old_labels=re.findall(r'\\label\{(eq:[^}]+)\}',(OUT/'before_reader_revision/method.tex').read_text())
assert labels==old_labels and len(labels)==6
static=json.loads((OUT/'latex_source_checks.json').read_text())
assert not static['duplicate_labels'] and not static['missing_labels'] and not static['missing_citation_keys']
assert all(x['exists'] for x in static['includes'])
online=json.loads((OUT/'online_prediction_verification.json').read_text())
assert online['cached_probability_max_abs_difference']==0 and online['threshold_prediction_disagreements']==0
report=json.loads((OUT/'delivery_verification.json').read_text())
assert not report['overfull_warnings']
checks={'actual_cited_references':23,'prior_actual_cited_references':16,'scenario_page':2,
        'internal_codes_in_published_text':remaining,'masked_comparison_methods':5,
        'full_pixel_comparison_methods':5,'matched_objective_variants':4,'search_table_rows':7,
        'method_equations_retained':6,'references_last':True,'online_checkpoint_to_cache_exact':True}
report['human_visual_review']='Assistant inspected five rendered pages; scientific scene adjudication remains pending.'
report['strict_page_limit']=5;report['reader_revision']=checks
(OUT/'delivery_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'reader_delivery_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
p=PACK/'CONTINUATION_REPORT_20260930.md';s=p.read_text()
s+='''
## Publication-reader revision

The formal paper remains five pages. The task schematic starts page 2; only the detailed architecture spans both columns. All 23 bibliography entries are now cited in substantive related work and dataset discussion. The former PDF contained 16 actual references. All six core equations and Teacher/Student paths are retained.

The masked comparison shows both orbit backscatter images, both distortion regions, landslide reference, support, and five methods. The full-pixel comparison retains distorted predictions and shows five methods plus four matched objective variants. Table II now has seven rows in two groups. Published names are consistent across text, figures, and tables. Internal model/sample/region codes are confined to provenance files. All measured controls and four-region numerical records remain in the evidence CSVs.

Retained online-distillation branch checkpoints reproduce the cached probabilities exactly, with zero threshold prediction disagreements. Recomputed summary metrics differ slightly from the recorded source summary, but all displayed table cells agree at two-decimal percent precision; recorded main-table values were preserved. Sources: reader_figure_sources.json, online_prediction_verification.json, and reader_revision_manifest.json.

The bibliography was restored from the existing 23-entry library. Selected metadata and method summaries were checked on primary publisher pages; this is not a claim that all article full texts were audited. Human scientific image adjudication remains pending.
'''
p.write_text(s,encoding='utf-8')
print('PUBLICATION_CHECKS_PASS: 5 pages, 23 actual references, scenario on page 2, six equations retained')
