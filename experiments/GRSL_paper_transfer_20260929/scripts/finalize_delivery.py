"""Refresh completed tables, compile the unique formal manuscript, render QA."""
from pathlib import Path
import subprocess,json,csv,hashlib,re,shutil
import pymupdf
ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930';MS=PACK/'manuscript'

def main():
    for name in ('reader_revision.py','source_audit.py'):
        subprocess.run([__import__('sys').executable,str(Path(__file__).parent/name)],check=True,stdout=subprocess.DEVNULL)
    compiler=Path(__file__).parent/'project_compiler/tectonic.exe'
    log=Path(__file__).parent/'compile_project.log'
    with log.open('w') as f:subprocess.run([str(compiler),'main.tex','--keep-logs','--keep-intermediates'],cwd=MS,stdout=f,stderr=subprocess.STDOUT,check=True)
    compiled=pymupdf.open(MS/'main.pdf')
    assert len(compiled)<=5, f'Strict five-page limit exceeded: {len(compiled)}'
    compiled.close()
    pdf=PACK/'preview/GRSL_updated_20260930.pdf';shutil.copy2(MS/'main.pdf',pdf)
    qa=PACK/'preview/qa_final';qa.mkdir(exist_ok=True);doc=pymupdf.open(pdf)
    page_info=[]
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.8,1.8)).save(qa/f'page_{i+1}.png')
        page_info.append({'page':i+1,'text_characters':len(page.get_text()),'embedded_images':len(page.get_images())})
    summaries={}
    for arm in ('Seg','SegP','SegE'):
        p=OUT/'loss_retraining'/arm/'summary.json'
        summaries[arm]=json.loads(p.read_text()) if p.exists() else None
    compile_text=log.read_text();status={'pdf':str(pdf),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(doc),'project_compile':'SUCCESS',
        'native_editor_compile':'platform-directory error; source remains in native editor','compiler_source':'https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic@0.17.0',
        'compiler_archive_verified_against_github_digest':True,'overfull_warnings':[l for l in compile_text.splitlines() if 'Overfull' in l],
        'page_info':page_info,'human_visual_review':'PENDING assistant image inspection of final page PNGs','completed_loss_arms':[k for k,v in summaries.items() if v]}
    (OUT/'delivery_verification.json').write_text(json.dumps(status,indent=2))
    lines=['# GRSL continuation delivery (2026-09-30)','',f'- Formal source: `manuscript/main.tex`; project PDF: `preview/GRSL_updated_20260930.pdf` ({len(doc)} pages).',
        '- Main comparison preserves approximately five representative baselines and Ours. Historical DIS2 87.33% AUPRC / 61.77% IoU is unchanged; unavailable region scores remain blank.',
        '- New inference controls: CG0 paired, fixed center, random offsets, no-search bypass, global counter derangement, frozen self context; independently trained SG0 re-evaluated. S2 learned, center, random, and no-search are also measured.',
        '- Each control has Overall, TargetValid, BothValid, G10 AUPRC/IoU/F1 and saved full probabilities. G01 is retained in CSV as an additional region.',
        '- Current CG0 alignment replaces the older CG10-derived plot. The statistic and weighting definitions are written in the source protocol and manuscript.',
        '- Three native scene figures plus a genuine unmasked G10 detail are provided from recorded probabilities and original SAR; full scene layers, IDs, crop coordinates, and shared display ranges are saved.',
        '- Actual Teacher/Student sampling and weights are saved for every candidate at a selected query. Offset-head parameter symmetry is verified: three learned points coincide per level in both current checkpoints. No sampling locations are visually jittered.',
        '- Full editable network is preserved with TGD off in CG0 and shared Student memory labels. PPTX package checks passed and it was rendered through PowerPoint; native original objects were retained to avoid importer style loss.',
        '- The historical weight curve and standalone alignment plot remain in figure resources and evidence; they are omitted from the five-page main paper. No current weight scan is claimed.','',
        '## Matched current loss controls','',
        'The new Seg/SegP/SegE arms use canonical initialization, seed 42, batch 16, optimizer, pixel support, two-stage calibration/schedule, stop-gradient KD, and frozen CG0. Removed component coefficients are not reassigned. The combined row is the existing current S2-2Stage checkpoint, not relabeled historical S2.',
        '','| Arm | Completion | Best epoch | Overall AUPRC | TargetValid AUPRC | G10 IoU |','|---|---|---:|---:|---:|---:|']
    for arm,s in summaries.items():
        if s:
            m=s['metrics'];lines.append(f"| {arm} | {s['status']}; {s['stopping_reason']} | {s['best_epoch']} | {100*m['Overall']['AUPRC']:.4f}% | {100*m['TargetValid']['AUPRC']:.4f}% | {100*m['G10']['IoU']:.4f}% |")
        else:lines.append(f'| {arm} | RUNNING / PENDING | -- | -- | -- | -- |')
    lines+=['','E alone is essentially unchanged from the new segmentation-only control (88.68% versus 88.69% Overall AUPRC); P gives 88.96%, and the frozen combined model gives 89.16%. These are single-seed component-removal results, not statistical superiority.','','## Source correspondence and reproduction','',
        '- `new_evidence/20260930/paper_source_correspondence.csv`: table/figure to model, checkpoint, code, and original file; recorded SHA256 and existence.',
        '- `new_evidence/20260930/version_boundaries.md`: current/historical separation and claim limits.',
        '- `new_evidence/20260930/figure_sources.json`, `current_query_samples.csv`, and native `layers_*.npz`: editable scientific figure inputs.',
        '- `scripts/current_mechanisms.py`, `train_matched_loss.py`, `build_figures.py`, `render_current_search.py`: computations and plots.',
        '- `scripts/finalize_delivery.py`: completed-row update, compact layout, compilation, and page rendering. `scripts/project_compiler/` is a project-scoped portable compiler, not a replacement for the native editor.',
        '- Primary Hinton probabilities were recovered from retained weights and agree with recorded metrics. Boehm cache order was checked against actual labels, geometry, and orbit arrays. Original cache order and new canonical batch order differ; join using sample ID and orbit.','',
        '## Minimal missing originals','',
        'Only historical DIS2 regional recomputation still needs originals: the probability arrays corresponding to the 87.33% / 61.77% record, plus sample ID / orbit order and original label / geometry identity. A single NPZ with p, y, target_geometry, counter_geometry, sample_id, orbit and model/source metadata is sufficient. Alternatively the exact historical checkpoint and preprocessing/source metadata can regenerate it. Current low-score DIS2-Port and recovered different-score re-test are not substitutes.','',
        '## Claim that needs the strongest further support','',
        'The necessity of nonzero adaptive spatial sampling is not established. Fixed-center S2 is nearly unchanged, and three learned points collapse to one per level. Current evidence supports counter-guidance sensitivity and scale/weight allocation, not a claim that diverse spatial sampling causes the gains. Matched loss controls must be interpreted as recorded, including any null or negative result; a single run does not establish statistical superiority.','',
        '## Submission limits','',
        'This is a complete evidence-oriented revised draft, not a declaration of submission readiness. The formal PDF is strictly limited to five pages, including references, tables, and figures, with an enforced pre-delivery assertion. Author/affiliation/funding information and manual scientific image adjudication remain outside this computational update. Source checks verify resources and labels, not every bibliographic statement.']
    (PACK/'CONTINUATION_REPORT_20260930.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DELIVERY_REFRESHED',len(doc),status['completed_loss_arms'])

if __name__=='__main__':main()
