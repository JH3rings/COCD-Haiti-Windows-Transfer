"""Verify bibliography identifiers using primary metadata; preserve manuscript bib."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,re,requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper_release'
SOURCE=ROOT/'experiments/GRSL_paper_transfer_20260929/manuscript/references.bib'

def fetch(doi):
    url='https://api.crossref.org/works/'+doi
    try:
        r=requests.get(url,timeout=30);r.raise_for_status();m=r.json()['message']
        return {'doi':doi,'url':url,'status':'verified_primary_metadata','metadata':m}
    except Exception as e:return {'doi':doi,'url':url,'status':'unverified_network','error':str(e)}

def main():
    raw=SOURCE.read_text();entries=re.split(r'(?=@\w+\{)',raw);entries=[e.strip() for e in entries if e.strip()]
    pairs=[(re.search(r'@\w+\{([^,]+),',e)[1],e) for e in entries]
    dois=[re.search(r'doi\s*=\s*\{([^}]+)',e,re.I)[1] for _,e in pairs if re.search(r'doi\s*=\s*\{([^}]+)',e,re.I)]
    with ThreadPoolExecutor(max_workers=6) as ex:metadata=list(ex.map(fetch,dois))
    (OUT/'provenance/reference_doi_metadata.json').write_text(json.dumps(metadata,indent=2))
    audit=[];bydoi={r['doi'].lower():r for r in metadata}
    for key,e in pairs:
        d=re.search(r'doi\s*=\s*\{([^}]+)',e,re.I);m=bydoi[d[1].lower()] if d else None
        title=m['metadata'].get('title',[None])[0] if m and 'metadata' in m else None
        dates={k:m['metadata'][k] for k in ('published','published-online','published-print','issued') if m and 'metadata' in m and k in m['metadata']}
        audit.append({'key':key,'doi':d[1] if d else None,'primary_title':title,'primary_dates':json.dumps(dates),
                      'verification':m['status'] if m else 'primary page verified separately' if key in ('Boehm2022','Hinton2015','Kieu2026DIS2') else 'no DOI in original; retained context citation, metadata not reverified',
                      'source':m['url'] if m else {'Boehm2022':'https://arxiv.org/abs/2211.02869','Hinton2015':'https://arxiv.org/abs/1503.02531',
                          'Kieu2026DIS2':'https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.html'}.get(key)})
    # Keep every original reference available to the future manuscript writer.
    (OUT/'context_references.bib').write_text(raw)
    chosen=[]
    for key,e in pairs:
        if key not in ('Boehm2022','Hinton2015','Liu2024','Sun2027','Kieu2026DIS2','Chen2024MFEWF','Sikdar2025'):continue
        if key=='Sun2027':
            e=e.replace('Sun2027','Sun2026Online').replace('volume = {138}, pages = {104696}, year = {2027}',
                 'pages = {104696}, year = {2026}, note = {Available online 9 August 2026; assigned to volume 138, February 2027}')
            e=e[:-1]+',\n  url = {https://www.sciencedirect.com/science/article/abs/pii/S1566253526005725}\n}'
        if key in ('Boehm2022','Hinton2015') and 'doi =' not in e:
            arxiv='2211.02869' if key=='Boehm2022' else '1503.02531'
            e=e[:-1]+',\n  doi = {10.48550/arXiv.'+arxiv+'}\n}'
        chosen.append(e)
    chosen += [r'''@inproceedings{Ronneberger2015UNet,
  author = {Olaf Ronneberger and Philipp Fischer and Thomas Brox},
  title = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention},
  year = {2015}, doi = {10.1007/978-3-319-24574-4_28}
}''',r'''@article{Chen2017DeepLab,
  author = {Liang-Chieh Chen and George Papandreou and Florian Schroff and Hartwig Adam},
  title = {Rethinking Atrous Convolution for Semantic Image Segmentation},
  journal = {arXiv preprint arXiv:1706.05587}, year = {2017}, doi = {10.48550/arXiv.1706.05587}
}''']
    extra=fetch('10.1007/978-3-319-24574-4_28')
    (OUT/'provenance/UNet_reference_metadata.json').write_text(json.dumps(extra,indent=2))
    (OUT/'baseline_references.bib').write_text('\n\n'.join(chosen)+'\n')
    with (OUT/'reference_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    print('Original references retained:',len(pairs),'Crossref verified:',sum(r['status']=='verified_primary_metadata' for r in metadata),'/',len(metadata),flush=True)

if __name__=='__main__':main()
