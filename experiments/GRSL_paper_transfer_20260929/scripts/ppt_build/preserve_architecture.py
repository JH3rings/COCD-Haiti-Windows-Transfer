"""Repair importer style drift by retaining native source objects verbatim.

Artifact-tool import/render has been inspected. Only three original text nodes
are changed; no shape, connector, run style, master, or slide geometry is rebuilt.
"""
from pathlib import Path
import zipfile,json,hashlib
PACK=Path(__file__).resolve().parents[2]
src=PACK/'editable/architecture_optional_TGD_draft.pptx'
out=PACK/'editable/architecture_current_shared_memory.pptx'
changes={'Teacher search (on target features)':'Teacher policy on Student memory',
         'Student search (on target features)':'Student policy on same memory',
         'TGD (optional)':'TGD (off in CG0)'}
with zipfile.ZipFile(src) as zin,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as zout:
    for entry in zin.infolist():
        data=zin.read(entry.filename)
        if entry.filename=='ppt/slides/slide1.xml':
            text=data.decode('utf-8')
            for old,new in changes.items():
                assert text.count('>'+old+'<')==1,old
                text=text.replace('>'+old+'<','>'+new+'<')
            data=text.encode('utf-8')
        zout.writestr(entry,data)
report={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'changes':changes,'repair_reason':'artifact-tool rendering changed formula styles and shadows; source objects/styles retained','native_parts_unchanged_except':['ppt/slides/slide1.xml text nodes']}
(Path(__file__).parent/'native_preservation.json').write_text(json.dumps(report,indent=2))
print('NATIVE_STYLE_PRESERVED')
