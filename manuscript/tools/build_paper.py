"""Compile the current manuscript without leaving auxiliary files in the project."""
from pathlib import Path
import os,shutil,subprocess,tempfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
TEXBIN=Path.home()/"Library/TinyTeX/bin/universal-darwin"
def executable(name):
    found=shutil.which(name)
    if found:return found
    path=TEXBIN/name
    if path.is_file():return str(path)
    raise SystemExit(f"Missing LaTeX program: {name}. Install a TeX distribution or use the Codex LaTeX editor.")
def run(args,cwd,env):
    result=subprocess.run(args,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.stdout[-8000:])
    return result.stdout
env=os.environ.copy()
for key in ("BIBINPUTS","BSTINPUTS"):
    env[key]=str(ROOT)+os.pathsep+env.get(key,"")
with tempfile.TemporaryDirectory(prefix="grsl-paper-") as directory:
    tmp=Path(directory)
    command=[executable("pdflatex"),"-interaction=nonstopmode","-halt-on-error","-output-directory="+str(tmp),"main.tex"]
    run(command,ROOT,env)
    run([executable("bibtex"),"main"],tmp,env)
    run(command,ROOT,env)
    run(command,ROOT,env)
    log=(tmp/"main.log").read_text(errors="replace")
    if "undefined" in log or "Overfull" in log:
        raise SystemExit("Unresolved references or overflowing boxes. Check the saved source before release.")
    pdf=tmp/"main.pdf"
    pages=len(PdfReader(pdf).pages)
    if pages>5:
        for i, page in enumerate(PdfReader(pdf).pages):
            content = page.extract_text()
            print(f"Page {i+1}: {len(content)} characters; start={content[:180]!r}; end={content[-300:]!r}")
        raise SystemExit(f"Page limit exceeded: {pages} pages; final PDF was not replaced.")
    shutil.copy2(pdf,ROOT/"GRSL_final.pdf")
    shutil.copy2(tmp/"main.bbl",ROOT/"GRSL_final.bbl")
    print(f"Compiled {pages} pages: {ROOT/'GRSL_final.pdf'}")
