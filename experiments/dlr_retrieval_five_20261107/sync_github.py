"""Publish only source and compact study records to the dedicated Git branch."""
import ast
import json
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STUDY=ROOT/"experiments/dlr_unet_study_20261104"
WORKTREE=ROOT.parent/"COCD_DLR_retrieval_publish_20261107"
BRANCH="codex/dlr-retrieval-five-20261107"


def git(*args):
    return subprocess.run(["git","-C",str(WORKTREE),*args],capture_output=True,text=True,check=True).stdout.strip()


def dependencies():
    found=set()
    pending=list(HERE.glob("*.py"))
    while pending:
        source=pending.pop()
        if source in found:
            continue
        found.add(source)
        tree=ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]
            elif isinstance(node,ast.Import):
                names=[x.name for x in node.names]
            else:
                continue
            for name in names:
                for base in (HERE,STUDY,ROOT/"cocd"):
                    path=base/Path(*name.split(".")).with_suffix(".py")
                    if path.exists() and path not in found:
                        pending.append(path)
    return found


def publish(label):
    assert (WORKTREE/".git").is_file()
    assert git("branch","--show-current")==BRANCH
    files=dependencies()
    files.update(p for p in HERE.iterdir() if p.suffix in (".md",".svg",".png",".pdf",".json")
                 and not p.name.startswith("github_"))
    for p in (HERE/"results").rglob("*.json"):
        if p.name not in ("progress.json","history.json") and not p.name.startswith("github_"):
            files.add(p)
    for p in (HERE/"results").glob("*.csv"):
        files.add(p)
    for p in (HERE/"results").glob("*.png"):
        files.add(p)
    relative=[]
    for source in sorted(files):
        rel=source.relative_to(ROOT)
        target=WORKTREE/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        relative.append(rel.as_posix())
    git("add","-f","--",*relative)
    changed=subprocess.run(["git","-C",str(WORKTREE),"diff","--cached","--quiet"]).returncode
    if changed:
        git("commit","-m",f"DLR retrieval study: {label}")
        git("push","origin",f"HEAD:refs/heads/{BRANCH}")
    result={"branch":BRANCH,"commit":git("rev-parse","HEAD"),"files":len(relative),"stage":label}
    (HERE/"github_sync.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    publish("mechanism checks and initial M0 training")
