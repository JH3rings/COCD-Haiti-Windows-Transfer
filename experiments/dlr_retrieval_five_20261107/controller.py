"""Monitor the running queue and publish measured stage records."""
import argparse
import json
import subprocess
import time

from build_report import main as report
from sync_github import HERE, publish

TERMINAL={"COMPLETE","FAILED","TEACHER_SEARCH_GATE_NOT_PASSED"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--attach",type=int)
    args=parser.parse_args()
    if args.attach is None:
        process=subprocess.Popen(["D:/deeplearning/anaconda3/python.exe","-X","utf8","-u","run.py"],cwd=HERE)
        pid=process.pid
    else:
        process=None
        pid=args.attach
    prior=set()
    while True:
        path=HERE/"results/status.json"
        if not path.exists():
            time.sleep(10)
            continue
        state=json.loads(path.read_text())
        frozen={p.parent.name for p in (HERE/"results").glob("*/selection.json")}
        key=tuple(sorted(frozen))
        terminal=state["state"] in TERMINAL
        if key not in prior or terminal:
            report()
            try:
                publish(state["state"] if terminal else f"{len(frozen)} frozen models")
                prior.add(key)
            except Exception as error:
                (HERE/"github_sync_error.txt").write_text(str(error))
                print("Git sync pending",str(error),flush=True)
        if terminal:
            break
        alive=subprocess.run(["powershell","-NoProfile","-Command",
                              f"if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"],
                             capture_output=True).returncode==0
        if not alive:
            raise RuntimeError("Training process exited before a terminal state")
        time.sleep(15)


if __name__=="__main__":
    main()
