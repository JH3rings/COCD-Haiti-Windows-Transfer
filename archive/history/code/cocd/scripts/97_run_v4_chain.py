"""
v4 full retrain chain — the dependent half.

Two teachers exist and the arms do not all distil from the same one:

  * ``new-teacher`` feeds ``calib`` only: lambda is measured against the New
    COCD Teacher's complement, because the protocol's calibration needs a
    TRAINED teacher.  At initialisation the student-side complement C_S is
    exactly zero, so L_dist0 would be zero and the order-of-magnitude rule would
    have nothing to compare.  ``run_calib`` refuses to run in that state.
  * the LEGACY teacher feeds SO / VKD / DIS2 / R3, which are the legacy control
    arms and load ``teacher_wm.pt`` through ``v2.load_teacher``.

That second dependency is easy to miss: the legacy teacher is normally trained
in its own job, so a chain that starts while it is still running reaches the
first arm and dies on a missing ``teacher_wm.pt``.  ``wait_for`` below blocks on
the file actually existing, so the chain cannot outrun it.

So this script runs, strictly in order:

    1. new-teacher                        (New COCD Teacher, v4 early stopping)
    2. calib                              (freeze lambda against the trained teacher)
    3. every internal student arm          (SO, VKD, DIS2, R3, N1, N2, N2noorbit)

Each step is awaited in the foreground.  A non-zero exit stops the chain, so a
failed teacher never silently produces a table of arms trained on nothing.

Usage:
    python cocd/scripts/97_run_v4_chain.py
    python cocd/scripts/97_run_v4_chain.py --skip-teacher      # teacher already done
    python cocd/scripts/97_run_v4_chain.py --arms N1 N2        # a subset
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MAIN = ROOT / 'cocd' / 'windows_main' / 'main.py'
LOGDIR = ROOT / 'experiments' / '_v4_logs'
OURS = ROOT / 'experiments' / 'ours_v2'
WMAIN = ROOT / 'experiments' / 'windows_main'

# The order matters: the teacher first, then lambda, then the arms.
# SO needs no distillation but still loads the legacy teacher through the same
# code path as the other legacy arms, so it waits on the same file.
DEFAULT_ARMS = ['SO', 'VKD', 'DIS2', 'R3', 'N1', 'N2', 'N2noorbit']

# Which checkpoint each stage cannot start without.  A stage whose dependency is
# missing is the exact failure this table exists to prevent.
REQUIRES = {
    'calib': [WMAIN / 'NewTeacher_seed{seed}.pt'],
    'SO': [OURS / 'teacher_wm.pt'],
    'VKD': [OURS / 'teacher_wm.pt'],
    'DIS2': [OURS / 'teacher_wm.pt'],
    'R3': [OURS / 'teacher_wm.pt'],
    'N1': [WMAIN / 'NewTeacher_seed{seed}.pt'],
    'N2': [WMAIN / 'NewTeacher_seed{seed}.pt'],
    'N2noorbit': [WMAIN / 'NewTeacher_seed{seed}.pt'],
}


def wait_for(paths, label, poll=30, timeout_h=4.0):
    """Block until every path exists, so a still-training teacher is respected.

    The legacy teacher writes ``teacher_wm_latest.pt`` every epoch and only
    ``teacher_wm.pt`` once it finishes.  Waiting on the final name is therefore
    both necessary and sufficient: it means training is complete.
    """
    deadline = time.time() + timeout_h * 3600
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return True
    print(f'  waiting for {label} ({len(missing)} file(s) not yet written):', flush=True)
    for p in missing:
        print(f'    {p}', flush=True)
    while time.time() < deadline:
        time.sleep(poll)
        if all(p.exists() for p in paths):
            print(f'  {label} ready after {poll}s poll', flush=True)
            return True
        still = [p.name for p in paths if not p.exists()]
        print(f'  ... still waiting on {still}', flush=True)
    print(f'  TIMEOUT after {timeout_h}h waiting for {label}', flush=True)
    return False


def run_stage(name, stage_args, logfile):
    """Run one runner stage to completion, teeing stdout+stderr to a log file."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / logfile
    cmd = [PY, str(MAIN), *stage_args]
    print(f'\n{"=" * 78}\n=== {name}\n=== {" ".join(stage_args)}\n'
          f'=== log -> {log}\n{"=" * 78}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f'=== {name}: exit={proc.returncode}  {dt / 60:.1f} min  '
          f'({dt / 3600:.2f} h)', flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-teacher', action='store_true',
                    help='the New COCD Teacher is already trained')
    ap.add_argument('--skip-calib', action='store_true',
                    help='lambda is already frozen')
    ap.add_argument('--arms', nargs='+', default=None,
                    help=f'arms to train; default {" ".join(DEFAULT_ARMS)}')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--wait-hours', type=float, default=4.0,
                    help='how long to wait for a still-training dependency')
    args = ap.parse_args()
    arms = args.arms or DEFAULT_ARMS

    chain = []
    if not args.skip_teacher:
        chain.append(('New COCD Teacher', ['new-teacher', '--seed', str(args.seed)],
                      'new_teacher.log', 'new-teacher'))
    if not args.skip_calib:
        chain.append(('lambda calibration', ['calib', '--seed', str(args.seed)],
                      'calib.log', 'calib'))
    for arm in arms:
        chain.append((f'arm {arm}', [arm, '--seed', str(args.seed)],
                      f'arm_{arm}.log', arm))

    print(f'v4 chain: {len(chain)} stage(s), seed {args.seed}')
    for i, (name, argv, *_rest) in enumerate(chain, 1):
        dep = REQUIRES.get(argv[0], [])
        note = ''
        if dep:
            fmt = [str(p).format(seed=args.seed) for p in dep]
            note = '   needs ' + ', '.join(Path(p).name for p in fmt)
            note += '' if all(Path(p).exists() for p in fmt) else '  [NOT YET PRESENT]'
        print(f'  {i}. {name:24s} main.py {" ".join(argv)}{note}')

    t_start = time.time()
    for i, (name, argv, logfile, stage) in enumerate(chain, 1):
        print(f'\n>>> stage {i}/{len(chain)}', flush=True)
        dep = [Path(str(p).format(seed=args.seed)) for p in REQUIRES.get(stage, [])]
        if dep and not wait_for(dep, f'{name} prerequisites', timeout_h=args.wait_hours):
            print(f'\n!!! chain stopped: {name} prerequisites never appeared', flush=True)
            return 1
        rc = run_stage(name, argv, logfile)
        if rc != 0:
            print(f'\n!!! chain stopped: {name} exited {rc}. '
                  f'See {LOGDIR / logfile}', flush=True)
            return rc
    total = time.time() - t_start
    print(f'\n=== v4 chain finished: {len(chain)} stage(s) in {total / 3600:.2f} h ===',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
