#!/usr/bin/env python3
"""Cross-platform launcher for the unified protocol.  Replaces the .sh wrapper.

The shell scripts of the original project used `caffeinate` and BSD `stat`, which
do not exist on Windows.  Nothing else about them was load-bearing, so the sweep
order and the guards are reproduced here in Python.

    python cocd/run_protocol_v2.py plan               # show the plan, run nothing
    python cocd/run_protocol_v2.py train              # teacher if absent, then the arms
    python cocd/run_protocol_v2.py train --stages SO  # one arm, no teacher needed
    python cocd/run_protocol_v2.py test               # the teacher's one-shot test table

Every child is a plain `python <script> ...` call, so anything that runs here can
also be run directly; this file only decides the order and refuses to start a
second run on top of a live one.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'cocd'))

from paths import OUT_ROOT, describe, resolve_checkpoint  # noqa: E402

SCRIPT = PKG / 'cocd' / 'scripts' / '23_train_ours_v2.py'
VAL_REPORT = PKG / 'cocd' / 'scripts' / '50_protocol_val_report.py'
OUT = OUT_ROOT / 'ours_v2'
LOG = OUT / 'protocol_v2_log.txt'

# A checkpoint written this recently may still be being written by another run.
SETTLE_SECONDS = 300


def log(message: str) -> None:
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def assert_nothing_running(settle: int) -> None:
    """Refuse to start when a checkpoint was touched moments ago."""
    newest, newest_path = 0.0, None
    if OUT.exists():
        for p in OUT.glob('*_latest.pt'):
            m = p.stat().st_mtime
            if m > newest:
                newest, newest_path = m, p
    if newest_path is None:
        return
    age = time.time() - newest
    if age < settle:
        raise SystemExit(
            f'still writing: {newest_path.name} was modified {age:.0f}s ago. '
            f'Another run is probably active. Pass --settle 0 to override, or wait.')


def run(cmd: list[str], dry_run: bool) -> None:
    log('RUN: ' + ' '.join(str(c) for c in cmd))
    if dry_run:
        return
    with LOG.open('a') as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        log(f'FAILED with exit code {rc}: {cmd[0]}')
        raise SystemExit(rc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['plan', 'train', 'test'], help='what to do')
    ap.add_argument('--stages', default='', help='comma-separated arms; default is the plan in the script')
    ap.add_argument('--tag', default='_v2', help='suffix for every artefact; keeps a new run away from the shipped teacher')
    ap.add_argument('--epochs', type=int, default=20, help='protocol maximum')
    ap.add_argument('--batch', type=int, default=16, help='physical batch; must satisfy batch x accum == 16')
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-early-stop', action='store_true',
                    help='run the full schedule; required when arms are compared epoch for epoch')
    ap.add_argument('--python', default=sys.executable, help='interpreter for the child processes')
    ap.add_argument('--settle', type=int, default=SETTLE_SECONDS,
                    help='seconds of checkpoint quiet required before starting')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    args = ap.parse_args()

    print(describe())
    print(f'[run] mode={args.mode} tag={args.tag} epochs={args.epochs} '
          f'batch={args.batch} accum={args.accum} seed={args.seed}')
    print(f'[run] outputs -> {OUT}')

    if args.batch * args.accum != 16:
        raise SystemExit(f'protocol requires physical x accumulate == 16, got '
                         f'{args.batch} x {args.accum} = {args.batch * args.accum}')

    common = ['--tag', args.tag, '--epochs', str(args.epochs), '--batch', str(args.batch),
              '--accum', str(args.accum), '--seed', str(args.seed)]
    if args.no_early_stop:
        common.append('--no-early-stop')

    teacher = resolve_checkpoint(f'teacher{args.tag}.pt')
    stages = [s for s in args.stages.split(',') if s]

    if args.mode == 'plan':
        print(f'\n  teacher{args.tag}.pt : '
              f'{"present, will be reused" if teacher.exists() else "absent, will be trained"}  ({teacher})')
        print(f'  arms : {stages or "(the script's own PLAN)"}')
        print(f'  log  : {LOG}')
        print('\nnothing was run (plan mode)')
        return

    assert_nothing_running(args.settle)

    if args.mode == 'test':
        run([args.python, '-u', SCRIPT, 'teacher-test', *common], args.dry_run)
        log('test table written')
        return

    # ---- train ---------------------------------------------------------------
    # The distillation arms read the teacher at run time, so it has to exist under
    # this tag first.  A chain that only runs the baseline does not need it.
    needs_teacher = not stages or any(s not in ('SO',) for s in stages)
    if needs_teacher:
        if teacher.exists():
            log(f'teacher{args.tag}.pt already present; not retraining it')
        else:
            run([args.python, '-u', SCRIPT, 'teacher', *common], args.dry_run)
    else:
        log('only the baseline was requested; the teacher is not needed')

    arm_cmd = [args.python, '-u', SCRIPT, 'all']
    if stages:
        arm_cmd += ['--stages', ','.join(stages)]
    arm_cmd += common
    run(arm_cmd, args.dry_run)

    log('sweep finished')
    if not args.dry_run:
        log('validation-only readout: '
            f'{args.python} {VAL_REPORT} --tag {args.tag} teacher SO VKD DIS2 R3')


if __name__ == '__main__':
    main()
