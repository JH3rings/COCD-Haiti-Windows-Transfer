#!/usr/bin/env python3
"""Validation-only readout for a protocol-v2 run.  **Never touches the test set.**

    python scripts/50_protocol_val_report.py --tag _v2 teacher
    python scripts/50_protocol_val_report.py --tag _v2 SO R3 R3D

Prints, per requested method, the validation-selected F1 threshold and the metrics at that
threshold, alongside the threshold-free AUPRC and the 0.5 readout for continuity.  The point
of the separate entry point is that deciding whether a run is worth continuing must not
require reading the test partition (protocol: one frozen read per method, at the end).

For the Teacher it reports two things: each head with its own validation-selected threshold,
and the reading the test table will actually use -- one threshold selected on the deployment
head (z34) and applied to all three.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def row(method, name, p, y, thr):
    m = v2.metrics(p, y, thr)
    return {'method': method, 'reading': name, 'threshold': thr, 'source': 'validation/max_f1',
            'iou': m['iou'], 'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'],
            'auprc': m['auprc'], 'iou@0.5': m['iou@0.5'], 'f1@0.5': m['f1@0.5']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('methods', nargs='+', help='teacher and/or arm names from CONFIGS')
    ap.add_argument('--tag', default='_v2')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tid, vid, eid = v2.split()
    train_set = v2.HaitiPairs(tid, True)
    val_set = v2.HaitiPairs(vid, False, train_set.stats)
    rows = []
    for method in args.methods:
        if method == 'teacher':
            t = v2.load_teacher(tag=args.tag)
            q = v2.predict_teacher_heads(t, val_set)
            shared = None
            for head, nm in (('z0', 'self'), ('z4', 'p4'), ('z34', 'dual')):
                thr, f1 = v2.best_f1_threshold(q[head], q['y'])
                rows.append(row(f'Teacher{args.tag}', f'{nm} (own thr)', q[head], q['y'], thr))
                if nm == 'dual':
                    shared = thr
            for head, nm in (('z0', 'self'), ('z4', 'p4'), ('z34', 'dual')):
                rows.append(row(f'Teacher{args.tag}', f'{nm} (dual thr)', q[head], q['y'], shared))
            print(f'[Teacher{args.tag}] dual-selected threshold (to be carried to test) = {shared:.3f}')
        else:
            cfg = v2.CONFIGS[method]
            m = v2.OursV3Student(cfg['variant']).to(v2.DEV)
            ckpt = v2.OUT / f'{method}{args.tag}_seed{args.seed}.pt'
            if not ckpt.exists():
                print(f'[skip] {ckpt.name} does not exist')
                continue
            m.load_state_dict(__import__('torch').load(ckpt, map_location=v2.DEV, weights_only=True))
            q = v2.predict_student(m, val_set)
            thr, f1 = v2.best_f1_threshold(q['p'], q['y'])
            rows.append(row(cfg.get('label', method) + args.tag, 'deployed', q['p'], q['y'], thr))
            print(f'[{method}{args.tag}] validation-selected threshold = {thr:.3f} (val F1 {f1:.4f})')
    frame = pd.DataFrame(rows)
    out = v2.OUT / f'val_report{args.tag}.csv'
    if out.exists():
        frame = pd.concat([pd.read_csv(out), frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=['method', 'reading'], keep='last')
    frame.to_csv(out, index=False)
    pd.set_option('display.width', 220)
    print(frame.round(4).to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
