"""Collect the finished single-orbit baseline seats into one comparison table.

Every seat under the unified v4 protocol writes the same files:

    experiments/single_orbit_baselines_v3/<seat>/metrics.csv
    experiments/single_orbit_baselines_v3/<seat>/progress.json
    experiments/single_orbit_baselines_v3/logs/<seat>.log

(The output directory keeps its historical ``_v3`` name so the seat scripts and
this table never disagree about where a run lives.  The protocol it reads is v4.)

``metrics.csv`` has one row per partition (``overall`` / ``asc`` / ``desc``) at a
single fixed threshold, so this script only has to concatenate them and lay the
partitions out side by side.  Seats that have not been run yet are reported as
missing rather than silently dropped, because a partially filled table is easy
to mistake for a complete one.

Under v4 each seat may stop at a different epoch, so the selected epoch and the
number of epochs actually run are printed alongside the metrics.

Usage
-----
    python cocd/scripts/24_external_seat_table.py
    python cocd/scripts/24_external_seat_table.py --seeds 42 123 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V3_OUT = ROOT / 'experiments' / 'single_orbit_baselines_v3'
PROT = json.loads((ROOT / 'cocd' / 'configs' / 'protocol_windows_main.json').read_text(encoding='utf-8'))

# seat key -> display label, parameter count, loss kind, implementation note
# FC-Siam is deliberately absent: the user excluded it from the v4 run.
SEATS = [
    ('boehm', 'Boehm SAR U-Net++', 26.0850, 'CE', 'U-Net++ with the paper setting'),
    ('cdnette', 'CDNetE Early Fusion', 24.4428, 'CE', 'smp Unet(resnet34), no official code'),
    ('mfewf', 'MFEWF adapted', 26.9821, 'CE', 'local reimplementation, no official code'),
]
EXCLUDED = [('fc_siam', 'FC-Siam (official impl)')]

PARTITIONS = ('overall', 'asc', 'desc')
METRICS = ('iou', 'f1', 'precision', 'recall', 'auprc')


def read_seat(seat: str, seed: int | None) -> dict | None:
    """Return the per-partition metrics of one seat, or None if it has not run."""
    cand = V3_OUT / seat / 'metrics.csv'
    if seed is not None and not cand.exists():
        cand = V3_OUT / f'{seat}_seed{seed}' / 'metrics.csv'
    if not cand.exists():
        return None
    frame = pd.read_csv(cand)
    out = {'path': cand}
    for row in frame.itertuples():
        out[row.partition] = {m: float(getattr(row, m)) for m in METRICS
                              if hasattr(row, m)}
    return out


def seat_meta(seat: str, seed: int | None) -> dict:
    """Read the run metadata the seat wrote, if it left one."""
    for name in ('progress.json', 'summary.json'):
        for base in ((V3_OUT / seat), V3_OUT / f'{seat}_seed{seed}'):
            p = base / name
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42],
                    help='seeds to look for; the protocol names 42, 123 and 2026')
    ap.add_argument('--out', default='', help='where to write the combined CSV')
    args = ap.parse_args()

    frames, missing = [], []
    for seat, label, params, loss, note in SEATS:
        got = read_seat(seat, None)
        if got is None:
            missing.append(label)
            continue
        meta = seat_meta(seat, None)
        for part in PARTITIONS:
            if part not in got:
                continue
            frames.append({'method': label, 'parameters_M': params, 'loss': loss,
                           'implementation': note, 'partition': part,
                           'selected_epoch': meta.get('best_epoch'),
                           'epochs_run': meta.get('epoch'), **got[part]})

    if not frames:
        print(f'no seat has finished yet under {V3_OUT}')
        print('expected one of: ' + ', '.join(s for s, *_ in SEATS))
        if EXCLUDED:
            print('excluded from the v4 run: ' + ', '.join(l for _, l in EXCLUDED))
        return

    frame = pd.DataFrame(frames)
    order = [label for _, label, *_ in SEATS if label in set(frame.method)]
    frame['method'] = pd.Categorical(frame.method, order, ordered=True)
    frame = frame.sort_values(['method', 'partition'], key=lambda c: c
                              if c.name != 'partition'
                              else c.map({p: i for i, p in enumerate(PARTITIONS)}))

    out = Path(args.out) if args.out else V3_OUT / 'comparison_table.csv'
    frame.to_csv(out, index=False)

    sch = PROT['schedule']
    print('=' * 108)
    print(f'Single-orbit baselines, unified {PROT["revision"]} protocol')
    print(f'  train 8220 views/epoch (1370 locations x 3 modes x 2 orbits), '
          f'test 686 views (343 x 2 orbits)')
    print(f'  Adam lr {PROT["optimizer"]["lr"]:g}, wd {PROT["optimizer"]["weight_decay"]:g}, '
          f'batch {PROT["batch"]["physical"]}, ceiling {sch["max_epochs"]}, '
          f'early stop patience {sch["early_stop_patience"]} on {sch["monitor"]}, '
          f'threshold fixed {PROT["threshold"]["value"]}')
    print(f'  !! {PROT["monitor_risk"]["statement"]}')
    print('=' * 108)
    head = f'{"method":26s} {"M":>8s} {"loss":>5s} {"part":>8s} ' + ' '.join(
        f'{m:>9s}' for m in METRICS) + f' {"sel_ep":>6s} {"ran":>5s}'
    print(head)
    print('-' * len(head))
    for row in frame.itertuples():
        sel = '' if pd.isna(row.selected_epoch) else str(int(row.selected_epoch))
        ran = '' if pd.isna(row.epochs_run) else str(int(row.epochs_run))
        print(f'{row.method:26s} {row.parameters_M:8.4f} {row.loss:>5s} '
              f'{row.partition:>8s} ' + ' '.join(f'{getattr(row, m):9.4f}'
                                                 for m in METRICS) + f' {sel:>6s} {ran:>5s}')
    print()
    if EXCLUDED:
        print('excluded from the v4 run: ' + ', '.join(l for _, l in EXCLUDED))
    if missing:
        print('not yet run: ' + ', '.join(missing))
        print('  a seat appears here only once its metrics.csv exists; '
              'the table above is partial.')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
