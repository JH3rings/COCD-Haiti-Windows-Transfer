"""Stage-1 read-out: frozen Teacher vs Teacher-T2a, validation only.

The two models are run through exactly the same procedure
(``predict_teacher_heads`` -> ``teacher_curve_row``) so the comparison is a
comparison of weights, not of measurement code.  Nothing here trains, and the
test partition is not touched: whether T2a is even worth a test run is decided
by the pre-registered validation rule.

It also back-fills ``teacher_validation.csv`` for the frozen Teacher, whose
three-head validation table was never written when it was trained (only its
best dual AUPRC survived), by recomputing it here.

Usage:  python scripts/43_compare_teacher_t2a.py
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
spec = importlib.util.spec_from_file_location('v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

FROZEN_BEST_VAL_Z34 = 0.89626   # the number the pre-registered rule is written against
MARGIN = 0.0020


def curve_for(ckpt, val_set):
    model = v2.OursV2Teacher().to(v2.DEV)
    model.load_state_dict(torch.load(ckpt, map_location=v2.DEV, weights_only=True))
    model.eval()
    return model, v2.teacher_curve_row(model, val_set, 0)


def main():
    _, vid, _ = v2.split()
    stats = v2.HaitiPairs(v2.split()[0], True).stats
    val_set = v2.HaitiPairs(vid, False, stats)

    frozen, cf = curve_for(v2.OUT / 'teacher.pt', val_set)
    t2a, ct = curve_for(v2.OUT / 'teacher_T2a.pt', val_set)

    heads = ('z0', 'z4', 'z34')
    print(f'{"head":6s}{"metric":10s}{"frozen":>10s}{"T2a":>10s}{"delta":>10s}')
    for h in heads:
        for k in ('iou', 'f1', 'auprc'):
            a, b = cf[f'{h}_{k}'], ct[f'{h}_{k}']
            print(f'{h:6s}{k:10s}{a:10.5f}{b:10.5f}{b - a:+10.5f}')
        print()
    print(f'{"gain z34-z0 auprc":24s}{cf["gain_auprc_z34_minus_z0"]:+8.5f}'
          f'{ct["gain_auprc_z34_minus_z0"]:+8.5f}')
    print(f'{"gain z34-z4 auprc":24s}{cf["gain_auprc_z34_minus_z4"]:+8.5f}'
          f'{ct["gain_auprc_z34_minus_z4"]:+8.5f}')
    print()

    print('dual-orbit gain over the self path, by region (val, AUPRC):')
    print(f'{"region":8s}{"frozen":>10s}{"T2a":>10s}{"delta":>10s}   (z34 - z0)')
    for region in ('G00', 'G01', 'G10', 'G11'):
        a, b = cf[f'gain_auprc_{region}'], ct[f'gain_auprc_{region}']
        print(f'{region:8s}{a:10.5f}{b:10.5f}{b - a:+10.5f}')
    print()

    print('correction magnitudes (validation mean):')
    for tag, c in (('frozen', cf), ('T2a', ct)):
        print(f'  {tag:7s} |r3|={c["p3_corr_norm_mean"]:9.4f}  |r4|={c["p4_corr_norm_mean"]:9.4f}')
    print(f'\ntap-path self check (must be 0.0): frozen {cf["tap_path_max_abs_diff"]}, '
          f'T2a {ct["tap_path_max_abs_diff"]}')

    curve = pd.read_csv(v2.OUT / 'teacher_T2a_curve.csv')
    best_t2a = float(curve.z34_auprc.max())
    best_epoch = int(curve.loc[curve.z34_auprc.idxmax(), 'epoch'])
    print('\npre-registered rule: best val z34 AUPRC >= '
          f'{FROZEN_BEST_VAL_Z34 + MARGIN:.5f} (frozen {FROZEN_BEST_VAL_Z34:.5f} + {MARGIN:.4f})')
    print(f'frozen recomputed here: {cf["z34_auprc"]:.5f} '
          f'(must reproduce {FROZEN_BEST_VAL_Z34:.5f} -> '
          f'{"OK" if abs(cf["z34_auprc"] - FROZEN_BEST_VAL_Z34) < 5e-5 else "MISMATCH"})')
    print(f'T2a best-in-curve z34 AUPRC: {best_t2a:.5f} at epoch {best_epoch} '
          f'-> delta {best_t2a - FROZEN_BEST_VAL_Z34:+.5f}'
          f'  => {"IMPROVED" if best_t2a - FROZEN_BEST_VAL_Z34 >= MARGIN else "NOT IMPROVED"}')

    # Back-fill the frozen Teacher's validation table, which its own run never wrote.
    rows = v2.teacher_validation_rows(frozen, val_set, '')
    out = v2.OUT / 'teacher_validation.csv'
    if not out.exists():
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'\nwrote {out.relative_to(A)} (back-filled, frozen Teacher)')
    else:
        print(f'\n{out.relative_to(A)} already present; left alone')


if __name__ == '__main__':
    main()
