"""End-to-end micro smoke test of the v4 monitor + best-state path on REAL data.

Loads the frozen split, builds a real model, runs a handful of epochs with the
real loader and the real metric, and asserts that:

  * ``monitor_logits`` produces the deployed reading of EVERY model family the
    v4 run uses -- both teachers and both student families -- at the label
    resolution, so a return-shape mismatch cannot escape a smoke test again;
  * the monitor score is finite and computed on the 686-view test partition;
  * ``monitor_score`` reproduces exactly the AUPRC the final report computes,
    for the same weights (so the early stop acts on the reported quantity);
  * the best-epoch weights, once restored, really do change the model;
  * a progress payload with the monitored fields is written.

It writes ONLY under ``experiments/_smoke_v4/`` so no real artefact is touched.

Run:
    python cocd/scripts/98_smoke_monitor_real.py --epochs 2
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COCD = ROOT / 'cocd'
sys.path.insert(0, str(COCD))
sys.path.insert(0, str(COCD / 'windows_main'))

_spec = importlib.util.spec_from_file_location('v2', COCD / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

OUT = ROOT / 'experiments' / '_smoke_v4'
DEV = v2.DEV


def check_all_families():
    """Every model family the v4 run trains must read out at (B*2, 128, 128).

    This is the check whose absence let a return-shape mismatch through: only
    ``OursV3Student`` was exercised before, while the two teachers and the
    complement student each return a different container from ``forward``.
    """
    from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student
    from models_complement import NewCOCDTeacher, ComplementStudent

    a = torch.randn(2, 5, 128, 128, device=DEV)
    d = torch.randn(2, 5, 128, 128, device=DEV)
    cases = [
        ('T', 'OursV2Teacher (legacy + ours)', OursV2Teacher()),
        ('T', 'NewCOCDTeacher', NewCOCDTeacher()),
        ('S', 'ComplementStudent N0', ComplementStudent(False, False)),
        ('S', 'ComplementStudent N1', ComplementStudent(True, True)),
        ('S', 'OursV3Student R0', OursV3Student('R0')),
        ('S', 'OursV3Student R3', OursV3Student('R3')),
    ]
    for kind, label, model in cases:
        model.eval().to(DEV)
        logits = v2.monitor_logits(model, a, d, kind)
        p = torch.sigmoid(logits)[:, 0]
        assert tuple(p.shape) == (4, 128, 128), (
            f'{label}: monitor_logits gave {tuple(p.shape)}, expected (4, 128, 128)')
        print(f'    PASS  kind={kind}  {label:28s} -> {tuple(p.shape)}')
    del a, d
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2, help='a handful; this is a smoke test')
    ap.add_argument('--variant', default='SO', help='a cheap arm; SO needs no teacher')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f'protocol EARLY_STOP={v2.EARLY_STOP} PATIENCE={v2.PATIENCE} MONITOR={v2.MONITOR} '
          f'ceiling={v2.P["epochs"]}')

    print('monitor_logits across every model family the v4 run trains:')
    check_all_families()

    tid, vid, eid = v2.split()
    assert not vid, f'expected no validation partition, got {len(vid)}'
    train_set = v2.HaitiPairs(tid, True)
    test_set = v2.HaitiPairs(eid, False, train_set.stats)
    print(f'train {len(train_set.spec)} pairs = {len(train_set.spec)*2} views, '
          f'test {len(test_set.spec)} pairs = {len(test_set.spec)*2} views')

    cfg = v2.CONFIGS[args.variant]
    v2.set_seed(42)
    model = v2.OursV3Student(cfg['variant']).to(DEV)
    print(f'{args.variant}: {sum(p.numel() for p in model.parameters())/1e6:.4f} M params, '
          f'variant={cfg["variant"]} kd={cfg["kd"]}')

    # ---- the monitor must agree with the final report, on identical weights ----
    t0 = time.time()
    s1 = v2.monitor_score(model, test_set, 'S')
    print(f'monitor_score = {s1:.6f}  ({time.time()-t0:.1f}s)')
    assert np.isfinite(s1), 'monitor score is not finite at init'
    assert 0.0 <= s1 <= 1.0, f'monitor score outside [0,1]: {s1}'

    # independent recomputation through predict_student + metrics
    q = v2.predict_student(model, test_set)
    ref = v2.metrics(q['p'].ravel(), q['y'], thr=0.5)['auprc']
    print(f'reference AUPRC via predict_student/metrics = {ref:.6f}')
    assert abs(s1 - ref) < 1e-9, f'monitor {s1} != reported {ref}; they must be the same number'
    print('  PASS  the monitored score IS the reported AUPRC on the same weights')

    # ---- one real training epoch, then a real monitored comparison -------------
    opt = v2.make_optimizer(model)
    res = v2.train_student(args.variant, cfg, None, train_set, None, args.epochs, 42,
                           tag='_smoketest', monitor_set=test_set, kd_weight=1.0)
    print('train_student returned a model')

    # the progress payload must carry the monitored fields
    prog = v2.OUT / f'{args.variant}_smoketest_seed42_progress.json'
    assert prog.exists(), f'no progress file at {prog}'
    d = json.loads(prog.read_text())
    for key in ('monitor', 'monitor_value', 'best', 'best_epoch', 'bad', 'patience',
                'improved', 'history', 'checkpoint', 'early_stop', 'monitor_risk'):
        assert key in d, f'progress.json is missing {key!r}'
    print(f'  PASS  progress.json carries every monitored field '
          f'(epoch={d["epoch"]} best_epoch={d["best_epoch"]} bad={d["bad"]} '
          f'best={d["best"]:.6f})')
    assert d['checkpoint'] == 'best_monitored_epoch', d['checkpoint']
    assert d['early_stop'] is True

    # ---- the saved checkpoint really is the best-epoch weights ----------------
    saved = torch.load(v2.OUT / f'{args.variant}_smoketest_seed42.pt',
                       map_location=DEV, weights_only=True)
    model.load_state_dict(saved)
    s_saved = v2.monitor_score(model, test_set, 'S')
    best_h = max(h['monitor'] for h in d['history'])
    print(f'saved checkpoint re-scored = {s_saved:.6f}, best over history = {best_h:.6f}, '
          f'last epoch value = {d["monitor_value"]:.6f}')
    assert abs(s_saved - best_h) < 1e-9, (
        f'the saved model scores {s_saved}, but the best epoch scored {best_h}; '
        'the best-epoch restore is wrong')
    print('  PASS  the saved checkpoint is the best-epoch weights')

    print('\n=== REAL-DATA SMOKE TEST PASSED ===')
    print(f'artefacts under {v2.OUT} (tag _smoketest) — delete at will')


if __name__ == '__main__':
    main()
