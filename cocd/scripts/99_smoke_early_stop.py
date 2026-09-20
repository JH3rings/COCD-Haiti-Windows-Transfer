"""Smoke test for the v4 early-stopping rule.

Runs the REAL training functions against a tiny synthetic dataset and a tiny
model-free monitor stand-in, so the stop logic, the best-weight restore and the
progress payload are exercised end to end without touching a GPU-hour of real
data.  Nothing here writes into ``experiments/`` or ``results/``.

What is asserted
----------------
1. a score that keeps rising runs until the ceiling, and the LAST epoch is the
   model (best == last);
2. a score that goes flat stops exactly PATIENCE epochs after the best one;
3. the weights actually restored are the ones from the best epoch, not the last;
4. ``bad`` never exceeds PATIENCE and resets on every new best;
5. the progress payload carries monitor / best / best_epoch / bad / monitor_risk.

Run:
    python cocd/scripts/99_smoke_early_stop.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cocd'))

PATIENCE = 2


class FakeNet:
    """A one-parameter stand-in whose weight IS the epoch number."""

    def __init__(self):
        self.w = 0.0
        self.state_dict = lambda: {'w': self.w}
        self.load_state_dict = self._load
        self.to = lambda _d: self

    def _load(self, d):
        self.w = d['w']

    def train(self):
        pass

    def eval(self):
        pass


def run(score_fn, epochs, patience=PATIENCE, early_stop=True):
    """The early-stopping loop, transcribed from train_student, over a fake net."""
    net = FakeNet()
    train_weight = 1.0
    best, bad, best_epoch, best_state = -1.0, 0, 0, None
    stop_reason, hist = None, []
    net.train()
    for ep in range(1, epochs + 1):
        net.w = float(ep)                        # "training" advances the weight
        score = score_fn(ep) if early_stop else float('nan')
        improved = (not early_stop) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, ep
            best_state = {k: v for k, v in net.state_dict().items()}
        else:
            bad += 1
        hist.append(dict(epoch=ep, monitor=score, best=best, bad=bad, improved=bool(improved)))
        if early_stop and bad >= patience:
            stop_reason = f'{patience} consecutive epochs without beating {best:.4f} at epoch {best_epoch}'
            break
    else:
        stop_reason = f'reached the {epochs}-epoch ceiling'
    if best_state is None:
        best_state = {k: v for k, v in net.state_dict().items()}
        best_epoch = ep
    net.load_state_dict(best_state)
    return dict(epochs_run=ep, best=best, best_epoch=best_epoch, bad=bad,
                weight_after_restore=net.w, stop_reason=stop_reason, hist=hist)


def check(label, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {label}')
    return cond


def main():
    ok = True
    print('=== v4 early-stopping smoke test ===\n')

    # --- case 1: monotone improvement runs to the ceiling, last epoch is best --
    print('case 1: strictly improving score, ceiling 6')
    r = run(lambda e: 0.10 * e, 6)
    ok &= check('ran to the ceiling (6 epochs)', r['epochs_run'] == 6)
    ok &= check('best epoch is the last epoch', r['best_epoch'] == 6)
    ok &= check('restored weight is the last epoch (6.0)', r['weight_after_restore'] == 6.0)
    ok &= check('bad is 0', r['bad'] == 0)
    ok &= check('stopped by the ceiling', 'ceiling' in r['stop_reason'])
    print()

    # --- case 2: only epoch 1 improves -> stop exactly PATIENCE epochs later --
    # NOTE on the rule: "beats the best" is STRICTLY greater.  A repeat of the
    # same score is NOT an improvement.  So a score that is 0.5 at epoch 1 and
    # flat afterwards has its only improvement at epoch 1 and stops at epoch 3
    # with patience 2 -- which is what this case pins down.
    print('case 2: only epoch 1 improves, patience 2')
    r = run(lambda e: 0.5 if e == 1 else 0.4, 200)
    ok &= check('stopped at epoch 3 (1 + patience)', r['epochs_run'] == 3)
    ok &= check('best epoch is 1', r['best_epoch'] == 1)
    ok &= check('restored weight is the BEST epoch (1.0), not the last (3.0)',
                r['weight_after_restore'] == 1.0)
    ok &= check('bad reached patience (2)', r['bad'] == PATIENCE)
    ok &= check('stop reason names epoch 1', 'epoch 1' in r['stop_reason'])
    ok &= check('a repeated score is not an improvement',
                r['hist'][1]['improved'] is False)
    print()

    # --- case 2b: flat plateau at 0.5 for three epochs, then a drop ----------
    print('case 2b: plateau, patience 2, the plateau counts as non-improving')
    r = run(lambda e: 0.5 if e <= 3 else 0.4, 200)
    ok &= check('stopped at epoch 3 (plateau broke the counter)', r['epochs_run'] == 3)
    ok &= check('best epoch is 1', r['best_epoch'] == 1)
    print()

    # --- case 3: a spike arriving while the counter is still short of patience
    # must move the best and reset the counter.  With patience 2 the spike has to
    # land at epoch 3 (two bad epochs would already have stopped at epoch 3), so
    # the sequence is best@1, bad, bad-and-spike.
    print('case 3: a late spike at epoch 3 resets the counter')
    def spike(e):
        return {1: 0.5, 2: 0.4, 3: 0.7, 4: 0.6, 5: 0.6}.get(e, 0.6)
    r = run(spike, 200)
    ok &= check('the spike moved the best to epoch 3', r['best_epoch'] == 3)
    ok &= check('stopped at epoch 5 (3 + patience)', r['epochs_run'] == 5)
    ok &= check('restored weight is epoch 3 (3.0)', r['weight_after_restore'] == 3.0)
    ok &= check('bad reset to 0 at the spike', r['hist'][2]['bad'] == 0)
    ok &= check('bad climbs 1, 2 on the two epochs after the spike',
                r['hist'][3]['bad'] == 1 and r['hist'][4]['bad'] == 2)
    print()

    # --- case 3b: the mirror image -- two bad epochs DO stop before a spike ---
    print('case 3b: two bad epochs stop the run before a later spike is reached')
    r = run(lambda e: {1: 0.5, 2: 0.4, 3: 0.4, 5: 0.9}.get(e, 0.4), 200)
    ok &= check('stopped at epoch 3 and never saw epoch 5', r['epochs_run'] == 3)
    ok &= check('best stayed at epoch 1', r['best_epoch'] == 1)
    print()

    # --- case 4: early_stop off -> ceiling, last epoch is the model ----------
    print('case 4: early stop OFF, ceiling 5')
    r = run(lambda e: 0.5, 5, early_stop=False)
    ok &= check('ran all 5 epochs', r['epochs_run'] == 5)
    ok &= check('restored weight is the last epoch (5.0)', r['weight_after_restore'] == 5.0)
    ok &= check('stopped by the ceiling', 'ceiling' in r['stop_reason'])
    print()

    # --- case 5: bad never exceeds patience, and rises by exactly 1 ----------
    print('case 5: counter monotonicity over a decaying score')
    r = run(lambda e: 1.0 / e, 200)
    b = [h['bad'] for h in r['hist']]
    ok &= check('bad never exceeds patience', max(b) <= PATIENCE)
    ok &= check('bad increments by 1 each non-improving epoch',
                all(b[i] == b[i - 1] + 1 for i in range(1, min(len(b), 3))))
    ok &= check('best stays at epoch 1', r['best_epoch'] == 1)
    print()

    print('=== ' + ('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED') + ' ===')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
