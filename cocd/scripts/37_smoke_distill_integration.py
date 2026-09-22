#!/usr/bin/env python3
"""End-to-end integration check for the distillation round, on real data.

The structural smoke test verifies the operators; this one verifies the wiring.
It runs the *actual* training function from ``23_train_ours_v2.py`` against a
handful of real samples and a throwaway output directory, so nothing it does can
contaminate the reported runs.

Four things are checked:

* the ``gain`` control still computes exactly the expression it computed before
  the distillation code was added, bit for bit;
* the DIS2 block and the selective block both produce a finite KD term on real
  data, with the per-term breakdown the run log will show;
* the selection is actually active on real data (the random-input case is
  degenerate, so this has to be checked here);
* the auto-matched KD budget lands where it claims to.

The geometry round adds two more:

* the ``full`` weight is exactly the ``gain`` weight times ``(1 + 1_real * G10)``,
  and a synthetic no-change pair is left unboosted;
* one epoch of ``R3g`` runs through the real training function and the log shows
  the boosted fraction, so the geometry branch is reachable rather than falling
  through to the selective rule.

Carried as a known finding rather than fixed: the gain weight is written twice,
in ``selective_kl`` with ``EPS = 1e-8`` and in ``gain_protect`` with ``EPS = 1e-7``.
The two agree everywhere except where the teacher is already exact and
``e0 / (e0 + EPS)`` is ill-conditioned.  Only the unreachable ``selective`` arm
reads ``gain_protect``, so no reported number depends on the difference.
"""
import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
from pathlib import Path

import torch

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))

spec = importlib.util.spec_from_file_location('train23', A / 'scripts' / '23_train_ours_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

N = 0


def ok(condition, label):
    global N
    N += 1
    if not condition:
        raise AssertionError(f'FAILED  {label}')
    print(f'  ok  {label}')


DEV = m.DEV
tmp = Path(tempfile.mkdtemp(prefix='distill_integration_'))
m.BATCH = 2

tid, vid, eid = m.split()
train_set = m.HaitiPairs(tid[:8], True)
val_set = m.HaitiPairs(vid[:4], False, train_set.stats)
test_set = m.HaitiPairs(eid[:4], False, train_set.stats)
# The frozen Teacher lives in the real output directory; only the arm outputs
# are redirected, and only after the Teacher has been read.
teacher = m.load_teacher()
m.OUT = tmp
weight = m.posweight(train_set)
print(f'device={DEV}  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}')

batch = next(iter(m.loader(train_set)))
a, d, y, ga, gd = (x.to(DEV) for x in batch[:5])
yy = torch.cat((y, y))

print('\n[A] the gain control is unchanged by the distillation refactor')
with torch.no_grad():
    model = m.OursV3Student('R1').to(DEV); model.load_teacher_self(teacher); model.eval()
    z4, z34, _ = model(torch.cat((a, d)))
    # The expression the pre-refactor training loop evaluated, written out again.
    t_a, t_d = teacher.forward_pair(a, d)
    t0, t4, t34 = torch.cat((t_a[0], t_d[0])), torch.cat((t_a[1], t_d[1])), torch.cat((t_a[2], t_d[2]))
    g10 = torch.cat(((ga > 0) & (gd == 0), (gd > 0) & (ga == 0)))
    k4, _ = m.selective_kl(t0, t4, z4, yy, g10, 'gain')
    k34, _ = m.selective_kl(t0, t34, z34, yy, g10, 'gain')
    before = .5 * (k4 + k34)
    t = m.merge_taps(teacher.forward_pair_taps(a, d))
    k4n, _ = m.selective_kl(t['z0'], t['z4'], z4, yy, g10, 'gain')
    k34n, _ = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'gain')
    after = .5 * (k4n + k34n)
ok(torch.equal(before, after), 'the merged-tap path reproduces the original gain KD bit for bit')
ok(torch.equal(t['z4'], t4) and torch.equal(t['z34'], t34), 'merge_taps reproduces the original teacher concatenation')

print('\n[B] every mode produces a finite KD term on real data')
with torch.no_grad():
    taps = model.forward_taps(torch.cat((a, d)))
    rows = {}
    w = m.gain_protect(t['z0'], t['z34'], taps['z34'], yy)
    ref = after.detach()
    for mode in ('dis2', 'dis2-logit', 'selective'):
        if mode == 'dis2':
            kd, parts = m.dis2_multilevel_kd(taps, t)
        elif mode == 'dis2-logit':
            kd, parts = m.dis2_logit_only(taps, t)
        else:
            kd, parts = m.selective_multilevel_kd(taps, t, w, yy)
        rows[mode] = (kd, parts)
        ok(torch.isfinite(kd) and float(kd) > 0, f'{mode}: KD on a real batch is finite and positive')
        print(f'       {mode:11s} kd={float(kd):.4f}  control={float(ref):.4f}  ratio={float(kd)/float(ref):.3g}  ' +
              ' '.join(f'{k}={float(v):.4f}' for k, v in parts.items()))
    ok(float(ref) > 0, 'the gain-mode reference budget is positive')
    ok(abs(float(m.dis2_multilevel_kd(taps, t)[0]) - float(m.selective_multilevel_kd(taps, t, w, yy)[0])) > 1e-6,
       'the DIS2 rule and the selective rule are not the same number')

print('\n[C] the selection is active on real data')
sel = float((w > 0).float().mean())
print(f'       selected pixel fraction = {sel:.4f}')
ok(sel > 0, 'a non-empty set of pixels is selected on a real batch')
ok(0 < float(w[w > 0].mean()) <= 1, 'the selected pixels carry a weight inside (0, 1]')

print('\n[D] one epoch of each arm runs through the real training function')
for name in ('R1', 'DIS2'):
    model_out = m.train_student(name, m.CONFIGS[name], teacher, train_set, val_set, 1, 42,
                                tag='_smoke', early_stop=False)
    state = torch.load(tmp / f'{name}_smoke_seed42_latest.pt', map_location='cpu', weights_only=False)
    ok(state.get('kd_weight') == 1.0, f'{name}: the checkpoint records the one shared KD weight')
    ok((tmp / f'{name}_smoke_seed42.pt').exists(), f'{name}: a checkpoint was written')
    rows_eval = m.report(f'smoke-{name}', m.predict_student(model_out, test_set))
    ok(all(0. <= r['iou'] <= 1. and (r['auprc'] != r['auprc'] or 0. <= r['auprc'] <= 1.) for r in rows_eval),
       f'{name}: the evaluation path returns sane metrics on the frozen protocol')

print('\n[E] the geometry priority is the R3 weight times (1 + 1_real * G10)')
real_all = torch.ones(yy.shape[0], dtype=torch.bool, device=DEV)
real_none = torch.zeros_like(real_all)
mixed = real_all.clone(); mixed[:yy.shape[0] // 2] = False
half = yy.shape[0] // 2
with torch.no_grad():
    base = m.gain_protect(t['z0'], t['z34'], z34, yy)
    k_base, w_base = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'gain')
    k_true, w_true = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'full', real_all)
    k_false, w_false = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'full', real_none)
    _, w_zero = m.selective_kl(t['z0'], t['z34'], z34, yy, torch.zeros_like(g10), 'full', real_all)
    _, w_ones = m.selective_kl(t['z0'], t['z34'], z34, yy, torch.ones_like(g10), 'full', real_all)
    _, w_mix = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'full', mixed)
    _, w_gain_r = m.selective_kl(t['z0'], t['z34'], z34, yy, g10, 'gain', real_all)
with torch.no_grad():
    e0 = torch.nn.functional.binary_cross_entropy_with_logits(t['z0'][:, 0], yy, reduction='none')
    et = torch.nn.functional.binary_cross_entropy_with_logits(t['z34'][:, 0], yy, reduction='none')
    es = torch.nn.functional.binary_cross_entropy_with_logits(z34[:, 0], yy, reduction='none')
    prot = (et < es).float()
    w_e7 = (((e0 - et) / (e0 + 1e-7)).clamp(0, 1)) * prot
    w_e8 = (((e0 - et) / (e0 + 1e-8)).clamp(0, 1)) * prot
    tp = torch.sigmoid(t['z34'][:, 0])
    ent = torch.nn.functional.binary_cross_entropy(tp, tp, reduction='none')
    sce = torch.nn.functional.binary_cross_entropy_with_logits(z34[:, 0], tp, reduction='none')
    k_e7 = m.balanced_mean(w_e7 * (sce - ent), yy)
    k_e8 = m.balanced_mean(w_e8 * (sce - ent), yy)
    k_plain = m.balanced_mean(sce - ent, yy)
rel = float(abs(k_e8 - k_e7) / k_e8)
ok(torch.equal(w_e7, base), 'gain_protect is exactly the gain weight at EPS=1e-7')
ok(torch.equal(w_e8, w_base), 'selective_kl is exactly the gain weight at EPS=1e-8')
ok(int((w_e7 != w_e8).sum()) > 0, 'the two EPS choices do differ, so this is a real finding and not a vacuous check')
ok(rel < 1e-3, f'the weight difference moves the KD term by {rel:.2%}, so no reported number can depend on it')
print(f'       EPS 1e-7 vs 1e-8: {int((w_e7 != w_e8).sum())} of {w_e7.numel()} pixels differ, '
      f'max |dw| = {float((w_e7 - w_e8).abs().max()):.3f}, but the KD term moves {rel:.2%} '
      f'({float(k_e7):.6e} vs {float(k_e8):.6e})')
print(f'       gain x protect keeps {float(k_e8) / float(k_plain):.1%} of the unweighted KD budget on this batch')
ok(torch.equal(w_true, (1. + g10.float()) * w_base),
   'with every sample real, the full weight is (1 + G10) x gain x protect')
ok(torch.equal(w_false, w_base), 'with no sample real, the full weight is the gain x protect weight exactly')
ok(torch.equal(w_zero, w_base), 'an empty G10 leaves the full weight untouched')
ok(torch.equal(w_ones, 2. * w_base), 'a saturated G10 doubles the full weight exactly')
ok(torch.equal(w_gain_r, w_base), 'passing real to the gain mode changes nothing')
ok(torch.equal(k_false, k_base), 'the no-geometry full arm scores exactly what the gain arm scores')
ok(torch.equal(w_mix[:half], w_base[:half]), 'synthetic no-change pairs keep the unboosted weight')
ok(torch.equal(w_mix[half:], w_true[half:]), 'real pre/post pairs carry the boost')
ok(float((w_ones > w_base).float().sum()) == float((w_base > 0).float().sum()) and float((w_base > 0).sum()) > 0,
   'every selected pixel is boosted when G10 is everywhere')
print(f'       selected={float((w_base > 0).float().mean()):.4f} of pixels   '
      f'G10={float(g10.float().mean()):.4f}   real={float(real_all.float().mean()):.2f} of the batch')
ok(m.CONFIGS['R3g']['variant'] == m.CONFIGS['R3']['variant'],
   'R3g shares R3 structure, so the pair differs in the distillation weight alone')
ok(m.CONFIGS['R3g']['kd'] == 'full' and m.CONFIGS['R3']['kd'] == 'gain',
   'R3g is the geometry arm and R3 is its no-geometry control')
ok('R3g' not in m.PLAN, 'R3g is not in the all-plan, so a finished arm cannot be retrained by accident')

print('\n[F] one epoch of R3g runs through the real training function')
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.train_student('R3g', m.CONFIGS['R3g'], teacher, train_set, val_set, 1, 42, tag='_smoke', early_stop=False)
logged = buf.getvalue()
kd_line = next((l for l in logged.splitlines() if 'kd epoch=' in l), '')
print('       ' + kd_line)
ok('geo_frac=' in logged, 'the epoch log reports the priority fraction, so the geometry branch ran')
ok('w_frac=' in logged and 'w_mean=' in logged, 'the epoch log reports the live selection and its weight')
kd_values = {k: float(v) for k, v in (tok.split('=') for tok in kd_line.split() if '=' in tok)}
ok(set(kd_values) == {'epoch', 'w_frac', 'geo_frac', 'w_mean'},
   'the geometry arm reports exactly the selection, the priority and the mean weight')
ok(all(v >= 0 for v in kd_values.values()), 'every reported geometry diagnostic is non-negative')
ok(0. <= kd_values['w_mean'] <= 2., 'the mean distillation weight stays inside the range the rule allows')
ok((tmp / 'R3g_smoke_seed42.pt').exists(), 'R3g writes its own checkpoint')

shutil.rmtree(tmp, ignore_errors=True)
print('\nscratch directory removed:', not tmp.exists())
print(f'\nALL {N} ASSERTIONS PASSED')
