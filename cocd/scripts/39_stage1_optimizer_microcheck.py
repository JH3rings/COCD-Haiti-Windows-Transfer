#!/usr/bin/env python3
"""Stage-1 pre-flight: is a zero-gradient batch really ``no-op`` for AdamW?

The Stage-0 audit measured that a synthetic no-change pair contributes a loss and
a gradient of exactly zero.  That answers the question for the *loss*, not for the
*optimiser*: AdamW also applies decoupled weight decay and multiplicatively decays
both moment estimates every ``step()``.  Whether that matters depends on one
detail -- whether the inactive parameters carry a zero-valued ``.grad`` tensor or
no ``.grad`` at all, because ``Optimizer.step`` skips parameters whose grad is
``None`` but happily steps any parameter that has a tensor.

This script is read-only.  It builds its own Teacher, takes a handful of steps and
throws it away: no checkpoint is written, no training run is started and no
training strategy is changed.  What it records is exactly what was asked for:

  * whether the grad is ``None`` or a zero tensor, per parameter group;
  * the largest absolute parameter change after one ``optimizer.step()``;
  * whether ``exp_avg`` / ``exp_avg_sq`` moved, and by how much.

It runs four probes.  S1 is a fresh optimiser on a synthetic-only batch.  S2 first
accumulates moments on real batches, then takes the synthetic step, which is the
case where the decay of a non-zero moment can matter.  S3 is the real-batch
control, so the size of the synthetic effect can be read against the size of a
genuine step.  S4 asks the only question with a training consequence: does a
synthetic step change what the *next* real step does?
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

A = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ours_v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def build_teacher():
    model = v2.OursV2Teacher().to(v2.DEV)
    model.load_s0(torch.load(A / 'experiments' / 'rapid_landslide_cocd' / 'S0.pt',
                             map_location=v2.DEV, weights_only=True))
    return model


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


def params_of(model):
    return {n: p for n, p in model.named_parameters()}


def snap_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def snap_moments(model, opt):
    """exp_avg/exp_avg_sq for every parameter the optimiser has state for."""
    out = {}
    for n, p in model.named_parameters():
        st = opt.state.get(p, None)
        if st and 'exp_avg' in st:
            out[n] = (st['exp_avg'].detach().clone(), st['exp_avg_sq'].detach().clone())
    return out


def grad_report(model):
    none = zero = nonzero = 0
    maxabs = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            none += 1
            continue
        m = float(p.grad.detach().abs().max())
        if m == 0.0:
            zero += 1
        else:
            nonzero += 1
            maxabs = max(maxabs, m)
    return {'grad_none': none, 'grad_zero_tensor': zero, 'grad_nonzero': nonzero,
            'grad_max_abs': maxabs}


def delta_report(before, model):
    worst_name, worst, changed = '', 0.0, 0
    sq_before = sq_delta = 0.0
    rel = 0.0
    for n, p in model.named_parameters():
        d = (p.detach() - before[n])
        mx = float(d.abs().max())
        sq_delta += float((d ** 2).sum())
        sq_before += float((before[n] ** 2).sum())
        if mx > 0.0:
            changed += 1
            scale = float(before[n].abs().max())
            if mx > worst:
                worst, worst_name = mx, n
            if scale > 0:
                rel = max(rel, mx / scale)
    return {'param_max_abs_delta': worst, 'param_max_abs_delta_name': worst_name,
            'params_changed': changed, 'param_max_rel_delta': rel,
            'param_delta_l2': float(np.sqrt(sq_delta)), 'param_l2_before': float(np.sqrt(sq_before))}


def moment_report(before, model, opt):
    after = snap_moments(model, opt)
    d1 = d2 = 0.0
    changed = 0
    n1_before = n2_before = n1_after = n2_after = 0.0
    for n, (m0, v0) in before.items():
        m1, v1 = after.get(n, (None, None))
        if m1 is None:
            continue
        dm, dv = float((m1 - m0).abs().max()), float((v1 - v0).abs().max())
        d1, d2 = max(d1, dm), max(d2, dv)
        if dm > 0.0 or dv > 0.0:
            changed += 1
        n1_before += float((m0 ** 2).sum()); n1_after += float((m1 ** 2).sum())
        n2_before += float((v0 ** 2).sum()); n2_after += float((v1 ** 2).sum())
    return {'moments_tracked': len(before), 'moments_changed': changed,
            'exp_avg_max_abs_delta': d1, 'exp_avg_sq_max_abs_delta': d2,
            'exp_avg_l2_before': float(np.sqrt(n1_before)), 'exp_avg_l2_after': float(np.sqrt(n1_after)),
            'exp_avg_sq_l2_before': float(np.sqrt(n2_before)), 'exp_avg_sq_l2_after': float(np.sqrt(n2_after))}


def state_summary(model, opt):
    """What the optimiser holds *after* a step, including a step on zero grads."""
    tracked = created_zero = 0
    m_max = v_max = 0.0
    for _, p in model.named_parameters():
        st = opt.state.get(p, None)
        if not st or 'exp_avg' not in st:
            continue
        tracked += 1
        mm, vv = float(st['exp_avg'].abs().max()), float(st['exp_avg_sq'].abs().max())
        m_max, v_max = max(m_max, mm), max(v_max, vv)
        if mm == 0.0 and vv == 0.0:
            created_zero += 1
    return {'state_tracked': tracked, 'state_created_but_zero': created_zero,
            'exp_avg_max_abs': m_max, 'exp_avg_sq_max_abs': v_max}


def flat_delta(before, model):
    """All parameter deltas as one float32 CPU vector, for direction comparisons."""
    return torch.cat([(p.detach() - before[n]).reshape(-1) for n, p in model.named_parameters()]).float().cpu()


def direction(a, b):
    na, nb = float(a.norm()), float(b.norm())
    cos = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float('nan')
    return {'cos_delta_vs_previous_real_step': cos,
            'delta_norm_ratio': (nb / na) if na > 0 else float('nan'),
            'delta_l2_this_step': nb, 'delta_l2_previous_real_step': na}


def named_deltas(before, model, names):
    out = {}
    for n in names:
        p = dict(model.named_parameters())[n]
        out[n] = {'grad': None if p.grad is None else float(p.grad.detach().abs().max()),
                  'max_abs_delta': float((p.detach() - before[n]).abs().max()),
                  'max_abs_param': float(before[n].abs().max())}
    return out


def take_step(model, opt, batch, weight):
    a, d, y = batch[0].to(v2.DEV), batch[1].to(v2.DEV), batch[2].to(v2.DEV)
    opt.zero_grad()
    loss = v2.teacher_batch(model, a, d, y, weight)
    loss.backward()
    return float(loss.detach())


def main():
    tid, vid, eid = v2.split()
    train_set = v2.HaitiPairs(tid, True)
    weight = v2.posweight(train_set)
    print(f'train samples/epoch={len(train_set)} real rows={len(train_set.rows)} posweight={weight:.3f}', flush=True)

    # The spec is one real row followed by its two synthetic views, so with
    # batch 2 a synthetic-only batch exists but an all-real one never does.
    # Take the synthetic-only batch as it is, and slice the real samples out of a
    # mixed batch for the control.
    syn = real = None
    for a, d, y, ga, gd, r in v2.loader(train_set, False):
        if syn is None and not bool(r.any()):
            syn = (a, d, y, ga, gd, r)
        if real is None and bool(r.any()):
            sel = r.bool()
            real = (a[sel], d[sel], y[sel], ga[sel], gd[sel], r[sel])
        if syn is not None and real is not None:
            break
    if syn is None or real is None:
        raise SystemExit('could not isolate a synthetic-only and a real-only batch')
    print(f'synthetic-only batch real={syn[5].tolist()} (n={len(syn[0])})  '
          f'real control batch real={real[5].tolist()} (n={len(real[0])})', flush=True)

    # Representative parameters: one from each functional block.
    keys = ['a3.net.4.weight', 'a4.net.4.weight', 'backbone.head.2.weight',
            'backbone.l4.weight', 'backbone.features.0.0.weight',
            'backbone.features.7.2.block.5.weight']
    all_names = [n for n, _ in v2.OursV2Teacher().named_parameters()]
    keys = [k for k in keys if k in all_names]
    if not keys:  # fall back to a spread over the parameter list
        keys = all_names[::max(1, len(all_names) // 5)][:5]
    print(f'representative params: {keys}', flush=True)

    res = {'representative_params': keys, 'posweight': weight, 'samples': len(train_set)}

    # --- S1: fresh optimiser, one synthetic-only step ---------------------------
    model = build_teacher(); opt = new_optimizer(model)
    before = snap_params(model)
    loss = take_step(model, opt, syn, weight)
    g = grad_report(model)
    state_before = state_summary(model, opt)
    opt.step()
    res['S1_fresh_synthetic'] = {'loss': loss, **g, **delta_report(before, model),
                                 'state_before_step': state_before, 'state_after_step': state_summary(model, opt),
                                 'params': named_deltas(before, model, keys)}

    # --- S2: warm moments (real batches first), then the synthetic step ---------
    model = build_teacher(); opt = new_optimizer(model)
    for _ in range(3):
        take_step(model, opt, real, weight); opt.step()
    before = snap_params(model)
    take_step(model, opt, real, weight); opt.step()
    d_real = flat_delta(before, model)
    loss = take_step(model, opt, syn, weight)
    g = grad_report(model); before = snap_params(model); m4 = snap_moments(model, opt)
    opt.step()
    d_syn = flat_delta(before, model)
    res['S2_after3real_synthetic'] = {'loss': loss, **g, **delta_report(before, model),
                                      'state_before_step': state_summary(model, opt)}
    res['S2_after3real_synthetic'].update({'step_' + k: v for k, v in
                                           {**delta_report(before, model), **moment_report(m4, model, opt)}.items()})
    res['S2_after3real_synthetic'].update(direction(d_real, d_syn))
    res['S2_after3real_synthetic']['state_after_step'] = state_summary(model, opt)
    res['S2_after3real_synthetic']['params_after_step'] = named_deltas(before, model, keys)

    # --- S3: real-batch control ------------------------------------------------
    model = build_teacher(); opt = new_optimizer(model)
    loss = take_step(model, opt, real, weight)
    g = grad_report(model); before = snap_params(model); m4 = snap_moments(model, opt)
    opt.step()
    res['S3_fresh_real'] = {'loss': loss, **g, **delta_report(before, model), **moment_report(m4, model, opt),
                            'state_after_step': state_summary(model, opt),
                            'params': named_deltas(before, model, keys)}

    # --- S4: does the synthetic step move the following real step? --------------
    m_a = build_teacher(); o_a = new_optimizer(m_a)
    for _ in range(3):
        take_step(m_a, o_a, real, weight); o_a.step()
    before_a = snap_params(m_a); take_step(m_a, o_a, real, weight); o_a.step()
    control = delta_report(before_a, m_a)

    m_b = build_teacher(); o_b = new_optimizer(m_b)
    for _ in range(3):
        take_step(m_b, o_b, real, weight); o_b.step()
    take_step(m_b, o_b, syn, weight); o_b.step()          # the synthetic step in between
    before_b = snap_params(m_b); take_step(m_b, o_b, real, weight); o_b.step()
    perturbed = delta_report(before_b, m_b)
    res['S4_next_real_step'] = {'control': control, 'after_synthetic_insert': perturbed,
                                'delta_max_abs_ratio': (perturbed['param_max_abs_delta'] /
                                                        max(control['param_max_abs_delta'], 1e-30))}

    # --- S5: why S1's parameter delta came out exactly zero --------------------
    # Decoupled weight decay is p *= (1 - lr*wd) = p * (1 - 1e-8).  At these
    # parameter magnitudes that is below float32 resolution, so the *measured*
    # delta is zero even though the term was applied.
    probe = torch.tensor([0.6255, 0.2496, 3e-4, 1.7e-1])
    decayed = probe * (1 - 1e-4 * 1e-4)
    res['S5_weight_decay_fp32'] = {
        'lr_times_wd': 1e-4 * 1e-4,
        'values': [float(x) for x in probe],
        'value_changed': [bool(x != y) for x, y in zip(probe, decayed)],
        'max_abs_change': float((decayed - probe).abs().max()),
        'fp32_eps': float(torch.finfo(torch.float32).eps)}

    out = A / 'experiments' / 'ours_v2' / 'stage1_optimizer_microcheck.json'
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float), flush=True)
    print(f'\nwritten: {out}', flush=True)


if __name__ == '__main__':
    main()
