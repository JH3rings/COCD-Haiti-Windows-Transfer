#!/usr/bin/env python3
"""Protocol probe: can physical batch 16 be run on this machine?

Read-only, no checkpoint is written and no optimiser step is taken on any real
weight.  Measures, for the shapes the new protocol needs:

  * Student single-orbit `(B,5,128,128)` forward + backward
  * Teacher paired forward (both target directions, both Omega references)
  * peak MPS allocation and step time, so the per-epoch budget can be projected

Prints a table; exits non-zero if a configuration cannot be allocated.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1].parent
A = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student  # noqa: E402
from losses.landslide_cocd import land_loss  # noqa: E402

DEV = v2.DEV
print(f'device={DEV}  mps={DEV.type == "mps"}')
if DEV.type == 'mps':
    budget = torch.mps.recommended_max_memory()
    print(f'recommended_max_memory = {budget / 2**30:.1f} GiB')
weight = 14.356170298161542


def peak():
    return torch.mps.driver_allocated_memory() / 2**30 if DEV.type == 'mps' else float('nan')


def trial(tag, fn, repeats=3):
    torch.mps.empty_cache()
    base = peak()
    outs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if DEV.type == 'mps':
            torch.mps.synchronize()
        outs.append(time.perf_counter() - t0)
    hi = peak()
    fn() if False else None
    print(f'  {tag:38s} driver_peak={hi:7.2f} GiB (base {base:6.2f})  '
          f'step={min(outs):.3f}s (median {sorted(outs)[len(outs) // 2]:.3f})')
    return hi, sorted(outs)[len(outs) // 2]


print('\n=== Student, single-orbit forward+backward ===')
for B in (8, 16, 32):
    m = OursV3Student('R3').to(DEV)
    m.train()
    x = torch.randn(B, 5, 128, 128, device=DEV)
    y = torch.zeros(B, 128, 128, device=DEV)

    def step(m=m, x=x, y=y):
        m.zero_grad(set_to_none=True)
        z4, z34, _ = m(x)
        loss = .5 * land_loss(z4, y, weight) + .5 * land_loss(z34, y, weight)
        loss.backward()

    trial(f'Student B={B}', step)
    del m, x, y
    torch.mps.empty_cache()

print('\n=== Teacher, paired forward (2 target directions x 2 Omega references) ===')
for B in (4, 8, 16):
    m = OursV2Teacher().to(DEV)
    m.train()
    a = torch.randn(B, 5, 128, 128, device=DEV)
    d = torch.randn(B, 5, 128, 128, device=DEV)
    y = torch.zeros(B, 128, 128, device=DEV)

    def step(m=m, a=a, d=d, y=y):
        m.zero_grad(set_to_none=True)
        oa, od = m.forward_pair(a, d)
        loss = sum(v2.TEACHER_W[i] * land_loss(o[i], y, weight) for o in (oa, od) for i in range(3)) * .5
        loss.backward()

    trial(f'Teacher pair B={B} (= {2 * B} single-orbit targets)', step)
    del m, a, d, y
    torch.mps.empty_cache()

# --------------------------------------------------------------------------- #
# Real-data step timing, so the per-epoch budget is measured rather than
# extrapolated from synthetic tensors.
# --------------------------------------------------------------------------- #
print('\n=== real-data timing at the protocol batch (16) ===')
tid, vid, eid = v2.split()
train_set = v2.HaitiPairs(tid, True)
val_set = v2.HaitiPairs(vid, False, train_set.stats)
print(f'  spec={len(train_set.spec)} samples/epoch  batches/epoch={-(-len(train_set.spec) // 16)}  '
      f'val samples={len(val_set.spec)}')
weight = v2.posweight(train_set)
STEPS = 20


def time_loader(desc, step_fn, n=STEPS):
    it = iter(v2.loader(train_set, True))
    ts = []
    for _ in range(n):
        batch = next(it)
        t0 = time.perf_counter()
        step_fn(batch)
        if DEV.type == 'mps':
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    per = sorted(ts)[len(ts) // 2]
    nbatches = -(-len(train_set.spec) // 16)
    print(f'  {desc:26s} median={per:.3f}s/batch  ->  {per * nbatches / 60:.1f} min/epoch  '
          f'->  {per * nbatches * 20 / 3600:.2f} h for 20 epochs  peak={peak():.2f} GiB')


s = OursV3Student('R3').to(DEV); s.train()
s_opt = v2.make_optimizer(s)


def s_step(batch):
    a, d, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
    x, yy = torch.cat((a, d)), torch.cat((y, y))
    s.zero_grad(set_to_none=True)
    z4, z34, _ = s(x)
    loss = .5 * v2.seg(z4, yy, weight) + .5 * v2.seg(z34, yy, weight)
    loss.backward(); s_opt.step()


time_loader('Student R3 (2x16 targets)', s_step)
del s, s_opt; torch.mps.empty_cache()

t_model = OursV2Teacher().to(DEV); t_model.train()
t_opt = v2.make_optimizer(t_model)


def t_step(batch):
    a, d, y = batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV)
    t_model.zero_grad(set_to_none=True)
    oa, od = t_model.forward_pair(a, d)
    loss = .5 * sum(v2.TEACHER_W[i] * v2.seg(o[i], y, weight) for o in (oa, od) for i in range(3))
    loss.backward(); t_opt.step()


time_loader('Teacher (pair, both dirs)', t_step)
