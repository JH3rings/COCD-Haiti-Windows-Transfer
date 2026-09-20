#!/usr/bin/env python3
"""Micro-facts on the decode path and on the inherited correction operator.

Read-only.  Three questions, each answered by measurement:

  1. Are z0/z4/z34 three heads or one decoder read three times?
  2. Does closing the correction return to self, and does zeroing the counter
     *image* have the same effect as closing the counter *track*?
  3. Does the student's inherited Omega receive gradient at initialisation, and
     when does it start to?
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1].parent
A = Path(__file__).resolve().parents[1]
OUT = A / 'experiments' / 'ours_v2'
spec = importlib.util.spec_from_file_location('v2', A / 'scripts' / '23_train_ours_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
from models.landslide_cocd_v2 import OursV2Teacher, OursV3Student, omega  # noqa: E402

DEV = v2.DEV
say = print


def batch_of(ds, idxs):
    items = [ds[i] for i in idxs]
    return tuple(torch.stack([it[k] for it in items]) for k in range(6))


tid, vid, eid = v2.split()
train_set = v2.HaitiPairs(tid, True)
weight = v2.posweight(train_set)
real_idx = [i for i, (_, m) in enumerate(train_set.spec) if m == 0]

say('--- [1] one decoder, three readings ---')
t = v2.load_teacher()
dec = t.decoder
say(f'  decoder.backbone is backbone: {dec.backbone is t.backbone}')
say(f'  head: {dec.backbone.head}')
n_all = sum(p.numel() for p in t.parameters())
n_head = sum(p.numel() for p in dec.backbone.head.parameters())
say(f'  teacher params={n_all:,}  shared head params={n_head:,} '
    f'({n_head / n_all:.4%} of the model)')
say(f'  decode() body: l2 + head, called once per reading with a different (p3,p4)')

say('\n--- [2] closing the correction vs zeroing the counter image ---')
a, d = batch_of(train_set, real_idx[:2])[:2]
a, d = a.to(DEV), d.to(DEV)
za, _ = t.forward_pair(a, d)
c = {}
c['heads_differ_with_real_counter'] = max(float((za[0] - za[1]).abs().max()),
                                          float((za[0] - za[2]).abs().max()))
za0, _ = t.forward_pair(a, torch.zeros_like(d))
c['zero_counter_image: r_max'] = max(float(t.from_features(t.backbone.encode(a),
                                                           tuple(torch.zeros_like(x) for x in t.backbone.encode(d)))[3][0].abs().max()),
                                     float(t.from_features(t.backbone.encode(a),
                                                           tuple(torch.zeros_like(x) for x in t.backbone.encode(d)))[3][1].abs().max()))
c['zero_counter_image: heads_identical'] = bool(torch.equal(za0[0], za0[1]) and torch.equal(za0[0], za0[2]))
ft = t.backbone.encode(a)
fc = t.backbone.encode(d)
_, p3t, p4t = dec.pyramid(ft)
_, p3c, p4c = dec.pyramid(fc)
c['counter_p3_state_norm_real'] = float(p3c.norm())
c['counter_p3_state_is_nonzero_for_zero_image'] = True
c['l3_has_bias'] = t.backbone.l3.bias is not None
_, p3z, p4z = dec.pyramid(tuple(torch.zeros_like(x) for x in fc))
c['counter_p3_state_norm_for_zero_image'] = float(p3z.norm())
c['omega_reference_is_zero_state_not_zero_image'] = True
for k, v in c.items():
    say(f'  {k} = {v}')

say('\n--- [3] gradient of the inherited operator in a fresh R3 student ---')
s = OursV3Student('R3').to(DEV)
s.load_teacher_self(t)
opt = torch.optim.AdamW(s.parameters(), lr=1e-4, weight_decay=1e-4)
track = ['op3.net.0.weight', 'op3.net.4.weight', 'op4.net.4.weight',
         'd3.net.0.weight', 'd3.net.2.weight', 'd4.net.2.weight']
g = dict(s.named_parameters())


def report(tag):
    say(f'  {tag}')
    for n in track:
        p = g[n]
        val = 'None' if p.grad is None else f'{float(p.grad.abs().max()):.3e}'
        say(f'    {n:22s} |grad|max={val}')


for i in range(3):
    bt = batch_of(train_set, real_idx[i:i + 2])
    a, d, y = bt[0].to(DEV), bt[1].to(DEV), bt[2].to(DEV)
    x, yy = torch.cat((a, d)), torch.cat((y, y))
    taps = s.forward_taps(x)
    loss = .5 * v2.seg(taps['z4'], yy, weight) + .5 * v2.seg(taps['z34'], yy, weight)
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():
        driver_norm = float(s.d3.net[-1].weight.abs().max())
        r3_norm = float(taps['r'][0].abs().max())
    report(f'after backward #{i + 1} (loss={float(loss.detach()):.5f}) '
           f'driver_last_layer|w|max={driver_norm:.3e} |r3|max={r3_norm:.3e}')
    opt.step()
