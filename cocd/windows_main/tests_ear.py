#!/usr/bin/env python3
"""Cheap structural checks for the EAR implementation (no dataset/training)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / 'cocd'))
from windows_main.models_complement import (  # noqa: E402
    EARStudent,
    EARTeacher,
    init_ear_student_from_teacher,
)


def check(name: str, condition: bool) -> None:
    print(('PASS' if condition else 'FAIL') + '  ' + name)
    if not condition:
        raise SystemExit(1)


torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.randn(2, 5, 128, 128, device=device)
d = torch.randn(2, 5, 128, 128, device=device)
a[:, 4] = 0
d[:, 4] = 1

teacher = EARTeacher().to(device).eval()
student = EARStudent().to(device).eval()
init_ear_student_from_teacher(student, teacher)

with torch.no_grad():
    oa, od = teacher.forward_pair(a, d)
    so = student(torch.cat((a, d)))
    zero = teacher.ear(oa['F_t'], torch.zeros_like(oa['bT']))

check('Teacher outputs 128x128 self/dual maps',
      tuple(oa['z_self'].shape) == (2, 1, 128, 128) and
      tuple(oa['z_dual'].shape) == (2, 1, 128, 128))
check('EAR uses nine actions at 32x32 P2', tuple(oa['rho'].shape) == (2, 9, 32, 32))
check('b=0 gives zero rho and zero correction',
      float(zero['rho'].abs().max()) == 0.0 and float(zero['C'].abs().max()) == 0.0)
check('initial Student equals copied Teacher self',
      torch.allclose(so['z'][:2], oa['z_self'], atol=2e-5, rtol=2e-5) and
      torch.allclose(so['z'][2:], od['z_self'], atol=2e-5, rtol=2e-5))
check('Student forward has no counter/GT/geometry arguments',
      list(__import__('inspect').signature(EARStudent.forward).parameters) == ['self', 'x'])
check('invalid 3x3 neighbours are masked',
      torch.equal(teacher.ear._valid_mask(1, 1, device).sum(), torch.tensor(1, device=device)))

# Make the Teacher guidance nonzero and verify a real action gradient reaches
# the driver and EAR write projection after one student loss.
with torch.no_grad():
    teacher.A.weight.normal_(0, 0.02)
    teacher.ear.wo.weight.normal_(0, 0.02)
teacher.train()
student.train()
oa, od = teacher.forward_pair(a, d)
out = student(torch.cat((a, d)))
loss = F.binary_cross_entropy_with_logits(out['z'][:, 0], torch.rand(4, 128, 128, device=device))
loss.backward()
check('Student action path has finite gradients',
      all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters()))
check('Student driver receives a gradient after the zero start',
      student.driver[-1].weight.grad is not None and
      float(student.driver[-1].weight.grad.abs().sum()) > 0.0)
print(f'PASS  device={device} teacher_params={sum(p.numel() for p in teacher.parameters())} '
      f'student_params={sum(p.numel() for p in student.parameters())}')
