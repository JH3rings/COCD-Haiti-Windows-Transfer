#!/usr/bin/env python3
"""Short structural checks for CGSearch (no dataset access or training)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "cocd"))
from windows_main.models_cgsearch import (  # noqa: E402
    CGSearchStudent,
    CGSearchTeacher,
    action_distance,
    copy_teacher_to_student,
    parameter_counts,
)


def check(name: str, condition: bool) -> None:
    print(("PASS" if condition else "FAIL") + "  " + name)
    if not condition:
        raise SystemExit(1)


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1, 5, 128, 128, device=device)
c1 = torch.randn(1, 5, 128, 128, device=device)
c2 = torch.randn(1, 5, 128, 128, device=device)

teacher = CGSearchTeacher().to(device).eval()
student = CGSearchStudent().to(device).eval()
copy_teacher_to_student(student, teacher)

with torch.no_grad():
    ot = teacher(x, c1)
    ot2 = teacher(x, c2)
    os = student(x)
    os0 = student.forward_zero_guidance(x)

check("Teacher returns 128x128 self and dual logits",
      tuple(ot["z_self"].shape) == (1, 1, 128, 128) and
      tuple(ot["z_dual"].shape) == (1, 1, 128, 128))
check("runtime FPN shapes are P2 32x32, P3 16x16, P4 8x8",
      ot["shapes"]["P2"][1:] == (128, 32, 32) and
      ot["shapes"]["P3"][1:] == (128, 16, 16) and
      ot["shapes"]["P4"][1:] == (128, 8, 8))
check("rho has [B, H3*W3, H4*W4] shape",
      tuple(ot["rho"].shape) == (1, 16 * 16, 8 * 8))
check("A0 and AT normalize over all target P4 candidates",
      torch.allclose(ot["A0"].sum(-1), torch.ones_like(ot["A0"].sum(-1)), atol=1e-6) and
      torch.allclose(ot["AT"].sum(-1), torch.ones_like(ot["AT"].sum(-1)), atol=1e-6))
check("zero-initialized GuideT makes dual exactly self",
      float((ot["z_dual"] - ot["z_self"]).abs().max()) < 1e-6 and
      float((ot["AT"] - ot["A0"]).abs().max()) < 1e-6)
check("Teacher K/V are target-only when counter changes",
      torch.allclose(ot["K"], ot2["K"], atol=0, rtol=0) and
      torch.allclose(ot["V"], ot2["V"], atol=0, rtol=0))
check("Student forward accepts only target tensor",
      list(inspect.signature(CGSearchStudent.forward).parameters) == ["self", "x"])
check("Student has no geometry/counter argument",
      not any(k in inspect.signature(CGSearchStudent.forward).parameters
              for k in ("counter", "geometry", "mask", "gt")))
check("Student copied self path equals Teacher self with bS=0",
      torch.allclose(os["z_self"], ot["z_self"], atol=2e-5, rtol=2e-5) and
      torch.allclose(os0["z"], ot["z_self"], atol=2e-5, rtol=2e-5))
check("zero-guidance Student still completes the full forward",
      tuple(os0["z"].shape) == (1, 1, 128, 128))

# The action helper is the only KD primitive.  Teacher is detached by the
# helper, so a loss gradient can only reach the Student action tensor.
rho_s = torch.randn(1, 16 * 16, 8 * 8, device=device, requires_grad=True)
rho_t = torch.randn(1, 16 * 16, 8 * 8, device=device, requires_grad=True)
action_distance(rho_s, rho_t).mean().backward()
check("search-action distance detaches Teacher",
      rho_s.grad is not None and rho_t.grad is None)
counts = parameter_counts(teacher, student)
check("parameter counts are recorded and Student extra capacity is light",
      counts["Student_total"] > counts["SO_total"] and counts["Student_extra_vs_SO"] < 300_000)
print(f"PASS  device={device} counts={counts}")
