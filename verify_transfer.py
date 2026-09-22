#!/usr/bin/env python3
"""Read-only acceptance check for the final GRSL repository.

The verifier checks the final data loader, imports the final Teacher/Student
and KD entry points, instantiates both models, runs CPU dummy forwards, and
checks the structural boundary that counter information reaches policy context
only while the decoder consumes retrieved evidence ``R``. It does not train,
rewrite checkpoints, or write result files.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / "cocd"))

FAILURES: list[str] = []
WARNINGS: list[str] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def warn(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok  " if ok else "warn"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        WARNINGS.append(name)


print("=" * 78)
print("COCD Haiti final-v2 repository acceptance check")
print("=" * 78)

print("\n[1] packages and final imports")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from paths import DATASET_ROOT, DEV, SPLIT_DIR, describe  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_da_search import (  # noqa: E402
    CHANNELS,
    DeformableSearch,
    LEVELS,
    POINTS,
    SearchContext,
    TargetOnlyBaseline,
    pyramid,
    pyramid_from_encoded,
)
from windows_main.models_student_target_search_v2 import (  # noqa: E402
    StudentTargetSearchV2,
)
from windows_main.models_teacher_target_search_v2 import (  # noqa: E402
    TargetSearchTeacherV2,
)
from windows_main import student_search_kd_v2  # noqa: E402,F401

print(f"  numpy={np.__version__} pandas={pd.__version__} torch={torch.__version__}")
print(f"  rasterio={rasterio.__version__}")
step("required packages import", True)
step("final KD module imports", True)
step("final search primitives import", all((CHANNELS, LEVELS, POINTS)))
print(describe())
step("package resolves its device", True, DEV.type)

print("\n[2] active split")
try:
    train_ids, val_ids, test_ids = split()
    print(f"  train={len(train_ids)} validation={len(val_ids)} test={len(test_ids)}")
    step(
        "default final split is 1370 / 0 / 343",
        (len(train_ids), len(val_ids), len(test_ids)) == (1370, 0, 343),
    )
    step(
        "active split partitions are disjoint",
        set(train_ids).isdisjoint(val_ids)
        and set(train_ids).isdisjoint(test_ids)
        and set(val_ids).isdisjoint(test_ids),
    )
except Exception as exc:  # noqa: BLE001
    train_ids, val_ids, test_ids = [], [], []
    step("active split loads", False, f"{type(exc).__name__}: {exc}")

print("\n[3] final loader smoke")
if train_ids:
    try:
        sample_set = HaitiPairs(train_ids[:1], train_modes=True)
        batch = next(iter(DataLoader(sample_set, batch_size=2, shuffle=False, num_workers=0)))
        a, d, y, ga, gd, real = batch
        step("HaitiPairs builds from active train ids", len(sample_set) == 3)
        step(
            "loader returns five-channel orbit inputs",
            tuple(a.shape[1:]) == (5, 128, 128)
            and tuple(d.shape[1:]) == (5, 128, 128),
            f"asc={tuple(a.shape)} desc={tuple(d.shape)}",
        )
        step(
            "loader returns labels and geometry masks",
            tuple(y.shape[1:]) == (128, 128)
            and tuple(ga.shape[1:]) == (128, 128)
            and tuple(gd.shape[1:]) == (128, 128),
        )
    except Exception as exc:  # noqa: BLE001
        a = d = None
        step("final loader smoke", False, f"{type(exc).__name__}: {exc}")
else:
    a = d = None
    warn("final loader smoke", False, "no train ids available")

print("\n[4] final model and structure smoke")
try:
    teacher = TargetSearchTeacherV2().cpu().eval()
    student = StudentTargetSearchV2().cpu().eval()
    x = torch.randn(1, 5, 128, 128)
    counter_a = torch.randn_like(x)
    counter_b = torch.randn_like(x)
    with torch.no_grad():
        teacher_out_a = teacher(x, counter_a)
        teacher_out_b = teacher(x, counter_b)
        student_out = student(x)
        target_pyramid = pyramid(teacher.backbone, x)
        counter_pyramid_a = pyramid(teacher.backbone, counter_a)
        counter_pyramid_b = pyramid(teacher.backbone, counter_b)
        context_a = teacher.context(target_pyramid, counter_pyramid_a)
        context_b = teacher.context(target_pyramid, counter_pyramid_b)
        teacher_decoded = teacher.forward_from_cached_R(teacher_out_a["R"], (128, 128))
        student_decoded = student.decode_from_R_for_test(student_out["R"], (128, 128))
    step("final Teacher instantiates", True)
    step("final Student instantiates", True)
    step(
        "Teacher output is target-value search based",
        teacher_out_a["z"].shape == (1, 1, 128, 128)
        and teacher_out_a["R"].shape == (1, CHANNELS, 32, 32),
    )
    step(
        "Student has no counter input",
        list(inspect.signature(StudentTargetSearchV2.forward).parameters) == ["self", "target"],
    )
    step(
        "decoder accepts retrieved evidence R",
        teacher_decoded.shape == teacher_out_a["z"].shape
        and student_decoded.shape == student_out["z"].shape,
    )
    step(
        "counter changes policy context while target memory is fixed",
        all(torch.equal(a, b) for a, b in zip(teacher_out_a["values"], teacher_out_b["values"]))
        and not torch.equal(context_a, context_b),
    )
    teacher_source = Path("cocd/windows_main/models_teacher_target_search_v2.py").read_text()
    step(
        "Teacher source routes counter through context only",
        "context = self.context(target, counter)" in teacher_source
        and "values = tuple(layer(feature) for layer, feature in zip(self.value_proj, target))" in teacher_source
        and "self._decode_from_search(retrieved, output_size)" in teacher_source,
    )
except Exception as exc:  # noqa: BLE001
    step("final model smoke", False, f"{type(exc).__name__}: {exc}")

print("\n[5] final shared-module boundary")
shared = Path("cocd/windows_main/models_da_search.py").read_text()
backbone = Path("cocd/models/landslide_cocd.py").read_text()
for symbol in (
    "CHANNELS", "LEVELS", "POINTS", "pyramid", "pyramid_from_encoded",
    "SearchContext", "DeformableSearch", "TargetOnlyBaseline",
):
    step(f"shared final symbol {symbol}", symbol in shared)
for retired in (
    "class FusionPyramid", "class DeformablePolicy", "class SearchWrite",
    "class DAStudent", "class NormalDualTeacher", "class DASearchTeacher",
):
    step(f"retired shared symbol absent: {retired[6:]}", retired not in shared)
step("backbone keeps ConvNeXtTinyFPN", "class ConvNeXtTinyFPN" in backbone)
for retired in ("class TeacherCorrection", "class StudentCorrection", "class DualOrbitTeacher", "class SingleOrbitStudent"):
    step(f"retired backbone symbol absent: {retired[6:]}", retired not in backbone)

print("\n" + "=" * 78)
if FAILURES:
    print(f"TRANSFER CHECK FAILED - {len(FAILURES)} problem(s):")
    for failure in FAILURES:
        print(f"  - {failure}")
    sys.exit(1)
print("TRANSFER CHECK PASSED")
if WARNINGS:
    print(f"  {len(WARNINGS)} warning(s):")
    for warning in WARNINGS:
        print(f"  - {warning}")
print("  final-v2 code only; no training, checkpoint rewrite, or result regeneration")
print("=" * 78)
