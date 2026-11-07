"""Shared protocol helpers for the final distillation benchmark."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
COCD = ROOT / "cocd"
if str(COCD) not in sys.path:
    sys.path.insert(0, str(COCD))

from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, split  # noqa: E402
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2  # noqa: E402
from windows_main.models_teacher_target_search_v2 import (  # noqa: E402
    TargetSearchTeacherV2,
    load_so_backbone,
)

SEED = 42
BATCH = 16
LR = 5e-5
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 100
PATIENCE = 2
SO_INIT = ROOT / "artifacts/ours/initialization/backbone.pt"
TEACHER_CHECKPOINT = ROOT / "artifacts/ours/teacher/best.pt"
EPS = 1e-8


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders():
    train_ids, val_ids, test_ids = split()
    if val_ids:
        raise RuntimeError(f"Locked benchmark requires zero validation IDs; got {len(val_ids)}")
    train = HaitiPairs(train_ids, train_modes=True)
    test = HaitiPairs(test_ids, train_modes=False, stats=train.stats)
    return train, test, {"train_locations": len(train_ids), "validation_locations": 0,
                         "test_locations": len(test_ids), "train_rows": len(train)}


def loader(dataset, shuffle: bool = False):
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def make_student() -> StudentTargetSearchV2:
    model = StudentTargetSearchV2().to(DEV)
    load_so_backbone(model, str(SO_INIT))
    return model


def load_teacher() -> TargetSearchTeacherV2:
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(TEACHER_CHECKPOINT)
    model = TargetSearchTeacherV2().to(DEV)
    model.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=DEV, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target)


def student_predictions(model, dataset):
    model.eval()
    predictions, labels, geom_target, geom_counter = [], [], [], []
    with torch.no_grad():
        for asc, desc, y, ga, gd, _ in loader(dataset):
            oa = model(asc.to(DEV))["z"][:, 0]
            od = model(desc.to(DEV))["z"][:, 0]
            predictions.extend((torch.sigmoid(oa).cpu().numpy(), torch.sigmoid(od).cpu().numpy()))
            labels.extend((y.numpy(), y.numpy()))
            geom_target.extend((ga.numpy(), gd.numpy()))
            geom_counter.extend((gd.numpy(), ga.numpy()))
    return tuple(np.concatenate(parts, axis=0) for parts in
                 (predictions, labels, geom_target, geom_counter))


def region_masks(gt: np.ndarray, gc: np.ndarray):
    return {
        "Overall": np.ones_like(gt, dtype=bool),
        "BothValid": (gt == 0) & (gc == 0),
        "TargetValid": gt == 0,
        "G10": (gt > 0) & (gc == 0),
        "G01": (gt == 0) & (gc > 0),
    }


def metric_rows(prediction):
    p, y, gt, gc = prediction
    result = {}
    for name, mask in region_masks(gt, gc).items():
        m = mask.reshape(-1)
        yy = y.reshape(-1)[m].astype(bool)
        pp = p.reshape(-1)[m]
        hard = pp >= 0.5
        tp = int(np.logical_and(hard, yy).sum())
        fp = int(np.logical_and(hard, ~yy).sum())
        fn = int(np.logical_and(~hard, yy).sum())
        result[name] = {
            "AUPRC": float(average_precision_score(yy, pp)),
            "IoU": float(tp / (tp + fp + fn + EPS)),
            "F1": float(2 * tp / (2 * tp + fp + fn + EPS)),
            "pixels": int(m.sum()),
        }
    return result


def overall_ap(prediction) -> float:
    p, y, _, _ = prediction
    return float(average_precision_score(y.reshape(-1).astype(bool), p.reshape(-1)))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def canonical_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=WEIGHT_DECAY)


def protocol_record(arm: str, extras=None):
    record = {
        "arm": arm,
        "seed": SEED,
        "batch_size": BATCH,
        "optimizer": {"name": "Adam", "lr": LR, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": WEIGHT_DECAY},
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_monitor": "test Overall AUPRC (locked project protocol)",
        "validation_locations": 0,
        "geometry_use": "post-hoc evaluation only; no geometry input/loss/search/KD",
        "initialization": str(SO_INIT),
        "paired_teacher": str(TEACHER_CHECKPOINT),
    }
    if extras:
        record.update(extras)
    return record
