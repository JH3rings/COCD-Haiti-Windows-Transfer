"""The one place where this package resolves its own locations and its device.

Every absolute path of the original macOS project is replaced by a resolution
relative to this file, so the package can be unpacked anywhere on any machine.
Two locations can still be pointed somewhere else from outside the code:

* ``COCD_DATASET_ROOT`` — the directory holding ``Pre_event/``, ``Post_event/``
  and ``processed/``.  Defaults to ``<package>/data/haiti``.
* ``COCD_OUT_ROOT`` — the directory runs write into.  Defaults to
  ``<package>/experiments``.

The device is chosen in the order CUDA -> MPS -> CPU, so the same code runs on
the Windows CUDA target and on the Apple-silicon machine it was developed on.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
COCD = Path(__file__).resolve().parent            # <package>/cocd
PACKAGE = COCD.parent                             # <package>


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


DATASET_ROOT = _env_path('COCD_DATASET_ROOT', PACKAGE / 'data' / 'haiti')
# v3: no validation partition.  train_ids.csv is the merged 1370 locations, so
# the frozen protocol's train set is read straight off disk.  The pre-v3
# three-way split is kept in data/splits/ for reproducing the older runs; point
# COCD_SPLIT_DIR at it to get the old behaviour back.
SPLIT_DIR = _env_path('COCD_SPLIT_DIR', PACKAGE / 'data' / 'splits_v3')
OUT_ROOT = _env_path('COCD_OUT_ROOT', PACKAGE / 'experiments')

CONFIGS = COCD / 'configs'
SCRIPTS = COCD / 'scripts'
MANIFESTS = COCD / 'manifests'
THIRD_PARTY = COCD / 'third_party'

PROCESSED = DATASET_ROOT / 'processed'

# Sentinel-1 stacks for the four acquisitions this study uses.  Each file is a
# 128x128 three-band float32 GeoTIFF: band 2 = VV, band 1 = VH, band 3 unused.
RAW = {
    ('asc', 'pre'): 'Pre_event/S1_ASC_20210805',
    ('desc', 'pre'): 'Pre_event/S1_DESC_20210803',
    ('asc', 'post'): 'Post_event/S1_ASC_20210817',
    ('desc', 'post'): 'Post_event/S1_DESC_20210815',
}

CHECKPOINT_DIRS = (
    OUT_ROOT / 'ours_v2',
    PACKAGE / 'checkpoints',
    PACKAGE / 'legacy_checkpoints' / 'v1_protocol',
    OUT_ROOT / 'rapid_landslide_cocd',
)


def resolve_checkpoint(name: str) -> Path:
    """Find a checkpoint by file name without caring which folder holds it.

    Returns the first existing candidate, and otherwise the canonical output
    path for a run that has not happened yet, so callers can use the result both
    to load an existing checkpoint and to test whether one exists.
    """
    for folder in CHECKPOINT_DIRS:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return CHECKPOINT_DIRS[0] / name


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    """CUDA first, then Apple MPS, then CPU.  Override with ``COCD_DEVICE``."""
    forced = os.environ.get('COCD_DEVICE')
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEV = pick_device()

# The legacy stage-1 loader defaulted to 2 on MPS because that device is launch
# bound.  A CUDA host wants a larger batch, so the default is bigger there and
# every entry point can still override it (`--batch`, or the env var below).
LOADER_BATCH = int(os.environ.get('COCD_LOADER_BATCH', 2 if DEV.type == 'mps' else 16))


def describe() -> str:
    """One block for logs, so a run's provenance is visible without the CLI."""
    lines = [f'[paths] package      {PACKAGE}',
             f'[paths] dataset      {DATASET_ROOT}',
             f'[paths] splits       {SPLIT_DIR}',
             f'[paths] outputs      {OUT_ROOT}',
             f'[device] {DEV.type}  (torch {torch.__version__})']
    if DEV.type == 'cuda':
        lines.append(f'[device] {torch.cuda.get_device_name(0)}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe())
