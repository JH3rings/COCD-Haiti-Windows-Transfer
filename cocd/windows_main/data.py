"""Dataset and frozen split loader for the active DA-search experiment.

This module deliberately contains no legacy Teacher/Student definitions.  The
active Windows line (T0, T1 and S0) imports the data path from here so retired
q3/q4 ablation code can be removed without changing the dataset protocol.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from paths import DATASET_ROOT, SPLIT_DIR

ROOT = DATASET_ROOT
PROCESSED = DATASET_ROOT / "processed"
EPS = 1e-8
RAW = {
    ("asc", "pre"): "Pre_event/S1_ASC_20210805",
    ("desc", "pre"): "Pre_event/S1_DESC_20210803",
    ("asc", "post"): "Post_event/S1_ASC_20210817",
    ("desc", "post"): "Post_event/S1_DESC_20210815",
}


def read(path: Path, bands=None):
    with rasterio.open(path) as src:
        return src.read(bands).astype("float32") if bands else src.read(1).astype("float32")


def split():
    """Return the pinned train, validation and test IDs.

    The active v3 protocol merges train and validation.  A legacy three-way
    split remains readable if COCD_SPLIT_DIR points at it, but no split is
    silently redrawn here.
    """
    pinned = {name: SPLIT_DIR / f"{name}_ids.csv" for name in ("train", "val", "test")}
    if pinned["train"].exists() and pinned["test"].exists():
        train = pd.read_csv(pinned["train"]).sample_id.astype(int).tolist()
        test = pd.read_csv(pinned["test"]).sample_id.astype(int).tolist()
        val = (pd.read_csv(pinned["val"]).sample_id.astype(int).tolist()
               if pinned["val"].exists() else [])
        return train, val, test
    manifest = Path(__file__).resolve().parents[1] / "manifests" / "spatial_80_20_split.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"pinned split files and fallback manifest are missing: {SPLIT_DIR}")
    table = pd.read_csv(manifest)
    train_ids = table[table.split == "train"].sample_id.astype(int).to_numpy()
    test_ids = table[table.split == "test"].sample_id.astype(int).tolist()
    rng = np.random.default_rng(42)
    val_ids = set(rng.choice(train_ids, size=round(0.1 * len(train_ids)), replace=False))
    return ([int(x) for x in train_ids if x not in val_ids],
            [int(x) for x in train_ids if x in val_ids], test_ids)


class HaitiPairs(Dataset):
    """Cached paired SAR observations with the existing geometry masks."""

    def __init__(self, ids, train_modes=False, stats=None):
        self.rows = []
        self.ids = [int(x) for x in ids]
        self.train_modes = train_modes
        for sample_id in tqdm(ids, desc="cache DA-search data", unit="location"):
            rasters = {}
            for orbit in ("asc", "desc"):
                for event in ("pre", "post"):
                    rasters[(orbit, event)] = read(
                        ROOT / RAW[(orbit, event)] / f"{sample_id}.tif", [2, 1])
            land = (read(PROCESSED / f"sample_{sample_id:06d}" / "landslide_mask.tif") > 0).astype("float32")
            geom_asc = read(PROCESSED / f"sample_{sample_id:06d}" / "asc_pre_geom.tif")
            geom_desc = read(PROCESSED / f"sample_{sample_id:06d}" / "desc_pre_geom.tif")
            self.rows.append((rasters, land, geom_asc, geom_desc))
        if stats is None:
            sm = np.zeros(2); ss = np.zeros(2); count = 0
            for rasters, _, _, _ in self.rows:
                for value in rasters.values():
                    sm += value.sum((1, 2)); ss += (value * value).sum((1, 2))
                    count += value.shape[1] * value.shape[2]
            mean = sm / count
            std = np.sqrt(ss / count - mean ** 2 + 1e-6)
            stats = mean, std
        self.stats = stats
        self.spec = [(i, mode) for i in range(len(self.rows))
                     for mode in ((0, 1, 2) if train_modes else (0,))]

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, index):
        row_index, mode = self.spec[index]
        rasters, land, geom_asc, geom_desc = self.rows[row_index]
        mean, std = self.stats

        def one(orbit):
            pre = rasters[(orbit, "pre")]
            post = rasters[(orbit, "post")]
            if mode == 1:
                post = pre
            elif mode == 2:
                pre = post
            orbit_id = np.full((1, 128, 128), 0.0 if orbit == "asc" else 1.0, np.float32)
            stacked = np.concatenate((pre, post, orbit_id))
            scale_mean = np.r_[mean, mean, 0.0][:, None, None]
            scale_std = np.r_[std, std, 1.0][:, None, None]
            return torch.from_numpy(((stacked - scale_mean) / scale_std).astype("float32"))

        y = land if mode == 0 else np.zeros_like(land)
        return (one("asc"), one("desc"), torch.from_numpy(y),
                torch.from_numpy(geom_asc), torch.from_numpy(geom_desc),
                torch.tensor(mode == 0))


def positive_weight(dataset: HaitiPairs) -> float:
    positive = sum(float(row[1].sum()) for row in dataset.rows)
    return (len(dataset.rows) * 128 * 128 - positive) / (positive + EPS)


__all__ = ["HaitiPairs", "positive_weight", "read", "split"]
