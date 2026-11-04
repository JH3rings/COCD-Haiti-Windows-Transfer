"""Read official DLR samples with train-only normalization and paired augmentation."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ("PRE1_gVV", "PRE1_gVH", "POST1_gVV", "POST1_gVH")

class Inputs(Dataset):
    def __init__(self, frame, stats, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.cache = []
        mean = np.asarray(stats["sar_mean"] * 2, np.float32)[:, None, None]
        std = np.asarray(stats["sar_std"] * 2, np.float32)[:, None, None]
        for row in self.frame.itertuples():
            with h5py.File(row.h5_path, "r") as src:
                i = int(row.h5_index)
                ids = src.attrs["IDs_order"]
                sid = ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i])
                assert sid == row.sample_id
                sar = np.stack([src[name][i, 0] for name in FIELDS]).astype(np.float32)
                label = src["None_MASK"][i].astype(np.float32)
            assert sar.shape == (4, 128, 128) and label.shape == (1, 128, 128)
            assert np.isin(label, (0, 1)).all()
            valid = (np.isfinite(sar) & (sar > 0)).all(axis=0)
            sar = (10 * np.log10(np.maximum(np.where(np.isfinite(sar), sar, 0), 1e-8)) - mean) / std
            sar[:, ~valid] = 0
            self.cache.append({"x": torch.from_numpy(np.clip(sar, -8, 8)),
                               "mask": torch.from_numpy(label),
                               "valid": torch.from_numpy(valid[None]),
                               "sample_id": sid, "event_id": row.event_id,
                               "official_split": row.official_split})
    def __len__(self):return len(self.cache)
    def __getitem__(self,i):
        b=dict(self.cache[i])
        if self.augment:
            turns=int(torch.randint(0,4,()).item());dims=[]
            if torch.rand(())<.5:dims.append(-1)
            if torch.rand(())<.5:dims.append(-2)
            for k,v in b.items():
                if isinstance(v,torch.Tensor):
                    if turns:v=torch.rot90(v,turns,(-2,-1))
                    if dims:v=v.flip(dims)
                    b[k]=v
        return b
