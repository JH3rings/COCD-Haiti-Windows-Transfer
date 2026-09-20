#!/usr/bin/env python3
"""Seed-42 random 80/20 split using every processed Haiti patch."""
from pathlib import Path
import json, numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
A=Path(__file__).resolve().parents[1];M=A/'manifests';OUT=A/'experiments'/'landslide_downstream_reproduction'
N=1713;SEED=42;N_TEST=343
rng=np.random.default_rng(SEED);test=set(rng.choice(np.arange(N),size=N_TEST,replace=False).tolist())
d=pd.DataFrame({'sample_id':np.arange(N,dtype=int)})
d['split']=np.where(d.sample_id.isin(test),'test','train');d['seed']=SEED
assert len(d)==N and (d.split=='test').sum()==N_TEST and (d.split=='train').sum()==N-N_TEST
d.to_csv(M/'spatial_80_20_split.csv',index=False)
summary={'seed':SEED,'method':'random patch-level 80/20 split; all 1,713 processed samples retained','train_count':int((d.split=='train').sum()),'test_count':int((d.split=='test').sum()),'dropped_count':0}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'future_random_80_20_split_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
