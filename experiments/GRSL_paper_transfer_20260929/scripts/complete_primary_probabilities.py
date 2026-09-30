"""Recover current Hinton scores and verify external cache sample indexing."""
from pathlib import Path
import sys,csv,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import DEV,loader,metric_rows
from windows_main.data import HaitiPairs,split
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2

torch.set_num_threads(4)
_,_,ids=split();norm=np.load(OUT/'normalization.npz');ds=HaitiPairs(ids,stats=(norm['mean'],norm['std']))
model=StudentTargetSearchV2().to(DEV).eval();ckpt=ROOT/'experiments/distillation_final/hinton_kd/Hinton-KD-Port_seed42_best.pt'
model.load_state_dict(torch.load(ckpt,map_location=DEV,weights_only=True))
parts=[[],[],[],[]];samples=[];orbits=[]
with torch.inference_mode():
    for bi,(a,d,y,ga,gd,_) in enumerate(loader(ds)):
        for x,gt,gc,orbit in ((a,ga,gd,'asc'),(d,gd,ga,'desc')):
            p=torch.sigmoid(model(x.to(DEV))['z'][:,0]).cpu().numpy()
            for accum,v in zip(parts,(p,y.numpy(),gt.numpy(),gc.numpy())):accum.append(v)
            samples.extend(ids[bi*16:bi*16+len(a)]);orbits.extend([orbit]*len(a))
prediction=tuple(np.concatenate(p) for p in parts)
np.savez_compressed(OUT/'Hinton_current_predictions.npz',p=prediction[0],y=prediction[1],target_geometry=prediction[2],counter_geometry=prediction[3],sample_id=samples,orbit=orbits)
metrics=metric_rows(prediction)
recorded=json.loads((ckpt.parent/'summary.json').read_text())['metrics']
delta={r:{k:metrics[r][k]-recorded[r][k] for k in ('AUPRC','IoU','F1')} for r in metrics}
assert max(abs(v) for region in delta.values() for v in region.values())<1e-7
boehm=np.load(ROOT/'experiments/single_orbit_baselines_v3/boehm/test_predictions.npz')
checks={name:bool(np.array_equal(boehm[name],prediction[j])) for name,j in [('y',1),('gt',2),('gc',3)]}
checks['orbit']=bool(np.array_equal(boehm['orbit'],np.array([0 if o=='asc' else 1 for o in orbits])))
assert all(checks.values()),checks
index=[{'row':i,'sample_id':sid,'orbit':o} for i,(sid,o) in enumerate(zip(samples,orbits))]
with (OUT/'Boehm_verified_prediction_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
(OUT/'primary_probability_verification.json').write_text(json.dumps({'Hinton_recomputed_metrics':metrics,'Hinton_minus_recorded':delta,'Boehm_index_checks':checks,
    'Boehm_order':'identical labels, target/counter geometry and orbit arrays to live pinned dataset in canonical batch order; sample IDs reconstructed from pinned index'},indent=2))
print('PRIMARY_PROBABILITIES_VERIFIED')
