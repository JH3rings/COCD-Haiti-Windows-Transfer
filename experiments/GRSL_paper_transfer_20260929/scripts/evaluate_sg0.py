from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'))
from common import DEV,loader,metric_rows
from windows_main.data import HaitiPairs,split
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2

torch.set_num_threads(4)
_,_,ids=split();norm=np.load(OUT/'normalization.npz');ds=HaitiPairs(ids,stats=(norm['mean'],norm['std']))
model=TargetSearchTeacherV2().to(DEV).eval();ckpt=ROOT/'experiments/windows_main/teacher_target_search_v2/SG0_seed42.pt'
model.load_state_dict(torch.load(ckpt,map_location=DEV,weights_only=True))
parts=[[],[],[],[]];index=[]
with torch.inference_mode():
    for bi,(a,d,y,ga,gd,_) in enumerate(loader(ds)):
        oa,od=model.forward_pair(a.to(DEV),d.to(DEV),'self')
        for o,gt,gc,orbit in ((oa,ga,gd,'asc'),(od,gd,ga,'desc')):
            for accum,v in zip(parts,(torch.sigmoid(o['z'][:,0]).cpu().numpy(),y.numpy(),gt.numpy(),gc.numpy())):accum.append(v)
            for sid in ids[bi*16:bi*16+len(a)]:index.append({'row':len(index),'sample_id':sid,'orbit':orbit})
prediction=tuple(np.concatenate(p) for p in parts)
np.savez_compressed(OUT/'SG0_predictions.npz',p=prediction[0],y=prediction[1],target_geometry=prediction[2],counter_geometry=prediction[3],sample_id=np.array([r['sample_id'] for r in index]),orbit=np.array([r['orbit'] for r in index]))
rows=[{'arm':'SG0:retrained-self-guidance','region':r,'evidence':'existing independently trained SG0 checkpoint, re-evaluated',**v} for r,v in metric_rows(prediction).items()]
with (OUT/'current_sg0_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'SG0_source.json').write_text(json.dumps({'checkpoint':str(ckpt.relative_to(ROOT)),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'code':'cocd/windows_main/teacher_target_search_v2.py: ARMS SG0=(self,0)','normalization':'current training set mean/std from normalization.npz','distinction':'SG0 is independently trained; CG0:self-context is a frozen CG0 inference intervention'},indent=2))
print('SG0_COMPLETE')
