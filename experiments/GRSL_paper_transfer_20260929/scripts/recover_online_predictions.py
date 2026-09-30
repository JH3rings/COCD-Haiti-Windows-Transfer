"""Re-infer retained online-distillation branches for publication figures."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
PACK=Path(__file__).resolve().parents[1];ROOT=PACK.parents[1];OUT=PACK/'new_evidence/20260930';SRC=ROOT/'experiments/GRSL_complete_paper_package/baselines/mmokd_port'
sys.path.insert(0,str(ROOT/'experiments/distillation_final'));sys.path.insert(0,str(SRC))
from common import DEV,loader,metric_rows,SO_INIT
from windows_main.data import HaitiPairs,split
from deploy import load_student
torch.set_num_threads(4)
_,_,ids=split();norm=np.load(OUT/'normalization.npz');ds=HaitiPairs(ids,stats=(norm['mean'],norm['std']))
models={orbit:load_student(orbit,SRC/f'student_{orbit}_best.pt',SO_INIT,DEV) for orbit in ('asc','desc')}
parts=[[],[],[],[]];samples=[];orbits=[]
with torch.inference_mode():
    for bi,(a,d,y,ga,gd,_) in enumerate(loader(ds)):
        for x,gt,gc,orbit in [(a,ga,gd,'asc'),(d,gd,ga,'desc')]:
            p=torch.softmax(models[orbit](x.to(DEV))['logits'],dim=1)[:,1].cpu().numpy()
            for accum,v in zip(parts,(p,y.numpy(),gt.numpy(),gc.numpy())):accum.append(v)
            samples.extend(ids[bi*16:bi*16+len(a)]);orbits.extend([orbit]*len(a))
prediction=tuple(np.concatenate(x) for x in parts);metrics=metric_rows(prediction)
np.savez_compressed(OUT/'Online_current_predictions.npz',p=prediction[0],y=prediction[1],target_geometry=prediction[2],counter_geometry=prediction[3],sample_id=samples,orbit=orbits)
cache=np.load(ROOT/'experiments/GRSL_complete_paper_package/02_qualitative/final_case_data/test_prediction_cache.npz');idx={(s,o):i for i,(s,o) in enumerate(zip(samples,orbits))};join=np.array([idx[(int(s),str(o))] for s,o in zip(cache['sample_id'],cache['orbit'])]);assert np.array_equal(cache['y'],prediction[1][join]);assert np.array_equal(cache['target_geometry'],prediction[2][join]);delta=prediction[0][join]-cache['p_MMOKD_Port'];recorded=json.loads((SRC/'summary.json').read_text())
report={'source_branches':{o:{'path':str(SRC/f'student_{o}_best.pt'),'sha256':hashlib.sha256((SRC/f'student_{o}_best.pt').read_bytes()).hexdigest()} for o in ('asc','desc')},'metrics':metrics,'cached_probability_max_abs_difference':float(np.abs(delta).max()),'cached_probability_mean_abs_difference':float(np.abs(delta).mean()),'threshold_prediction_disagreements':int(((prediction[0][join]>=.5)!=(cache['p_MMOKD_Port']>=.5)).sum()),'note':'Fresh retained-checkpoint inference for figures; recorded main-table numbers are preserved.'}
(OUT/'online_prediction_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
