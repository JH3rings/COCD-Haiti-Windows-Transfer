"""Read-only frozen-checkpoint evaluation. Never trains or overwrites run evidence."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/distillation_final'))
from common import DEV, loader, metric_rows
from windows_main.data import HaitiPairs, split
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2
from models.landslide_cocd import ConvNeXtTinyFPN

SOURCES = {
    'baseline': 'experiments/windows_main/student_target_search_v2/S0_seed42.pt',
    'student': 'experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt',
    'multi-to-uni': 'experiments/distillation_final/multi2uni_port/Multi2Uni-Port_deploy_student_seed42_best.pt',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list(SOURCES)+['all'], default='student')
    parser.add_argument('--smoke', action='store_true', help='One real location; not a benchmark result')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT/'paper_release/local_predictions'
    out.mkdir(parents=True, exist_ok=True)
    norm = json.loads((ROOT/'paper_release/protocol.json').read_text())['normalization']
    _, _, ids = split()
    ds = HaitiPairs(ids[:1] if args.smoke else ids, stats=(np.array(norm['mean']), np.array(norm['std'])))
    methods = list(SOURCES) if args.method=='all' else [args.method]
    for name in methods:
        model = (ConvNeXtTinyFPN() if name=='multi-to-uni' else StudentTargetSearchV2()).to(DEV).eval()
        source = ROOT/SOURCES[name]
        model.load_state_dict(torch.load(source, map_location=DEV, weights_only=True), strict=True)
        arrays = [[],[],[],[]]; sid=[]; orbit=[]
        with torch.inference_mode():
            for bi,(a,d,y,ga,gd,_) in enumerate(loader(ds)):
                for x,g,c,o in ((a,ga,gd,'asc'),(d,gd,ga,'desc')):
                    z = model.decode(*model.encode(x.to(DEV))) if name=='multi-to-uni' else model(x.to(DEV))['z']
                    for acc,v in zip(arrays,(torch.sigmoid(z[:,0]).cpu().numpy(),y.numpy(),g.numpy(),c.numpy())):
                        acc.append(v)
                    sid.extend(ds.ids[bi*16:bi*16+len(a)]); orbit.extend([o]*len(a))
        pred=tuple(np.concatenate(x) for x in arrays)
        stem=name+('-smoke' if args.smoke else '')
        report={'method':name,'smoke_only':args.smoke,'checkpoint':SOURCES[name],
                'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'metrics':metric_rows(pred),'locations':len(ds),'device':str(DEV),'batch_size':16}
        (out/(stem+'.json')).write_text(json.dumps(report,indent=2))
        if not args.smoke:
            np.savez_compressed(out/(stem+'.npz'),p=pred[0],y=pred[1],target_geometry=pred[2],counter_geometry=pred[3],sample_id=sid,orbit=orbit)
        print(name,report['metrics']['Overall'],flush=True)

if __name__=='__main__': main()
