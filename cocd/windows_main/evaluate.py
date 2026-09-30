"""Target-only checkpoint evaluation; geometry defines output subsets only."""
from pathlib import Path
import argparse,sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'cocd'))
from windows_main.protocol import DEV,make_loaders,student_predictions,metric_rows,write_json
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    _,test,_=make_loaders();model=StudentTargetSearchV2().to(DEV)
    state=torch.load(args.checkpoint,map_location=DEV,weights_only=True);model.load_state_dict(state,strict=True)
    write_json(args.output,metric_rows(student_predictions(model,test)))

if __name__=='__main__':main()
