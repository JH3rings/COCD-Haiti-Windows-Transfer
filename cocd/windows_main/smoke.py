"""Final API and dummy-forward smoke tests, with no optimizer or training."""
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'cocd'))
from windows_main.data import HaitiPairs,split
from windows_main.models_da_search import DeformableSearch
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2
from windows_main.models_student_target_search_v2 import StudentTargetSearchV2

def main():
    torch.set_num_threads(4);a=torch.randn(1,5,128,128);d=torch.randn_like(a)
    teacher=TargetSearchTeacherV2().eval();student=StudentTargetSearchV2().eval()
    with torch.no_grad():
        t=teacher(a,d);s=student(a)
        assert t['z'].shape==s['z'].shape==(1,1,128,128)
        assert torch.isfinite(t['z']).all() and torch.isfinite(s['z']).all()
        assert all(torch.equal(x,y) for x,y in zip(t['values'],teacher(a,d*0)['values']))
    print(json.dumps({'Teacher_import':True,'Student_import':True,'dataset_API_import':True,'search_import':True,'dummy_forward':True,'no_training':True}))

if __name__=='__main__':main()
