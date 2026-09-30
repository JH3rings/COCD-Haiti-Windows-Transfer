"""Four-point, sequential internal-ratio sweep at locked initial KD/Seg=10%."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows_main import student_search_kd_v2 as run
run.OUT = run.ROOT / 'experiments' / 'windows_main' / 'student_target_search_v2' / 'weight_sweep'
run.ARMS = {'P0E100':(0.0,1.0),'P25E75':(0.25,0.75),'P50E50':(0.5,0.5),'P100E0':(1.0,0.0)}
run.main()
