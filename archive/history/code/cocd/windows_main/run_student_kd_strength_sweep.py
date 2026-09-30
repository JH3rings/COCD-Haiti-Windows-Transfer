"""Sequential KD-strength sweep: alpha=.5, only missing 5% and 20% arms."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from windows_main import student_search_kd_v2 as run
run.OUT=run.ROOT/'experiments'/'windows_main'/'student_target_search_v2'/'strength_sweep'
run.ARMS={'KD05':(.5,.5),'KD20':(.5,.5)}
run.KD_FRACTIONS={'KD05':.05,'KD20':.20}
run.main()
