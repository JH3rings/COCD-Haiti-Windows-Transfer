import importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=R/'experiments/dataset_audit_final/99_cleanup_manifest/legacy_scripts/run_extended_audit.py'
s=importlib.util.spec_from_file_location('audit_engine',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.ROOT=R;m.DATA=R/'data/haiti';m.OLD=R/'experiments/dataset_audit_final/05_existing_results_reused/experiments/windows_main/physics_audit_20260924';m.DEM_DIR=m.OLD/'dem';m.OUT=R/'experiments/dataset_audit_final/01_statistics';m.main()
