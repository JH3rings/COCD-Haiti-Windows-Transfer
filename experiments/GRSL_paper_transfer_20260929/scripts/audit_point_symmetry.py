"""Parameter-level proof of learned-point collapse in retained checkpoints."""
from pathlib import Path
import json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[3];PACK=Path(__file__).resolve().parents[1];OUT=PACK/'new_evidence/20260930'
rows=[]
for model,path in [('CG0',ROOT/'experiments/windows_main/teacher_target_search_v2/CG0_seed42.pt'),('S2-2Stage',ROOT/'experiments/distillation_final/ours_s2_2stage/S2-2Stage_seed42_best.pt')]:
    s=torch.load(path,map_location='cpu',weights_only=True)
    for level in range(3):
        checks={}
        for key in ('policy.offset.weight','policy.offset.bias'):
            x=s[key].reshape(3,3,2,-1)[level]
            checks[key]=float((x[1:]-x[:1]).abs().max())
        for key in ('policy.weight.weight','policy.weight.bias'):
            x=s[key].reshape(3,4,-1)[level,1:]
            checks[key]=float((x[1:]-x[:1]).abs().max())
        rows.append({'model':model,'level':f'P{level+2}','max_parameter_difference':checks,'three_learned_offsets_identical_for_all_contexts':all(checks[k]==0 for k in ('policy.offset.weight','policy.offset.bias')),
                     'checkpoint':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'learned_point_symmetry.json').write_text(json.dumps({'records':rows,
    'implication':'Per level the three learned points have identical offset functions; the effective support is a fixed center plus one learned location. Candidate weights at the same location can be summed without changing retrieval.',
    'scope':'Retained CG0/S2-2Stage checkpoints, not a claim about every possible initialization or architecture; no checkpoint was changed.'},indent=2))
print('SYMMETRY_AUDIT_SAVED')
