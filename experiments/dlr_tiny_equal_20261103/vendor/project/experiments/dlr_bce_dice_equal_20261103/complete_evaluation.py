"""Complete omitted base test, then resume skipped/absent final evaluations only.
No training or checkpoint reselection. Preserve original failure and source.
"""
import json,torch
import run as r
r.b.setup();r.verify()
for arm in ['T_multimodal','T_optical','base','A','B','unet']:
 s=json.loads((r.OUT/arm/'selection.json').read_text())
 assert s['epochs_completed']==100 and r.b.sha(r.OUT/arm/'best.pt')==s['checkpoint_sha256']
dest=r.OUT/'base'
if not (dest/'test_metrics.json').exists():
 assert not (dest/'test_started.json').exists()
 r.b.write(dest/'test_started.json',{'all_six_frozen':True,'checkpoint_sha256':r.b.sha(dest/'best.pt'),'reason':'missing baseline evaluation in inherited final-tests routine'})
 m=r.ab.build_system('base');m.load_state_dict(torch.load(dest/'best.pt',weights_only=True,map_location='cpu'))
 stats=json.loads((r.OUT/'normalization.json').read_text())
 score=r.b.evaluate('base',m,r.b.DLRInputs(r.b.frames()['test'],stats,include_post=False),save=dest/'test_predictions.npz')
 r.b.write(dest/'test_metrics.json',score);del m;torch.cuda.empty_cache()
r.finish()
err=r.OUT/'error.json'
if err.exists():err.rename(r.OUT/'error.evaluation_preserved.json')
print((r.OUT/'all_six_comparison.csv').read_text(),flush=True)
