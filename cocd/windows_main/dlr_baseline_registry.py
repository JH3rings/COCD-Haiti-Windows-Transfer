"""Read and hash-verify the user-frozen DLR baseline; never train or retest it."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 return h.hexdigest()

def resolve(root=None,expected_initial_sha256=None,expected_split_sha256=None,
            expected_normalization_sha256=None):
 root=Path(root) if root is not None else ROOT
 registry=json.loads((root/'configs/active_dlr_baseline.json').read_text(encoding='utf-8'))
 assert registry['policy']=='REUSE_ONLY_NO_RETRAIN_NO_RETEST'
 for item in registry['locked_artifacts'].values():
  path=root/item['path']
  if not path.is_file() or sha256(path)!=item['sha256']:raise RuntimeError(f'Canonical DLR baseline artifact mismatch: {path}. Do not replace/retrain silently.')
 for key,wanted in [('initial',expected_initial_sha256),('split',expected_split_sha256),('normalization',expected_normalization_sha256)]:
  if wanted is not None and registry['locked_artifacts'][key]['sha256']!=wanted:
   raise ValueError(f'{key} differs from canonical baseline. Report unmatched protocol; do not auto-retrain.')
 return {'registry':registry,
         'checkpoint':root/registry['locked_artifacts']['checkpoint']['path'],
         'clean_initialization':root/registry['locked_artifacts']['initial']['path'],
         'metrics':json.loads((root/registry['locked_artifacts']['metrics']['path']).read_text()),
         'predictions':root/registry['locked_artifacts']['predictions']['path']}

def forbid_baseline_training():
 raise RuntimeError('User froze the DLR SAR baseline. Reuse configs/active_dlr_baseline.json; do not train/test another baseline without new explicit instruction.')

if __name__=='__main__':
 result=resolve();print(json.dumps({'status':'VERIFIED_REUSE_ONLY',
  'checkpoint':str(result['checkpoint']),
  'independent44_IoU':result['metrics']['per_official_subset']['testind']['IoU'],
  'all120_IoU':result['metrics']['IoU']},indent=2))
