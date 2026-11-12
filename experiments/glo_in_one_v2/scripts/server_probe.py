import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch

root = Path(__file__).resolve().parents[1]
data = {'python':sys.version, 'hostname':platform.node(), 'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
        'packages':{n:bool(importlib.util.find_spec(n)) for n in ['imgaug','tensorboardX','scipy','sklearn','skimage']}}
for name, cmd in {'gpu':['nvidia-smi'], 'disk':['df','-h',str(root)], 'source_commit':['git','-C',str(root/'upstream'),'rev-parse','HEAD']}.items():
    result = subprocess.run(cmd, capture_output=True, text=True)
    data[name] = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
path = root/'model.pth'
if path.exists():
    with path.open('rb') as handle:
        data['weight_sha256'] = hashlib.file_digest(handle,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(handle.read()).hexdigest()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    data['weight_bytes'] = path.stat().st_size
    data['checkpoint_type'] = type(checkpoint).__name__
    data['checkpoint_keys_count'] = len(checkpoint)
    data['checkpoint_first_keys'] = list(checkpoint)[:6]
(root/'logs/environment.json').write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2))
