"""Prepare an audited CPU-only copy of the upstream inference code."""
import difflib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source = root / 'upstream'
target = root / 'compat_cpu'
shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '__pycache__', 'apex'), dirs_exist_ok=True)
patches = []
for path in sorted(target.glob('*.py')):
    before = path.read_text()
    after = before
    if path.name in ('testing_2D_patch_v2.py', 'unet2D_ns_for_Glo_v2.py'):
        after = after.replace('.cuda()', '.cpu()')
        after = after.replace("torch.device('cuda:{}'.format(args.local_rank))", "torch.device('cpu')")
        after = after.replace('torch.cuda.FloatTensor(', 'torch.FloatTensor(')
    if path.name == 'testing_2D_patch_v2.py':
        after = after.replace('from apex import amp', '# Apex not required for FP32 single-process inference')
        after = after.replace('from apex.parallel import convert_syncbn_model', '')
        after = after.replace('num_worker = 8', 'num_worker = 0')
        after = after.replace("torch.load(args.reload_path, map_location=torch.device('cpu'))", "torch.load(args.reload_path, map_location=torch.device('cpu'), weights_only=True)")
        after = after.replace('import torch\n', 'import torch\ntorch.set_num_threads(4)\n', 1)
    if path.name == 'engine.py':
        start = after.index('try:\n    from apex.parallel')
        end = after.index('\n\nlogger', start)
        after = after[:start] + '# Apex is unused in single-process CPU inference.\n' + after[end:]
    if path.name in ('MOTSDataset_2D_Patch_normal_save_csv_Glo_v2.py', 'MOTSDataset_2D_Patch_supervise_normal_csv_Glo_v2.py'):
        after = after.replace("matplotlib.use('TkAgg')", "matplotlib.use('Agg')")
        after = after.replace('import nibabel as nib', '# unused nibabel import removed for patch inference')
        after = after.replace('import SimpleITK as sitk', '# unused SimpleITK import removed for patch inference')
    if path.name == 'Glo_v2_segmentation.py':
        after = after.replace('import subprocess', 'import subprocess\nimport sys', 1)
        after = after.replace('"python",', 'sys.executable,')
        after = after.replace('subprocess.run(save_csv)', 'subprocess.run(save_csv, check=True)')
        after = after.replace('subprocess.run(testing)', 'subprocess.run(testing, check=True)')
    if after != before:
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='upstream/'+path.name, tofile='compat_cpu/'+path.name))
(root/'reports').mkdir(exist_ok=True)
(root/'reports/cpu_compat.patch').write_text(''.join(patches))
demo = root/'data/official_demo'
demo.mkdir(parents=True, exist_ok=True)
for path in sorted((source/'Test_Patch').glob('*.png')):
    shutil.copy2(path, demo/path.name)
print(json.dumps({'compat':str(target),'demo_images':len(list(demo.glob('*.png')))}))
