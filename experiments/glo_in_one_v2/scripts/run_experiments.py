"""Run staged official inference with CPU compatibility and audit all channels."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT/'.venv/bin/python')
SOURCE = Path(os.environ.get('HZY_GLO_DATASET', '/home/duyanhong/Dataspace/HZY/processed/hzy_intrinsic_cell_upgrade_43_20261021'))


def run(name, inputs):
    output = ROOT/'outputs'/name
    command = [PYTHON, '-u', 'Glo_v2_segmentation.py', '--dataset_dir', str(inputs),
               '--reload_path', str(ROOT/'model.pth'), '--output_folder', str(output)]
    started = time.monotonic()
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT/'compat_cpu', stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'{name} failed with code {result.returncode}; see logs')
    images = sorted(inputs.glob('*.png'))
    outputs = sorted(output.rglob('*_preds_merged.npy'))
    assert len(images) == len(outputs), (name, len(images), len(outputs))
    audit = {'name':name, 'command':command, 'returncode':result.returncode,
             'wall_seconds':time.monotonic()-started, 'device':'cpu','threads':4,'images':[]}
    for image in images:
        path = next(p for p in outputs if p.name == image.name+'_preds_merged.npy')
        array = np.load(path, allow_pickle=False)
        w,h = Image.open(image).size
        assert array.shape == (h,w,15), (path,array.shape)
        assert array.dtype != object and np.isfinite(array).all()
        assert np.isin(array, [0,1]).all()
        audit['images'].append({'image':image.name,'output':str(path),'shape':list(array.shape),
                                'dtype':str(array.dtype), 'nonzero_by_channel':array.sum((0,1)).astype(int).tolist(),
                                'fraction_by_channel':array.mean((0,1)).tolist(),
                                'overlapping_pixel_fraction':float((array.sum(2)>1).mean()),
                                'sha256':hashlib.sha256(array.tobytes()).hexdigest()})
    (ROOT/'logs'/f'{name}_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='images'}), flush=True)
    return audit


def visualize(audit, inputs, limit):
    folder = ROOT/'figures'/audit['name']
    folder.mkdir(parents=True,exist_ok=True)
    for item in audit['images'][:limit]:
        image = np.asarray(Image.open(inputs/item['image']).convert('RGB'))
        masks = np.load(item['output'])
        sample = folder/Path(item['image']).stem
        sample.mkdir(exist_ok=True)
        fig, axes = plt.subplots(4,4,figsize=(12,12))
        axes.flat[0].imshow(image); axes.flat[0].set_title('Original image')
        for channel, ax in enumerate(list(axes.flat)[1:]):
            mask = masks[:,:,channel]
            overlay = image.astype(float)/255
            overlay[mask] = .55*overlay[mask]+.45*np.array([0,1,.55])
            ax.imshow(overlay)
            ax.set_title(f'channel {channel:02d} | {mask.mean():.1%}',fontsize=10)
            Image.fromarray((mask*255).astype('uint8')).save(sample/f'channel_{channel:02d}_mask.png')
            Image.fromarray((overlay*255).astype('uint8')).save(sample/f'channel_{channel:02d}_overlay.png')
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle('Independent binary channels; class names provisional',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(sample/'all_15_channels.png',dpi=150)
        plt.close(fig)


def main():
    first = json.loads((ROOT/'logs/demo_audit.json').read_text()) if (ROOT/'logs/demo_audit.json').exists() else None
    if first is None:
        # Audit the initial successful run through a repeatable official-entry run.
        first = run('demo', ROOT/'data/official_demo')
    repeated = run('demo_repeat', ROOT/'data/official_demo')
    stable = {a['image']: a['sha256'] for a in first['images']} == {a['image']:a['sha256'] for a in repeated['images']}
    assert stable, 'Repeated CPU inference differs'
    (ROOT/'logs/repeatability.json').write_text(json.dumps({'identical':stable,'images':len(first['images'])}))
    visualize(first,ROOT/'data/official_demo',limit=2)
    background = ROOT/'data/background'
    background.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(512,512),'white').save(background/'background.png')
    run('background',background)
    rows = list(csv.DictReader((SOURCE/'manifest.csv').open(encoding='utf-8-sig')))
    groups = {}
    for row in rows:
        if row['split']=='test':
            groups.setdefault(row['case_key'],[]).append(row)
    for group in groups.values():
        group.sort(key=lambda row:row['crop_id'])
    selected = []
    for index in range(max(map(len,groups.values()))):
        for case in sorted(groups):
            if index < len(groups[case]) and len(selected)<30:
                selected.append(groups[case][index])
    staging = ROOT/'data/local_patches'
    staging.mkdir(parents=True,exist_ok=True)
    manifest = []
    for number,row in enumerate(selected,1):
        src = SOURCE/'test_image'/(row['crop_id']+'.jpg')
        name = f'local_{number:03d}.png'
        with Image.open(src) as image:
            image.convert('RGB').save(staging/name)
        manifest.append({'image_id':name,'patient_id':row['case_key'],'wsi_id':row['slide_id'],
                         'species':'human','stain':'unverified','resolution_or_mpp':'unverified',
                         'source_path':str(src),'gt_path':str(SOURCE/'test_point'/(row['crop_id']+'.json')),
                         'annotation_type':'three-class cell points and glomerular ROI; no 15-channel pixel GT',
                         'approved_for_use':True,'conversion':'RGB JPEG to lossless PNG; dimensions unchanged'})
    with (ROOT/'data/local_manifest.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    audit=run('local_zero_shot',staging)
    visualize(audit,staging,limit=3)
    (ROOT/'logs/completed.json').write_text(json.dumps({'official_demo_count':len(first['images']),
            'local_count':len(manifest),'local_cases':len(groups),'repeat_identical':stable,
            'pixel_metrics':'N/A; no confirmed matching pixel labels'},indent=2))


if __name__=='__main__':
    main()
