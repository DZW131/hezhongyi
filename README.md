# HuBMAP Glomeruli Segmentation Pipeline

This repository is a PyTorch U-Net project adapted for HuBMAP kidney glomeruli segmentation on a Linux server.

If you want a more presentation-oriented explanation of the project goals, strategy, current results, and suggested reporting language for teachers or doctors, see [PROJECT_PRESENTATION_GUIDE.md](PROJECT_PRESENTATION_GUIDE.md).

The current workflow is designed for the dataset layout you already have on the server:

```text
/root/datasets/HuBMAP/train/
  0486052bb.tiff
  0486052bb.json
  0486052bb-anatomical-structure.json
  095bf7a1f.tiff
  095bf7a1f.json
  095bf7a1f-anatomical-structure.json
  ...
```

You do not need to rename or manually convert the HuBMAP dataset before using this project.

The repository now supports:

- raw TIFF + polygon JSON annotation input
- optional anatomical ROI filtering from `*-anatomical-structure.json`
- train/val tile generation
- training with richer metrics
- best/latest checkpoint saving
- local training curve and preview generation
- standalone checkpoint evaluation
- TIFF and HSPN inference scripts
- performance-oriented training options for a 4090 server

## 1. Repository layout

```text
.
|-- train.py
|-- evaluate.py
|-- evaluate_checkpoint.py
|-- predict.py
|-- predict_tiff.py
|-- predict_hspn_enhanced.py
|-- predict_hspn_stain_norm.py
|-- requirements.txt
|-- scripts
|   |-- prepare_hubmap_tiles.py
|   |-- plot_training_history.py
|   |-- download_data.sh
|   `-- download_data.bat
|-- unet
|   |-- unet_model.py
|   `-- unet_parts.py
`-- utils
    |-- data_loading.py
    |-- dice_score.py
    |-- segmentation_metrics.py
    |-- visualization.py
    `-- utils.py
```

## 2. What the HuBMAP files mean

For each slide, the project expects:

- `slide_id.tiff`
  - the raw whole-slide image
- `slide_id.json`
  - polygon annotations for glomeruli
  - the script looks for `properties.classification.name == "glomerulus"`
- `slide_id-anatomical-structure.json`
  - polygon annotations for anatomical structures
  - in your data these include labels such as `Cortex` and `Medulla`

This means the pipeline can:

1. read the TIFF slide
2. rasterize glomerulus polygons into a binary mask
3. optionally rasterize anatomical polygons such as `Cortex`
4. tile the slide into trainable patches
5. train U-Net on the generated tile dataset

## 3. Recommended server environment

Recommended setup:

- OS: Linux
- GPU: NVIDIA RTX 4090
- Python: 3.10+
- PyTorch: 2.x
- CUDA: version matched to the server driver

Notes:

- `requirements.txt` does not install `torch`. Install the correct CUDA-compatible PyTorch build first.
- `wandb` is optional. If it is missing, training still works.

Example environment setup:

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master

conda activate unet_kidney

# Install a CUDA-compatible torch build first, based on your server setup.
# Then install project dependencies.
pip install -r requirements.txt
```

## 4. Update the old server project

If your server still has the old project version, update it to the `unet_segm` branch:

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master
git fetch origin
git checkout unet_segm
git pull origin unet_segm
```

After that, the old workflow using in-repo `data/imgs` and `data/masks` is no longer required. You can keep the raw HuBMAP dataset unchanged in `/root/datasets/HuBMAP`.

## 5. Data preparation

### 5.1 Input data

Raw input slides stay where they already are:

```text
/root/datasets/HuBMAP/train
```

### 5.2 Output tiles

Recommended output tile directory:

```text
/root/datasets/HuBMAP_tiles
```

The script will create:

```text
/root/datasets/HuBMAP_tiles/
|-- train
|   |-- images
|   `-- masks
|-- val
|   |-- images
|   `-- masks
`-- manifests
    |-- tiles.csv
    |-- slides.csv
    `-- summary.json
```

### 5.3 Tile generation script

Main script:

```bash
python scripts/prepare_hubmap_tiles.py -h
```

What it does:

- reads each TIFF slide
- reads the matching glomerulus polygon JSON
- converts polygons to a binary mask
- optionally reads anatomical JSON and keeps only selected ROI labels such as `Cortex`
- tiles the slide into patches
- filters empty background
- keeps a balanced set of positive and negative patches
- splits slides into train and val sets
- saves manifests for reproducibility

### 5.4 Recommended command for your server dataset

This command is the best starting point for your current dataset layout:

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

Why this works for your dataset:

- `--images-dir` points to the existing directory that already contains `.tiff`, `.json`, and `-anatomical-structure.json`
- the script automatically detects `slide_id.json` as the glomerulus annotation source
- `--roi-labels Cortex` restricts tile generation to cortex regions using `slide_id-anatomical-structure.json`
- if a slide does not contain `Cortex`, the default policy is now to skip that slide instead of crashing the whole run
- TIFF slides with extra singleton dimensions or `CHW` channel order are normalized automatically before tiling

### 5.5 Important preprocessing options

- `--tile-size`
  - tile size before optional resizing
- `--stride`
  - sliding window step
- `--downsample`
  - optional resizing after tiling
- `--val-ratio`
  - validation split ratio at the slide level
- `--target-labels`
  - annotation labels to treat as positive, default is `glomerulus`
- `--roi-labels`
  - anatomical labels to keep, for example `Cortex`
- `--min-roi-coverage`
  - minimum ROI overlap required for a tile when ROI labels are used
- `--missing-roi-policy`
  - controls what happens if a requested ROI label is missing on a slide
  - `skip-slide` (default): skip the slide
  - `ignore-roi`: process the slide without ROI filtering
  - `error`: stop immediately
- `--min-tissue-coverage`
  - filters nearly empty white background patches
- `--min-positive-pixels`
  - minimum positive pixels required to force a tile to be positive
- `--negative-ratio`
  - number of negative tiles kept per positive tile

### 5.6 If you want to use all anatomical regions

If you do not want cortex filtering:

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --tile-size 1024 \
  --stride 1024
```

## 6. Training

### 6.1 Recommended checkpoint directory

Recommended checkpoint output directory:

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

This keeps the training artifacts inside the project directory while leaving the original HuBMAP dataset unchanged.

### 6.2 Recommended training command

```bash
python train.py \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --checkpoint-metric dice \
  --analysis-frequency 5 \
  --preview-frequency 5 \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

### 6.3 Metrics tracked during training

The training pipeline now records:

- validation loss
- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- epoch time
- training images per second

### 6.4 Training outputs

With the checkpoint directory above, training produces:

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/
|-- best.pth
|-- latest.pth
|-- epoch_001.pth                 # only if --save-every-epoch is used
`-- analysis
    |-- history.csv
    |-- training_curves.png
    |-- best_preview.png
    |-- best_metrics.json
    `-- val_previews
```

What they mean:

- `best.pth`
  - best checkpoint according to `--checkpoint-metric`
- `latest.pth`
  - most recent checkpoint
- `history.csv`
  - epoch-level metrics and timing
- `training_curves.png`
  - local overview plot of training progress
- `best_preview.png`
  - best validation example visualization
- `best_metrics.json`
  - saved metric summary for the best checkpoint

### 6.5 Throughput optimizations already included

The training script includes server-oriented speedups:

- AMP support
- TF32 enabled by default on CUDA
- cuDNN benchmark enabled by default
- non-blocking host-to-device copies
- dataloader `persistent_workers`
- dataloader `prefetch_factor`
- optional `torch.compile`
- mask-value caching in the mask directory
- configurable validation and preview frequency

### 6.6 Resume training

```bash
python train.py \
  --load /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/latest.pth \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --classes 2 \
  --amp \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

## 7. Standalone evaluation

Use this script to evaluate a saved checkpoint on a prepared image/mask dataset:

```bash
python evaluate_checkpoint.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --images-dir /root/datasets/HuBMAP_tiles/val/images \
  --masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --classes 2 \
  --batch-size 2 \
  --num-workers 8 \
  --amp \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/eval_best
```

Outputs:

- `metrics.json`
- `preview.png`

## 8. Visualization

### 8.1 Generated automatically during training

Training automatically generates:

- `analysis/history.csv`
- `analysis/training_curves.png`
- `analysis/best_preview.png`
- `analysis/val_previews/*.png`

### 8.2 Regenerate the training curve image

```bash
python scripts/plot_training_history.py \
  --history-csv /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/analysis/history.csv \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/analysis/training_curves_regenerated.png
```

## 9. Inference

### 9.1 Standard image inference

For regular `.jpg` or `.png` images:

```bash
python predict.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input demo.jpg \
  --output demo_mask.png \
  --classes 2
```

### 9.2 TIFF inference for large HuBMAP slides

```bash
python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/HuBMAP/test/2ec3f1bb9.tiff \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/output_2ec3f1bb9.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2
```

Use the same `--scale` value that was used during training. For the current HuBMAP baseline in this repository, that value is `0.5`.

### 9.3 HSPN inference with contrast enhancement

```bash
python predict_hspn_enhanced.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/hspn_enhanced_pred.png \
  --classes 2 \
  --scale 0.5 \
  --threshold 0.5 \
  --tile-size 1024
```

### 9.4 HSPN inference with stain normalization

```bash
python predict_hspn_stain_norm.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/hspn_stain_norm_pred.png \
  --classes 2 \
  --scale 0.5 \
  --threshold 0.5 \
  --tile-size 1024
```

### 9.5 Recommended testing strategy for in-hospital slides

When you test internal hospital slides, do not assume the HSPN-specific scripts will always be better than direct inference. In practice:

- `predict_tiff.py` is the baseline and should always be tested first
- `predict_hspn_enhanced.py` can improve recall on pale slides, but it can also over-segment
- `predict_hspn_stain_norm.py` is useful when the staining style is clearly different from HuBMAP, but it should still be compared against the direct baseline

Recommended order:

1. Run direct inference first with the same `--scale` used during training.
2. If the direct result is too conservative or misses obvious glomeruli, try `predict_hspn_enhanced.py`.
3. If the staining style is strongly shifted, also try `predict_hspn_stain_norm.py`.
4. Compare the masks side by side and prefer the result that is medically plausible, not simply the one with the largest positive area.

Recommended settings for the current HuBMAP baseline:

- always use `--classes 2`
- always use `--scale 0.5`
- start from `--threshold 0.5`
- if `predict_hspn_enhanced.py` produces too much foreground, increase the threshold and compare `0.6`, `0.7`, and `0.8`

Important note:

- for the current two-class checkpoint, the HSPN scripts interpret class 1 as the glomerulus class
- `predict_hspn_enhanced.py` now uses the positive-class probability threshold when `--classes 2`, so the threshold is meaningful for suppressing over-segmentation

## 10. End-to-end workflow for your server

### Step 1. Update code

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master
git fetch origin
git checkout unet_segm
git pull origin unet_segm
```

### Step 2. Install dependencies

```bash
conda activate unet_kidney
pip install -r requirements.txt
```

### Step 3. Prepare HuBMAP tiles

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

### Step 4. Train

```bash
python train.py \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

### Step 5. Evaluate the best checkpoint

```bash
python evaluate_checkpoint.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --images-dir /root/datasets/HuBMAP_tiles/val/images \
  --masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --classes 2 \
  --amp \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/eval_best
```

### Step 6. Run inference

```bash
python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/HuBMAP/test/2ec3f1bb9.tiff \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/output_2ec3f1bb9.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2
```

## 11. Practical notes

### 11.1 Why use `--roi-labels Cortex`

Your anatomical JSON files contain `Cortex` and `Medulla`. Glomeruli are expected in cortex regions, so using:

```bash
--roi-labels Cortex
```

usually reduces wasted tiles and speeds up downstream training.

### 11.2 Why slide-level splitting matters

Train/val split should happen at the slide level, not the patch level. Otherwise neighboring patches from the same slide may appear in both train and validation sets, which makes validation metrics look overly optimistic.

### 11.3 Old in-repo `data/imgs` and `data/masks`

Your older project version used:

```text
/root/Pytorch-UNet/Pytorch-UNet-master/data/imgs
/root/Pytorch-UNet/Pytorch-UNet-master/data/masks
```

The new recommended workflow no longer depends on these directories. The project can generate and read tiles directly from external dataset directories such as:

```text
/root/datasets/HuBMAP_tiles
```

### 11.4 Best checkpoint selection

The training loop now keeps:

- `best.pth`
- `latest.pth`

so you do not need to manually guess which `checkpoint_epochXX.pth` to use.

### 11.5 If `torch.compile` causes issues

Disable it with:

```bash
--compile off
```

### 11.6 If you want every epoch checkpoint

Use:

```bash
--save-every-epoch
```

## 12. Current project status

This repository is now structured as a complete HuBMAP glomeruli segmentation project for your server workflow:

- raw dataset stays unchanged
- the project performs annotation parsing and tiling
- the project trains directly from generated tiles
- the project saves best/latest checkpoints
- the project provides evaluation and visualization tools
- the project supports TIFF and HSPN inference

## 13. Upstream origin

The project started from `milesial/Pytorch-UNet` and has been adapted into a HuBMAP-focused engineering workflow.
