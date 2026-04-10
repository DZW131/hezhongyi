# HuBMAP Glomerulus SAM2 Final Pipeline

This repository is the final engineered SAM2 route for HuBMAP kidney glomerulus segmentation.

It is designed to run alongside the U-Net route:

- `hezhongyi-unet_segm`: coarse semantic segmentation baseline
- `hezhongyi-sam2_segm`: promptable instance refinement route based on SAM2

The current final version is intentionally streamlined:

- training uses the existing `/root/datasets/HuBMAP_tiles_v2` tile dataset
- SAM2 is trained as a single-frame promptable model with `num_frames=1`
- the default backbone is official SAM2.1 B+
- source binary masks are auto-converted into pseudo-instances by connected components
- training, validation summary, offline evaluation, and inference are all kept in one clean path

## 1. Final workflow

The recommended server workflow is:

1. prepare the conda environment
2. download the official SAM2.1 B+ checkpoint
3. adapt `/root/datasets/HuBMAP_tiles_v2` into SAM2 format
4. train with `scripts/train_hubmap_sam2.py`
5. evaluate with `scripts/evaluate_hubmap_sam2.py`
6. infer with `scripts/predict_hubmap_sam2.py`

This avoids slow WSI re-decoding during training data preparation and keeps the U-Net vs SAM2 comparison fair because both routes use the same tile split.

## 2. Repository layout

```text
.
|-- README.md
|-- setup.py
|-- checkpoints/
|-- hubmap_sam2/
|   |-- dataset.py
|   |-- inference.py
|   |-- metrics.py
|   |-- meters.py
|   |-- prompts.py
|   |-- training_summary.py
|   `-- visualization.py
|-- sam2/
|   |-- configs/
|   |   `-- sam2.1_training/
|   `-- ...
|-- scripts/
|   |-- prepare_hubmap_sam2_from_tiles.py
|   |-- train_hubmap_sam2.py
|   |-- summarize_hubmap_sam2_run.py
|   |-- evaluate_hubmap_sam2.py
|   `-- predict_hubmap_sam2.py
`-- training/
```

## 3. Environment

Recommended environment:

- environment name: `sam2_segm`
- Python: `3.10`
- PyTorch: `2.5.1`
- torchvision: `0.20.1`

Create it:

```bash
conda create -n sam2_segm python=3.10 pip -y
conda activate sam2_segm
```

Install PyTorch. If your server supports CUDA 12.4:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

If needed, replace `12.4` with the CUDA version matched to `nvidia-smi`.

Install the repository:

```bash
cd /root/sam2_segm
pip install -U pip setuptools wheel
pip install -e .
```

Notes:

- `imagecodecs` is included because HuBMAP TIFFs and some downstream TIFF outputs depend on it
- the optional SAM2 CUDA extension may fail to build on some machines; the rest of the project can still work
- if needed, you can disable extension build:

```bash
export SAM2_BUILD_CUDA=0
pip install -e .
```

## 4. Base checkpoint

Default checkpoint:

```text
/root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt
```

Download it:

```bash
mkdir -p /root/sam2_segm/checkpoints
cd /root/sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Default config:

```text
configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml
```

## 5. Input dataset

The final training route uses the existing U-Net tile dataset:

```text
/root/datasets/HuBMAP_tiles_v2/
|-- train
|   |-- images/*.jpg
|   `-- masks/*.png
|-- val
|   |-- images/*.jpg
|   `-- masks/*.png
`-- manifests
    |-- slides.csv
    |-- tiles.csv
    `-- summary.json
```

Your current dataset characteristics are:

- image tiles are `.jpg`
- masks are binary `.png`
- mask values are `0` and `255`
- filenames are `slide_id_x_y`
- train/val split is already fixed

This repository adapts those binary masks into SAM2 pseudo-instances by connected components.

## 6. Adapt tiles to SAM2 format

Run:

```bash
cd /root/sam2_segm
conda activate sam2_segm

python scripts/prepare_hubmap_sam2_from_tiles.py \
  --source-root /root/datasets/HuBMAP_tiles_v2 \
  --output-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --link-mode auto \
  --min-instance-area 32 \
  --min-positive-pixels 64
```

What this script does:

- reuses the existing `train/val` split
- reuses the existing tile images through hardlink or copy
- converts binary masks into instance ID maps
- generates prompt metadata for each instance
- writes the SAM2 training directory structure

Output dataset:

```text
/root/datasets/HuBMAP_tiles_v2_sam2/
|-- train
|   |-- JPEGImages/<sample_id>/00000.jpg
|   |-- Annotations/<sample_id>/00000.png
|   |-- Metadata/<sample_id>.json
|   `-- list.txt
|-- val
|   |-- JPEGImages/<sample_id>/00000.jpg
|   |-- Annotations/<sample_id>/00000.png
|   |-- Metadata/<sample_id>.json
|   `-- list.txt
`-- manifests
    |-- samples.csv
    |-- slides.csv
    `-- summary.json
```

Check it:

```bash
ls /root/datasets/HuBMAP_tiles_v2_sam2/train/JPEGImages | head
ls /root/datasets/HuBMAP_tiles_v2_sam2/train/Annotations | head
cat /root/datasets/HuBMAP_tiles_v2_sam2/manifests/summary.json
```

## 7. Training

Main config:

```text
configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml
```

Final default behavior:

- official SAM2.1 B+
- `num_frames = 1`
- image encoder frozen
- `num_maskmem = 0`
- resolution `896`
- bfloat16 AMP
- TF32 enabled by default on GPU
- validation every 2 epochs
- point prompts always used
- small amount of box prompts mixed in

Train:

```bash
cd /root/sam2_segm
conda activate sam2_segm

python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --num-gpus 1 \
  --num-nodes 1
```

Useful speed knobs:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --resolution 896 \
  --num-workers 12 \
  --num-epochs 24 \
  --val-epoch-freq 2 \
  --compile-image-encoder
```

Resume:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --resume-from /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/latest.pt
```

## 8. Training outputs

Run directory:

```text
/root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/
|-- checkpoints
|   |-- checkpoint.pt
|   |-- latest.pt
|   |-- best.pt
|   `-- val_all_glomerulus_metrics_dice.pt
|-- logs
|   |-- train_stats.json
|   |-- val_stats.json
|   |-- best_stats.json
|   `-- log.txt
|-- tensorboard
`-- analysis
    |-- history.csv
    |-- training_curves.png
    |-- best_metrics.json
    |-- final_val_metrics.json
    `-- summary.json
```

The final standardized training-side validation file is:

```text
analysis/final_val_metrics.json
```

It contains:

- `Dice`
- `IoU`
- `Precision`
- `Recall`
- `Specificity`
- `Accuracy`
- `validation_loss`

You can regenerate summaries at any time:

```bash
python scripts/summarize_hubmap_sam2_run.py \
  --run-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus
```

## 9. Evaluation

Supported evaluation modes:

- `oracle-point`
- `oracle-box`
- `oracle-point-box`
- `amg`
- `prior-mask`

Recommended checkpoint evaluation:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/eval_oracle_point
```

Evaluate the hybrid U-Net -> SAM2 route:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode prior-mask \
  --prior-mask-dir /root/hezhongyi-unet_segm/outputs/hubmap_val_binary \
  --prompt-mode point_box \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/eval_prior_mask
```

Evaluation output:

- `metrics.json`
- `per_sample_metrics.csv`
- `previews/*.png`

`metrics.json` contains the standardized core fields:

- `Dice`
- `IoU`
- `Precision`
- `Recall`
- `Specificity`
- `Accuracy`
- `validation_loss`

For offline evaluation, `validation_loss` is recorded as `null` because the script is predictor-based rather than training-loop loss evaluation.

## 10. Inference

### 10.1 Tile inference

Pure SAM2 automatic prompts:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode tile \
  --prompt-source amg \
  --image /root/datasets/HuBMAP_tiles_v2_sam2/val/JPEGImages/0486052bb_9216_2048/00000.jpg \
  --output-dir /root/sam2_segm/inference/tile_amg
```

Prior-mask-guided tile refinement:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode tile \
  --prompt-source mask \
  --prompt-mode point_box \
  --image /root/datasets/HuBMAP_tiles_v2_sam2/val/JPEGImages/0486052bb_9216_2048/00000.jpg \
  --prior-mask /root/hezhongyi-unet_segm/outputs/sample_binary.png \
  --output-dir /root/sam2_segm/inference/tile_prior_mask
```

Tile outputs:

- `image.png`
- `instance_map.png`
- `binary_mask.png`
- `summary.json`

### 10.2 Whole-slide inference

Recommended production route:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --tile-selection auto \
  --output-dir /root/sam2_segm/inference/wsi_prior_mask_0486052bb
```

Whole-slide outputs:

- `instance_map.tiff`
- `binary_mask.tiff`
- `summary.json`

## 11. Core metrics

The final standardized metrics aligned with the U-Net route are:

- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- validation loss

Where they appear:

- training summary: `analysis/final_val_metrics.json`
- offline evaluation: `eval_*/metrics.json`

## 12. Recommended server commands

```bash
cd /root/sam2_segm
git fetch origin
git checkout sam2_segm
git pull origin sam2_segm
conda activate sam2_segm
```

```bash
python scripts/prepare_hubmap_sam2_from_tiles.py \
  --source-root /root/datasets/HuBMAP_tiles_v2 \
  --output-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --link-mode auto \
  --min-instance-area 32 \
  --min-positive-pixels 64
```

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus
```

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/eval_oracle_point
```

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --tile-selection auto \
  --output-dir /root/sam2_segm/inference/wsi_prior_mask_0486052bb
```

## 13. Practical notes

- preprocessing is now lightweight because it starts from the existing tile dataset instead of raw WSI decoding
- training will still be slower than U-Net because SAM2.1 B+ is a larger promptable model
- for no-GT deployment, the most practical route is still `U-Net coarse mask -> SAM2 refinement`
- for model comparison during development, `oracle-point` on the validation split is the cleanest checkpoint-selection signal
