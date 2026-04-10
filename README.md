# HuBMAP Glomerulus SAM2 Pipeline

This repository is a SAM2-based HuBMAP kidney glomerulus segmentation project designed for Linux server training and inference.

It is intended to be the second technical route alongside your U-Net project:

- `hezhongyi-unet_segm`: coarse semantic segmentation / candidate generation
- `hezhongyi-sam2_segm`: promptable instance refinement with SAM2

The repository has been adapted from the original SAM2 training stack into a HuBMAP-focused engineering workflow:

- raw HuBMAP slides stay outside the repository
- the repository provides dataset preparation scripts
- train/val split is done by slide
- SAM2 is trained as a single-frame promptable model with `num_frames=1`
- validation, checkpointing, summaries, and inference scripts are included
- the hybrid route `U-Net coarse mask -> SAM2 refinement` is supported

## 1. Current scope

This repository now supports:

- complete in-repo `sam2/` package and training config tree
- HuBMAP TIFF + polygon JSON preprocessing
- slide-level train/val split
- optional anatomical ROI filtering from `*-anatomical-structure.json`
- instance-preserving tile generation
- automatic prompt generation from GT instances
- single-frame SAM2 fine-tuning
- validation metrics: Dice, IoU, Precision, Recall, Specificity, Accuracy
- instance-level matching metrics for evaluation
- `best` / `latest` checkpoint saving
- training history export and curve plotting
- tile inference
- whole-slide patch-based inference
- two no-GT inference routes:
  - pure SAM2 automatic mask generation (`amg`)
  - coarse-mask-driven refinement (`mask`)

## 2. Recommended deployment layout

You can clone this repository anywhere on the server. In the examples below we use:

```text
/root/sam2_segm
```

Raw HuBMAP data stays outside the repository:

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

Recommended generated dataset root:

```text
/root/datasets/HuBMAP_sam2
```

Recommended training output root:

```text
/root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

## 3. Core design choices

### 3.1 Single-frame SAM2

This project treats each tile as a one-frame video:

- `num_frames = 1`
- the original SAM2 training framework is reused
- the model remains a promptable segmentation model, not a plain binary segmenter

### 3.2 Instance-level training targets

HuBMAP glomerulus polygons are preserved as tile-local instance IDs:

- each visible glomerulus becomes one instance in the training annotation PNG
- prompt metadata is stored per instance
- slide-level splitting prevents train/val leakage

### 3.3 Automatic prompt generation

Training and oracle evaluation prompts are derived automatically from GT masks:

- point prompt: distance-transform interior point
- box prompt: tight bounding box of the same instance

This is more robust than using a naive centroid because the point is more likely to remain inside irregular masks.

### 3.4 Recommended production path

For real no-GT inference, the recommended path is:

1. U-Net predicts a coarse binary glomerulus mask.
2. Connected components in that mask are converted into prompts.
3. SAM2 refines them into cleaner instance masks.

This route is implemented in the inference scripts through `--prompt-source mask`.

Pure `amg` inference is also available, but should be treated as a baseline rather than the default production route.

## 4. Repository layout

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
|   |   |-- sam2.1/
|   |   `-- sam2.1_training/
|   `-- ...
|-- scripts/
|   |-- prepare_hubmap_sam2_dataset.py
|   |-- train_hubmap_sam2.py
|   |-- summarize_hubmap_sam2_run.py
|   |-- evaluate_hubmap_sam2.py
|   `-- predict_hubmap_sam2.py
`-- training/
```

## 5. Environment setup

### 5.1 Recommended conda environment

Recommended environment:

- environment name: `sam2_segm`
- Python: `3.10`
- PyTorch: `2.5.1`
- torchvision: `0.20.1`

Create the environment:

```bash
conda create -n sam2_segm python=3.10 pip -y
conda activate sam2_segm
```

### 5.2 Install PyTorch

Install a CUDA build that matches your server. If your driver supports CUDA 12.4:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

If your server only supports CUDA 12.1:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

You can confirm the driver side with:

```bash
nvidia-smi
```

### 5.3 Install TIFF decoding dependencies

HuBMAP TIFF files are often JPEG-compressed TIFFs. These require `imagecodecs`.

Install it explicitly:

```bash
conda install -c conda-forge imagecodecs -y
```

Recommended TIFF stack:

```bash
conda install -c conda-forge tifffile imagecodecs -y
```

### 5.4 Install the repository

```bash
cd /root/sam2_segm
pip install -U pip setuptools wheel
pip install -e .
```

Notes:

- the project depends on packages such as `hydra-core`, `submitit`, `tensordict`, `tensorboard`, `opencv-python`, `matplotlib`, and `tifffile`
- the optional SAM2 CUDA extension may fail to compile on some machines; the rest of the project can still work
- if needed, you can disable extension build explicitly:

```bash
export SAM2_BUILD_CUDA=0
pip install -e .
```

## 6. Pull the correct branch

If your server clone is older, update to the HuBMAP branch:

```bash
cd /root/sam2_segm
git fetch origin
git checkout sam2_segm
git pull origin sam2_segm
```

## 7. Base checkpoint

The default setup uses the SAM2.1 tiny checkpoint.

Recommended location:

```text
/root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt
```

Download:

```bash
mkdir -p /root/sam2_segm/checkpoints
cd /root/sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

Default config:

```text
configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml
```

## 8. Input data assumptions

For each slide, the project expects:

- `slide_id.tiff`
  - raw slide image
- `slide_id.json`
  - polygon annotations for glomeruli
  - the parser looks for labels such as `properties.classification.name == "glomerulus"`
- `slide_id-anatomical-structure.json`
  - polygon annotations for anatomical structures such as `Cortex` and `Medulla`

This allows the pipeline to:

1. read the WSI TIFF
2. parse glomerulus polygons as instances
3. optionally keep only cortex-region instances
4. tile the slide into trainable patches
5. produce SAM2-compatible single-frame training folders

## 9. Dataset preparation

### 9.1 Output dataset structure

The preparation script writes:

```text
/root/datasets/HuBMAP_sam2/
|-- train
|   |-- JPEGImages
|   |   `-- <sample_id>
|   |       `-- 00000.png
|   |-- Annotations
|   |   `-- <sample_id>
|   |       `-- 00000.png
|   |-- Metadata
|   |   `-- <sample_id>.json
|   `-- list.txt
|-- val
|   |-- JPEGImages
|   |-- Annotations
|   |-- Metadata
|   `-- list.txt
`-- manifests
    |-- samples.csv
    |-- slides.csv
    `-- summary.json
```

Although the folder is named `JPEGImages`, the actual images are stored as `PNG`. This preserves compatibility with the official SAM2 dataset interface.

### 9.2 Preparation command

Recommended command:

```bash
cd /root/sam2_segm
conda activate sam2_segm

python scripts/prepare_hubmap_sam2_dataset.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_sam2 \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-roi-coverage 0.05 \
  --min-positive-pixels 64
```

Useful options:

- `--roi-labels Cortex`
  - keeps only cortex-region instances
- `--missing-roi-policy`
  - `skip-slide`, `ignore-roi`, or `error`
- `--split-csv`
  - reuse a fixed slide split
- `--downsample`
  - resize tiles after extraction
- `--max-instances-per-tile`
  - keeps palette PNG IDs safe
- `--limit-slides`
  - useful for smoke tests

### 9.3 What the script saves per tile

Each saved sample includes:

- RGB tile image
- instance ID annotation PNG
- per-object metadata
  - visible area
  - sampled interior point
  - bounding box

## 10. Training

### 10.1 Main training config

Main config:

```text
configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml
```

The current default behavior is:

- `num_frames = 1`
- image encoder frozen by default
- prompt-based training enabled
- point prompts are always used
- box prompts are mixed in with a smaller probability
- validation metrics are written through `hubmap_sam2.meters.BinarySegmentationMeter`

### 10.2 Recommended training command

```bash
cd /root/sam2_segm
conda activate sam2_segm

python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --num-gpus 1 \
  --num-nodes 1
```

### 10.3 Adjusting training parameters

You can pass Hydra overrides repeatedly:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --hydra-override scratch.num_epochs=60 \
  --hydra-override scratch.train_batch_size=4 \
  --hydra-override scratch.num_train_workers=16
```

### 10.4 Resume training

Resume from the same run directory:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

If `checkpoint.pt` already exists under the run directory, training resumes automatically.

Start a new run from an older checkpoint:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny_v2 \
  --resume-from /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/latest.pt
```

### 10.5 Training outputs

Training outputs look like:

```text
/root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/
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
    `-- summary.json
```

Meaning:

- `checkpoint.pt`
  - canonical resume checkpoint
- `latest.pt`
  - latest alias
- `best.pt`
  - best checkpoint alias
- `history.csv`
  - merged train/val epoch history
- `training_curves.png`
  - loss and metric curves

### 10.6 Regenerate training summary

```bash
python scripts/summarize_hubmap_sam2_run.py \
  --run-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

## 11. Evaluation

### 11.1 Supported modes

`scripts/evaluate_hubmap_sam2.py` supports:

- `oracle-point`
- `oracle-box`
- `oracle-point-box`
- `amg`
- `prior-mask`

### 11.2 Recommended validation command

Recommended model-quality check:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_oracle_point
```

### 11.3 Evaluate pure automatic SAM2 mode

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode amg \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_amg
```

### 11.4 Evaluate U-Net -> SAM2 refinement

If you have coarse validation masks, place them under a prior-mask directory using one of:

- `<prior_mask_dir>/<sample_id>.png`
- `<prior_mask_dir>/<sample_id>.tif`
- `<prior_mask_dir>/<sample_id>/binary_mask.png`

Then run:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode prior-mask \
  --prior-mask-dir /root/hezhongyi-unet_segm/outputs/hubmap_val_binary \
  --prompt-mode point_box \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_prior_mask
```

### 11.5 Evaluation outputs

The evaluation script writes:

- `metrics.json`
- `per_sample_metrics.csv`
- `previews/*.png`

Metrics include:

- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- instance-level Precision / Recall / F1

## 12. Inference

### 12.1 Tile inference with pure SAM2 automatic prompts

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode tile \
  --prompt-source amg \
  --image /root/datasets/HuBMAP_sam2/val/JPEGImages/0486052bb_x00000_y00000/00000.png \
  --output-dir /root/sam2_segm/inference/tile_amg
```

Outputs:

- `image.png`
- `instance_map.png`
- `binary_mask.png`
- `summary.json`

### 12.2 Tile inference with prior mask prompts

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode tile \
  --prompt-source mask \
  --prompt-mode point_box \
  --image /root/datasets/HuBMAP_sam2/val/JPEGImages/0486052bb_x00000_y00000/00000.png \
  --prior-mask /root/hezhongyi-unet_segm/outputs/sample_binary.png \
  --output-dir /root/sam2_segm/inference/tile_prior_mask
```

### 12.3 Whole-slide inference with pure SAM2 AMG

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source amg \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/sam2_segm/inference/wsi_amg_0486052bb
```

### 12.4 Whole-slide inference with U-Net prior mask refinement

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/sam2_segm/inference/wsi_prior_mask_0486052bb
```

Whole-slide outputs:

- `instance_map.tiff`
- `binary_mask.tiff`
- `summary.json`

## 13. Complete server workflow

### Step 1. Pull code

```bash
cd /root/sam2_segm
git fetch origin
git checkout sam2_segm
git pull origin sam2_segm
```

### Step 2. Create and activate environment

```bash
conda create -n sam2_segm python=3.10 pip -y
conda activate sam2_segm
```

### Step 3. Install PyTorch and TIFF dependencies

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c conda-forge tifffile imagecodecs -y
```

If your machine is not on CUDA 12.4, replace `pytorch-cuda=12.4` with the version matched to `nvidia-smi`.

### Step 4. Install the repository

```bash
cd /root/sam2_segm
pip install -U pip setuptools wheel
pip install -e .
```

### Step 5. Download the SAM2 checkpoint

```bash
mkdir -p /root/sam2_segm/checkpoints
cd /root/sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

### Step 6. Prepare HuBMAP SAM2 dataset

```bash
cd /root/sam2_segm
conda activate sam2_segm

python scripts/prepare_hubmap_sam2_dataset.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_sam2 \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-roi-coverage 0.05 \
  --min-positive-pixels 64
```

### Step 7. Train

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --num-gpus 1 \
  --num-nodes 1
```

### Step 8. Evaluate

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_oracle_point
```

### Step 9. Run no-GT inference

Recommended production route:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/sam2_segm/inference/wsi_prior_mask_0486052bb
```

## 14. Monitoring and useful checks

Check GPU usage during training:

```bash
watch -n 2 nvidia-smi
```

Check whether TIFF decoding dependency is installed:

```bash
python -c "import imagecodecs; print('imagecodecs ok')"
```

Check whether the prepared dataset exists:

```bash
ls /root/datasets/HuBMAP_sam2/train/JPEGImages | head
ls /root/datasets/HuBMAP_sam2/train/Annotations | head
cat /root/datasets/HuBMAP_sam2/manifests/summary.json
```

## 15. Troubleshooting

### 15.1 TIFF preprocessing error: `requires the 'imagecodecs' package`

If preprocessing fails with an error like:

```text
ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package
```

install:

```bash
conda install -c conda-forge imagecodecs -y
```

Then rerun preprocessing.

### 15.2 `image data are not memory-mappable`

This is not the actual failure by itself. It only means `tifffile.memmap(...)` could not memory-map the slide and the loader fell back to normal reading.

### 15.3 SAM2 CUDA extension build fails during `pip install -e .`

If the optional extension fails to build, you can usually still use the project. If needed:

```bash
export SAM2_BUILD_CUDA=0
pip install -e .
```

### 15.4 Why train/val must be split by slide

If neighboring tiles from the same slide appear in both train and validation, validation metrics will look artificially optimistic. This repository always treats slide-level splitting as the correct default.

## 16. Practical recommendation

For the first full server run, the recommended baseline is:

- `Cortex` ROI filtering enabled
- `1024 x 1024` tiles
- SAM2.1 tiny initialization
- oracle-point validation during model development
- `U-Net coarse mask -> SAM2 refinement` for production inference

## 17. Status

This repository is now structured as a complete HuBMAP glomerulus SAM2 project:

- raw data stays outside the repository
- preprocessing is included in-repo
- training config is HuBMAP-specific
- validation metrics and summaries are available
- `best` and `latest` checkpoints are saved
- tile and whole-slide inference are available
- the hybrid U-Net -> SAM2 route is directly supported
