# HuBMAP Glomerulus SAM2 Engineering Pipeline

This repository is now a HuBMAP kidney glomerulus segmentation project built on top of SAM2 and adapted for a Linux server workflow.

It is intended to be the second technical route alongside your U-Net project:

- `hezhongyi-unet_segm`: semantic coarse segmentation / candidate generation
- `hezhongyi-sam2_segm`: promptable instance refinement with SAM2

The current codebase has been reworked around the HuBMAP data layout you already use on the server:

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

You do not need to move raw slides into the repository.

## 1. What this project now does

This repository now supports:

- full in-repo `sam2/` package and training configs
- HuBMAP TIFF + polygon JSON preprocessing
- slide-level train/val split
- optional anatomical ROI filtering from `*-anatomical-structure.json`
- instance-preserving tile generation for SAM2
- single-frame SAM2 training with `num_frames=1`
- automatic prompt generation from glomerulus instances
- validation metrics: Dice, IoU, Precision, Recall, Specificity, Accuracy
- instance-level matching metrics for evaluation
- best/latest checkpoint saving
- training history export and curve plotting
- tile inference
- whole-slide patch-based inference
- two automatic inference routes:
  - pure SAM2 automatic mask generation (`amg`)
  - coarse-mask-driven SAM2 refinement (`mask`), suitable for U-Net -> SAM2 chaining

## 2. Key design choices

### 2.1 Single-frame SAM2

This project treats each image tile as a one-frame video:

- `num_frames = 1`
- SAM2 training logic is reused instead of replacing the whole framework
- the model remains a promptable segmentation model rather than a plain semantic segmenter

### 2.2 Instance-level dataset, not merged binary-only training masks

HuBMAP glomerulus polygons are preserved as tile-local instance IDs during preprocessing:

- each glomerulus becomes one instance in the saved annotation PNG
- metadata stores per-instance prompt hints
- train/val split is done by slide, not by tile

### 2.3 Automatic prompt design

Training and evaluation use prompt generation derived from instance masks:

- positive point: distance-transform peak inside each instance
- optional box: tight bounding box of the same instance
- training config mixes point prompts with a smaller amount of box prompts

This is more stable than using a plain polygon centroid because the sampled point is guaranteed to stay deep inside the visible mask.

### 2.4 Recommended production route

For real no-GT inference, the recommended route is:

1. U-Net produces a coarse binary glomerulus candidate mask.
2. SAM2 converts connected components in that coarse mask into prompts.
3. SAM2 refines them into instance masks.

This hybrid route is implemented in the inference scripts through `--prompt-source mask`.

Pure SAM2 automatic mask generation is still available through `--prompt-source amg`, but it should be treated as a baseline rather than the default production strategy.

## 3. Repository layout

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
|   `-- configs/
|       |-- sam2.1/
|       `-- sam2.1_training/
|-- scripts/
|   |-- prepare_hubmap_sam2_dataset.py
|   |-- train_hubmap_sam2.py
|   |-- summarize_hubmap_sam2_run.py
|   |-- evaluate_hubmap_sam2.py
|   `-- predict_hubmap_sam2.py
`-- training/
```

## 4. Repository completeness

The original workspace was not a complete trainable SAM2 repository. It contained the training skeleton but was missing the top-level `sam2/` package and config tree required by:

- `setup.py`
- Hydra model construction
- image predictor / automatic mask generator
- checkpoint loading

This repository now includes the missing `sam2/` package and a HuBMAP-specific training config:

- `configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml`

So the current tree is now self-contained for training and inference.

## 5. Recommended server environment

Recommended setup:

- OS: Linux
- Python: 3.10+
- GPU: CUDA-capable GPU
- PyTorch: 2.5.1+ with a CUDA build matching the server driver

Example installation:

```bash
cd /root/hezhongyi-sam2_segm

conda activate sam2_kidney

# Install a CUDA-compatible torch build first according to your server.
# Then install the project itself.
pip install -e .
```

Notes:

- the repository now includes training dependencies such as `submitit`, `tensordict`, `tensorboard`, `fvcore`, `opencv-python`, `matplotlib`, and `tifffile`
- building the optional SAM2 CUDA extension may fail on some environments; the project can still run without it

## 6. Initial checkpoint placement

SAM2 fine-tuning starts from an official SAM2.1 checkpoint.

Recommended checkpoint location:

```text
/root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt
```

Example download:

```bash
cd /root/hezhongyi-sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

The default HuBMAP config is currently built around the tiny model:

- config: `configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml`
- init checkpoint: `sam2.1_hiera_tiny.pt`

You can later swap to `small`, `base_plus`, or `large`, but the tiny model is the cleanest starting point for engineering validation.

## 7. Raw HuBMAP inputs

For each slide, the project expects:

- `slide_id.tiff`
  - raw whole-slide image
- `slide_id.json`
  - polygon annotations for glomeruli
  - the parser looks for labels such as `properties.classification.name == "glomerulus"`
- `slide_id-anatomical-structure.json`
  - polygon annotations for anatomical regions such as `Cortex` and `Medulla`

This allows the project to:

1. read the whole-slide TIFF
2. parse glomerulus polygons as instances
3. optionally keep only cortex-region glomeruli
4. tile the slide into trainable samples
5. save SAM2-compatible single-frame training folders

## 8. Data preparation

### 8.1 Output dataset structure

Recommended output directory:

```text
/root/datasets/HuBMAP_sam2
```

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

`JPEGImages` is kept because the official SAM2 dataset interface expects that naming pattern, even though the actual tile files are stored as PNG.

### 8.2 Preparation script

Main script:

```bash
python scripts/prepare_hubmap_sam2_dataset.py -h
```

What it does:

- reads TIFF slides directly from the external dataset directory
- parses glomerulus polygons as instances
- optionally filters instances by anatomical ROI such as `Cortex`
- cuts tiles using slide coordinates
- keeps only positive tiles with enough tissue and glomerulus pixels
- stores tile-local instance maps instead of one merged binary mask
- writes prompt metadata per object
- creates a slide-level train/val split

### 8.3 Recommended command for your server

```bash
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
  - reuse a fixed slide split from a CSV
- `--downsample`
  - resize tiles after extraction
- `--max-instances-per-tile`
  - caps tile instance count so palette PNG IDs stay valid

## 9. Training

### 9.1 Default training config

Main HuBMAP config:

```text
configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml
```

This config:

- uses `num_frames=1`
- freezes the image encoder by default
- fine-tunes the promptable segmentation path
- uses point prompts for training
- mixes in box prompts with lower probability
- writes validation metrics through `hubmap_sam2.meters.BinarySegmentationMeter`

### 9.2 Recommended training command

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --num-gpus 1 \
  --num-nodes 1
```

Optional Hydra overrides can be passed repeatedly:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --hydra-override scratch.num_epochs=60 \
  --hydra-override scratch.train_batch_size=4 \
  --hydra-override scratch.num_train_workers=16
```

### 9.3 Resume training

Two common resume modes:

1. Resume from the same run directory:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

If `checkpoint.pt` already exists in the run directory, the trainer resumes automatically.

2. Start a new run initialized from an older run checkpoint:

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny_v2 \
  --resume-from /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/latest.pt
```

### 9.4 Training outputs

With the run directory above, training produces:

```text
/root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/
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
  - best checkpoint alias for the tracked validation meter
- `history.csv`
  - merged epoch-level train/val history
- `training_curves.png`
  - local overview figure for loss and validation metrics

### 9.5 Regenerate summary artifacts

```bash
python scripts/summarize_hubmap_sam2_run.py \
  --run-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

Or:

```bash
python scripts/train_hubmap_sam2.py \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny \
  --summary-only
```

## 10. Evaluation

### 10.1 Supported evaluation modes

`scripts/evaluate_hubmap_sam2.py` supports:

- `oracle-point`
  - GT-derived interior point per instance
- `oracle-box`
  - GT-derived box per instance
- `oracle-point-box`
  - GT-derived point + box
- `amg`
  - pure SAM2 automatic mask generation
- `prior-mask`
  - coarse-mask-driven refinement, suitable for U-Net -> SAM2 evaluation

### 10.2 Recommended promptable validation command

This is the best measure of how well the fine-tuned SAM2 responds to ideal automatic prompts derived from the instance masks:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_oracle_point
```

### 10.3 Evaluate pure automatic SAM2 mode

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode amg \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_amg
```

### 10.4 Evaluate U-Net prior -> SAM2 refinement

If you have coarse binary masks for the validation tiles, place them in a directory using either:

- `<prior_mask_dir>/<sample_id>.png`
- `<prior_mask_dir>/<sample_id>.tif`
- `<prior_mask_dir>/<sample_id>/binary_mask.png`

Then run:

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode prior-mask \
  --prior-mask-dir /root/hezhongyi-unet_segm/outputs/hubmap_val_binary \
  --prompt-mode point_box \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_prior_mask
```

### 10.5 Metrics written by evaluation

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
- instance-level Precision / Recall / F1 through greedy IoU matching

## 11. Inference

### 11.1 Tile inference with pure SAM2 automatic prompts

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode tile \
  --prompt-source amg \
  --image /root/datasets/HuBMAP_sam2/val/JPEGImages/0486052bb_x00000_y00000/00000.png \
  --output-dir /root/hezhongyi-sam2_segm/inference/tile_amg
```

Outputs:

- `image.png`
- `instance_map.png`
- `binary_mask.png`
- `summary.json`

### 11.2 Tile inference with coarse-mask prompts

This is the recommended SAM2 refinement mode when you already have a U-Net coarse mask.

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode tile \
  --prompt-source mask \
  --prompt-mode point_box \
  --image /root/datasets/HuBMAP_sam2/val/JPEGImages/0486052bb_x00000_y00000/00000.png \
  --prior-mask /root/hezhongyi-unet_segm/outputs/sample_binary.png \
  --output-dir /root/hezhongyi-sam2_segm/inference/tile_prior_mask
```

### 11.3 Whole-slide inference with SAM2 automatic mask generation

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source amg \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/hezhongyi-sam2_segm/inference/wsi_amg_0486052bb
```

### 11.4 Whole-slide inference with U-Net prior mask refinement

If U-Net has already produced a full-slide binary mask:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/hezhongyi-sam2_segm/inference/wsi_prior_mask_0486052bb
```

Whole-slide outputs:

- `instance_map.tiff`
- `binary_mask.tiff`
- `summary.json`

## 12. Recommended server workflow

### Step 1. Pull the code

```bash
cd /root/hezhongyi-sam2_segm
git fetch origin
git checkout sam2_segm
git pull origin sam2_segm
```

### Step 2. Install dependencies

```bash
cd /root/hezhongyi-sam2_segm
conda activate sam2_kidney
pip install -e .
```

### Step 3. Download the base checkpoint

```bash
cd /root/hezhongyi-sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

### Step 4. Prepare the HuBMAP SAM2 dataset

```bash
python scripts/prepare_hubmap_sam2_dataset.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_sam2 \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64
```

### Step 5. Train

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_sam2 \
  --init-checkpoint /root/hezhongyi-sam2_segm/checkpoints/sam2.1_hiera_tiny.pt \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny
```

### Step 6. Evaluate

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/eval_oracle_point
```

### Step 7. Run no-GT inference

Recommended hybrid route:

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_t_hubmap_glomerulus.yaml \
  --checkpoint /root/hezhongyi-sam2_segm/checkpoints/hubmap_glomerulus_sam2_tiny/checkpoints/best.pt \
  --mode wsi \
  --prompt-source mask \
  --prompt-mode point_box \
  --slide /root/datasets/HuBMAP/train/0486052bb.tiff \
  --prior-mask /root/hezhongyi-unet_segm/outputs/0486052bb_binary_mask.tiff \
  --anatomical-json /root/datasets/HuBMAP/train/0486052bb-anatomical-structure.json \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --output-dir /root/hezhongyi-sam2_segm/inference/wsi_prior_mask_0486052bb
```

## 13. Practical notes

### 13.1 Why validation has multiple modes

SAM2 is a promptable model, so there are two different questions:

- how well does the fine-tuned model respond when given a good automatic prompt?
- how well does the full no-GT automatic prompting pipeline perform?

That is why both oracle-prompt evaluation and no-GT automatic inference paths are kept.

### 13.2 Why the recommended production route uses U-Net first

Glomeruli are small objects in large histology slides. A coarse semantic detector is often better at finding candidate regions cheaply, while SAM2 is better used as an instance-aware refinement model once prompts already exist.

So the intended complement is:

- U-Net: broad candidate recall
- SAM2: promptable instance refinement

### 13.3 Why slide-level split matters

Train/val split must happen by slide, not by tile. Otherwise nearby tiles from the same WSI leak into both sets and inflate validation scores.

### 13.4 About automatic prompts used here

This project uses a distance-transform interior point rather than a naive centroid because:

- it is more likely to stay inside irregular polygon masks
- it is less sensitive to elongated or concave shapes
- it is directly usable as a positive SAM2 point prompt

### 13.5 Current default recommendation

For the first full server run, the recommended baseline is:

- `Cortex` ROI filtering enabled
- `1024 x 1024` tiles
- SAM2.1 tiny initialization
- oracle-point validation during model development
- U-Net prior mask refinement for production inference

## 14. Status

This repository is now structured as a complete HuBMAP glomerulus SAM2 engineering project:

- raw dataset stays outside the repository
- preprocessing is included in-repo
- the training config is HuBMAP-specific
- validation metrics and summaries are available
- best/latest checkpoints are saved
- tile and whole-slide inference are available
- the U-Net -> SAM2 hybrid route is directly supported
