# HuBMAP Glomeruli Segmentation with PyTorch U-Net

本项目基于 PyTorch U-Net，面向 **HuBMAP 肾小球分割** 与后续 **HSPN / 病理 TIFF 推理** 场景做了工程化改造。  
当前仓库的目标不是做一个通用 demo，而是作为你在 Linux 4090 云服务器上可长期维护、可重复训练、可稳定推理的分割工程。

## 1. 项目特点

- 面向 **HuBMAP WSI 数据** 的外部数据目录工作流，数据和权重不需要放进 Git 仓库。
- 提供 **WSI 切片脚本**，支持从 TIFF + RLE 标注或 TIFF + 现成 mask 生成训练 patch。
- 提供 **训练、评估、推理、可视化** 的完整脚本链路。
- 训练阶段支持 **Dice / IoU / Precision / Recall / Specificity / Accuracy / val loss**。
- checkpoint 保存逻辑完善，默认保存：
  - `best.pth`：当前最佳权重
  - `latest.pth`：最近一次训练权重
  - `epoch_xxx.pth`：可选逐 epoch 保存
- 自动输出训练过程分析结果：
  - `analysis/history.csv`
  - `analysis/training_curves.png`
  - `analysis/best_preview.png`
- 针对大规模 HuBMAP 训练做了性能优化：
  - AMP
  - TF32
  - cuDNN benchmark
  - non-blocking 数据搬运
  - persistent workers
  - prefetch factor
  - `torch.compile`
  - mask value 扫描缓存

## 2. 项目结构

```text
.
├── train.py                          # 主训练脚本
├── evaluate.py                       # 训练阶段内部验证逻辑
├── evaluate_checkpoint.py            # 独立 checkpoint 评估脚本
├── predict.py                        # 普通图像推理
├── predict_tiff.py                   # 大尺寸 TIFF 滑窗推理
├── predict_hspn_enhanced.py          # HSPN 推理，带对比度增强
├── predict_hspn_stain_norm.py        # HSPN 推理，带 stain normalization
├── requirements.txt                  # Python 依赖
├── scripts
│   ├── prepare_hubmap_tiles.py       # HuBMAP WSI 切片脚本
│   ├── plot_training_history.py      # 从 history.csv 重绘训练曲线
│   ├── download_data.sh
│   └── download_data.bat
├── unet
│   ├── unet_model.py                 # U-Net 主体
│   └── unet_parts.py                 # 编码器/解码器模块
└── utils
    ├── data_loading.py               # 数据读取与预处理
    ├── dice_score.py                 # Dice loss / Dice 计算
    ├── segmentation_metrics.py       # 分割指标统计
    ├── visualization.py              # 训练曲线与分割预览图生成
    └── utils.py                      # 简单图像可视化
```

## 3. 推荐运行环境

推荐场景：

- OS：Linux
- GPU：NVIDIA RTX 4090
- Python：3.10 或更高
- PyTorch：2.x
- CUDA：与服务器驱动匹配

说明：

- `requirements.txt` **不包含** `torch`，因为它需要根据服务器 CUDA 版本安装。
- 请先按 PyTorch 官方方式安装与你服务器 CUDA 匹配的 `torch`，再安装项目依赖。

推荐安装流程：

```bash
git clone -b unet_segm https://github.com/DZW131/hezhongyi.git
cd hezhongyi

python -m venv .venv
source .venv/bin/activate

# 先按你的 CUDA 版本安装 torch
# 例如：参考 https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

如果需要记录在线实验：

```bash
pip install wandb
wandb login
```

如果不想使用 W&B，也可以直接跑，训练脚本会自动退化为本地日志模式。

## 4. 数据与目录约定

### 4.1 原始数据

本项目假设 **HuBMAP 原始 TIFF 数据、CSV 标注、切片后的 patch 数据、训练得到的 checkpoints** 都存放在仓库之外，例如：

```text
/data/hubmap_raw/
/data/hubmap_tiles/
/data/checkpoints/
```

这样做的目的是：

- Git 仓库保持轻量
- 服务器训练与本地代码同步解耦
- 便于多次实验复用数据

### 4.2 训练数据目录格式

训练脚本默认读取如下结构：

```text
/data/hubmap_tiles/
├── train
│   ├── images
│   │   ├── xxx.jpg
│   │   └── ...
│   └── masks
│       ├── xxx.png
│       └── ...
├── val
│   ├── images
│   │   ├── xxx.jpg
│   │   └── ...
│   └── masks
│       ├── xxx.png
│       └── ...
└── manifests
    ├── tiles.csv
    ├── slides.csv
    └── summary.json
```

要求：

- `images` 中是 `.jpg`
- `masks` 中是 `.png`
- 同名 image / mask 一一对应
- 每个 image 和 mask 尺寸必须一致

## 5. 数据处理脚本

### 5.1 脚本

```bash
python scripts/prepare_hubmap_tiles.py -h
```

### 5.2 功能

该脚本负责把 HuBMAP 原始 WSI 切成可直接训练的 patch 数据。

支持两种输入模式：

1. `TIFF + annotations CSV`
2. `TIFF + 独立 mask 目录`

核心逻辑：

- 按整张 slide 做 `train/val` 划分，而不是按 patch 随机拆分
- 过滤空白背景 tile
- 统计 mask 覆盖率
- 控制正负样本比例
- 输出 patch 数据和 manifest 文件

### 5.3 常用命令

#### 方式 A：使用 HuBMAP CSV 中的 RLE 标注

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /data/hubmap_raw/train \
  --annotations-csv /data/hubmap_raw/train.csv \
  --output-dir /data/hubmap_tiles \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

#### 方式 B：使用已有 mask 文件

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /data/hubmap_raw/train \
  --mask-dir /data/hubmap_raw/masks \
  --output-dir /data/hubmap_tiles \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2
```

#### 方式 C：使用预先指定的 slide 划分文件

`split.csv` 格式：

```csv
slide_id,split
slide_001,train
slide_002,val
```

命令：

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /data/hubmap_raw/train \
  --annotations-csv /data/hubmap_raw/train.csv \
  --split-csv /data/hubmap_raw/split.csv \
  --output-dir /data/hubmap_tiles
```

### 5.4 关键参数说明

- `--tile-size`：切片尺寸，通常与训练尺寸保持一致，默认 `1024`
- `--stride`：滑窗步长，等于 `tile-size` 表示无重叠切片
- `--downsample`：切片保存前的缩放比例
- `--val-ratio`：按 slide 划分验证集比例
- `--min-tissue-coverage`：组织覆盖率阈值，用于过滤空白 tile
- `--min-positive-pixels`：最少阳性像素数，低于它会被视为负样本
- `--negative-ratio`：每个正样本保留多少负样本
- `--max-background-tiles-per-slide`：没有正样本的 slide 最多保留多少背景 tile

### 5.5 输出结果

脚本会生成：

- `train/images/*.jpg`
- `train/masks/*.png`
- `val/images/*.jpg`
- `val/masks/*.png`
- `manifests/tiles.csv`
- `manifests/slides.csv`
- `manifests/summary.json`

## 6. 训练脚本

### 6.1 脚本

```bash
python train.py -h
```

### 6.2 推荐训练命令

```bash
python train.py \
  --images-dir /data/hubmap_tiles/train/images \
  --masks-dir /data/hubmap_tiles/train/masks \
  --val-images-dir /data/hubmap_tiles/val/images \
  --val-masks-dir /data/hubmap_tiles/val/masks \
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
  --checkpoint-dir /data/checkpoints/hubmap_unet
```

### 6.3 训练产物

假设 `--checkpoint-dir /data/checkpoints/hubmap_unet`，训练完成后目录通常如下：

```text
/data/checkpoints/hubmap_unet/
├── best.pth
├── latest.pth
├── epoch_001.pth                  # 仅在 --save-every-epoch 时生成
├── epoch_002.pth
└── analysis
    ├── history.csv
    ├── training_curves.png
    ├── best_preview.png
    ├── best_metrics.json
    └── val_previews
        ├── epoch_005.png
        ├── epoch_010.png
        └── ...
```

文件说明：

- `best.pth`：按 `--checkpoint-metric` 选出的最佳模型
- `latest.pth`：最新一次 epoch 的模型
- `history.csv`：每个 epoch 的损失、指标、耗时、吞吐量
- `training_curves.png`：训练曲线总览
- `best_preview.png`：最佳 epoch 的预测可视化
- `best_metrics.json`：最佳模型对应的指标记录

### 6.4 当前支持的验证指标

训练和评估阶段会输出：

- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- Validation Loss

### 6.5 性能优化说明

训练脚本默认已经打开以下加速选项：

- CUDA 下启用 `TF32`
- CUDA 下启用 `cuDNN benchmark`
- `pin_memory + non_blocking`
- dataloader `persistent_workers`
- dataloader `prefetch_factor`
- `torch.compile(auto)`
- `mask_values` 缓存到 mask 目录中的 `.mask_values_cache.json`

### 6.6 常用训练参数

- `--optimizer {rmsprop,adamw}`：优化器选择
- `--compile {auto,off,default,reduce-overhead,max-autotune}`：是否开启 `torch.compile`
- `--disable-tf32`：关闭 TF32
- `--disable-cudnn-benchmark`：关闭 cuDNN benchmark
- `--num-workers`：dataloader worker 数量
- `--prefetch-factor`：每个 worker 预取 batch 数量
- `--val-frequency`：每隔多少个 epoch 做一次验证
- `--analysis-frequency`：每隔多少个 epoch 刷新一次训练曲线
- `--preview-frequency`：每隔多少个 epoch 存一张验证预览图
- `--save-every-epoch`：额外保存每个 epoch 的 checkpoint

### 6.7 断点续训

```bash
python train.py \
  --load /data/checkpoints/hubmap_unet/latest.pth \
  --images-dir /data/hubmap_tiles/train/images \
  --masks-dir /data/hubmap_tiles/train/masks \
  --val-images-dir /data/hubmap_tiles/val/images \
  --val-masks-dir /data/hubmap_tiles/val/masks \
  --amp \
  --checkpoint-dir /data/checkpoints/hubmap_unet
```

## 7. 指标评估脚本

### 7.1 脚本

```bash
python evaluate_checkpoint.py -h
```

### 7.2 作用

独立地对某个 checkpoint 在指定数据集上做评估，并输出：

- `metrics.json`
- `preview.png`

### 7.3 示例

```bash
python evaluate_checkpoint.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --images-dir /data/hubmap_tiles/val/images \
  --masks-dir /data/hubmap_tiles/val/masks \
  --classes 2 \
  --batch-size 2 \
  --num-workers 8 \
  --amp \
  --output-dir /data/checkpoints/hubmap_unet/eval_best
```

## 8. 可视化脚本

### 8.1 自动生成

训练过程中会自动生成：

- `analysis/history.csv`
- `analysis/training_curves.png`
- `analysis/best_preview.png`
- `analysis/val_previews/*.png`

这些文件已经覆盖了常见的训练可视化需求。

### 8.2 重绘训练曲线

如果你修改了 `history.csv` 或者想单独重新出图：

```bash
python scripts/plot_training_history.py \
  --history-csv /data/checkpoints/hubmap_unet/analysis/history.csv \
  --output /data/checkpoints/hubmap_unet/analysis/training_curves_regenerated.png
```

## 9. 推理脚本

### 9.1 普通图片推理

适用于 `.jpg` / `.png` 等普通图像：

```bash
python predict.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --input demo.jpg \
  --output demo_mask.png \
  --classes 2
```

支持参数：

- `--viz`：显示结果
- `--no-save`：只看不保存
- `--scale`：输入图像缩放比例

### 9.2 大尺寸 TIFF 推理

适用于 WSI 或大尺寸 TIFF：

```bash
python predict_tiff.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --input sample.tiff \
  --output sample_mask.png \
  --tile-size 1024 \
  --classes 2
```

### 9.3 HSPN 增强推理

对输入 tile 进行线性对比度增强，适合染色偏淡的图像：

```bash
python predict_hspn_enhanced.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --input sample_hspn.tiff \
  --output sample_hspn_pred.png \
  --tile-size 1024 \
  --threshold 0.5 \
  --scale 1.0 \
  --classes 2
```

### 9.4 HSPN stain normalization 推理

对输入 tile 做颜色归一化，再进行分割：

```bash
python predict_hspn_stain_norm.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --input sample_hspn.tiff \
  --output sample_hspn_norm_pred.png \
  --tile-size 1024 \
  --threshold 0.5 \
  --classes 2
```

## 10. 权重文件管理建议

推荐仅长期保留以下文件：

- `best.pth`
- `latest.pth`
- `analysis/history.csv`
- `analysis/training_curves.png`
- `analysis/best_preview.png`
- `analysis/best_metrics.json`

如果磁盘紧张，不建议长期保留所有 `epoch_xxx.pth`。

推荐目录：

```text
/data/checkpoints/
└── hubmap_unet/
    ├── best.pth
    ├── latest.pth
    └── analysis/
```

## 11. 端到端推荐流程

### 步骤 1：准备环境

```bash
git pull origin unet_segm
source .venv/bin/activate
pip install -r requirements.txt
```

### 步骤 2：切片 HuBMAP 数据

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /data/hubmap_raw/train \
  --annotations-csv /data/hubmap_raw/train.csv \
  --output-dir /data/hubmap_tiles \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2
```

### 步骤 3：训练模型

```bash
python train.py \
  --images-dir /data/hubmap_tiles/train/images \
  --masks-dir /data/hubmap_tiles/train/masks \
  --val-images-dir /data/hubmap_tiles/val/images \
  --val-masks-dir /data/hubmap_tiles/val/masks \
  --epochs 50 \
  --batch-size 2 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --checkpoint-dir /data/checkpoints/hubmap_unet
```

### 步骤 4：独立评估最佳模型

```bash
python evaluate_checkpoint.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --images-dir /data/hubmap_tiles/val/images \
  --masks-dir /data/hubmap_tiles/val/masks \
  --classes 2 \
  --amp \
  --output-dir /data/checkpoints/hubmap_unet/eval_best
```

### 步骤 5：进行推理

```bash
python predict_tiff.py \
  --model /data/checkpoints/hubmap_unet/best.pth \
  --input /data/inference/sample.tiff \
  --output /data/inference/sample_mask.png \
  --tile-size 1024 \
  --classes 2
```

## 12. 常见注意事项

### 12.1 `--classes` 如何设置

- 二分类前景/背景分割：通常设为 `2`
- 如果你训练的是单通道 sigmoid 版本，则需保持训练和推理配置一致

当前项目默认更推荐使用二分类 softmax 方案。

### 12.2 为什么要按 slide 划分 train / val

如果同一张 WSI 的相邻 patch 同时进入训练集和验证集，验证指标会明显偏乐观。  
本项目的数据处理脚本默认按 **整张 slide** 切分，这一点对病理分割任务非常重要。

### 12.3 `mask_values` 缓存文件是什么

在 `masks` 目录下会生成：

```text
.mask_values_cache.json
```

这是为了避免每次启动训练都重新扫描整套 mask，提高重复实验的启动速度。

### 12.4 `torch.compile` 是否一定更快

不一定。  
在多数 Linux + CUDA + PyTorch 2.x 场景下会有收益，但首次编译会增加启动开销。  
如果你的实验很短，或者环境兼容性不好，可以使用：

```bash
--compile off
```

### 12.5 W&B 是否必须

不是。  
即使没有安装 W&B，项目仍可完整训练，只是不会上传在线实验记录。

## 13. 当前项目状态

当前仓库已经具备以下工程能力：

- 外部 HuBMAP 数据目录支持
- WSI 切片脚本
- 训练指标完善
- 自动最佳权重保存
- 本地训练可视化输出
- 独立 checkpoint 评估脚本
- 普通图像 / TIFF / HSPN 推理脚本
- 面向 4090 服务器的训练性能优化

如果后续还要继续扩展，下一步最值得做的通常是：

- 更强的数据增强策略
- 更细粒度的实验配置文件管理
- 更完整的测试集评估与结果汇总
- 面向多实验的自动化训练调度

## 14. 致谢

项目最初基于 `milesial/Pytorch-UNet`，后续围绕 HuBMAP 肾小球分割场景做了定制化与工程化扩展。
