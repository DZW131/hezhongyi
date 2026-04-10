# HuBMAP 肾小球分割 SAM2 工程

本仓库是面向 **HuBMAP 肾小球分割任务** 的最终版 **SAM2 工程路线**。  
它不是原始官方 SAM2 README 的简单保留，而是已经按当前项目实际需求做过整理、裁剪和工程化改造的版本。

这条路线的目标是与现有 U-Net 路线并行存在、形成互补：

- `hezhongyi-unet_segm`：语义分割主路线，负责稳定的粗分割
- `hezhongyi-sam2_segm`：基于 SAM2 的提示式实例分割路线，负责实例级建模与可扩展精修

当前仓库已经固定为一条清晰的最终主路径：

1. 复用已经存在的 `/root/datasets/HuBMAP_tiles_v2`
2. 将二值 tile mask 轻量适配为 SAM2 单帧实例训练格式
3. 使用官方 `SAM2.1 Hiera B+` 预训练权重进行微调
4. 输出统一的训练、评估、推理结果与指标文件

---

## 1. 这个项目已经做了哪些工作

和原始 SAM2 仓库相比，这个项目已经完成了以下工程化改造。

### 1.1 任务定义改造

- 将 SAM2 从视频分割任务收敛为 **单帧图像任务**
- 训练时使用 `num_frames = 1`
- 保留 SAM2 的 promptable segmentation 思路，而不是把它改成普通 U-Net 式纯二值分割器

### 1.2 数据路线改造

- 不再要求从原始 WSI 每次慢速重切
- 直接复用已有的 `/root/datasets/HuBMAP_tiles_v2/`
- 保持和 U-Net 路线相同的 `train/val` tile 划分，便于公平对比
- 将二值 mask 通过 connected components 自动转换为 **伪实例标注**

### 1.3 Prompt 机制改造

- 训练阶段不依赖人工点击
- 自动从实例 mask 中生成 point / box prompt
- 当前默认策略是：
  - 训练时始终使用 prompt
  - 以点提示为主
  - 少量混入 box 提示
- 推理阶段支持两条路：
  - `amg`：纯 SAM2 自动提议
  - `prior-mask`：先有粗 mask，再由 SAM2 做提示式精修

### 1.4 训练工程改造

- 增加 HuBMAP 任务专用数据适配层
- 提供统一训练脚本、评估脚本、推理脚本
- 标准化 checkpoint 输出
- 增加训练过程总结与曲线文件
- 统一和 U-Net 路线对齐的核心指标

### 1.5 指标与结果输出改造

训练总结与离线评估统一输出：

- `Dice`
- `IoU`
- `Precision`
- `Recall`
- `Specificity`
- `Accuracy`
- `validation_loss`

### 1.6 服务器兼容性改造

- 适配 headless 服务器环境
- 数据适配阶段避免重复读取大 TIFF
- 对 `cv2/libGL` 相关问题增加兼容处理
- 默认依赖改为 `opencv-python-headless`

---

## 2. 当前最终推荐流程

最终推荐在服务器上按下面顺序执行：

1. 创建独立 conda 环境
2. 安装 PyTorch 和仓库依赖
3. 下载官方 `sam2.1_hiera_base_plus.pt`
4. 将 `/root/datasets/HuBMAP_tiles_v2` 适配为 SAM2 训练格式
5. 启动训练
6. 训练完成后做离线评估
7. 按 tile 或整张 WSI 执行推理

这条路径的特点是：

- 数据准备快
- 与 U-Net 路线可直接比较
- 保留 SAM2 的 promptable / instance-level 特性
- 工程入口清晰，适合服务器直接执行

---

## 3. 仓库结构

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

---

## 4. 环境准备

推荐环境：

- 环境名：`sam2_segm`
- Python：`3.10`
- PyTorch：`2.5.1`
- torchvision：`0.20.1`

### 4.1 创建 conda 环境

```bash
conda create -n sam2_segm python=3.10 pip -y
conda activate sam2_segm
```

### 4.2 安装 PyTorch

如果服务器适合 CUDA 12.4：

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

如果你的 `nvidia-smi` 显示更适合 CUDA 12.1，则把 `12.4` 换成 `12.1`。

### 4.3 安装本仓库

```bash
cd /root/sam2_segm
pip install -U pip setuptools wheel
pip install -e .
```

### 4.4 说明

- 仓库默认依赖已经改为 `opencv-python-headless`
- 这样更适合没有桌面环境的 Linux 服务器
- 即使服务器缺少 `libGL.so.1`，当前代码也已经对验证阶段的点采样提供了无 OpenCV 回退逻辑
- 如果 CUDA 扩展编译失败，很多场景下仍可继续使用

如果不想编译 CUDA 扩展：

```bash
export SAM2_BUILD_CUDA=0
pip install -e .
```

---

## 5. 官方预训练权重

当前默认使用官方 **SAM2.1 Hiera B+** 权重。

推荐放置位置：

```text
/root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt
```

下载方式：

```bash
mkdir -p /root/sam2_segm/checkpoints
cd /root/sam2_segm/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

当前主配置文件：

```text
configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml
```

---

## 6. 输入数据集要求

当前最终训练路线不再从原始 WSI 重新做完整慢速预处理，而是直接复用已经存在的 tile 数据集：

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

当前这套数据的特点已经确认：

- 图像是 `.jpg` tile
- mask 是二值 `.png`
- mask 像素值为 `0 / 255`
- 文件名格式为 `slide_id_x_y`
- `train/val` 已经按 slide 固定划分

这意味着：

- U-Net 和 SAM2 可以共用同一套 tile 数据
- 数据公平可比
- 不需要为 SAM2 单独做一套慢速 WSI 切片流程

---

## 7. 数据适配：从 U-Net tile 数据转成 SAM2 格式

执行：

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

这个脚本会做的事：

- 复用现有 `train/val` 划分
- 通过硬链接或复制复用原始 tile 图像
- 将二值 mask 自动转换为实例 ID 图
- 从实例图中生成 point / box prompt 元数据
- 只保留适合 SAM2 训练的正样本 tile

适配后输出目录形态：

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

快速检查：

```bash
ls /root/datasets/HuBMAP_tiles_v2_sam2/train/JPEGImages | head
ls /root/datasets/HuBMAP_tiles_v2_sam2/train/Annotations | head
cat /root/datasets/HuBMAP_tiles_v2_sam2/manifests/summary.json
```

---

## 8. 训练配置说明

当前主配置：`configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml`

当前默认设计为：

- 使用官方 `SAM2.1 Hiera B+`
- `num_frames = 1`
- 冻结 image encoder
- `num_maskmem = 0`
- 输入分辨率 `896`
- 使用 `bfloat16 AMP`
- GPU 默认开启 `TF32`
- 默认每 2 个 epoch 验证一次
- 训练始终使用 prompt
- 训练以点提示为主，少量混入 box 提示
- `num_correction_pt_per_frame = 0`

最后这一点很重要：

- 当前版本不是交互式多轮纠错训练
- 它是“单轮 promptable segmentation”训练
- 这样更适合当前 HuBMAP tile 场景，也更稳、更省时

---

## 9. 启动训练

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

单张 4090 上更稳的首轮训练方式：

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --num-gpus 1 \
  --num-nodes 1 \
  --num-workers 4
```

可选加速参数：

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

---

## 10. 恢复训练

如果训练中途中断，优先检查下面这个文件是否存在：

```text
/root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/latest.pt
```

如果存在，可以恢复：

```bash
python scripts/train_hubmap_sam2.py \
  --dataset-root /root/datasets/HuBMAP_tiles_v2_sam2 \
  --init-checkpoint /root/sam2_segm/checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --resume-from /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/latest.pt
```

如果 `latest.pt` 还没生成，就直接重新启动训练即可。

---

## 11. 训练输出目录

运行目录：

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

重点关注：

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `analysis/final_val_metrics.json`
- `analysis/training_curves.png`

如果你想重新生成总结文件：

```bash
python scripts/summarize_hubmap_sam2_run.py \
  --run-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus
```

---

## 12. 统一指标输出

当前训练总结与离线评估统一输出以下指标：

- `Dice`
- `IoU`
- `Precision`
- `Recall`
- `Specificity`
- `Accuracy`
- `validation_loss`

训练阶段的标准文件是：

```text
analysis/final_val_metrics.json
```

离线评估阶段的标准文件是：

```text
eval_*/metrics.json
```

说明：

- 训练总结里的 `validation_loss` 来自训练/验证循环
- 离线评估脚本中的 `validation_loss` 固定记为 `null`
- 这是因为离线评估走的是 predictor 推理，不是训练阶段 loss 计算

---

## 13. 离线评估

支持的评估模式：

- `oracle-point`
- `oracle-box`
- `oracle-point-box`
- `amg`
- `prior-mask`

### 13.1 推荐：验证集 checkpoint 评估

```bash
python scripts/evaluate_hubmap_sam2.py \
  --dataset-dir /root/datasets/HuBMAP_tiles_v2_sam2 \
  --split val \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode oracle-point \
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/eval_oracle_point
```

### 13.2 推荐：U-Net -> SAM2 混合路线评估

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

评估输出包括：

- `metrics.json`
- `per_sample_metrics.csv`
- `previews/*.png`

---

## 14. 推理

### 14.1 Tile 推理

#### 纯 SAM2 自动 prompt

```bash
python scripts/predict_hubmap_sam2.py \
  --config configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml \
  --checkpoint /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/best.pt \
  --mode tile \
  --prompt-source amg \
  --image /root/datasets/HuBMAP_tiles_v2_sam2/val/JPEGImages/0486052bb_9216_2048/00000.jpg \
  --output-dir /root/sam2_segm/inference/tile_amg
```

#### 先有粗 mask，再由 SAM2 精修

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

输出文件：

- `image.png`
- `instance_map.png`
- `binary_mask.png`
- `summary.json`

### 14.2 整张 WSI 推理

当前更推荐的生产路线是：

- U-Net 先做 coarse binary mask
- SAM2 再对候选区域做 promptable refinement

命令示例：

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

输出文件：

- `instance_map.tiff`
- `binary_mask.tiff`
- `summary.json`

---

## 15. 为什么这条 SAM2 路线会比 U-Net 更慢

这是正常现象。

当前你在单张 4090 上看到 SAM2 训练时间明显长于 U-Net，主要原因是：

- `SAM2.1 Hiera B+` 本身就比 U-Net 更重
- 它不是简单前向二值分割，而是带 prompt 编码的分割框架
- 验证阶段还会进入 prompt 采样和 predictor 式流程
- 当前虽然已经把任务收敛为单帧，并关闭了 correction clicks，但模型主体仍然更复杂

所以：

- 训练速度慢于 U-Net 是预期内的
- 但它能提供更强的提示式实例建模能力
- 这也是两条技术路线并行保留的价值所在

---

## 16. 当前配置下，效果大概会怎样

当前这套默认配置属于一个比较稳妥的第一版工程配置：

- 用官方 B+ 预训练权重
- 只做单帧任务
- 不做多轮纠错点训练
- 以 point prompt 为主
- 保留少量 box prompt 混合

这意味着：

- 训练稳定性会比较好
- 与 U-Net 做公平对比更方便
- 在 `oracle-point` 或 `oracle-point-box` 评估下，效果通常应该是有竞争力的
- 但如果你未来想把“交互纠错能力”也纳入优化目标，再引入 correction points 会更完整

换句话说，当前版本是一个很适合作为 **第一条可落地 SAM2 工程路线** 的配置，而不是最终极限版。

---

## 17. 常见问题

### 17.1 训练能跑，第一个 epoch 结束后验证时报错：`libGL.so.1`

原因：

- 不是训练本身炸了
- 是验证阶段点采样逻辑尝试导入 `cv2`
- 服务器没有桌面图形依赖，导致 `opencv-python` 动态加载失败

当前仓库已经做了两层处理：

- 默认依赖改为 `opencv-python-headless`
- 即使没有可用 `cv2`，验证阶段也会自动回退到无 OpenCV 的采样逻辑

如果你本地代码较旧，先执行：

```bash
cd /root/sam2_segm
git fetch origin
git checkout sam2_segm
git pull origin sam2_segm
```

### 17.2 训练中断后怎么继续

优先看：

```text
/root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus/checkpoints/latest.pt
```

如果存在，使用 `--resume-from` 恢复；否则重新开始即可。

### 17.3 为什么这里没有强制使用纠错点

因为当前目标是先做一条稳定、清晰、可工程化的 SAM2 路线：

- 单帧
- 单轮 prompt
- 可训练
- 可评估
- 可推理
- 可与 U-Net 公平对比

多轮纠错点不是不能做，而是被放到了后续增强阶段。

---

## 18. 推荐服务器命令清单

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
  --output-dir /root/sam2_segm/checkpoints/hubmap_glomerulus_sam2_bplus \
  --num-gpus 1 \
  --num-nodes 1 \
  --num-workers 4
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

---

## 19. 一句话总结

这个仓库现在已经不是“原始 SAM2 代码骨架”，而是一条面向 HuBMAP 肾小球分割任务、可直接在服务器上训练、评估、推理，并且能与 U-Net 路线并行对比的 **最终版 SAM2 工程实现**。
