# 河中医紫癜性肾炎增生大类检测实验记录（2026-06-18）

## 目标

本轮目标是把“系膜细胞增生 + 毛细血管内细胞增生”先作为一个大类病变处理，从像素级 Dice 评价转到肾小球 crop 级别的 bbox 检出评价，优先解决“有没有增生、在哪里”的展示和量化问题。

## 数据与评价口径

- 服务器：5090，`duyanhong@100.81.116.53`
- 检测数据集：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/proliferation_binary`
- 数据规模：train 347 张、val 106 张；train 380 个框、val 108 个框
- 模型：torchvision Faster R-CNN MobileNetV3 FPN
- bbox 命中：IoU >= 0.1，或预测框中心落在 GT 框内
- presence 指标：以每个肾小球 crop 是否存在增生作为二分类
- box 指标：以预测框和 GT 框的一对一匹配统计 precision / recall / F1

## Baseline：从零训练 Faster R-CNN

远端目录：

`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/proliferation_binary_fasterrcnn`

- best epoch：22，按 `box_f1` 保存
- best checkpoint：`best.pth`
- baseline 预览图：`/home/duyanhong/Dataspace/HZY/outputs/proliferation_detection_previews_thr04`

| 阈值 | Presence Precision | Presence Recall | Presence Specificity | Presence F1 | Box Precision | Box Recall | Box F1 | 说明 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.7312 | 0.9189 | 0.2188 | 0.8144 | 0.1568 | 0.5370 | 0.2427 | 召回较高，但误报和重复框很多 |
| 0.40 | 0.7538 | 0.6622 | 0.5000 | 0.7050 | 0.3611 | 0.3611 | 0.3611 | baseline 最适合展示的折中阈值 |

结论：baseline 能说明 bbox 检出路线可行，但从零训练的误报和漏检控制都不够稳。

## 优化版：COCO 预训练 + 基础增强 + 候选框限制

远端目录：

`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/proliferation_binary_fasterrcnn_coco_aug_presence`

关键设置：

- COCO 预训练初始化：`--pretrained coco`
- 训练增强：`--augmentation basic`
- 前 3 轮冻结 backbone：`--freeze-backbone-epochs 3`
- 每张图最多 20 个候选框：`--detections-per-img 20`
- NMS：`--nms-thresh 0.35`
- 训练 60 epoch，学习率 `5e-5`，StepLR 25 epoch 衰减 0.5
- 主保存指标：`presence_f1`，主阈值 `0.2`

### Best Presence 模型

- checkpoint：`best.pth`
- epoch：9
- 用途：更适合“少漏检”的筛查/医生展示初筛口径

| 阈值 | Presence Precision | Presence Recall | Presence Specificity | Presence F1 | Box Precision | Box Recall | Box F1 | 说明 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 0.7766 | 0.9865 | 0.3438 | 0.8690 | 0.2355 | 0.7500 | 0.3584 | 几乎不漏检，但框偏多 |
| 0.20 | 0.7935 | 0.9865 | 0.4063 | 0.8795 | 0.2809 | 0.6944 | 0.4000 | 推荐作“少漏检”阈值 |
| 0.40 | 0.8125 | 0.8784 | 0.5313 | 0.8442 | 0.3678 | 0.5926 | 0.4539 | 更平衡，仍保持较高召回 |
| 0.60 | 0.8571 | 0.7297 | 0.7188 | 0.7883 | 0.4701 | 0.5093 | 0.4889 | 推荐作“少误报/展示更干净”阈值 |

### Latest 模型

- checkpoint：`latest.pth`
- epoch：60
- 用途：定位更稳，presence 召回比第 9 轮低

| 阈值 | Presence Precision | Presence Recall | Presence Specificity | Presence F1 | Box Precision | Box Recall | Box F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.8261 | 0.7703 | 0.6250 | 0.7972 | 0.5093 | 0.5093 | 0.5093 |
| 0.40 | 0.8571 | 0.7297 | 0.7188 | 0.7883 | 0.5417 | 0.4815 | 0.5098 |
| 0.50 | 0.8667 | 0.7027 | 0.7500 | 0.7761 | 0.5543 | 0.4722 | 0.5100 |

历史日志显示第 32 轮的主阈值 box F1 最高（约 0.5411），但上一版脚本只保存 `best.pth` 和 `latest.pth`，因此第 32 轮 checkpoint 没有保留下来。当前脚本已补充 `--save-best-metrics`，后续训练会额外保存 `best_box_f1.pth`、`best_presence_f1.pth`，避免再次丢失定位最优模型。

## 推荐结论

- 临床筛查、尽量少漏检：用优化版 `best.pth`，阈值 `0.2`。验证集 presence recall 0.9865，只漏 1 个阳性 crop；代价是 specificity 0.4063，误报仍偏多。
- 汇报展示、误报少一些：用优化版 `best.pth`，阈值 `0.6`，或用 `latest.pth` 阈值 `0.4/0.5`。这两类设置的框级 F1 接近 0.49-0.51，画出来会比低阈值干净。
- 下一步最值得做：重新跑一版保留多指标 best checkpoint 的训练，或者从 `latest.pth` / `best.pth` 各出一组可视化预览，供医生对比“少漏检”和“少误报”两种策略。

## 2026-06-19 追加优化复跑

新月体检测任务中，`--ensure-positive-batches` 和 detection-first 主线明显提升了框指标。为验证同样策略是否适用于增生类病变，本轮追加跑了三组增生检测实验：

1. `box_balanced`：原始增生检测数据集，开启 `--ensure-positive-batches`，主指标 `box_f1@0.4`。
2. `box_clean`：原始增生检测数据集，不做 positive-balanced 采样，主指标 `box_f1@0.5`，用于找更干净的展示阈值。
3. `hardneg`：重新生成 768 crop，将 `negative_ratio` 从 `0.25` 提高到 `1.0`，再训练检测模型。

### 新增数据口径

原始增生检测数据集：

- train：347 张图，380 个框，242 张阳性图，105 张阴性图
- val：106 张图，108 个框，74 张阳性图，32 张阴性图

hard-negative 随机阴性增强数据集：

- train：555 张图，380 个框，242 张阳性图，313 张阴性图
- val：162 张图，108 个框，74 张阳性图，88 张阴性图
- 说明：框数量不变，随机阴性图显著增加。

### 追加实验结果

| 实验 | 数据 | 主保存阈值 | best epoch | Presence Precision | Presence Recall | Presence Specificity | Presence F1 | Box Precision | Box Recall | Box F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 旧优化版 best presence | 原始数据 | 0.20 | 9 | 0.7935 | 0.9865 | 0.4063 | 0.8795 | 0.2809 | 0.6944 | 0.4000 | 最适合少漏检筛查 |
| 旧优化版 latest | 原始数据 | 0.50 | 60 | 0.8667 | 0.7027 | 0.7500 | 0.7761 | 0.5543 | 0.4722 | 0.5100 | 当前较干净展示口径 |
| 旧日志定位峰值 | 原始数据 | 0.20 | 32 | 0.8438 | 0.7297 | 0.6875 | 0.7826 | 0.5657 | 0.5185 | 0.5411 | 历史最高 box F1，但当时未保存 checkpoint |
| 新 box_balanced | 原始数据 | 0.40 | 33 | 0.8429 | 0.7973 | 0.6563 | 0.8194 | 0.4771 | 0.4815 | 0.4793 | 不如旧版，positive-balanced 不适合增生 |
| 新 box_clean | 原始数据 | 0.50 | 20 | 0.8730 | 0.7432 | 0.7500 | 0.8029 | 0.5435 | 0.4630 | 0.5000 | 接近旧 latest，但没有提升 |
| 新 hardneg | 随机阴性增强 | 0.50 | 11 | 0.6623 | 0.6892 | 0.7045 | 0.6755 | 0.4793 | 0.5370 | 0.5066 | 随机加阴性没有提升，还降低 presence |

### 阶段判断

本轮优化说明：增生病变的主要瓶颈不是训练 batch 是否包含阳性，也不是简单增加随机阴性图。增生类病变更接近“肾小球内部结构/细胞密度判别”问题，正常高细胞区、系膜区密集区域、切片染色差异都会造成误报；随机负例没有精准覆盖模型最容易误报的结构。

当前推荐仍然保持：

- 少漏检筛查：旧优化版 `best.pth`，阈值 `0.2`。
- 展示/定位更干净：旧优化版 `latest.pth`，阈值 `0.4/0.5`。
- 不采用：`box_balanced`、`box_clean`、随机 `hardneg` 作为主结果。

下一步真正值得做的是“模型误报驱动”的 hard-negative mining：

- 用旧优化版模型在 train/val 或未参与训练的肾小球 crop 上跑预测。
- 收集高分假阳性区域，尤其是无增生但细胞密度高、系膜区纹理复杂的肾小球。
- 将这些误报图作为困难阴性加入训练，而不是随机增加普通阴性。
- 或者改成两阶段：Faster R-CNN 先给候选框，再接一个肾小球/候选区域分类器判断是否真增生。

## 复现命令

```bash
python scripts/train_hzy_detection_boxes.py \
  --data-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/proliferation_binary \
  --output-dir /home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/proliferation_binary_fasterrcnn_coco_aug_presence \
  --epochs 60 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --pretrained coco \
  --augmentation basic \
  --augmentation-seed 2026 \
  --freeze-backbone-epochs 3 \
  --detections-per-img 20 \
  --nms-thresh 0.35 \
  --lr-step-size 25 \
  --lr-gamma 0.5 \
  --score-thresholds 0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6 \
  --primary-threshold 0.2 \
  --checkpoint-metric presence_f1 \
  --save-best-metrics box_f1,presence_f1 \
  --match-iou 0.1 \
  --device cuda
```

追加复跑命令示例：

```bash
# box_clean：原始数据，按 box_f1@0.5 保存
CUDA_VISIBLE_DEVICES=1 python scripts/train_hzy_detection_boxes.py \
  --data-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/proliferation_binary \
  --output-dir /home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/proliferation_binary_fasterrcnn_coco_aug_box_clean_20260619 \
  --epochs 70 \
  --batch-size 4 \
  --learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --num-workers 4 \
  --pretrained coco \
  --augmentation basic \
  --augmentation-seed 2026 \
  --freeze-backbone-epochs 3 \
  --detections-per-img 20 \
  --nms-thresh 0.35 \
  --lr-step-size 25 \
  --lr-gamma 0.5 \
  --score-thresholds 0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --primary-threshold 0.5 \
  --checkpoint-metric box_f1 \
  --save-best-metrics box_f1,presence_f1 \
  --match-iou 0.1 \
  --device cuda
```
