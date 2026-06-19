# HZY HSPN 新月体二分类新版框评估（2026-06-19）

## 运行结论

已在 5090 服务器完成 `crescent_binary` 的新版流程：仍以分割模型为主线训练，然后把预测 mask 转成检测框做辅助评估。

主阈值 `0.50` 下，病例/crop 级别能把所有阳性样本召回出来，但假阳性偏多；框级别同样表现为高召回、低精度。

| threshold | presence_precision | presence_recall | presence_f1 | presence_specificity | box_precision | box_recall | box_f1 | matched / gt boxes | pred boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.3500 | 1.0000 | 0.5185 | 0.7214 | 0.1391 | 1.0000 | 0.2442 | 21 / 21 | 151 |
| 0.30 | 0.3684 | 1.0000 | 0.5385 | 0.7429 | 0.1489 | 1.0000 | 0.2593 | 21 / 21 | 141 |
| 0.40 | 0.4118 | 1.0000 | 0.5833 | 0.7857 | 0.1653 | 0.9524 | 0.2817 | 20 / 21 | 121 |
| 0.50 | 0.4286 | 1.0000 | 0.6000 | 0.8000 | 0.1818 | 0.9524 | 0.3053 | 20 / 21 | 110 |
| 0.60 | 0.4468 | 1.0000 | 0.6176 | 0.8143 | 0.1905 | 0.9524 | 0.3175 | 20 / 21 | 105 |
| 0.70 | 0.4773 | 1.0000 | 0.6462 | 0.8357 | 0.1939 | 0.9048 | 0.3193 | 19 / 21 | 98 |

## 主阈值 0.50 细节

- 验证 crop：161
- GT 阳性 crop：21；GT 阴性 crop：140
- 预测阳性 crop：49；预测阴性 crop：112
- crop 状态：`tp_localized=20`，`positive_wrong_location=1`，`fp_overcalled=28`，`tn=112`
- 框统计：`gt_boxes=21`，`pred_boxes=110`，`matched_boxes=20`，`unmatched_gt_boxes=1`，`unmatched_pred_boxes=90`

## 分割训练结果

- best epoch：12
- best Dice：0.6175
- IoU：0.4467
- Precision：0.5803
- Recall：0.6598
- Specificity：0.9963
- Accuracy：0.9937

## Detection-first 新月体检测结果

已将增生类病变上的 Faster R-CNN detection-first 方法应用到 `crescent_binary`。这条线不替代分割模型，而是作为“有没有、在哪里”的检测主线；分割 Dice/面积指标继续作为形态和面积辅助评估。

为适配新月体阳性少、阴性多的情况，训练脚本新增：

- `--ensure-positive-batches`：训练时过采样阳性样本，保证每个 batch 至少包含一个阳性框。
- `--fail-on-nonfinite-loss`：调试开关；默认遇到 NaN/Inf loss 会跳过该 batch 并记录 `nonfinite_batches`，开启后直接报错。

最终推荐使用 balanced 版结果：

- 训练数据：591 张 train crop，117 个 train boxes，112 张 train 阳性，479 张 train 阴性
- 验证数据：161 张 val crop，21 个 val boxes，21 张 val 阳性，140 张 val 阴性
- 模型：torchvision Faster R-CNN MobileNetV3 FPN，COCO 预训练，基础增强
- 训练设置：50 epoch，batch size 4，learning rate `5e-5`，`detections_per_img=20`，NMS `0.4`
- best epoch：34
- nonfinite batches：0

| threshold | presence_precision | presence_recall | presence_f1 | presence_specificity | box_precision | box_recall | box_f1 | matched / gt boxes | pred boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.7727 | 0.8095 | 0.7907 | 0.9643 | 0.6296 | 0.8095 | 0.7083 | 17 / 21 | 27 |
| 0.10 | 0.8500 | 0.8095 | 0.8293 | 0.9786 | 0.7083 | 0.8095 | 0.7556 | 17 / 21 | 24 |
| 0.20 | 0.8500 | 0.8095 | 0.8293 | 0.9786 | 0.7083 | 0.8095 | 0.7556 | 17 / 21 | 24 |
| 0.30 | 0.8947 | 0.8095 | 0.8500 | 0.9857 | 0.7727 | 0.8095 | 0.7907 | 17 / 21 | 22 |
| 0.40 | 0.8947 | 0.8095 | 0.8500 | 0.9857 | 0.7727 | 0.8095 | 0.7907 | 17 / 21 | 22 |
| 0.50 | 0.8947 | 0.8095 | 0.8500 | 0.9857 | 0.7727 | 0.8095 | 0.7907 | 17 / 21 | 22 |
| 0.60 | 0.8750 | 0.6667 | 0.7568 | 0.9857 | 0.7778 | 0.6667 | 0.7179 | 14 / 21 | 18 |
| 0.70 | 0.8667 | 0.6190 | 0.7222 | 0.9857 | 0.7647 | 0.6190 | 0.6842 | 13 / 21 | 17 |

和分割转框主阈值 `0.50` 相比：

| 方法 | threshold | presence_precision | presence_recall | presence_f1 | box_precision | box_recall | box_f1 | pred boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 分割转框 | 0.50 | 0.4286 | 1.0000 | 0.6000 | 0.1818 | 0.9524 | 0.3053 | 110 |
| Faster R-CNN balanced | 0.30-0.50 | 0.8947 | 0.8095 | 0.8500 | 0.7727 | 0.8095 | 0.7907 | 22 |

判断：detection-first 对新月体非常有效，显著减少多余框和阴性误报；代价是召回从 `0.95/1.00` 降到 `0.81`。因此当前推荐交付口径是：

- 新月体筛查/不可漏检：保留分割转框的低阈值高召回作为候选。
- 新月体定位展示/病例级统计：优先用 Faster R-CNN balanced，阈值 `0.30-0.50`。
- 最终组合：检测框负责候选区域和阳性肾小球判断，分割模型在框内细化面积与形态。

## 数据与训练口径

- 任务：`crescent_binary`（新月体二分类）
- slides：90
- glomeruli：1364
- crop size：512
- margin：64
- min source crop size：384
- negative ratio：0.5
- train crops：591
- val crops：161
- positive crops：133
- negative crops：619
- 观察到的新月体特征计数：细胞性 56，纤维细胞性 42，纤维性 22
- 类别像素：细胞性 1,707,193，纤维细胞性 1,484,185，纤维性 483,381

说明：这次是在 5090 服务器重新生成数据并训练的复跑版。阳性数量和类别像素与旧摘要基本一致，但当前脚本/数据生成得到的负样本 crop 数比 2026-06-15 旧摘要更多，因此这份结果适合评估新版框指标趋势，不作为严格复现旧 split 的结果。

## 产物路径

- 运行日志：`/home/duyanhong/Dataspace/HZY/runs/crescent_binary_box_eval_match0615_20260619/run.log`
- 数据目录：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_crescent_binary_match0615_20260619/crescent_binary`
- best checkpoint：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_glomerulus_lesion_tasks_crescent_binary_match0615_20260619/crescent_binary/best.pth`
- 分割训练指标：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_glomerulus_lesion_tasks_crescent_binary_match0615_20260619/crescent_binary/analysis/best_metrics.json`
- 框评估目录：`/home/duyanhong/Dataspace/HZY/eval/crescent_binary_box_eval_match0615_20260619`
- 框评估指标：`/home/duyanhong/Dataspace/HZY/eval/crescent_binary_box_eval_match0615_20260619/metrics.csv`
- 框评估逐例结果：`/home/duyanhong/Dataspace/HZY/eval/crescent_binary_box_eval_match0615_20260619/cases.csv`
- 预览图：`/home/duyanhong/Dataspace/HZY/eval/crescent_binary_box_eval_match0615_20260619/previews`
- detection box 数据集：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/crescent_binary_match0615_20260619`
- detection 探路版 checkpoint：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_20260619`
- detection 稳定版 checkpoint：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_stable_20260619`
- detection balanced 推荐 checkpoint：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_balanced_20260619`
- detection balanced 运行日志：`/home/duyanhong/Dataspace/HZY/runs/crescent_binary_detection_fasterrcnn_balanced_20260619/run.log`

## 复现命令

```bash
python scripts/prepare_hzy_detection_boxes.py \
  --dataset-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_crescent_binary_match0615_20260619/crescent_binary \
  --output-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/crescent_binary_match0615_20260619 \
  --class-mode binary \
  --class-name crescent \
  --min-area 16 \
  --box-margin 4 \
  --link-mode symlink \
  --overwrite

CUDA_VISIBLE_DEVICES=1 python scripts/train_hzy_detection_boxes.py \
  --data-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/crescent_binary_match0615_20260619 \
  --output-dir /home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_balanced_20260619 \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --num-workers 4 \
  --pretrained coco \
  --augmentation basic \
  --augmentation-seed 42 \
  --ensure-positive-batches \
  --detections-per-img 20 \
  --nms-thresh 0.4 \
  --score-thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
  --primary-threshold 0.3 \
  --checkpoint-metric box_f1 \
  --save-best-metrics box_f1,presence_f1 \
  --match-iou 0.1
```
