# HZY HSPN 新月体子类三分类 ROI classifier（2026-06-19）

## 目标

本轮目标是在已经跑通 `crescent_binary` 大类检测/分割的基础上，验证“先定位新月体候选区域，再做子类判别”的路线。这里不再把细胞性、纤维细胞性、纤维性新月体直接作为三分类像素分割主交付，而是从三类 mask 中抽取 lesion ROI，训练一个 ResNet18 三分类器。

## 数据生成

先从原始 HZY HSPN 标注重新生成三类新月体 glomerulus crop：

```bash
python scripts/prepare_hzy_glomerulus_lesion_crops.py \
  --images-dir /home/duyanhong/Dataspace/HZY/HZY_HSPN_export_ds025/images \
  --annotations-dir /home/duyanhong/Dataspace/HZY/HZY_HSPN_export_ds025/annotations \
  --output-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_crescent_subtype_20260619 \
  --tasks crescent \
  --crop-size 512 \
  --margin 64 \
  --min-source-crop-size 384 \
  --negative-ratio 0.5 \
  --max-background-crops-per-slide 8 \
  --seed 42 \
  --overwrite
```

再从三类 mask 中按连通域抽 ROI：

```bash
python scripts/prepare_hzy_crescent_subtype_rois.py \
  --dataset-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_crescent_subtype_20260619/crescent \
  --output-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619 \
  --min-area 32 \
  --box-margin 48 \
  --crop-size 224 \
  --overwrite
```

ROI 数据集：

| split | cellular | fibrocellular | fibrous | total |
|---|---:|---:|---:|---:|
| train | 59 | 40 | 17 | 116 |
| val | 9 | 5 | 7 | 21 |
| total | 68 | 45 | 24 | 137 |

说明：验证集三类都有样本，但总数只有 21 个 ROI，因此本轮结果只作为路线可行性基线，不作为稳定泛化结论。

## 训练设置

模型：torchvision ResNet18，ImageNet 预训练，最后一层改为 3 类输出。训练时使用 class weight、weighted sampler 和基础增强，以缓解纤维性新月体样本偏少。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_hzy_crescent_subtype_classifier.py \
  --data-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619 \
  --output-dir /home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619 \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --num-workers 4 \
  --model resnet18 \
  --pretrained imagenet \
  --allow-random-init \
  --freeze-backbone-epochs 3 \
  --weighted-sampler \
  --class-weights balanced \
  --augmentation basic \
  --save-montage \
  --device cuda
```

## 结果

Best epoch：21。

| Metric | Value |
|---|---:|
| Accuracy | 0.8095 |
| Balanced Accuracy | 0.8116 |
| Macro Precision | 0.7996 |
| Macro Recall | 0.8116 |
| Macro F1 | 0.8026 |
| Val loss | 0.6543 |

分类型指标：

| 类别 | Val support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 细胞性新月体 | 9 | 0.8750 | 0.7778 | 0.8235 |
| 纤维细胞性新月体 | 5 | 0.6667 | 0.8000 | 0.7273 |
| 纤维性新月体 | 7 | 0.8571 | 0.8571 | 0.8571 |

混淆矩阵（行是真值，列是预测）：

| true \ pred | cellular | fibrocellular | fibrous |
|---|---:|---:|---:|
| cellular | 7 | 2 | 0 |
| fibrocellular | 0 | 4 | 1 |
| fibrous | 1 | 0 | 6 |

## 阶段判断

- ROI 三分类路线是可行的，指标明显比此前直接解释三分类像素分割更适合作为“亚型判别”入口。
- 纤维细胞性新月体仍是最容易混淆的类别，符合其作为过渡形态的病理特点。
- 当前验证集只有 21 个 ROI，指标波动会很大；后续需要医生复核更多候选 ROI，尤其是细胞性 vs 纤维细胞性、纤维细胞性 vs 纤维性的边界样本。
- 推荐技术路线保持：`crescent_binary` 检测/分割负责“有没有、在哪里”；ROI classifier 负责“属于哪一型”；像素级亚型 mask 暂不作为主交付指标。

## 产物路径

- ROI 数据集：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619`
- ROI manifest：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619/manifest.csv`
- ROI summary：`/home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619/summary.json`
- classifier checkpoint：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/best.pth`
- best metrics：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_metrics.json`
- confusion matrix：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_confusion_matrix.csv`
- prediction detail：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_predictions.csv`
- montage：`/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_montage.jpg`
