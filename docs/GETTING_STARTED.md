# 接手运行指南

从仓库根目录执行，先完成README中的安装。路径均为示例，命令是启动配置，不声称逐项复现最佳实验。

## 数据准备

```bash
python scripts/export_hzy_slice_db_annotations.py --raw-root data/raw --output-dir data/export --split-scenes --coordinate-scale 0.25
python scripts/convert_hzy_czi_to_tiff.py --manifest-csv data/export/hzy_hspn_manifest.csv --output-dir data/export/images --downsample 0.25
python scripts/prepare_hubmap_tiles.py --images-dir data/export/images --annotations-dir data/export/annotations --annotation-format geojson --target-labels 未废弃肾小球 废弃肾小球 --output-dir data/glom_tiles --split-csv data/splits.csv --tile-size 1024 --stride 1024
python scripts/prepare_hzy_glomerulus_lesion_crops.py --images-dir data/export/images --annotations-dir data/export/annotations --output-root data/lesions --tasks proliferation crescent --split-csv data/splits.csv --crop-size 768
```

按实际输出核对manifest和annotations文件名。图像与polygon必须使用同一缩放和scene原点。`label.png`等玻片标签图不是mask。裁图脚本历史上主要面向train/val，正式独立测试需要另外维护固定患者级划分。

## 肾小球大图推理

```bash
python predict_tiff.py --model checkpoints/hzy_hubmap_finetuned_glomerulus_unet_best.pth --input data/export/images/example.tiff --output outputs/example_glom_mask.png --tile-size 1024 --scale 0.5 --classes 2 --threshold 0.5 --postprocess --fill-holes --hole-repair-radius 10 --smooth-radius 2 --min-component-area 800
```

后续需按mask连通实例裁出肾小球，保存slide_id和全局坐标。当前联合报告入口从crop manifest开始，尚无任意WSI的一键包装器。

## 新月体检测与三分类

```bash
python scripts/prepare_hzy_detection_boxes.py --dataset-root data/lesions/crescent --output-root data/crescent_boxes --class-mode binary --class-name crescent --link-mode copy
python scripts/train_hzy_detection_boxes.py --data-root data/crescent_boxes --output-dir checkpoints/crescent_detector --pretrained coco --augmentation basic --ensure-positive-batches --epochs 30 --batch-size 2 --device cuda
python scripts/prepare_hzy_crescent_subtype_rois.py --dataset-root data/lesions/crescent --output-root data/crescent_subtypes
python scripts/train_hzy_crescent_subtype_classifier.py --data-root data/crescent_subtypes --output-dir checkpoints/crescent_subtype --model resnet18 --pretrained imagenet --epochs 40 --batch-size 16 --device cuda
```

项目检测框评估默认IoU 0.1或中心命中，不等于COCO mAP，报告时需明确口径。

## 萎缩分割与T输入

`prepare_hzy_lesion_tiles.py --tasks tubular_atrophy` 支持train/val/test；训练使用根目录 `train.py`。

```bash
python scripts/evaluate_hzy_tubular_atrophy.py --dataset-root data/tubular_atrophy --checkpoint checkpoints/tubular_atrophy/best.pth --output-dir outputs/atrophy_test --split test --threshold 0.4
python scripts/infer_hzy_tubular_atrophy_scenes.py --scenes-dir data/export/images --checkpoint checkpoints/tubular_atrophy/best.pth --output-dir outputs/atrophy --threshold 0.2
```

评估/场景推理阈值需要按冻结配置设置。场景脚本输出预测萎缩面积/组织面积，亮度阈值排除背景，此量作为T代理输入，不等于完整IFTA。

## 联合推理、评分与对照

```bash
python scripts/run_hspn_full_report_inference.py \
  --data-root data/lesions/proliferation \
  --manifest data/lesions/proliferation/manifests/crops.csv \
  --split val --output-dir outputs/full_report \
  --m-checkpoint checkpoints/m_finetuned/best.pth \
  --e-checkpoint checkpoints/e_finetuned/best.pth \
  --m-init-model checkpoints/efficientnetv2-m_M_V3.pth \
  --e-init-model checkpoints/efficientnetv2-m_E_V3.pth \
  --detection-checkpoint checkpoints/crescent_detector/best.pth \
  --subtype-checkpoint checkpoints/crescent_subtype/best.pth \
  --mesangial-threshold 0.35 --endocapillary-threshold 0.20 \
  --crescent-threshold 0.30 --device cuda
```

可附加 `--tubular-atrophy-slide-csv` 接入场景萎缩结果。所有默认服务器路径都需显式覆盖。产出逐小球预测、新月体ROI、独立ISKDC/Oxford CSV/JSON、真实crop证据图和汇总。无真实底图时统计示意不能当成实际定位证据。

```bash
python scripts/aggregate_hspn_inference_for_scoring.py --manifest-csv data/lesions/proliferation/manifests/crops.csv --proliferation-csv outputs/proliferation.csv --crescent-detection-csv outputs/crescent_detection.csv --crescent-subtype-csv outputs/crescent_subtype.csv --output-csv outputs/per_glomerulus_predictions.csv --score-output-dir outputs/pathology
python scripts/build_hspn_scoring_validation_manifest.py --score-xlsx data/reference_scores.xlsx --atrophy-root data/raw_atrophy --task2-root data/raw_cells --output-dir outputs/validation_manifest
python scripts/compare_hspn_doctor_scores.py --case-manifest outputs/validation_manifest/unique_case_manifest.csv --prediction-csv outputs/full_report/per_glomerulus_predictions.csv --output-dir outputs/comparison
```

对照前核对病例ID和重复，输出包括 `case_comparison.csv`、`metric_summary.csv`、混淆表及汇总JSON。

## 弱监督研究支线

```bash
python scripts/infer_hzy_weakseg_hfrm.py --checkpoint checkpoints/weakseg_crescent/best.pth --task crescent --data-root data/lesions/crescent --split val --output-csv outputs/weakseg.csv --output-dir outputs/weakseg --save-masks --save-overlays --device cuda
```

HFRM-CAM为SSHR启发的本项目实现，以图像级标签监督多层分类/CAM融合，加入辅助分类分支和弱约束，mask主要用于验证。CAM阈值mask不能表述成精确分割真值；该研究支线不自动替换正式模型。

## 固有细胞点检测

```bash
python scripts/export_hzy_intrinsic_cell_annotations.py --raw-root data/raw_cells --output-dir data/cell_export
python scripts/prepare_hzy_intrinsic_cell_p2pnet.py --export-dir data/cell_export --output-dir data/cells --image-size 1024 --seed 20261008
pip install -r requirements-cells.txt
cd vendor/ihc-main
WANDB_MODE=disabled python train.py --model dpa --backbone vit_adapter --dataset ../../data/cells --num_classes 3 --space 8 --batch_size 1 --epochs 80 --start_eval 0 --eval_interval 5 --save_interval 10 --match_dis 10 --lr 0.0001 --num_workers 4 --name hspn_cells_80e
cd ../..
```

上游loader的 `test_image/test_point` 实际用于验证，准备脚本复制的是val；独立测试单独保存在 `heldout_test_image/heldout_test_point`。
准备脚本默认mean/std为初始值，80轮实验使用训练集实际统计 `[0.662,0.533,0.647]` / `[0.128,0.205,0.148]`，复现前应按训练集重算 `mean_std.npy`。

```bash
# 可视化脚本读取test_*；独立测试时单独建立目录放入heldout内容及相同mean_std.npy
python scripts/visualize_hzy_intrinsic_cell_dpa.py --source-root vendor/ihc-main --dataset data/cells_heldout_eval --checkpoint checkpoints/cells/best.pth --output-dir outputs/cells --match-distance 10
```

类别顺序为内皮、系膜、足细胞。这里评估的是点检测与分类，不是细胞轮廓分割。

## 常见问题

- 不要直接使用第三方原始requirements，它含旧GPU依赖和无效包名。使用仓库根目录两个新requirements。
- 形变注意力扩展缺失时检查是否使用本分支fallback，避免导入旧build文件。
- 加载完整MESCNN对象只用已知来源的权重，不能改变标签顺序。
- 原始数据、病例表和模型权重均不在普通Git文件中，见权重说明。
