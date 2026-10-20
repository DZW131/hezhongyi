# 紫癜性肾炎数字病理分析

更新：2026-10-20。本分支 `mescnn-transfer` 汇集当前本地可用的项目代码，供继续开发、训练和推理。

项目包含肾小球分割、肾小球病变识别、肾小管萎缩分割、固有细胞点检测，以及分别输出的ISKDC和Oxford评分报告。

新同学先读：[运行指南](docs/GETTING_STARTED.md)、[权重清单](docs/MODEL_WEIGHTS.md)、[真实指标与口径](docs/RESULTS.md)。历史实验日志完整保留在 [LEGACY_WORKLOG.md](docs/LEGACY_WORKLOG.md)。

## 1. 获取代码和安装环境

```bash
git clone --branch mescnn-transfer https://github.com/DZW131/hezhongyi.git
cd hezhongyi
conda create -n hspn python=3.10 -y
conda activate hspn
# 先安装与显卡/CUDA匹配的PyTorch和torchvision。以下仅为CPU检查示例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-project.txt
# 需要读取CZI时安装
pip install aicspylibczi czifile
python -m pytest tests -q
```

RTX 5090需要支持其架构的PyTorch/CUDA版本，不能照搬旧版cu121或第三方requirements中的torch 1.13。细胞任务另安装 `pip install -r requirements-cells.txt`，代码包含可微PyTorch形变注意力回退。当前交付未重新核对服务器GPU环境，因此不提供伪造的环境锁文件。

## 2. 当前技术路线和入口

| 工作 | 模型/方案 | 入口 |
| --- | --- | --- |
| CZI/标注导出 | scene拆分、坐标变换、GeoJSON | `scripts/export_hzy_slice_db_annotations.py`、`scripts/convert_hzy_czi_to_tiff.py` |
| 肾小球分割 | HuBMAP预训练后院内微调U-Net | `train.py`、`predict_tiff.py`、`evaluate_checkpoint.py` |
| 增生判断 | MESCNN EfficientNetV2-M，M/E多标签与M单任务 | `scripts/train_hzy_mescnn_proliferation_multilabel.py`、`scripts/train_hzy_mescnn_mesangial_binary.py` |
| 新月体定位与亚型 | Faster R-CNN + ResNet18 ROI三分类 | `scripts/train_hzy_detection_boxes.py`、`scripts/train_hzy_crescent_subtype_classifier.py` |
| 肾小管萎缩 | U-Net及场景面积统计 | `scripts/evaluate_hzy_tubular_atrophy.py`、`scripts/infer_hzy_tubular_atrophy_scenes.py` |
| 弱监督研究支线 | SSHR启发的HFRM-CAM | `scripts/train_hzy_weakseg_hfrm.py`、`scripts/infer_hzy_weakseg_hfrm.py` |
| 固有细胞 | DPA-P2PNet，内皮/系膜/足细胞点检测 | `vendor/ihc-main/train.py`、`scripts/visualize_hzy_intrinsic_cell_dpa.py` |
| 联合模型推理 | M/E + 新月体检测/亚型 + 可选萎缩CSV | `scripts/run_hspn_full_report_inference.py` |
| 统计与分型对照 | 独立ISKDC/Oxford报告、参考分型比较 | `scripts/aggregate_hspn_inference_for_scoring.py`、`scripts/score_hspn_pathology.py`、`scripts/compare_hspn_doctor_scores.py` |

联合报告脚本从肾小球crop及manifest开始，不是任意CZI的一键入口。WSI读取、肾小球分割、实例裁图、病变推理需要分阶段衔接。标注ROI裁图可验证病变模块，不能替代自动肾小球分割的端到端测试。

## 3. 数据与目录

```text
data/                     # 不进入Git，内部获取
  raw/case_x/slide.czi
  raw/case_x/slice.db
  export/images/          # scene TIFF
  export/annotations/     # 同坐标系GeoJSON
  glom_tiles/train/images/
  glom_tiles/train/masks/
  glom_tiles/val/images/
  glom_tiles/val/masks/
  lesions/proliferation/manifests/crops.csv
  lesions/proliferation/train/images/
  lesions/proliferation/val/images/
checkpoints/              # 权重从Release或内部获取
outputs/                  # 本机结果
scripts/                  # 数据、训练、推理、评分
unet/ utils/              # U-Net及共享组件
vendor/ihc-main/           # DPA-P2PNet本地适配代码
tests/ docs/              # 测试与接手说明
```

crop manifest至少包含 `slide_id,crop_id,split,image_path`。训练增生模型还需要 `class_pixels`，例如 `{"1":120,"2":0}`，1为系膜增生、2为毛细血管内增生。图像路径相对任务根目录或为绝对路径。坐标可填 `bbox_x_min,bbox_y_min,bbox_x_max,bbox_y_max`。

同一患者/切片的全部scene/crop保持同组。部分历史准备脚本默认按scene随机划分，新实验应传入经患者级核对的 `--split-csv`。测试集不能参与选模与调阈值。

## 4. 先跑评分器和测试

```bash
python scripts/score_hspn_pathology.py --write-template outputs/score_input_template.csv
# 填好逐肾小球预测表后
python scripts/score_hspn_pathology.py --input-csv outputs/per_glomerulus_predictions.csv --output-dir outputs/pathology
python -m pytest tests -q
```

评分器输出独立的Oxford/ISKDC CSV、JSON、病变统计与证据图。Oxford C统计活动性细胞性/纤维细胞性新月体，ISKDC默认统计全部新月体。分母为纳入统计的肾小球，同一小球多ROI须去重。实现见 `score_oxford`、`score_iskdc`。

Oxford结果是M/E/S/T/C组合，不另合并成单一等级。缺失IFTA时为T_NA；预测萎缩面积代理量与完整IFTA须区分。缺失S来源时不能把默认S0视为经过模型验证。

## 5. 训练与推理

```bash
# 肾小球U-Net，后续可加 --load 指定微调初始化
python train.py --images-dir data/glom_tiles/train/images --masks-dir data/glom_tiles/train/masks --val-images-dir data/glom_tiles/val/images --val-masks-dir data/glom_tiles/val/masks --classes 2 --scale 1.0 --epochs 40 --batch-size 4 --checkpoint-dir checkpoints/glom --wandb-mode disabled --compile off

# M/E多标签微调，同一肾小球可同时有两类增生
python scripts/train_hzy_mescnn_proliferation_multilabel.py --data-root data/lesions/proliferation --output-dir checkpoints/proliferation --init-model checkpoints/efficientnetv2-m_E_V3.pth --epochs 30 --image-size 384 --batch-size 8 --loss focal_bce --weighted-sampler --class-weights none --device cuda

# 弱监督CAM研究支线，示例启动配置
python scripts/train_hzy_weakseg_hfrm.py --task crescent --data-root data/lesions/crescent --output-dir checkpoints/weakseg_crescent --epochs 40 --image-size 384 --batch-size 8 --pretrained --device cuda
```

其他完整命令见 [GETTING_STARTED.md](docs/GETTING_STARTED.md)。原始MESCNN权重用于迁移基线/初始化，正式联合推理还需要院内M/E微调、新月体检测/亚型权重，具体见 [MODEL_WEIGHTS.md](docs/MODEL_WEIGHTS.md)。数据和权重未齐时，仅克隆代码无法复现完整模型报告。

## 6. 历史结果与本次检查

| 模块 | 已有结果 |
| --- | --- |
| 肾小球，院内验证 | Dice 0.9397，IoU 0.8863 |
| M/E多标签，170crop验证 | 宏F1 0.7375，平均AUC 0.8818 |
| 新月体检测 | 存在性F1 0.850，项目宽松框匹配F1 0.791 |
| 新月体ROI三分类 | 准确率0.810，宏F1 0.803 |
| 肾小管萎缩，独立测试 | Dice 0.6005，Recall 0.6376 |
| 固有细胞，80轮最优独立测试 | 点检测F1 0.608，三类宏F1 0.379 |
| 95例ISKDC对照 | 准确率0.6316，宏F1 0.3939 |

模块验证、独立测试和病例级对照的口径不同，数据划分与来源见 [RESULTS.md](docs/RESULTS.md)。本次本地评分、汇总、训练扩展与弱监督组件27项测试通过。服务器恢复连接后核对并下载了正式模型权重，未重跑GPU训练或95例推理。

## 7. 接手顺序

1. 克隆本分支、安装依赖、跑测试与评分器模板。
2. 下载已核实权重，跑肾小球大图推理和MESCNN迁移基线。
3. 补齐院内微调权重、原始数据和固定划分后跑联合报告。
4. 核对标签顺序、阈值、病例映射和分母，再做参考分型比较。

病例图像、明细、权重和凭据不进入普通Git提交。第三方来源见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。
