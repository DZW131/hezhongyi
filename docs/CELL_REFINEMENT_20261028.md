# 肾小球固有细胞检测与分类优化（2026-10-28）

本次已在服务器完成训练及固定参数后的测试复核。保持原DPA-P2PNet候选点检测器权重不变，新增细胞局部图像与周围组织联合分类模块，同时减少误检并改善三类细胞识别。以下指标为点检测结果，不是细胞分割结果。

## 数据与评估口径

- 沿用43例、547个肾小球图像的病例级划分：训练29例/377图，验证7例/95图，测试7例/75图。同病例不跨集合。
- 训练33,247个标注细胞点；验证8,580点；测试6,961点。43例与既往数据重叠24例，真正新增19例。
- 划分文件：`data/hzy_intrinsic_cell_upgrade_43_20261021/case_splits.csv`；SHA-256：`16cae49a091fad243a396084c7f253b1ebbbb36ba3369c23c89fa4fc80cf6f77`。
- 训练及评估均限制在已标注肾小球ROI内，忽略未标注区域。本实验使用已有ROI，不代表自动肾小球分割后整张切片的端到端能力。
- 在1024×1024肾小球图像坐标中，预测点和标注点距离严格小于10像素，采用最大二分匹配，一对一计数。整体检测忽略类别；各类检测还要求类别相同。宏平均F1为三类F1的算术平均。
- 每图计数平均绝对误差为 `mean(abs(预测数-标注数))`，单位为个/肾小球图像。计数接近不等于定位和分类正确。
- 这7例测试数据此前已分析过，因此本次结果属于相同队列上的回顾性比较，不能表述为新增盲测或外部验证。本次训练、选择方案和阈值仅使用训练/验证集。

## 方法与训练

1. **候选细胞位置**：冻结2026-10-21升级版DPA-P2PNet；先按前景/背景概率比0.4和4像素距离去重，保留较多候选位置。
2. **双视野分类**：每个候选点截取64×64局部图像和192×192上下文图像；后者缩放到64×64，两幅RGB图拼成6通道。使用ImageNet初始化的ResNet18提取512维特征，并与原检测器4类概率、相对ROI中心的二维坐标、归一化半径共7维特征拼接；519→128→4分类头输出内皮、系膜、足细胞和背景概率。推理不使用真实细胞标签或真实细胞数量。
3. **难负样本与类别平衡**：仅从训练集构建62,207个派生图块，其中内皮16,370、系膜17,902、足细胞14,208、背景13,727。包括标注中心、距标注中心不超过5像素的候选正样本，以及距任一标注点至少12像素的难负候选。派生图块不是新增独立病例或新增标注细胞。
4. **优化**：共10轮，批量192，混合精度BF16；AdamW，编码器学习率1e-5、分类头3e-4，权重衰减1e-3；逆频率平方根类别权重、标签平滑0.03，随机水平/垂直翻转。首轮冻结编码器，后续联合微调；验证集选择第10轮权重。
5. **验证集锁定参数**：最终概率为`0.75×新分类器概率+0.25×原检测器概率`，最大前景概率须不低于`1.4×背景概率`，10像素距离去重。并未更新候选点坐标回归器。

另比较了原始基线与升级版DPA的概率融合方案。两条方案均只在验证集选择，目标为`0.35×检测F1+0.45×宏平均F1+0.20×系膜F1−0.05×归一化计数误差`，附检测精确率/召回率约束。双视野方案验证目标0.58185，高于纯概率融合0.51088，因此只对双视野方案进行本轮测试复核。锁定记录见结果目录`selected_before_test.json`，记录时间早于测试输出。

## 相同7例、75图的结果

对照为2026-10-21的43例升级模型；所有数值由已保存的JSON结果提取。

| 指标 | 优化前 | 本次 | 变化 |
|---|---:|---:|---:|
| 整体检测精确率 | 0.690 | 0.753 | +6.3个百分点 |
| 整体检测召回率 | 0.767 | 0.789 | +2.2个百分点 |
| 整体检测F1 | 0.726 | 0.771 | +4.4个百分点 |
| 内皮细胞检测F1 | 0.454 | 0.610 | +15.5个百分点 |
| 系膜细胞检测F1 | 0.514 | 0.631 | +11.7个百分点 |
| 足细胞检测F1 | 0.504 | 0.614 | +11.1个百分点 |
| 三类宏平均F1 | 0.491 | 0.618 | +12.8个百分点 |

| 细胞类别 | 本次精确率 | 本次召回率 | 优化前计数误差 | 本次计数误差 |
|---|---:|---:|---:|---:|
| 内皮细胞 | 0.575 | 0.648 | 23.31 | 8.71 |
| 系膜细胞 | 0.651 | 0.613 | 27.47 | 11.17 |
| 足细胞 | 0.587 | 0.644 | 12.88 | 5.59 |

本次预测7,291个细胞点，真实标注6,961个；忽略类别时匹配5,492点。预测总数接近真实总数仍不能替代逐点检测指标。

验证集整体检测F1从0.644升至0.684，宏平均F1从0.445升至0.543，系膜细胞F1从0.504升至0.563。测试集中，既往4例/48图检测F1为0.790；真正新增3例/27图检测F1由0.677升至0.730，系膜F1由0.493升至0.598，系膜计数误差由38.37降至10.78。新增子集系膜预测1,016个、标注749个，仍有过检，不能认为跨病例差异已解决。

以病例为重采样单位进行2,000次配对自助法分析，整体检测F1提升的95%区间为[0.0208, 0.0677]，系膜F1提升区间为[0.0524, 0.1894]，宏平均F1提升区间为[0.1053, 0.1543]。召回率变化区间[-0.0046, 0.0431]跨0，不能据此认定召回率提升稳定；仅7例带来的不确定性仍然较大。

## 文件与复现

- 新脚本：`scripts/refine_hzy_intrinsic_cells.py`；概率融合对照：`scripts/calibrate_hzy_intrinsic_cells.py`。
- 结构配置与权重哈希：`configs/intrinsic_refinement_20261028.json`。
- 结果：`reports/hzy_intrinsic_refinement_20261028/`，含训练历史、验证参数搜索、测试JSON、逐图/逐病例CSV、对照CSV、配对自助法结果及真实图像证据。
- 证据图命名为`test_<crop_id>_evidence.png`。左侧圆圈为标注，右侧十字为预测，红色为内皮、绿色为系膜、蓝色为足细胞，青色轮廓为评估ROI。
- 逐点正确性核对图：`reports/hzy_intrinsic_refinement_20261028/matched_visualizations/`，含三张完整实例及三类正确检出局部。绿色圆圈代表与参考标注位置和类别均匹配，其他符号保留类别错误、误检、漏检。脚本为`scripts/visualize_hzy_intrinsic_cell_matches.py`，使用同类优先的一对一匹配；全75图同类匹配数量与结果JSON核对一致，新增匹配逻辑测试3项通过。
- 必须同时使用冻结的DPA权重和本次ResNet18权重，不能用新分类器单独替代完整检测器。
- 本地完整测试42项通过（1项既有依赖弃用警告），两份权重SHA-256核对通过；GPU训练及推理在RTX 5090D、PyTorch 2.11/CUDA 12.8环境完成。

服务器使用以下路径；`ROOT`须指向包含当前脚本和`vendor/ihc-main`的代码目录，数据按既有目录组织。需要依赖`requirements-cells.txt`及torchvision。新的实验应使用新的输出目录，并清空或隔离旧候选点缓存，避免不同权重或不同数据共用缓存。

```bash
ROOT=/home/duyanhong/Workspace/dpa-p2pnet-upgrade-20261021
DATA=/home/duyanhong/Dataspace/HZY/processed/hzy_intrinsic_cell_upgrade_43_20261021
DPA=/home/duyanhong/Dataspace/HZY/runs/intrinsic_upgrade_43_20261021/recall_mesangial/best.pth
OUT=/home/duyanhong/Dataspace/HZY/runs/intrinsic_refinement_20261028
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
IMAGENET=/home/duyanhong/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
cd "$ROOT"
# 复现训练须用新的OUT目录，避免覆盖已保存的最佳权重和冻结参数。
$PY scripts/refine_hzy_intrinsic_cells.py --mode fit \
  --source-root "$ROOT/vendor/ihc-main" --dataset "$DATA" \
  --dpa-checkpoint "$DPA" --imagenet-checkpoint "$IMAGENET" \
  --proposal-cache "$OUT/proposal_cache" --output-dir "$OUT" \
  --epochs 10 --batch-size 192 --workers 2 --seed 20261028
# 仅在权重和验证参数锁定后执行；读取OUT中的best.pth与frozen_calibration.json。
$PY scripts/refine_hzy_intrinsic_cells.py --mode evaluate \
  --source-root "$ROOT/vendor/ihc-main" --dataset "$DATA" \
  --dpa-checkpoint "$DPA" --imagenet-checkpoint "$IMAGENET" \
  --proposal-cache "$OUT/proposal_cache" --output-dir "$OUT" --batch-size 192
```

这一步完成的是肾小球ROI内细胞模块优化，尚未重新接入整张切片推理及病理评分，也未开展新增独立病例验证。后续优先固定本版本，对新病例做完整切片验证，并检查系膜过检、密集细胞漏检和三类混淆；新增标注应先明确是否为ROI内穷尽标注，再用于难负样本训练。

完成带参考标注的测试后，可运行以下命令生成逐点核对图。脚本会按每图总数绝对误差选取最小、中位和最大实例，显示正确检出、类别错误、误检和漏检。

```bash
python scripts/visualize_hzy_intrinsic_cell_matches.py --dataset data/hzy_intrinsic_cell_upgrade_43_20261021 --results reports/hzy_intrinsic_refinement_20261028 --output reports/hzy_intrinsic_refinement_20261028/matched_visualizations
```

病例数据、逐病例结果、真实病理图像及模型权重通过项目内部渠道获取。
