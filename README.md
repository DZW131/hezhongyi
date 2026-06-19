# 2026-05-11 院内 HSPN 标注数据继续训练日志

本次更新新增了院内紫癜性肾炎（HSPN）标注数据接入流程，目标是在原 HuBMAP 肾小球分割基线模型基础上，继续使用医生标注的院内数据进行微调训练。医生提供的数据不是现成的 `image + mask` 结构，而是每例包含 `.czi` 原始扫描文件和 `slice.db` 标注数据库，因此本次补充了两个脚本：

- `scripts/export_hzy_slice_db_annotations.py`
  - 从每个 `slice.db` 中读取医生 polygon 标注，导出为 `prepare_hubmap_tiles.py` 可读取的 GeoJSON 标注。
- `scripts/convert_hzy_czi_to_tiff.py`
  - 按导出的 manifest 将 `.czi` 原始图转换为与标注同名的 `.tiff` 图像。

本地检查当前数据集时，`label.png` / `slice_label.jpg` 被确认只是医院玻片标签或标签框预览，不是训练 mask；真正可用于训练的医生标注位于 `slice.db` 中。当前 `slice.db` 中已识别到的规范标注类别包括：

```text
未废弃肾小球
废弃肾小球
毛细血管内细胞增生
肾小球系膜细胞增生
细胞性新月体
纤维细胞性新月体
纤维性新月体
节段硬化
节段球囊粘连
纤维素样坏死
纤维素性血栓
```

本地脚本验证结果：共检测到 `52` 个 `slice.db`，成功导出 `52` 例标注 JSON，共 `1946` 个规范 polygon 标注。其中主要类别数量如下：

```text
未废弃肾小球：1287
废弃肾小球：33
肾小球系膜细胞增生：300
毛细血管内细胞增生：162
细胞性新月体：51
纤维细胞性新月体：42
纤维性新月体：20
节段硬化：30
节段球囊粘连：16
纤维素样坏死：4
纤维素性血栓：1
```

## A. 服务器数据放置

上传医生原始数据时，建议保持原始层级不变：

```text
/root/datasets/HZY_HSPN_raw/
  data/
    2026_03_10_11_41_01_153748/
      slices/
        202603101139282630/
          2026001.czi
          slice.db
          label.png
          slice_label.jpg
          thumbnail.jpeg
```

上传后先检查数量：

```bash
find /root/datasets/HZY_HSPN_raw -name "*.czi" | wc -l
find /root/datasets/HZY_HSPN_raw -name "slice.db" | wc -l
```

当前本地数据预期约为：

```text
52 个 .czi
52 个 slice.db
```

## B. 安装环境与可选 CZI 读取依赖

进入服务器项目目录：

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master
conda activate unet_kidney
pip install -r requirements.txt
```

`.czi` 读取依赖不是原 HuBMAP 流程的必需项，因此没有强制写入 `requirements.txt`。如果服务器无法直接读取 CZI，优先安装：

```bash
pip install aicspylibczi
```

如果 `aicspylibczi` 安装失败，也可以尝试：

```bash
pip install czifile
```

如果服务器环境无法使用 `aicspylibczi` 裁剪 scene，可使用扫描仪软件或其他病理图像工具按 scene 导出 TIFF，再放入 `/root/datasets/HZY_HSPN_export_ds025/images`，文件名需要与 manifest 中的 `slide_id` 对齐。

## C. 从 slice.db 导出医生标注

先将医生标注数据库转换为 GeoJSON。当前已确认医生标注坐标是 CZI mosaic 全局坐标，因此推荐直接按 scene 拆分导出，并同步生成 `0.25` 降采样坐标：

```bash
python scripts/export_hzy_slice_db_annotations.py \
  --raw-root /root/datasets/HZY_HSPN_raw \
  --output-dir /root/datasets/HZY_HSPN_export_ds025 \
  --split-scenes \
  --coordinate-scale 0.25 \
  --overwrite
```

该命令会生成：

```text
/root/datasets/HZY_HSPN_export_ds025/
  annotations/
    2026001_s0.json
    2026002_s0.json
    2026002_s1.json
    ...
  hzy_hspn_manifest.csv
  hzy_hspn_annotation_summary.json
```

其中：

- `annotations/*.json`：scene 内坐标系下的医生 polygon 标注，供后续切 patch 使用。
- `hzy_hspn_manifest.csv`：每个 scene 的 `slide_id`、`.czi` 路径、`slice.db` 路径、标注 JSON 路径和 scene region。
- `hzy_hspn_annotation_summary.json`：标注数量和类别统计。

检查导出结果：

```bash
cat /root/datasets/HZY_HSPN_export_ds025/hzy_hspn_annotation_summary.json
head -n 5 /root/datasets/HZY_HSPN_export_ds025/hzy_hspn_manifest.csv
ls /root/datasets/HZY_HSPN_export_ds025/annotations | head
```

默认只导出 `Mark_label_None` 表，因为本批数据中该表包含规范病理类别；`Mark_human` 中多为未分组的黄色人工痕迹，默认不纳入训练标注。如果后续确认 `Mark_human` 中有需要纳入的最终修订标注，可手动增加参数：

```bash
python scripts/export_hzy_slice_db_annotations.py \
  --raw-root /root/datasets/HZY_HSPN_raw \
  --output-dir /root/datasets/HZY_HSPN_export_ds025 \
  --mark-tables Mark_label_None Mark_human \
  --split-scenes \
  --coordinate-scale 0.25 \
  --overwrite
```

## D. 将 CZI 转为 TIFF

根据 manifest 将 `.czi` 转为可训练 `.tiff`。manifest 中的每一行现在对应一个 scene，转换脚本会读取该行的 `scene_x / scene_y / scene_width / scene_height`，只导出该 scene 区域，避免整张 mosaic 的黑色空区混入训练。

如果之前已经生成过未降采样的大 TIFF，建议删除后重新生成：

```bash
rm -rf /root/datasets/HZY_HSPN_export_ds025/images
```

建议先试转前 2 个 scene 确认读取正常：

```bash
python scripts/convert_hzy_czi_to_tiff.py \
  --manifest-csv /root/datasets/HZY_HSPN_export_ds025/hzy_hspn_manifest.csv \
  --output-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --downsample 0.25 \
  --isolate \
  --limit 2 \
  --overwrite
```

确认没有问题后再全量转换：

```bash
python scripts/convert_hzy_czi_to_tiff.py \
  --manifest-csv /root/datasets/HZY_HSPN_export_ds025/hzy_hspn_manifest.csv \
  --output-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --downsample 0.25 \
  --isolate \
  --overwrite
```

转换完成后检查：

```bash
find /root/datasets/HZY_HSPN_export_ds025/images -name "*.tiff" | wc -l
```

如果 CZI 读取失败，通常是服务器缺少可用的 CZI 解析库。处理顺序建议为：

```text
1. pip install aicspylibczi
2. scene 拆分依赖 aicspylibczi 的 read_mosaic(region=...)，czifile 不能替代 scene 裁剪
3. 若仍失败，用扫描仪软件按 scene 导出 TIFF
4. 保证 TIFF 文件名与 annotations/*.json 的 slide_id 一致，例如 2026002_s0.tiff / 2026002_s0.json
5. 如果使用了 --downsample，必须在导出标注时使用相同的 --coordinate-scale
```

如果全量转换过程中出现 `Segmentation fault (core dumped)`，通常是某一张 CZI 触发底层 CZI 读取库崩溃。请使用 `--isolate` 模式逐张子进程转换；这样单张失败不会中断整批转换，失败样本会写入：

```text
/root/datasets/HZY_HSPN_export_ds025/images/conversion_failures.csv
```

## E. 生成院内肾小球分割 tiles

当前建议先做“肾小球二分类分割微调”，即：

```text
前景：未废弃肾小球 + 废弃肾小球 + 肾小球
背景：其他组织和背景区域
```

因为本批院内数据没有 HuBMAP 那种 anatomical structure JSON，所以这里不使用 `--roi-labels Cortex`，只使用组织覆盖率和正负样本比例控制。

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --annotation-format json-polygons \
  --annotation-json-suffix .json \
  --target-labels 未废弃肾小球 废弃肾小球 肾小球 \
  --output-dir /root/datasets/HZY_HSPN_tiles_glom \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

说明：

- `--tile-size 1024 --stride 1024` 与 HuBMAP 基线保持一致。
- 训练时继续使用 `--scale 0.5`，因此模型实际输入约为 `512 x 512`。
- `--val-ratio 0.2` 表示按 slide 级别约 `8:2` 划分训练集和验证集。
- `--negative-ratio 2.0` 表示每个正样本最多保留约 2 个负样本。

生成后检查：

```bash
find /root/datasets/HZY_HSPN_tiles_glom/train/images -name "*.jpg" | wc -l
find /root/datasets/HZY_HSPN_tiles_glom/val/images -name "*.jpg" | wc -l
cat /root/datasets/HZY_HSPN_tiles_glom/manifests/summary.json
```

如果 `summary.json` 中 `positive_tiles` 为 0，或者日志提示多数 slide 没有阳性 patch，优先检查以下问题：

```text
1. CZI 转出的 TIFF 是否为全分辨率图像，而不是 thumbnail 或 label 区域。
2. TIFF 文件名是否与 annotations/*.json 的 slide_id 一致。
3. 标注 polygon 坐标是否落在 scene TIFF 图像尺寸范围内。
4. --target-labels 是否与 hzy_hspn_annotation_summary.json 中的类别名完全一致。
```

## F. 基于 HuBMAP best.pth 继续微调

建议不要从零训练，而是加载 HuBMAP 训练得到的 `best.pth` 作为预训练权重，在院内数据上低学习率微调：

```bash
python train.py \
  --load /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --images-dir /root/datasets/HZY_HSPN_tiles_glom/train/images \
  --masks-dir /root/datasets/HZY_HSPN_tiles_glom/train/masks \
  --val-images-dir /root/datasets/HZY_HSPN_tiles_glom/val/images \
  --val-masks-dir /root/datasets/HZY_HSPN_tiles_glom/val/masks \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 5e-6 \
  --scale 0.5 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --checkpoint-metric dice \
  --analysis-frequency 5 \
  --preview-frequency 5 \
  --wandb-mode disabled \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom
```

训练输出目录：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom/
  best.pth
  latest.pth
  analysis/
    history.csv
    training_curves.png
    best_preview.png
    best_metrics.json
    val_previews/
```

如果显存不足，可优先尝试：

```bash
--batch-size 1
```

如果 `torch.compile` 兼容性不好，可关闭：

```bash
--compile off
```

## G. 院内验证集评估

训练完成后，对院内验证集上的最佳权重做独立评估：

```bash
python evaluate_checkpoint.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom/best.pth \
  --images-dir /root/datasets/HZY_HSPN_tiles_glom/val/images \
  --masks-dir /root/datasets/HZY_HSPN_tiles_glom/val/masks \
  --classes 2 \
  --batch-size 2 \
  --num-workers 8 \
  --scale 0.5 \
  --amp \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom/eval_best
```

输出内容：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom/eval_best/
  metrics.json
  preview.png
```

重点查看：

```text
Dice
IoU
Precision
Recall
Specificity
Accuracy
```

其中 Dice、IoU、Precision、Recall 更适合汇报；Accuracy 容易受大量背景像素影响，只作为辅助参考。

## H. 使用微调后的模型做院内大图推理

如果需要对转换后的院内 TIFF 做整图推理，可先选择一张图测试。当前院内模型推荐使用 `predict_tiff.py`，并在整图 mask 拼接完成后启用后处理：

```bash
mkdir -p /root/Pytorch-UNet/Pytorch-UNet-master/predictions/hzy_scene_ds025

python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --input /root/datasets/HZY_HSPN_export_ds025/images/2026001_s0.tiff \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/hzy_scene_ds025/2026001_s0_pred_mask_post.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.5 \
  --postprocess \
  --fill-holes \
  --hole-repair-radius 10 \
  --smooth-radius 2 \
  --min-component-area 800
```

这组后处理只做三件事：删除很小的噪声连通域、填补肾小球内部空洞、轻度平滑边界。当前 baseline 推理没有明显大块误检，因此不再使用最大面积、最大宽高、组织 mask 或粘连拆分之类的额外过滤。

后处理参数含义：

```text
--fill-holes：填补肾小球 mask 内部空洞。
--hole-repair-radius：填洞前临时闭合小缺口，用来修补和外界有窄开口相连的“半开洞”；推荐从 6、10、14 依次试。
--smooth-radius：边界平滑半径，推荐从 1 或 2 开始；过大可能改变肾小球轮廓。
--min-component-area：删除小碎片噪声；若仍有小白点或小短条残留，可从 800 提到 1000 或 1500。
```

生成“原图 + 医生标注 + 模型预测”的叠加图：

```bash
python scripts/render_hzy_gt_pred_overlay.py \
  --slide-ids 2026001_s0 \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --predictions-dir /root/Pytorch-UNet/Pytorch-UNet-master/predictions/hzy_scene_ds025 \
  --prediction-suffix _pred_mask_post.png \
  --output-suffix _overlay_gt_pred_post.jpg \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/predictions/hzy_scene_ds025 \
  --preview-max-size 8000
```

如果 `predictions-dir` 中已有多个 `*_pred_mask.png`，也可以不传 `--slide-ids`，脚本会自动批量生成对应 overlay。叠加图中红色轮廓是医生标注，蓝色半透明区域是模型预测。

## I. 院内病变分割训练

本次微调优先解决“院内肾小球分割”问题。基础分割稳定后，可利用同一批 `slice.db` 中的病变标注继续开展病变分割。当前脚本已经能导出以下病变 polygon：

```text
废弃肾小球
肾小球系膜细胞增生
毛细血管内细胞增生
细胞性新月体
纤维细胞性新月体
纤维性新月体
节段硬化
节段球囊粘连
纤维素样坏死
纤维素性血栓
```

这些病变不建议一上来塞进一个大的互斥多分类模型：同一个肾小球可能同时有多种病变，而且类别数量差异很大。当前推荐先按形态和临床含义拆成 3 个较小的多类别任务组，每个任务组训练一个模型；这样既保留同组内部的类别区分，也避免一个模型同时处理过多稀有标签。

在训练病变模型前，建议先生成各类别局部可视化图集，快速熟悉每类病变在图中的形态：

```bash
python scripts/render_hzy_label_gallery.py \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --output-dir /root/datasets/HZY_HSPN_export_ds025/debug_label_gallery_color \
  --max-per-label 24 \
  --pad 512
```

输出目录中会包含 `label_color_legend.jpg`、每类 `gallery_*.jpg` 以及对应的局部 crop 文件夹。不同病变类别会使用不同颜色标出。

如果图例中的中文显示为乱码或方块，请先安装中文字体后重新运行：

```bash
apt-get update
apt-get install -y fonts-noto-cjk fonts-wqy-microhei
fc-cache -fv
```

也可以显式指定字体路径：

```bash
python scripts/render_hzy_label_gallery.py \
  --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

正式生成 tile 前，先统计一下三组任务的数据分布：

```bash
python scripts/analyze_hzy_lesion_tasks.py \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --output-dir /root/datasets/HZY_HSPN_lesion_tasks/analysis
```

输出中重点看：

```text
/root/datasets/HZY_HSPN_lesion_tasks/analysis/task_distribution.csv
/root/datasets/HZY_HSPN_lesion_tasks/analysis/label_distribution.csv
```

批量生成 3 个病变任务组的多类别 tile 数据集：

```bash
python scripts/prepare_hzy_lesion_tiles.py \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --output-root /root/datasets/HZY_HSPN_lesion_tasks \
  --tile-size 512 \
  --stride 512 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 16 \
  --negative-ratio 3.0
```

默认会生成 3 个任务组：

```text
proliferation: 背景 + 肾小球系膜细胞增生 + 毛细血管内细胞增生，--classes 3
crescent:      背景 + 细胞性新月体 + 纤维细胞性新月体 + 纤维性新月体，--classes 4
other_lesions: 背景 + 节段硬化 + 节段球囊粘连 + 纤维素样坏死 + 纤维素性血栓，--classes 5
```

其中 `other_lesions` 的 `纤维素样坏死` 只有 4 个标注、`纤维素性血栓` 只有 1 个标注，可以训练流程上先跑通，但指标和泛化能力需要谨慎解释。

批量训练这 3 个病变任务组模型：

```bash
python scripts/train_hzy_lesion_task_models.py \
  --tiles-root /root/datasets/HZY_HSPN_lesion_tasks \
  --checkpoint-root /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks \
  --base-checkpoint /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --scale 1.0 \
  --amp \
  --optimizer adamw \
  --wandb-mode disabled
```

脚本会自动根据任务组设置 `--classes`，并通过 `train.py --load-partial` 只加载与当前模型 shape 匹配的权重，因此可以从 2 类肾小球模型初始化 3/4/5 类病变模型。

也可以只准备和训练某一个任务组，例如只训练新月体：

```bash
python scripts/prepare_hzy_lesion_tiles.py \
  --tasks crescent \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --output-root /root/datasets/HZY_HSPN_lesion_tasks \
  --tile-size 512 \
  --stride 512 \
  --overwrite

python scripts/train_hzy_lesion_task_models.py \
  --tasks crescent \
  --tiles-root /root/datasets/HZY_HSPN_lesion_tasks \
  --checkpoint-root /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks \
  --base-checkpoint /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --scale 1.0 \
  --amp \
  --optimizer adamw \
  --wandb-mode disabled
```

训练完成后，每个任务组会有独立 checkpoint，例如：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks/proliferation/best.pth
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks/crescent/best.pth
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks/other_lesions/best.pth
```

下一步量化统计时，再把“肾小球实例分割结果”和“各任务组模型预测结果”进行空间匹配，得到单个肾小球是否存在新月体、节段硬化、系膜增生等指标。

### I.1 推荐：基于单个肾小球 crop 的病变分割

增生类病变面积很小，直接在整张 scene 的滑窗 tile 里训练时，前景像素容易被大量背景稀释。更推荐先用医生肾小球标注裁出单个肾小球局部 crop，再在 crop 内生成病变 mask 训练：

```bash
python scripts/prepare_hzy_glomerulus_lesion_crops.py \
  --tasks proliferation crescent \
  --images-dir /root/datasets/HZY_HSPN_export_ds025/images \
  --annotations-dir /root/datasets/HZY_HSPN_export_ds025/annotations \
  --output-root /root/datasets/HZY_HSPN_glomerulus_lesion_tasks \
  --crop-size 512 \
  --margin 96 \
  --min-source-crop-size 384 \
  --val-ratio 0.2 \
  --min-positive-pixels 8 \
  --negative-ratio 1.0 \
  --overwrite
```

输出结构仍与 `train_hzy_lesion_task_models.py` 兼容：

```text
/root/datasets/HZY_HSPN_glomerulus_lesion_tasks/proliferation/train/images
/root/datasets/HZY_HSPN_glomerulus_lesion_tasks/proliferation/train/masks
/root/datasets/HZY_HSPN_glomerulus_lesion_tasks/crescent/train/images
/root/datasets/HZY_HSPN_glomerulus_lesion_tasks/crescent/train/masks
```

训练时只需要把 `--tiles-root` 指到新的 crop 数据集：

```bash
python scripts/train_hzy_lesion_task_models.py \
  --tasks proliferation \
  --tiles-root /root/datasets/HZY_HSPN_glomerulus_lesion_tasks \
  --checkpoint-root /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_glomerulus_lesion_tasks \
  --base-checkpoint /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --scale 1.0 \
  --amp \
  --optimizer adamw \
  --wandb-mode disabled
```

训练阶段使用医生肾小球标注裁 crop；推理和量化阶段再使用肾小球模型预测结果生成实例 crop，并将病变预测映射回原图做单肾小球统计。

### I.2 新实验选项：二分类新月体、数据增强和稀疏分割 loss

当前病变分割支持 4 类 loss 配方之外的稀疏目标训练选项：

```text
--augmentation basic|strong        训练集增强，验证集不增强
--loss-mode focal_dice             Focal CE + Dice
--loss-mode tversky                CE + Tversky
--loss-mode focal_tversky          Focal CE + Focal Tversky
--loss-mode generalized_dice       Generalized Dice
--loss-mode ce_generalized_dice    CE + Generalized Dice
```

数据增强包含翻转、90 度旋转、小幅仿射变换、亮度/对比度/Gamma/颜色扰动。`basic` 适合先做稳定 baseline；`strong` 用于后续更激进实验。

新月体可以先做二分类任务，把细胞性、纤维细胞性、纤维性新月体统一映射为 class 1：

```bash
cd /root/workspace/hezhongyi-unet_segm
conda activate unet_kidney
DATA=/root/rivermind-data/dataspace

python scripts/prepare_hzy_glomerulus_lesion_crops.py \
  --tasks crescent_binary \
  --images-dir $DATA/HZY_HSPN_export_ds025/images \
  --annotations-dir $DATA/HZY_HSPN_export_ds025/annotations \
  --output-root $DATA/HZY_HSPN_glomerulus_lesion_tasks_crescent_binary \
  --crop-size 512 \
  --margin 64 \
  --min-source-crop-size 384 \
  --val-ratio 0.2 \
  --min-positive-pixels 8 \
  --negative-ratio 0.5 \
  --max-background-crops-per-slide 6 \
  --overwrite
```

推荐先用 Focal CE + Tversky 跑二分类新月体：

```bash
mkdir -p logs

PYTHONUNBUFFERED=1 nohup python scripts/train_hzy_lesion_task_models.py \
  --tasks crescent_binary \
  --tiles-root $DATA/HZY_HSPN_glomerulus_lesion_tasks_crescent_binary \
  --checkpoint-root $DATA/checkpoints/hzy_hspn_glomerulus_lesion_tasks_crescent_binary \
  --base-checkpoint $DATA/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --scale 1.0 \
  --amp \
  --optimizer adamw \
  --background-weight 0.10 \
  --foreground-weight 1.0 \
  --foreground-dice-only \
  --loss-mode focal_tversky \
  --focal-gamma 2.0 \
  --tversky-alpha 0.3 \
  --tversky-beta 0.7 \
  --tversky-gamma 1.33 \
  --augmentation basic \
  --wandb-mode disabled \
  > logs/train_hzy_glom_crop_crescent_binary.log 2>&1 &
```

增生类病变仍建议优先提高分辨率并使用 Focal Tversky：

```bash
python scripts/prepare_hzy_glomerulus_lesion_crops.py \
  --tasks proliferation \
  --images-dir $DATA/HZY_HSPN_export_ds025/images \
  --annotations-dir $DATA/HZY_HSPN_export_ds025/annotations \
  --output-root $DATA/HZY_HSPN_glomerulus_lesion_tasks_proliferation_768 \
  --crop-size 768 \
  --margin 24 \
  --min-source-crop-size 256 \
  --val-ratio 0.2 \
  --min-positive-pixels 4 \
  --negative-ratio 0.25 \
  --max-background-crops-per-slide 2 \
  --overwrite

PYTHONUNBUFFERED=1 nohup python scripts/train_hzy_lesion_task_models.py \
  --tasks proliferation \
  --tiles-root $DATA/HZY_HSPN_glomerulus_lesion_tasks_proliferation_768 \
  --checkpoint-root $DATA/checkpoints/hzy_hspn_glomerulus_lesion_tasks_proliferation_768 \
  --base-checkpoint $DATA/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --scale 1.0 \
  --amp \
  --optimizer adamw \
  --background-weight 0.03 \
  --foreground-weight 1.5 \
  --foreground-dice-only \
  --loss-mode focal_tversky \
  --focal-gamma 2.0 \
  --tversky-alpha 0.25 \
  --tversky-beta 0.75 \
  --tversky-gamma 1.33 \
  --augmentation basic \
  --wandb-mode disabled \
  > logs/train_hzy_glom_crop_proliferation_768.log 2>&1 &
```

训练日志中应确认：

```text
loss_mode: focal_tversky
augmentation: basic
foreground dice: True
```

### I.3 新月体 detection-first 框检测流程

当分割模型已经能召回新月体，但分割转框出现过多假阳性框时，可以复用增生类病变的 detection-first 流程：先把 `crescent_binary` mask 转为 bbox，再训练 Faster R-CNN 做“有没有、在哪里”的检测主线。分割模型继续用于框内形态和面积辅助评估。

先生成新月体 box 数据集：

```bash
python scripts/prepare_hzy_detection_boxes.py \
  --dataset-root $DATA/HZY_HSPN_glomerulus_lesion_tasks_crescent_binary/crescent_binary \
  --output-root $DATA/HZY_HSPN_detection_boxes/crescent_binary \
  --class-mode binary \
  --class-name crescent \
  --min-area 16 \
  --box-margin 4 \
  --link-mode symlink \
  --overwrite
```

再训练 balanced Faster R-CNN。`--ensure-positive-batches` 会过采样阳性样本，保证每个训练 batch 至少包含一个阳性框，适合新月体这种阳性少、阴性多的数据分布：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_hzy_detection_boxes.py \
  --data-root $DATA/HZY_HSPN_detection_boxes/crescent_binary \
  --output-dir $DATA/checkpoints/hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_balanced \
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

如果需要调试训练不稳定，可加 `--fail-on-nonfinite-loss`；默认行为是遇到 NaN/Inf loss 时跳过该 batch，并在 `history.csv` 中记录 `nonfinite_batches`。

### I.4 增生类病变 detection-first 当前推荐

增生类病变也可以走 detection-first 路线，把“肾小球系膜细胞增生 + 毛细血管内细胞增生”合并为一个 `proliferation_binary` 前景类别。当前最稳的结果仍来自 COCO 预训练 Faster R-CNN 优化版：

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

当前使用建议：

- 少漏检筛查：使用 `best.pth`，阈值 `0.2`。
- 展示/定位更干净：使用 `latest.pth`，阈值 `0.4/0.5`。
- 已验证但不推荐作为主结果：随机增加阴性样本、直接套用 `--ensure-positive-batches`。增生任务更需要模型误报驱动的 hard-negative mining，或检测候选框后的二阶段分类器。

### I.5 新月体子类三分类 ROI classifier

新月体细分类不建议直接回到三分类像素分割作为主线。当前更稳的做法是先用新月体大类检测/分割确定候选区域，再对候选 ROI 做子类分类：

- 细胞性新月体
- 纤维细胞性新月体
- 纤维性新月体

先准备三类新月体 glomerulus crop：

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

再从三类 mask 中抽取 lesion ROI 分类数据：

```bash
python scripts/prepare_hzy_crescent_subtype_rois.py \
  --dataset-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_crescent_subtype_20260619/crescent \
  --output-root /home/duyanhong/Dataspace/HZY/HZY_HSPN_crescent_subtype_rois_20260619 \
  --min-area 32 \
  --box-margin 48 \
  --crop-size 224 \
  --overwrite
```

最后训练 ResNet18 三分类器：

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

5090 探路结果：

| 数据 | 训练 ROI | 验证 ROI | Best epoch | Accuracy | Balanced Accuracy | Macro F1 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `HZY_HSPN_crescent_subtype_rois_20260619` | 116 | 21 | 21 | 0.8095 | 0.8116 | 0.8026 | 验证集很小，仅作为 ROI 三分类路线可行性基线 |

各类 F1：细胞性 `0.8235`，纤维细胞性 `0.7273`，纤维性 `0.8571`。混淆矩阵和预测明细见：

- `/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_confusion_matrix.csv`
- `/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_crescent_subtype_classifier_resnet18_20260619/analysis/best_predictions.csv`

---

# HuBMAP 肾小球分割工程说明

该仓库是在 PyTorch U-Net 基础上，针对 HuBMAP 肾脏病理数据改造的肾小球分割工程化版本，面向 Linux 服务器训练与推理流程。

如果需要一份更适合向老师、医生或课题组做汇报的文档，请参阅 [PROJECT_PRESENTATION_GUIDE.md](PROJECT_PRESENTATION_GUIDE.md)。

当前工作流已适配如下服务器数据布局：

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

使用该项目时，无需手工重命名，也无需预先将 HuBMAP 原始数据转换为 mask 图。

当前仓库已经支持：

- 原始 `TIFF + polygon JSON` 标注输入
- 基于 `*-anatomical-structure.json` 的解剖 ROI 过滤
- 训练/验证 patch 自动生成
- 更完整的训练评估指标
- `best.pth` / `latest.pth` 权重管理
- 本地训练曲线与验证可视化输出
- 独立 checkpoint 评估脚本
- 大图 TIFF 推理与院内图推理脚本
- 面向 4090 服务器的训练提速选项

## 0. 快速开始

如果只希望用最短路径完成一次 HuBMAP 训练闭环，可按以下顺序执行：

1. 将服务器仓库更新到 `unet_segm` 分支
2. 从 `/root/datasets/HuBMAP/train` 生成训练 patch
3. 使用 `--scale 0.5` 训练模型
4. 评估 `best.pth`
5. 对新 slide 先使用 `predict_tiff.py` 做直接基线推理

## 0.1 当前已验证基线

该项目已经在服务器上完成过一轮有效训练与验证，结果如下：

- tiles 目录：`/root/datasets/HuBMAP_tiles_v2`
- checkpoint 目录：`/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2`
- 最优 epoch：`13`
- 最优验证 Dice：`0.9292`
- 验证 IoU：`0.8678`
- 验证 Precision：`0.9291`
- 验证 Recall：`0.9294`
- 验证 Specificity：`0.9966`
- 验证 Accuracy：`0.9935`

对应最佳权重为：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth
```

如果需要直接复现当前已验证基线，建议沿用 `HuBMAP_tiles_v2` 和 `hubmap_unet_run2` 这组目录命名。

## 1. 仓库结构

```text
.
|-- train.py
|-- evaluate.py
|-- evaluate_checkpoint.py
|-- predict.py
|-- predict_tiff.py
|-- predict_hspn_enhanced.py
|-- predict_hspn_stain_norm.py
|-- requirements.txt
|-- scripts
|   |-- prepare_hubmap_tiles.py
|   |-- plot_training_history.py
|   |-- download_data.sh
|   `-- download_data.bat
|-- unet
|   |-- unet_model.py
|   `-- unet_parts.py
`-- utils
    |-- checkpoint_io.py
    |-- data_loading.py
    |-- dice_score.py
    |-- inference_postprocessing.py
    |-- segmentation_metrics.py
    |-- visualization.py
    `-- utils.py
```

## 2. HuBMAP 文件含义

对每张 slide，项目默认读取以下三类文件：

- `slide_id.tiff`
  - 原始全视野病理图像
- `slide_id.json`
  - 肾小球 polygon 标注
  - 脚本默认寻找 `properties.classification.name == "glomerulus"`
- `slide_id-anatomical-structure.json`
  - 解剖结构 polygon 标注
  - 当前数据中已确认包含 `Cortex`、`Medulla`

因此，该项目的数据处理流程是：

1. 读取 TIFF
2. 将肾小球 polygon 栅格化为二值 mask
3. 可选读取解剖 ROI，例如 `Cortex`
4. 将 WSI 切成可训练 patch
5. 基于 patch 数据训练 U-Net

## 3. 推荐服务器环境

推荐环境：

- 系统：Linux
- GPU：NVIDIA RTX 4090
- Python：3.10+
- PyTorch：2.x
- CUDA：与服务器驱动匹配

注意事项：

- `requirements.txt` 不负责安装 `torch`，应先安装与服务器 CUDA 匹配的 PyTorch 版本
- `wandb` 为可选依赖，即使未登录也可正常训练

示例环境配置：

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master

conda activate unet_kidney

# 先根据服务器 CUDA 环境安装合适的 torch
# 再安装本项目依赖
pip install -r requirements.txt
```

## 4. 更新服务器旧项目

如果服务器仍保留旧版仓库，可用以下命令更新到当前版本：

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master
git fetch origin
git checkout unet_segm
git pull origin unet_segm
```

更新后，不再依赖仓库内旧式的 `data/imgs`、`data/masks` 流程，原始 HuBMAP 数据可继续保持在 `/root/datasets/HuBMAP` 下不变。

## 5. 数据准备

### 5.1 原始输入目录

原始数据保持在已有位置：

```text
/root/datasets/HuBMAP/train
```

### 5.2 推荐输出目录

建议将切片结果输出到：

```text
/root/datasets/HuBMAP_tiles
```

脚本会自动生成如下结构：

```text
/root/datasets/HuBMAP_tiles/
|-- train
|   |-- images
|   `-- masks
|-- val
|   |-- images
|   `-- masks
`-- manifests
    |-- tiles.csv
    |-- slides.csv
    `-- summary.json
```

### 5.3 主脚本

```bash
python scripts/prepare_hubmap_tiles.py -h
```

脚本完成的工作包括：

- 读取每张 TIFF
- 读取对应肾小球 polygon JSON
- 栅格化为二值 mask
- 可选读取 anatomical JSON 并仅保留指定 ROI，例如 `Cortex`
- 滑窗切 patch
- 过滤大面积空白背景
- 控制正负样本比例
- 按 slide 级别划分 train / val
- 保存 manifest 便于复现

### 5.4 当前服务器推荐切片命令

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

### 5.5 重要切片参数

- `--tile-size`：切片尺寸
- `--stride`：滑窗步长
- `--downsample`：切片后额外缩放
- `--val-ratio`：按 slide 划分的验证集比例
- `--target-labels`：视为阳性的标注类别，默认 `glomerulus`
- `--roi-labels`：保留的 anatomical ROI 标签，例如 `Cortex`
- `--min-roi-coverage`：patch 需要满足的最小 ROI 覆盖比例
- `--missing-roi-policy`：ROI 缺失时的处理策略，可选 `skip-slide`、`ignore-roi`、`error`
- `--min-tissue-coverage`：过滤接近纯白背景的 patch
- `--min-positive-pixels`：强制判定正样本所需的最少阳性像素
- `--negative-ratio`：每个正样本保留多少个负样本

### 5.6 如不使用 Cortex 过滤

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --tile-size 1024 \
  --stride 1024
```

## 6. 模型训练

### 6.1 推荐 checkpoint 目录

建议输出到：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

### 6.2 推荐训练命令

```bash
python train.py \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --scale 0.5 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --checkpoint-metric dice \
  --analysis-frequency 5 \
  --preview-frequency 5 \
  --wandb-mode disabled \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

### 6.3 训练中记录的指标

训练阶段会记录：

- validation loss
- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- epoch 用时
- 训练吞吐量（images / second）

### 6.4 训练输出

训练完成后通常会生成：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/
|-- best.pth
|-- latest.pth
|-- epoch_001.pth                 # 仅在 --save-every-epoch 时生成
`-- analysis
    |-- history.csv
    |-- training_curves.png
    |-- best_preview.png
    |-- best_metrics.json
    `-- val_previews
```

### 6.5 已集成的训练提速项

训练脚本已集成以下提速策略：

- AMP 混合精度
- CUDA 下默认启用 TF32
- 默认启用 cuDNN benchmark
- non-blocking 数据拷贝
- DataLoader `persistent_workers`
- DataLoader `prefetch_factor`
- 可选 `torch.compile`
- mask 值缓存
- 可调的验证频率与可视化频率

### 6.6 继续训练

```bash
python train.py \
  --load /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/latest.pth \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --scale 0.5 \
  --classes 2 \
  --amp \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

## 7. 独立评估

如果需要对某个已保存权重单独评估：

```bash
python evaluate_checkpoint.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --images-dir /root/datasets/HuBMAP_tiles/val/images \
  --masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --classes 2 \
  --batch-size 2 \
  --num-workers 8 \
  --scale 0.5 \
  --amp \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/eval_best
```

输出内容包括：

- `metrics.json`
- `preview.png`

## 8. 可视化

### 8.1 训练时自动生成

训练时会自动生成：

- `analysis/history.csv`
- `analysis/training_curves.png`
- `analysis/best_preview.png`
- `analysis/val_previews/*.png`

### 8.2 重新绘制训练曲线

```bash
python scripts/plot_training_history.py \
  --history-csv /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/analysis/history.csv \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/analysis/training_curves_regenerated.png
```

## 9. 推理

### 9.1 普通图片推理

```bash
python predict.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input demo.jpg \
  --output demo_mask.png \
  --classes 2
```

### 9.2 HuBMAP 大图 TIFF 推理

```bash
python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/HuBMAP/test/2ec3f1bb9.tiff \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/output_2ec3f1bb9.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2
```

推理阶段应尽量保持与训练时一致的 `--scale`。当前已验证基线使用的是 `0.5`。

### 9.3 院内图增强版推理

```bash
python predict_hspn_enhanced.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/hspn_enhanced_pred.png \
  --classes 2 \
  --scale 0.5 \
  --threshold 0.7 \
  --tile-size 1024
```

### 9.4 院内图染色归一化推理

```bash
python predict_hspn_stain_norm.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/hspn_stain_norm_pred.png \
  --classes 2 \
  --scale 0.5 \
  --threshold 0.5 \
  --tile-size 1024
```

### 9.5 院内图推荐测试策略

院内图推理时，不建议默认认为 HSPN 变体一定优于直接推理。根据当前测试情况：

- `predict_tiff.py` 应作为首选基线
- `predict_hspn_enhanced.py` 可能提升召回，但也更容易过分割
- `predict_hspn_stain_norm.py` 在染色差异明显时可尝试，但仍需与 direct 基线对照

推荐顺序：

1. 先用 direct 方式推理，保持与训练一致的 `--scale`
2. 若 direct 结果过于保守，再测试 `predict_hspn_enhanced.py`
3. 若染色风格与 HuBMAP 差异明显，再补充测试 `predict_hspn_stain_norm.py`
4. 最终应以医学上合理的结果为准，而不是单纯选择前景面积更大的 mask

当前这套基线的推荐设置：

- 始终使用 `--classes 2`
- 始终使用 `--scale 0.5`
- `predict_tiff.py` 推荐从 `--threshold 0.5` 开始
- `predict_hspn_enhanced.py` 推荐从 `--threshold 0.7` 开始
- `predict_hspn_stain_norm.py` 推荐从 `--threshold 0.5` 开始
- 若 `enhanced` 前景污染过多，可继续比较 `0.6`、`0.7`、`0.8` 甚至更高阈值
- 若出现小碎片、肾小球内部空洞或边界毛糙，可启用简化后处理

HSPN 脚本还支持以下简化后处理参数：

- `--fill-holes`
- `--hole-repair-radius`
- `--smooth-radius`
- `--min-component-area`

### 9.6 当前院内图示例命令

对于当前院内图 `/root/datasets/diyingjia/202601260012.tif`，可按以下方式测试：

```bash
# direct 基线
python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/202601260012_direct_v2.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.5

# 增强版 HSPN 推理
python predict_hspn_enhanced.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/202601260012_hspn_enhanced_t07.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.7

# 染色归一化 HSPN 推理
python predict_hspn_stain_norm.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/202601260012_hspn_stainnorm_t05.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.5
```

如果 HSPN 结果出现小碎片、内部空洞或边界毛糙，可尝试简化后处理版：

```bash
# 增强版 + 简化后处理
python predict_hspn_enhanced.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/202601260012_hspn_enhanced_filtered.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.7 \
  --fill-holes \
  --hole-repair-radius 10 \
  --smooth-radius 2 \
  --min-component-area 800

# 染色归一化版 + 简化后处理
python predict_hspn_stain_norm.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet_run2/best.pth \
  --input /root/datasets/diyingjia/202601260012.tif \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/predictions/202601260012_hspn_stainnorm_filtered.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2 \
  --threshold 0.5 \
  --fill-holes \
  --hole-repair-radius 10 \
  --smooth-radius 2 \
  --min-component-area 800
```

以上参数是当前院内图测试中的经验起点，不是所有病例的固定常数，最终仍需结合可视化结果进行判断。

## 10. 当前服务器的一条完整流程

### 第一步：更新代码

```bash
cd /root/Pytorch-UNet/Pytorch-UNet-master
git fetch origin
git checkout unet_segm
git pull origin unet_segm
```

### 第二步：安装依赖

```bash
conda activate unet_kidney
pip install -r requirements.txt
```

### 第三步：准备 HuBMAP tiles

```bash
python scripts/prepare_hubmap_tiles.py \
  --images-dir /root/datasets/HuBMAP/train \
  --output-dir /root/datasets/HuBMAP_tiles \
  --roi-labels Cortex \
  --tile-size 1024 \
  --stride 1024 \
  --val-ratio 0.2 \
  --min-tissue-coverage 0.05 \
  --min-positive-pixels 64 \
  --negative-ratio 2.0
```

### 第四步：训练

```bash
python train.py \
  --images-dir /root/datasets/HuBMAP_tiles/train/images \
  --masks-dir /root/datasets/HuBMAP_tiles/train/masks \
  --val-images-dir /root/datasets/HuBMAP_tiles/val/images \
  --val-masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --scale 0.5 \
  --classes 2 \
  --amp \
  --optimizer adamw \
  --num-workers 16 \
  --prefetch-factor 4 \
  --compile auto \
  --wandb-mode disabled \
  --checkpoint-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet
```

### 第五步：评估最佳权重

```bash
python evaluate_checkpoint.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --images-dir /root/datasets/HuBMAP_tiles/val/images \
  --masks-dir /root/datasets/HuBMAP_tiles/val/masks \
  --classes 2 \
  --scale 0.5 \
  --amp \
  --output-dir /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/eval_best
```

### 第六步：推理

```bash
python predict_tiff.py \
  --model /root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hubmap_unet/best.pth \
  --input /root/datasets/HuBMAP/test/2ec3f1bb9.tiff \
  --output /root/Pytorch-UNet/Pytorch-UNet-master/output_2ec3f1bb9.png \
  --tile-size 1024 \
  --scale 0.5 \
  --classes 2
```

## 11. 实用说明

### 11.1 为什么推荐 `--roi-labels Cortex`

当前 anatomical JSON 中包含 `Cortex` 与 `Medulla`。肾小球主要位于皮质区，因此启用：

```bash
--roi-labels Cortex
```

通常会带来以下收益：

- 减少无关区域
- 提升阳性样本密度
- 降低背景噪声
- 提高训练效率

### 11.2 为什么必须按 slide 划分 train / val

训练集与验证集应按 slide 划分，而不是按 patch 随机打散。否则同一张大图的相邻 patch 可能同时进入 train 和 val，导致验证结果过于乐观。

### 11.3 旧版 `data/imgs` 与 `data/masks`

旧版项目依赖：

```text
/root/Pytorch-UNet/Pytorch-UNet-master/data/imgs
/root/Pytorch-UNet/Pytorch-UNet-master/data/masks
```

当前推荐流程已不再依赖这些目录，而是直接使用外部 tile 目录，例如：

```text
/root/datasets/HuBMAP_tiles
```

### 11.4 最优权重选择

训练后会自动保留：

- `best.pth`
- `latest.pth`

因此不再需要人工从 `checkpoint_epochXX.pth` 中猜测哪一轮最好。

### 11.5 若 `torch.compile` 有兼容问题

可关闭：

```bash
--compile off
```

### 11.6 若需要每个 epoch 都保存

可增加：

```bash
--save-every-epoch
```

### 11.7 为什么 Accuracy 很高

该任务是像素级分割，前景/背景极不平衡：

- 背景像素远多于肾小球像素
- 因此 `Accuracy` 和 `Specificity` 容易显得很高

所以在该项目中，更有代表性的指标是：

- Dice
- IoU
- Precision
- Recall

当前已验证基线中，最值得汇报的主指标是验证 Dice `0.9292`。

## 12. 当前项目状态

当前仓库已经形成一套较完整的 HuBMAP 肾小球分割工程流程：

- 原始数据集保持不变
- 项目内完成 polygon 标注解析与切片
- 支持直接基于外部 tiles 训练
- 自动保存最佳/最新权重
- 提供独立评估与可视化工具
- 支持 HuBMAP 大图与院内图推理

## 13. 上游来源

该项目最初来源于 `milesial/Pytorch-UNet`，当前版本已被改造成面向 HuBMAP 肾小球分割与服务器部署流程的工程化项目。
