# Glo-In-One-v2复现与能力复用

实验日期为2026-11-11，本次同步日期为2026-11-12。详细结果见[阶段报告](reports/FINAL_REPORT.md)，测量值见[实验汇总JSON](reports/experiment_summary.json)和[通道汇总CSV](reports/channel_statistics.csv)。

## 已完成的实验

| 项目 | 结果 |
|---|---|
| 官方权重 | 安全加载、严格键匹配成功，8479114个模型参数 |
| 官方样例 | 2张，均输出512×512×15，CPU运行17.66秒 |
| 重复推理 | 2张逐字节相同 |
| 院内探索 | 7个既有病例、30图，均输出1024×1024×15，CPU运行917.86秒 |
| 背景检查 | 纯白图通道10前景98.45%，需输入质量控制与人工核对 |
| 通道语义 | task顺序0–14已确认，候选类别名称待确认 |
| 分割指标 | Dice与IoU为N/A，缺少匹配的内部结构像素标注 |

实验采用4线程CPU、FP32。GPU当时已有其他任务。CPU与GPU的数值等价性尚待实测。本次工作保留为独立探索分支，现有细胞检测与M/E报告流程继续使用既有模型。

## 目录与数据获取

scripts包含CPU兼容副本生成、环境检查、推理与可视化、报告生成。configs保存来源、权重哈希、阈值和候选通道表。reports包含源码审计、语义审计、复用判断、兼容补丁及不含病例编号的汇总统计。

真实图像、单图预测、点标注、病例清单、原始日志和权重保存在内部工作区。服务器路径为`/home/duyanhong/Workspace/glo_in_one_v2_repro`，本地工作区为`D:/work/hezhongyi/glo_in_one_v2_repro`。在这些环境中可直接查看figures和outputs。仓库提供生成这些文件的脚本。

## 在服务器重跑

以下命令在本实验目录内运行。已有环境应具备PyTorch、torchvision、numpy、scipy、pandas、matplotlib、Pillow、opencv-python、scikit-image、scikit-learn、tensorboardX。此次测量使用Python 3.10.20、PyTorch 2.11.0+cu128、numpy 1.23.5及imgaug 0.4.0。

```bash
cd experiments/glo_in_one_v2
git clone https://github.com/hrlblab/Glo-In-One_v2.git upstream
git -C upstream checkout 71a6298538d3af2c2fc2983f078e6290c66e59d3

# 用已有环境创建项目独立venv。新服务器按实际路径设置。
BASE_PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
"$BASE_PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install imgaug==0.4.0 gdown
.venv/bin/python -m gdown 'https://drive.google.com/uc?id=108uVHgrNBhn-jICiF7WG9SjpmAHJdU1W' -O model.pth
echo 'c7a042638cd44fa44533e6ab621cc19b4bf60cfcf620fda9e8e12cbd5ae7d3b5  model.pth' | sha256sum -c -

mkdir -p logs reports
.venv/bin/python scripts/prepare_cpu_compat.py
.venv/bin/python scripts/server_probe.py
HZY_GLO_DATASET=/home/duyanhong/Dataspace/HZY/processed/hzy_intrinsic_cell_upgrade_43_20261021 .venv/bin/python -u scripts/run_experiments.py
.venv/bin/python scripts/build_reports.py
```

`HZY_GLO_DATASET`指向此前43例任务的内部数据目录，需含manifest.csv、test_image和test_point。按病例轮流选取最多30图，原尺寸转存RGB PNG。独立staging目录用于写入输入CSV，数据源保持只读。完整流程会重跑demo、背景和院内试验并刷新生成的报告。

仅重跑官方样例可在准备兼容副本后运行以下命令。

```bash
cd compat_cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python Glo_v2_segmentation.py --dataset_dir ../data/official_demo --reload_path ../model.pth --output_folder ../outputs/demo_manual
```

所有通道均按独立二值掩膜保存，阈值为前景logit大于背景logit。scale固定为官方默认0，其物理倍率含义仍待核实。新病例使用前应核对染色、MPP、通道名称及标注定义。微调、特征下游评估和WSI入口尚待开展。

## 上游来源与许可

官方代码提交为`71a6298538d3af2c2fc2983f078e6290c66e59d3`。CPU兼容补丁及生成的兼容源码保留上游GPL-2.0许可，许可全文见COPYING.upstream。权重与Docker独立使用条款待核实。[官方仓库](https://github.com/hrlblab/Glo-In-One_v2)
