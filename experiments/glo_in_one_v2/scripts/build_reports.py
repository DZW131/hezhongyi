"""Build measured reproduction reports from server audit records."""
import json
from pathlib import Path
import statistics
import csv

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
REPORTS.mkdir(exist_ok=True)
read=lambda name: json.loads((ROOT/'logs'/name).read_text())
env=read('environment.json')
demo=read('demo_audit.json')
repeat=read('repeatability.json')
local=read('local_zero_shot_audit.json')
background=read('background_audit.json')
manifest=list(csv.DictReader((ROOT/'data/local_manifest.csv').open()))
names=['adhesion','capsular drop','global sclerosis','hyaline cap','hyalinosis','mesangial lysis','mesangial expansion','microaneurysm','nodular sclerosis','segmental sclerosis',"Bowman capsule",'mesangium','glomerular tuft','podocyte','mesangial cell']
config={'upstream_commit':env['source_commit']['stdout'].strip(),'weight_sha256':env['weight_sha256'],
        'source':'https://github.com/hrlblab/Glo-In-One_v2',
        'weights':'https://drive.google.com/file/d/108uVHgrNBhn-jICiF7WG9SjpmAHJdU1W/view',
        'device':'cpu','threads':4,'task_ids':list(range(15)),'scale_id':0,'threshold':'logit_foreground > logit_background',
        'channels':[{'channel':i,'task_id':i,'candidate_name':n,'semantic_status':'community_provisional','formal_evaluation_allowed':False} for i,n in enumerate(names)]}
(ROOT/'configs').mkdir(exist_ok=True)
(ROOT/'configs/run_metadata.json').write_text(json.dumps(config,indent=2))
(ROOT/'configs/channel_map_provisional.json').write_text(json.dumps(config['channels'],indent=2))
rows=[]
for group,audit in [('official_demo',demo),('background',background),('local_zero_shot',local)]:
    for i in range(15):
        fractions=[im['fraction_by_channel'][i] for im in audit['images']]
        rows.append({'cohort':group,'channel':i,'semantic_status':'unconfirmed','N_images':len(fractions),
                     'nonempty_images':sum(x>0 for x in fractions),'mean_predicted_area_fraction':statistics.mean(fractions),
                     'min_fraction':min(fractions),'max_fraction':max(fractions),'Dice':'N/A','IoU':'N/A'})
with (REPORTS/'channel_statistics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write=lambda n,t:(REPORTS/n).write_text(t,encoding='utf-8')
write('ENVIRONMENT.md',f'''# 环境记录

执行日期为2026-11-11。服务器为{env['hostname']}，Python {env['python']}，PyTorch {env['torch']}，CUDA运行时{env['cuda_runtime']}。

GPU检查时已有3个计算进程，共占用16776 MiB。实验采用CPU、4线程、FP32。GPU显存峰值记为N/A，GPU推理未执行。项目虚拟环境通过sshr5090的Python创建并继承其只读依赖，仅在项目.venv安装imgaug 0.4.0。

磁盘可用空间初始为131 GB。详细驱动、进程和磁盘记录在logs/environment.json。

官方代码提交为{config['upstream_commit']}。服务器直连GitHub发生证书校验失败，使用本地从官方HTTPS克隆的git bundle传输，同一提交已核对。上游保存在upstream，兼容副本位于compat_cpu。

官方权重为OrderedDict，{env['checkpoint_keys_count']}个键，{env['weight_bytes']}字节。SHA-256为{env['weight_sha256']}，本地和服务器一致。模型strict=True加载成功，参数量8479114。加载使用weights_only=True。

代码许可证为上游GPL-2.0。权重和Docker镜像的单独条款仍待核实。权重来源为官方README的Google Drive链接，记录见configs/run_metadata.json。
''')
write('SOURCE_CODE_AUDIT.md','''# 源码与兼容审计

真实调用链为Glo_v2_segmentation.py → MOTSDataset_2D_Patch_normal_save_csv_Glo_v2.py生成data_list.csv → testing_2D_patch_v2.py → MOTSValDataSet_normal → unet2D_ns_for_Glo_v2.UNet2D → 15次task条件前向 → 每任务比较两类logit → 按task 0到14拼接保存。

代码位置均指upstream原件。

- Glo_v2_segmentation.py末尾两个subprocess.run未检查退出码。兼容副本增加check=True，并用sys.executable保证子进程环境一致。
- CSV生成文件第128–130行将task/scale均设为0，第167行将CSV写入输入目录。实验只传入独立staging副本。
- MOTSDataset_2D_Patch_supervise_normal_csv_Glo_v2.py第503–527行使用plt.imread，取前三通道并转float32 CHW。PNG输入为RGB [0,1]，不减均值、不除标准差，无缩放、padding、裁剪或验证增强。命令行input_size=512,512未在该类实施resize。
- testing_2D_patch_v2.py第121–125行默认input_size=512,512、batch_size=4、FP16=False。外层DataLoader batch=1，内部ImagePool按4图分组，尾组走剩余分支。两个官方输入均为512方图。
- 第597–611行依次调用task 0–14。第706–716行拼接为H×W×15，前景比较为logit[1]>logit[0]。尾组同样保存15通道。各类允许重叠，未做argmax。
- 模型第158–172行编码15维task与4维scale。scale=0的物理倍率含义缺少来源确认。
- 模型中硬编码CUDA分配，CPU兼容副本只将设备分配改为CPU。未改变网络层、参数、task、scale和阈值。
- FP32单进程推理用不到Apex训练/分布式功能，兼容副本移除相应导入。删除两个数据文件中未使用的nibabel/SimpleITK导入，将TkAgg改Agg，DataLoader workers设0。完整diff见cpu_compat.patch。

requirements.txt是Linux conda导出规格。当前采用项目独立venv继承既有环境，避免直接pip安装该文件。未执行上游训练脚本。

两张demo重复推理逐字节一致。CPU与CUDA输出等价性尚未实测，必须与部署性能区别记录。
''')
table='\n'.join(f'| {i} | {n} | 已确认task={i} | 社区推测，待作者确认 |' for i,n in enumerate(names))
write('CHANNEL_AUDIT.md',f'''# 通道语义审计

已由代码和真实输出确认15个独立二分类任务，数组轴为H、W、task，task顺序0–14。正式确认的临床类别映射为0个。

| 通道 | 候选名称 | 索引证据 | 名称证据 |
|---|---|---|---|
{table}

上游train_2D_patch.py使用数字task与外部训练数据CSV，本次仓库未提供可用于逐个确认类别名称的完整标注CSV。官方示意图列有5种正常结构及病变名称，但不能证明输出索引对应关系。README的14类描述与15任务模型存在差异，hyaline cap的归属仍有歧义。

候选表来自[Issue #1社区讨论](https://github.com/hrlblab/Glo-In-One_v2/issues/1)。本轮图片仅显示channel编号，候选名称不参与正式定量评价。
''')
write('DEMO_SMOKE_TEST.md',f'''# 官方样例验证

两张512×512官方PNG成功输出。数组均为512×512×15、bool，无NaN/Inf，值域仅0与1。严格加载全部模型键。

第一次审计运行含子进程启动耗时{demo['wall_seconds']:.2f}秒。重复推理逐字节相同为{repeat['identical']}，复核图像数{repeat['images']}。

纯白512×512背景输入已执行。通道10的前景占比为98.45%，通道5为25.46%，通道8为10.81%。这说明缺少目标组织时仍可能出现大面积前景。各通道假阳性像素比例记录在channel_statistics.csv的background行，不能用其代替常规病理精度。

原始掩膜见outputs/demo，15张单通道mask与叠加图见figures/demo，命令及完整日志在logs。初次试运行日志demo.log由可复跑审计覆盖，以demo_audit.json为计时依据。
''')
write('LOCAL_EVALUATION.md',f'''# 院内零样本实验

本轮使用{len(manifest)}个肾小球图块，来自{len(set(x['patient_id'] for x in manifest))}个既有测试病例。按病例轮流选取，保留源1024×1024尺寸，JPEG解码为RGB后无损保存PNG，模型读入为[0,1]。原图、坐标和标注文件保持原样。

物种为human。染色、MPP及倍率尚未确认，local_manifest.csv明确记录unverified。30图是探索性队列，不能视为新增独立病例验证。

全部输出为1024×1024×15，CPU推理含进程启动耗时{local['wall_seconds']:.2f}秒。逐通道面积、非空图数及重叠比例是预测统计。Dice和IoU为N/A。当前GT为3类细胞点和肾小球ROI，无法据此直接评价内部区域像素分割，也无法在语义待确认时正式评价候选细胞通道。

官方Arm A保留原始图块尺寸和默认scale=0。由于输入均为方图，本轮未额外构造padding臂或染色标准化臂。图块尺寸和scale物理含义的匹配仍需核实。

全部30图保存完整掩膜及统计，前3图保存逐通道可视化，用于检视候选结构、细胞碎片与结构外溢。尚无专家修正率或修正耗时测量。
''')
write('REUSE_DECISION.md','''# 能力复用判断

| 路线 | 当前结论 | 实测依据和下一步条件 |
|---|---|---|
| 直接结构分割 | 黄色，候选能力 | 官方和院内推理成功，语义与区域GT仍缺，白背景通道10前景98.45%，需要输入质量控制和专家核对 |
| 辅助标注 | 黄色，供核对草稿 | 可逐通道展示，需核实候选系膜区/足细胞含义，再测人工修正量 |
| 迁移微调 | 黄色，待条件具备 | 官方权重strict加载，尚无对应像素GT及已确认语义，微调未启动 |
| 特征提取 | 黄色，可继续验证 | forward返回logits和x_feat，尚未跑线性探针或下游性能对照 |
| WSI集成 | 黄色，等待图块可靠性 | 本轮只验证patch入口，Docker和坐标回贴尚未实测 |

对现有项目优先评估候选系膜区作为局部计数边界、候选足细胞作为漏检提示。候选mask应先与点标注及专家核对。现有三类点检测与M/E分类尚未被本轮模型替换。官方类别候选清单缺少内皮细胞及毛细血管内增生，因此E辅助仍需既有模型及局部管腔证据。系膜增宽与系膜细胞增生也需分别核实。
''')
write('FINAL_REPORT.md',f'''# Glo-In-One-v2复现与能力复用阶段报告

日期为2026-11-11。本轮完成独立服务器CPU部署、官方预训练权重strict加载、2张官方样例和30张院内图块的15通道推理。官方样例重复推理逐字节一致。

| 阶段 | 状态 | 图块数 | 实测结果 |
|---|---|---:|---|
| 官方demo | 完成CPU兼容入口 | 2 | 512×512×15，{demo['wall_seconds']:.2f}秒 |
| 通道索引审计 | 完成 | 15通道 | task顺序0–14已确认，临床名称均待确认 |
| 重复推理 | 完成 | 2 | 输出逐字节一致 |
| 背景检查 | 完成 | 1 | 假阳性面积见channel_statistics.csv |
| 院内零样本 | 完成 | 30 | 7例，1024×1024×15，{local['wall_seconds']:.2f}秒 |
| 分割Dice/IoU | 待具备GT | 0个匹配像素GT | N/A |
| 微调及WSI | 未启动 | 0 | 先解决语义、标注及资源条件 |

权重来自官方README，SHA-256为{env['weight_sha256']}。代码提交为{config['upstream_commit']}，GPL-2.0。环境、来源、CPU兼容补丁和限制见ENVIRONMENT.md及SOURCE_CODE_AUDIT.md。

可展示的成果为官方与院内真实通道拼图、完整独立二值掩膜、逐通道预测面积和可重跑脚本。论文报告的平均Dice不作为本次实验指标。当前结果支持继续核对候选内部结构，直接用于足细胞或M/E临床语义评价仍需通道确认与专家标注。

## 项目汇报专页

1. 官方预训练模型成功加载，模型参数8479114，输出15个独立二分类通道。
2. 两张官方样例重复预测完全一致，完成纯背景检查。
3. 院内7例30图探索性零样本推理成功，保留1024尺寸。
4. 已形成逐通道核对材料，足细胞、系膜区候选含义待核实。

```mermaid
flowchart LR
  A[独立图块副本 RGB] --> B[官方UNet与动态分类头]
  B --> C[task 0到14和scale 0]
  C --> D[双logit比较与15通道掩膜]
  D --> E[逐通道核对与预测统计]
```

![官方样例](../figures/demo/test_1/all_15_channels.png)

![院内探索样例](../figures/local_zero_shot/local_001/all_15_channels.png)

纯白背景上通道10前景98.45%，模型在无组织输入上存在明显假阳性。待解决问题为通道语义证据、染色/MPP及scale匹配、系膜区和足细胞像素GT。下一阶段先核对这些问题，再决定辅助标注或小规模微调。

详细记录分别见CHANNEL_AUDIT.md、DEMO_SMOKE_TEST.md、LOCAL_EVALUATION.md和REUSE_DECISION.md。权重、图像、输出均保存在独立目录，不改动现有项目模型。
''')
print('Reports written', REPORTS)
