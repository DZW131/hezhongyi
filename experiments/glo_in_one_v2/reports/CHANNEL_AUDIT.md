# 通道语义审计

已由代码和真实输出确认15个独立二分类任务，数组轴为H、W、task，task顺序0–14。正式确认的临床类别映射为0个。

| 通道 | 候选名称 | 索引证据 | 名称证据 |
|---|---|---|---|
| 0 | adhesion | 已确认task=0 | 社区推测，待作者确认 |
| 1 | capsular drop | 已确认task=1 | 社区推测，待作者确认 |
| 2 | global sclerosis | 已确认task=2 | 社区推测，待作者确认 |
| 3 | hyaline cap | 已确认task=3 | 社区推测，待作者确认 |
| 4 | hyalinosis | 已确认task=4 | 社区推测，待作者确认 |
| 5 | mesangial lysis | 已确认task=5 | 社区推测，待作者确认 |
| 6 | mesangial expansion | 已确认task=6 | 社区推测，待作者确认 |
| 7 | microaneurysm | 已确认task=7 | 社区推测，待作者确认 |
| 8 | nodular sclerosis | 已确认task=8 | 社区推测，待作者确认 |
| 9 | segmental sclerosis | 已确认task=9 | 社区推测，待作者确认 |
| 10 | Bowman capsule | 已确认task=10 | 社区推测，待作者确认 |
| 11 | mesangium | 已确认task=11 | 社区推测，待作者确认 |
| 12 | glomerular tuft | 已确认task=12 | 社区推测，待作者确认 |
| 13 | podocyte | 已确认task=13 | 社区推测，待作者确认 |
| 14 | mesangial cell | 已确认task=14 | 社区推测，待作者确认 |

上游train_2D_patch.py使用数字task与外部训练数据CSV，本次仓库未提供可用于逐个确认类别名称的完整标注CSV。官方示意图列有5种正常结构及病变名称，但不能证明输出索引对应关系。README的14类描述与15任务模型存在差异，hyaline cap的归属仍有歧义。

候选表来自[Issue #1社区讨论](https://github.com/hrlblab/Glo-In-One_v2/issues/1)。本轮图片仅显示channel编号，候选名称不参与正式定量评价。
