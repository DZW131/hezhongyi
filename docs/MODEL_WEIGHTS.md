# 模型权重与运行前检查

更新日期：2026-10-20。权重放在同仓库 Release，代码分支为 `mescnn-transfer`。

发布地址：<https://github.com/DZW131/hezhongyi/releases/tag/hspn-local-weights-20261020>。
下载后的文件放入仓库根目录 `checkpoints/`。各文件的 SHA-256、大小和来源见 `weights_manifest.json`。

| 权重 | 本地核对状态 | 用途 |
| --- | --- | --- |
| `hzy_hubmap_finetuned_glomerulus_unet_best.pth` | 已找到实际文件 | HuBMAP预训练后院内微调的肾小球U-Net |
| `efficientnetv2-m_M_V3.pth` | 已找到实际文件 | 原始MESCNN M分类器/微调初始化 |
| `efficientnetv2-m_E_V3.pth` | 已找到实际文件 | 原始MESCNN E分类器/微调初始化 |
| `densenet161_S_V3.pth` | 已找到实际文件 | 原始MESCNN S迁移基线 |
| `mobilenetv2_C_V3.pth` | 已找到实际文件 | 原始MESCNN C迁移基线，不是新月体ROI三分类 |
| `mesangial_finetuned_best.pth` | 已从恢复连接的服务器下载 | 院内M单任务最终模型 |
| `endocapillary_finetuned_best.pth` | 已从服务器下载 | 院内E多标签最终模型 |
| `crescent_detector_best.pth` | 已从服务器下载 | Faster R-CNN新月体定位 |
| `crescent_subtype_resnet18_best.pth` | 已从服务器下载 | ResNet18新月体三亚型 |
| `tubular_atrophy_v3_best.pth` | 已从服务器下载 | 95例流程使用的v3召回优化萎缩模型 |
| `intrinsic_cells_dpa_80e_inference.pth` | 已从服务器导出并下载 | 80轮验证最优权重，去除优化器状态，保留完整model/epoch/metrics/structure |
| HFRM-CAM各任务权重 | 已定位服务器目录，未纳入本次正式推理权重包 | 弱监督分割研究支线 |

原始MESCNN权重不能代替院内微调后的最终权重，不能据此复现正式95例结果。
完整报告入口还需要 `--m-init-model`、`--e-init-model`，与 `--m-checkpoint`、`--e-checkpoint` 分别对应原始初始化与院内微调权重。
脚本中保留的服务器默认路径仅是历史位置；新机器运行时请显式传入全部权重路径。

```bash
# 安装GitHub CLI后，可下载整个权重包；私有仓库先 gh auth login
gh release download hspn-local-weights-20261020 --repo DZW131/hezhongyi --dir checkpoints
```

每个模型应同时保存训练参数、标签顺序、输入尺寸、归一化、阈值和验证记录。
服务器已恢复连接，本次核对了正式权重来源与关键代码SHA-256，没有重新训练或重新评估模型。最终统一校验清单为 `weights_manifest.json`，包括11个权重的大小和SHA-256。服务器正式模型下载后均完成源文件SHA-256比对。

服务器历史权重根目录为 `/home/duyanhong/Dataspace/HZY/checkpoints/`；细胞80轮原始checkpoint位于 `/home/duyanhong/Workspace/dpa-p2pnet/checkpoint/hzy_intrinsic_dpa_80e/best.pth`。推理版仅剔除训练优化器状态，模型参数未转换精度。

当前95例场景萎缩流程使用v3模型、阈值0.15；RESULTS.md中的0.6005 Dice来自此前v2独立测试，不能直接当作v3新测结果。弱监督HFRM-CAM权重另在服务器 `hzy_hspn_weakseg_hfrm/` 下，正式MESCNN加新月体检测/分类报告不依赖它们。
