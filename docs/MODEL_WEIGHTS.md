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
| 院内M单任务与E多标签 `best.pth` | 本地未找到，待服务器恢复后补齐 | 统一报告正式增生模型 |
| Faster R-CNN新月体检测 `best.pth` | 本地未找到，待补齐 | 统一报告新月体定位 |
| ResNet18新月体三分类 `best.pth` | 本地未找到，待补齐 | 新月体三亚型 |
| 肾小管萎缩U-Net `best.pth` | 本地未找到，待补齐 | 切片萎缩负荷 |
| DPA-P2PNet 80轮验证最优权重 | 本地未找到，待补齐 | 固有细胞检测 |
| HFRM-CAM各任务权重 | 本地未找到，待补齐 | 弱监督分割研究支线 |

原始MESCNN权重不能代替院内微调后的最终权重，不能据此复现正式95例结果。
完整报告入口还需要 `--m-init-model`、`--e-init-model`，与 `--m-checkpoint`、`--e-checkpoint` 分别对应原始初始化与院内微调权重。
脚本中保留的服务器默认路径仅是历史位置；新机器运行时请显式传入全部权重路径。

```bash
# 安装GitHub CLI后，可下载整个权重包；私有仓库先 gh auth login
gh release download hspn-local-weights-20261020 --repo DZW131/hezhongyi --dir checkpoints
```

每个模型应同时保存训练参数、标签顺序、输入尺寸、归一化、阈值和验证记录。
本次没有连接服务器，也没有重新训练或重新评估模型。
