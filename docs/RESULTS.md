# 已有实验结果

下表为本地已有实验记录的历史结果，本次代码交付没有重新跑服务器推理。模块验证、独立测试与病例级结果的统计单位不同。

| 模块/数据口径 | 指标 |
| --- | --- |
| 院内微调肾小球U-Net，院内验证 | Dice 0.9397，IoU 0.8863，Precision 0.9523，Recall 0.9274 |
| MESCNN M/E多标签，验证170个肾小球crop | 宏F1 0.7375，平均AUC 0.8818，双标签完全一致率0.7588 |
| 上述多标签模型的M/E分项 | M F1 0.6602 / AUC 0.8379；E F1 0.8148 / AUC 0.9256 |
| 新月体Faster R-CNN | 存在性F1 0.850；框F1 0.791，框匹配采用项目宽松IoU/中心命中规则，不等同COCO mAP |
| 新月体ResNet18 ROI三分类 | 准确率0.810，平衡准确率0.812，宏F1 0.803 |
| 肾小管萎缩U-Net，8例/15场景/1942图块独立测试 | Dice 0.6005，IoU 0.4291，Recall 0.6376，Specificity 0.9608 |
| DPA-P2PNet，80轮验证最优，4张切片/46crop独立测试 | 点检测Precision 0.739，Recall 0.517，F1 0.608，三类宏F1 0.379 |
| 上述细胞分类分项 | 内皮F1 0.437，系膜F1 0.258，足细胞F1 0.441 |

固有细胞首批原始94张CZI中35张含细胞点标注，25张同时具备可用肾小球ROI，导出251个crop。按切片划分：训练17张/157crop，验证4张/48crop，独立测试4张/46crop。点匹配距离为1024尺寸crop中的10像素。

## 95例冻结模型与阈值的病例级对照

95例匹配病例，2692个肾小球。病例级结果是否完全独立于此前训练样本，仍需根据病例清单审计，不能直接称作外部独立测试。

| 终点 | 准确率 | 其他指标 |
| --- | ---: | --- |
| Oxford M | 0.7684 | 敏感度0.7500，F1 0.3529 |
| Oxford E | 0.6737 | 敏感度0.9268，F1 0.7103 |
| Oxford C三分类 | 0.6000 | 平衡准确率0.6061，宏F1 0.5393 |
| ISKDC分型 | 0.6316 | 平衡准确率0.4381，宏F1 0.3939 |
| T | 0.9579 | 敏感度0，F1 0；类别不平衡，准确率不代表阳性识别能力 |
| M/E/T/C可用分量完全一致 | 0.3263 | S未进入该轮模型来源的对照 |

T输入目前使用预测肾小管萎缩面积/组织面积代理量，并非完整IFTA标注。S若无模型或人工字段，不应将默认值解释成已验证阴性。

本地来源：`reports/hzy_results_20260615/report_notes.md`；`reports/mescnn_proliferation_multilabel_20260822/optimized_384_focal_sampler_noposw/analysis/run_summary.json`；`reports/hzy_tubular_atrophy_52_20260922/unet256_v2_test_t04/metrics.json`；`reports/hzy_intrinsic_cell_94_20261008/dpa_p2pnet_v1_run.md`；`reports/hspn_scoring_validation_20261014/all95/metric_summary.csv`。病例级明细与图像不随Git代码上传。
