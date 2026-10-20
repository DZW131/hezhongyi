# DETR模型TensorRT转换工具使用指南

## 简介

`export_to_tensorrt.py` 是一个完整的DETR模型TensorRT转换工具，提供从PyTorch模型到TensorRT引擎的一站式转换，包括精度验证和性能测试。

## 快速开始

### 1. 基本使用（使用默认参数）

```bash
conda activate py313
python export_to_tensorrt.py
```

这将使用默认配置转换模型，生成的文件将保存在 `exported/{GPU名称}_cuda{CUDA版本}/` 目录下。

### 2. 指定checkpoint

```bash
python export_to_tensorrt.py --checkpoint path/to/your/checkpoint.pth
```

### 3. 自定义输入尺寸

```bash
python export_to_tensorrt.py \
    --height 1024 \
    --width 1280 \
    --batch-size 1
```

### 4. 使用FP32精度

```bash
python export_to_tensorrt.py --precision fp32
```

## 完整参数说明

### 模型配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | None | 模型checkpoint路径 (.pth文件) |
| `--row` | 2 | Anchor points行数 |
| `--col` | 2 | Anchor points列数 |
| `--hidden-dim` | 256 | 隐藏层维度 |
| `--num-classes` | 10 | 类别数量 |
| `--dropout` | 0.1 | Dropout率 |
| `--nheads` | 8 | 注意力头数 |
| `--dim-feedforward` | 2048 | FFN维度 |
| `--enc-layers` | 6 | Encoder层数 |
| `--dec-layers` | 6 | Decoder层数 |
| `--backbone` | resnet50 | Backbone网络 (resnet50/resnet101) |

### 输入配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 1 | 批次大小 |
| `--height` | 800 | 输入图像高度 |
| `--width` | 1066 | 输入图像宽度 |

### 导出配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--precision` | fp16 | 推理精度 (fp16/fp32) |
| `--output-dir` | exported | 导出目录基础路径 |
| `--opset-version` | 17 | ONNX opset版本 |

### 测试配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-runs` | 100 | 性能测试运行次数 |
| `--warmup` | 10 | 预热次数 |
| `--skip-validation` | False | 跳过精度验证 |

## 使用示例

### 示例1：完整配置

```bash
python export_to_tensorrt.py \
    --checkpoint models/detr_checkpoint.pth \
    --num-classes 20 \
    --height 1024 \
    --width 1280 \
    --batch-size 2 \
    --precision fp16 \
    --num-runs 200
```

### 示例2：快速转换（跳过验证）

```bash
python export_to_tensorrt.py \
    --checkpoint models/detr_checkpoint.pth \
    --skip-validation
```

### 示例3：自定义模型架构

```bash
python export_to_tensorrt.py \
    --checkpoint models/detr_checkpoint.pth \
    --hidden-dim 512 \
    --nheads 16 \
    --enc-layers 12 \
    --dec-layers 12 \
    --backbone resnet101
```

### 示例4：自定义导出目录

```bash
python export_to_tensorrt.py \
    --checkpoint models/detr_checkpoint.pth \
    --output-dir my_exports
```

## 输出文件说明

转换完成后，会在 `exported/{GPU名称}_cuda{CUDA版本}/` 目录下生成以下文件：

```
exported/
└── RTX_3090_cuda12.6/
    ├── detr_model.onnx              # ONNX模型文件
    ├── detr_model_fp16.trt          # TensorRT FP16引擎
    └── conversion_results.json      # 转换结果报告
```

### conversion_results.json 内容示例

```json
{
  "timestamp": "2025-11-05T01:00:00.000000",
  "gpu_info": {
    "name": "RTX_3090",
    "cuda_version": "12.6"
  },
  "model_config": {
    "row": 2,
    "col": 2,
    "hidden_dim": 256,
    "num_classes": 10,
    "input_size": [1, 3, 800, 1066]
  },
  "performance": {
    "pytorch_time_ms": 88.42,
    "tensorrt_time_ms": 24.31,
    "speedup": 3.64
  },
  "accuracy": {
    "pnt_coords": {
      "max_abs_diff": 0.5,
      "mean_abs_diff": 0.08,
      "cosine_similarity": 1.0
    },
    "cls_logits": {
      "max_abs_diff": 2.08,
      "mean_abs_diff": 0.07,
      "cosine_similarity": 0.9997
    }
  },
  "precision": "fp16"
}
```

## 使用转换后的模型

### Python推理示例

```python
import numpy as np
from export_to_tensorrt import TRTInference

# 加载TensorRT引擎
engine_path = "exported/RTX_3090_cuda12.6/detr_model_fp16.trt"
engine = TRTInference(engine_path)

# 准备输入 (batch_size, channels, height, width)
input_data = np.random.randn(1, 3, 800, 1066).astype(np.float32)

# 推理
outputs = engine.infer(input_data)

# 重构输出
pnt_coords = outputs[0].reshape(1, 4250, 2)
cls_logits = outputs[1].reshape(1, 4250, 11)

print(f"Point coordinates shape: {pnt_coords.shape}")
print(f"Classification logits shape: {cls_logits.shape}")
```

## 性能优化建议

### 1. FP16 vs FP32

- **FP16**: 推荐用于生产环境
  - 速度提升: ~3.6x
  - 精度: 99.97%
  - 显存占用: 减半

- **FP32**: 需要最高精度时使用
  - 速度提升: ~1.5x
  - 精度: 接近100%
  - 显存占用: 较大

### 2. 批次大小

- 批次大小越大，吞吐量越高
- 但会增加延迟和显存占用
- 建议根据实际应用场景选择

### 3. 输入尺寸

- 较小的输入尺寸会显著提升速度
- 但可能影响检测精度
- 建议在精度和速度之间权衡

## 常见问题

### Q1: 如何查看GPU信息？

```bash
nvidia-smi
```

### Q2: 如何查看CUDA版本？

```bash
nvcc --version
# 或
python -c "import torch; print(torch.version.cuda)"
```

### Q3: 转换失败怎么办？

1. 检查CUDA和TensorRT版本是否兼容
2. 确保有足够的显存
3. 尝试使用FP32精度
4. 减小输入尺寸或批次大小

### Q4: 精度不够怎么办？

1. 使用FP32精度: `--precision fp32`
2. 检查模型checkpoint是否正确加载
3. 使用较新的opset版本: `--opset-version 18`

### Q5: 如何查看所有参数？

```bash
python export_to_tensorrt.py --help
```

## 注意事项

1. ⚠️ **重要**: 确保模型代码中没有使用numpy操作（已在AnchorPoints中修复）
2. ✅ 建议在转换前先测试PyTorch模型是否正常工作
3. ✅ 首次转换某个配置时，TensorRT引擎构建可能需要几分钟
4. ✅ 生成的TensorRT引擎是特定于GPU架构的，不能跨GPU使用

## 环境要求

- Python 3.13+
- PyTorch 2.9.0+
- TensorRT 10.13+
- CUDA 12.6+
- 足够的GPU显存 (建议8GB+)

## 技术支持

如果遇到问题，请查看：
- `PROBLEM_FIXED_REPORT.md` - 常见问题修复指南
- `TENSORRT_CONVERSION_REPORT.md` - 详细转换报告
- 或在项目中提issue

---

**最后更新**: 2025-11-05
**版本**: 1.0.0

