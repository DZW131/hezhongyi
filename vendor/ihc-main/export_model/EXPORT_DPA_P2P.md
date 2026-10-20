# DPA-P2P 模型 TensorRT 转换指南

本文档说明如何将 DPA-P2P 模型转换为 TensorRT 引擎，并进行精度和性能对比验证。

## 目录

1. [功能概述](#功能概述)
2. [环境要求](#环境要求)
3. [使用方法](#使用方法)
4. [参数说明](#参数说明)
5. [示例](#示例)
6. [输出文件](#输出文件)
7. [故障排除](#故障排除)

## 功能概述

`export_dpa_p2p.py` 脚本提供以下功能：

1. **PyTorch → ONNX 转换**: 将训练好的 DPA-P2P 模型导出为 ONNX 格式
2. **ONNX → TensorRT 转换**: 将 ONNX 模型转换为优化的 TensorRT 引擎
3. **精度验证**: 对比 PyTorch 和 TensorRT 模型的输出精度
4. **性能测试**: 测量两个模型的推理速度并计算加速比
5. **结果保存**: 自动保存转换结果和配置信息

## 环境要求

### 必需的依赖

- Python 3.8+
- PyTorch 1.12+
- TensorRT 10.0+
- CUDA 11.0+ / 12.0+
- PyCUDA
- ONNX Runtime (可选，用于验证 ONNX 模型)

### 安装依赖

```bash
pip install torch torchvision
pip install tensorrt
pip install pycuda
pip install onnxruntime-gpu
```

## 使用方法

### 基本用法

```bash
# 使用默认参数（不加载checkpoint）
python export_model/export_dpa_p2p.py

# 加载训练好的模型进行转换
python export_model/export_dpa_p2p.py --checkpoint ./checkpoint/your_model/best.pth
```

### 完整示例

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_model/best.pth \
    --batch-size 1 \
    --height 800 \
    --width 1066 \
    --precision fp16 \
    --num-runs 100 \
    --warmup 10
```

## 参数说明

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | None | 模型 checkpoint 路径 |
| `--hidden-dim` | int | 256 | 隐藏层维度 |
| `--num-classes` | int | 10 | 类别数量 |
| `--dropout` | float | 0.1 | Dropout 率 |
| `--num-levels` | int | 3 | 多尺度特征层数 |
| `--space` | int | 8 | Anchor points 间隔 |
| `--cell-backbone` | str | vit_adapter | Backbone 类型 (vit_adapter, resnet50, resnet101) |
| `--add-tf-encoder` | flag | False | 是否添加 Transformer Encoder |
| `--num-fov` | int | 0 | Field of view 数量 |
| `--ssl-type` | str | '' | SSL 预训练类型 |
| `--ssl-path` | str | '' | SSL 预训练权重路径 |

### 输入配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | int | 1 | 批次大小 |
| `--height` | int | 800 | 输入图像高度 |
| `--width` | int | 1066 | 输入图像宽度 |

### 导出配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--precision` | str | fp16 | 推理精度 (fp16 或 fp32) |
| `--output-dir` | str | exported | 导出目录基础路径 |
| `--opset-version` | int | 17 | ONNX opset 版本 |
| `--force-rebuild` | flag | False | 强制重新构建引擎 |

### 测试配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-runs` | int | 100 | 性能测试运行次数 |
| `--warmup` | int | 10 | 预热次数 |
| `--skip-validation` | flag | False | 跳过精度验证 |

## 示例

### 示例 1: 转换默认模型

```bash
# 使用默认配置转换模型（不加载权重）
python export_model/export_dpa_p2p.py
```

输出目录: `exported/RTX_3060_cuda12.6/` (根据你的 GPU 自动命名)

### 示例 2: 转换训练好的模型（FP16）

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_vit/best.pth \
    --precision fp16 \
    --cell-backbone vit_adapter \
    --num-levels 3 \
    --space 8
```

### 示例 3: 转换为 FP32 精度

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_resnet/best.pth \
    --precision fp32 \
    --cell-backbone resnet50 \
    --num-levels 3
```

### 示例 4: 自定义输入尺寸

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_model/best.pth \
    --height 1024 \
    --width 1024 \
    --batch-size 1
```

### 示例 5: 只导出不测试

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_model/best.pth \
    --skip-validation
```

### 示例 6: 更详细的性能测试

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_model/best.pth \
    --num-runs 500 \
    --warmup 50
```

## 输出文件

转换完成后，会在 `exported/{GPU_NAME}_cuda{CUDA_VERSION}/` 目录下生成以下文件：

### 1. ONNX 模型

**文件名**: `dpa_{checkpoint_name}.onnx` 或 `dpa_model.onnx`

这是中间格式的模型文件，可用于：
- 在不同框架间迁移
- 使用 ONNX Runtime 进行推理
- 转换到其他推理引擎

### 2. TensorRT 引擎

**文件名**: `dpa_{checkpoint_name}_{precision}.trt`

例如：
- `dpa_best_epoch_50_fp16.trt` (FP16 精度)
- `dpa_model_fp32.trt` (FP32 精度)

这是优化后的 TensorRT 引擎，用于高性能推理。

⚠️ **重要**: TensorRT 引擎是特定于 GPU 的，不同 GPU 或 CUDA 版本需要重新构建。

### 3. 结果报告

**文件名**: `dpa_conversion_results.json`

包含以下信息：
```json
{
  "timestamp": "2025-11-05T10:30:00",
  "model_type": "dpa_p2p",
  "gpu_info": {
    "name": "RTX_3060",
    "cuda_version": "12.6"
  },
  "model_config": {
    "hidden_dim": 256,
    "num_classes": 10,
    "num_levels": 3,
    "space": 8,
    "cell_backbone": "vit_adapter",
    "input_size": [1, 3, 800, 1066]
  },
  "performance": {
    "pytorch_time_ms": 45.32,
    "tensorrt_time_ms": 12.15,
    "speedup": 3.73
  },
  "accuracy": {
    "pnt_coords": {
      "max_abs_diff": 0.000123,
      "mean_abs_diff": 0.000045,
      "cosine_similarity": 0.99998765
    },
    "cls_logits": {
      "max_abs_diff": 0.000234,
      "mean_abs_diff": 0.000067,
      "cosine_similarity": 0.99997654
    }
  },
  "precision": "fp16"
}
```

## 工作流程

脚本执行以下步骤：

```
┌─────────────────────────────┐
│  1. 构建 PyTorch 模型        │
│     - 加载模型配置           │
│     - 加载 checkpoint (可选) │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  2. 导出 ONNX 模型           │
│     - 运行模型测试           │
│     - 导出为 ONNX 格式       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3. 构建 TensorRT 引擎       │
│     - 解析 ONNX 模型         │
│     - 应用优化               │
│     - 序列化引擎             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4. 性能测试                 │
│     - PyTorch 推理速度       │
│     - TensorRT 推理速度      │
│     - 计算加速比             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  5. 精度验证                 │
│     - 对比输出数值           │
│     - 计算误差指标           │
│     - 计算余弦相似度         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  6. 保存结果                 │
│     - 保存配置信息           │
│     - 保存性能数据           │
│     - 保存精度报告           │
└─────────────────────────────┘
```

## 精度指标说明

### 余弦相似度

余弦相似度用于衡量两个向量的方向相似性：

- **> 0.9999**: 精度优秀 ✓
- **> 0.999**: 精度良好 ✓
- **< 0.999**: 精度一般 ⚠️

### 绝对误差

- **最大绝对误差**: 两个输出中最大的差值
- **平均绝对误差**: 所有元素差值的平均值

FP16 模式下，通常会有较小的精度损失（最大误差 < 0.01），这在大多数应用中是可接受的。

## 故障排除

### 问题 1: ONNX 导出失败

**错误**: `RuntimeError: ONNX export failed`

**解决方案**:
1. 检查 PyTorch 和 ONNX 版本兼容性
2. 尝试降低 `--opset-version` (例如从 17 降到 13)
3. 确保模型中所有操作都支持 ONNX 导出

### 问题 2: TensorRT 构建失败

**错误**: `Error: Failed to build TensorRT engine`

**可能原因和解决方案**:

1. **GPU 显存不足**
   ```bash
   # 减小 batch size
   python export_model/export_dpa_p2p.py --batch-size 1
   
   # 减小图像尺寸
   python export_model/export_dpa_p2p.py --height 512 --width 512
   ```

2. **不支持的算子**
   - 检查 TensorRT 日志中的错误信息
   - 某些 PyTorch 操作可能不被 TensorRT 支持
   - 考虑修改模型架构或使用插件

3. **TensorRT 版本不兼容**
   ```bash
   # 检查 TensorRT 版本
   python -c "import tensorrt; print(tensorrt.__version__)"
   
   # 确保版本 >= 10.0
   ```

### 问题 3: 精度差异过大

**症状**: 余弦相似度 < 0.99

**解决方案**:

1. **使用 FP32 精度**
   ```bash
   python export_model/export_dpa_p2p.py --precision fp32
   ```

2. **检查输入数据**
   - 确保输入数据范围正确
   - 检查归一化是否一致

3. **验证模型权重**
   - 确认 checkpoint 加载成功
   - 检查是否所有层的权重都被正确加载

### 问题 4: 推理速度没有提升

**症状**: `speedup` < 1.5

**可能原因**:

1. **Batch size 太小**: TensorRT 在较大 batch size 时效果更好
2. **模型太简单**: 小模型可能不会有显著加速
3. **GPU 利用率低**: 检查 GPU 是否被充分利用

**解决方案**:
```bash
# 增加 batch size
python export_model/export_dpa_p2p.py --batch-size 4

# 使用更大的输入尺寸
python export_model/export_dpa_p2p.py --height 1024 --width 1024
```

### 问题 5: 输入尺寸不匹配

**错误**: `ValueError: Input size mismatch`

**解决方案**:

这通常发生在使用已存在的引擎文件，但输入尺寸不同时。

```bash
# 删除旧的引擎文件并重新构建
rm exported/RTX_3060_cuda12.6/*.trt

# 或使用 --force-rebuild 标志
python export_model/export_dpa_p2p.py --force-rebuild
```

### 问题 6: CUDA Out of Memory

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:

```bash
# 1. 减小 batch size
python export_model/export_dpa_p2p.py --batch-size 1

# 2. 减小输入尺寸
python export_model/export_dpa_p2p.py --height 512 --width 512

# 3. 减小测试次数
python export_model/export_dpa_p2p.py --num-runs 10 --warmup 3

# 4. 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

## 性能优化建议

### 1. 选择合适的精度

- **FP16**: 推荐用于部署，速度快，精度损失小
- **FP32**: 用于要求极高精度的场景

### 2. 优化输入尺寸

- 尽量使用 GPU 友好的尺寸（如 32 的倍数）
- 避免奇怪的尺寸组合

### 3. 增加 Batch Size

如果 GPU 显存允许，增加 batch size 可以显著提升吞吐量：

```bash
python export_model/export_dpa_p2p.py --batch-size 4
```

### 4. 使用 Workspace 优化

TensorRT 的 workspace 大小影响优化效果。默认是 1GB，可以在代码中调整：

```python
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
```

## DPA vs DETR 对比

### 模型特点对比

| 特性 | DPA-P2P | DETR-P2P |
|------|---------|----------|
| Backbone | ViT Adapter / ResNet | ResNet |
| 特征提取 | 多尺度 (3层) | 单一尺度 |
| Anchor Points | 动态生成 | 固定网格 |
| 输出 | pnt_coords + cls_logits | pnt_coords + cls_logits |
| 参数量 | 较大 | 较小 |

### 转换差异

1. **配置参数**:
   - DPA 需要配置 `--num-levels`, `--space`, `--cell-backbone`
   - DETR 需要配置 `--row`, `--col`, `--backbone`

2. **模型复杂度**:
   - DPA 模型通常更大，转换时间更长
   - DPA 需要更多 GPU 显存

3. **性能表现**:
   - 两者加速比类似（2-4x）
   - FP16 精度损失都很小

## 常见使用场景

### 场景 1: 研发阶段

使用默认模型快速验证转换流程：

```bash
python export_model/export_dpa_p2p.py --skip-validation
```

### 场景 2: 模型部署

转换训练好的模型并进行完整验证：

```bash
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/best_model.pth \
    --precision fp16 \
    --num-runs 200 \
    --warmup 20
```

### 场景 3: 性能对比

对比不同精度的性能：

```bash
# FP16
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/best_model.pth \
    --precision fp16

# FP32
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/best_model.pth \
    --precision fp32
```

### 场景 4: 批量处理

转换不同配置的模型：

```bash
# ViT Adapter backbone
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_vit.pth \
    --cell-backbone vit_adapter

# ResNet50 backbone
python export_model/export_dpa_p2p.py \
    --checkpoint ./checkpoint/dpa_resnet.pth \
    --cell-backbone resnet50
```

## 注意事项

1. **引擎不可移植**: TensorRT 引擎是特定于 GPU 架构的，不能在不同 GPU 间共享
2. **CUDA 版本**: 确保运行时的 CUDA 版本与构建时一致
3. **显存管理**: 转换过程可能需要大量显存，建议在空闲 GPU 上进行
4. **精度权衡**: FP16 会有微小精度损失，但通常可以接受
5. **Checkpoint 兼容性**: 确保 checkpoint 文件与模型配置匹配

## 参考资料

- [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX 官方文档](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## 许可证

本工具遵循项目主仓库的许可证。

