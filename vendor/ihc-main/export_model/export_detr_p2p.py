#!/usr/bin/env python3
"""
DETR模型TensorRT转换和验证工具

这个脚本提供完整的PyTorch -> ONNX -> TensorRT转换流程，
包括精度验证和性能测试。

作者: AI Assistant
日期: 2025-11-05
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detr_p2p.detr import build_model


# ============================================================================
# TensorRT相关类
# ============================================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
    """TensorRT推理引擎 (TensorRT 10+ API)"""
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        self.input_names = []
        self.output_names = []
        self._load_engine()
        
    def _load_engine(self):
        """加载TensorRT引擎"""
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # TensorRT 10+ 使用新的API
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # 计算大小
            size = trt.volume(shape)
            if size < 0:  # 动态形状
                size = 1
                for dim in shape:
                    size *= abs(dim) if dim != -1 else 1
            
            # 分配设备内存
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            host_mem = cuda.pagelocked_empty(size, dtype)
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'host': host_mem, 
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
                self.input_names.append(tensor_name)
            else:
                self.outputs.append({
                    'host': host_mem, 
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
                self.output_names.append(tensor_name)
    
    def get_input_shape(self):
        """获取引擎的输入shape"""
        if self.engine and len(self.inputs) > 0:
            return tuple(self.inputs[0]['shape'])
        return None
    
    def infer(self, input_data):
        """执行推理"""
        # 验证输入形状
        expected_size = self.inputs[0]['host'].size
        input_size = input_data.size
        if input_size != expected_size:
            raise ValueError(
                f"输入数据大小不匹配！\n"
                f"  期望: {expected_size} 元素 (shape: {self.inputs[0]['shape']})\n"
                f"  实际: {input_size} 元素 (shape: {input_data.shape})\n"
                f"  提示: 引擎可能是用不同的batch size或图像尺寸构建的，请删除旧引擎重新构建"
            )
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 设置tensor地址
        self.context.set_tensor_address(self.input_names[0], int(self.inputs[0]['device']))
        for i, output_name in enumerate(self.output_names):
            self.context.set_tensor_address(output_name, int(self.outputs[i]['device']))
        
        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 从GPU复制所有输出数据
        all_outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            all_outputs.append(output['host'])
        self.stream.synchronize()
        
        return all_outputs[0] if len(all_outputs) == 1 else all_outputs


# ============================================================================
# 工具函数
# ============================================================================

def get_gpu_info():
    """获取GPU和CUDA版本信息"""
    try:
        # 获取GPU名称
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'],
                              capture_output=True, text=True)
        gpu_name = result.stdout.strip().split('\n')[0]
        # 清理GPU名称，用于文件名
        gpu_name = gpu_name.replace(' ', '_').replace('NVIDIA_', '').replace('GeForce_', '')
        
        # 获取系统CUDA版本（从nvidia-smi）
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            output = result.stdout
            # 从nvidia-smi输出中提取CUDA版本
            # 格式类似: "CUDA Version: 12.6"
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    break
            else:
                # 如果nvidia-smi没有显示，尝试nvcc
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                output = result.stdout
                # 格式类似: "release 12.6, V12.6.20"
                for line in output.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[1].strip().split(',')[0]
                        break
                else:
                    cuda_version = torch.version.cuda  # 降级使用PyTorch的CUDA版本
        except:
            cuda_version = torch.version.cuda  # 降级使用PyTorch的CUDA版本
        
        return gpu_name, cuda_version
    except Exception as e:
        print(f"警告: 无法获取GPU信息: {e}")
        return "Unknown_GPU", "Unknown_CUDA"


def create_export_dir(base_dir="exported"):
    """创建导出目录"""
    gpu_name, cuda_version = get_gpu_info()
    export_dir = Path(base_dir) / f"{gpu_name}_cuda{cuda_version}"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def export_onnx(model, dummy_input, onnx_path, input_names=['input'], 
                output_names=['pnt_coords', 'cls_logits'], opset_version=17):
    """将PyTorch模型导出为ONNX格式"""
    print(f"\n{'='*70}")
    print("步骤1: 导出ONNX模型")
    print(f"{'='*70}")
    print(f"  目标文件: {onnx_path}")
    
    model.eval()
    
    # 测试模型输出
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  模型输出:")
    if isinstance(test_output, dict):
        for k, v in test_output.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}")
    elif isinstance(test_output, (tuple, list)):
        for i, v in enumerate(test_output):
            if isinstance(v, torch.Tensor):
                output_name = output_names[i] if i < len(output_names) else f"output_{i}"
                print(f"    {output_name}: {v.shape}")
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        dynamo=False,
        verbose=False
    )
    print(f"  ✓ ONNX模型导出成功!")
    return onnx_path


def build_tensorrt_engine(onnx_path, engine_path, precision='fp16'):
    """将ONNX模型转换为TensorRT引擎"""
    print(f"\n{'='*70}")
    print(f"步骤2: 构建TensorRT引擎 ({precision.upper()})")
    print(f"{'='*70}")
    print(f"  输入: {onnx_path}")
    print(f"  输出: {engine_path}")
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # 配置构建器
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"  启用FP16模式")
        elif precision == 'fp16':
            print(f"  警告: GPU不支持FP16，使用FP32模式")
        
        # 解析ONNX模型
        print(f"  解析ONNX模型...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print('  错误: 解析ONNX文件失败')
                for error in range(parser.num_errors):
                    print(f"    {parser.get_error(error)}")
                return None
        
        # 构建引擎
        print(f"  构建TensorRT引擎（可能需要几分钟）...")
        try:
            serialized_engine = builder.build_serialized_network(network, config)
        except Exception as e:
            print(f'  错误: 构建引擎时发生异常: {e}')
            return None
        
        if serialized_engine is None:
            print('  错误: 构建引擎失败')
            print('  可能的原因:')
            print('    1. GPU显存不足 - 尝试减小batch size或图像尺寸')
            print('    2. 模型太复杂 - 考虑简化模型结构')
            print('    3. TensorRT不支持某些算子 - 检查TensorRT日志')
            return None
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"  ✓ TensorRT引擎构建成功!")
        
        # 返回引擎对象
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine


def benchmark_pytorch(model, input_data, num_runs=100, warmup=10):
    """测试PyTorch模型的推理速度"""
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    
    # 计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = datetime.now()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = datetime.now()
    avg_time = (end_time - start_time).total_seconds() / num_runs * 1000  # ms
    
    return avg_time, output


def benchmark_tensorrt(trt_engine, input_data, num_runs=100, warmup=10):
    """测试TensorRT模型的推理速度"""
    # 预热
    for _ in range(warmup):
        _ = trt_engine.infer(input_data)
    
    # 计时
    start_time = datetime.now()
    for _ in range(num_runs):
        output = trt_engine.infer(input_data)
    end_time = datetime.now()
    
    avg_time = (end_time - start_time).total_seconds() / num_runs * 1000  # ms
    
    return avg_time, output


def compare_outputs(pytorch_output, trt_output, trt_engine):
    """对比PyTorch和TensorRT输出的精度"""
    print(f"\n{'='*70}")
    print("步骤3: 精度验证")
    print(f"{'='*70}")
    
    # 将TensorRT输出重构为字典
    trt_output_dict = {}
    for i, name in enumerate(trt_engine.output_names):
        data = trt_output[i] if isinstance(trt_output, list) else trt_output
        shape = trt_engine.outputs[i]['shape']
        trt_output_dict[name] = data.reshape(shape)
    
    # 将PyTorch输出转换为字典格式（支持dict和tuple/list）
    if isinstance(pytorch_output, dict):
        pytorch_output_dict = pytorch_output
    elif isinstance(pytorch_output, (tuple, list)):
        # 模型返回tuple/list，使用output名称作为key
        pytorch_output_dict = {}
        for i, tensor in enumerate(pytorch_output):
            if i < len(trt_engine.output_names):
                pytorch_output_dict[trt_engine.output_names[i]] = tensor
            else:
                pytorch_output_dict[f"output_{i}"] = tensor
    else:
        raise ValueError(f"不支持的PyTorch输出类型: {type(pytorch_output)}")
    
    # 对比每个输出
    results = {}
    for key in pytorch_output_dict.keys():
        if key in trt_output_dict:
            pytorch_array = pytorch_output_dict[key].cpu().numpy().flatten()
            trt_array = trt_output_dict[key].flatten()
            
            # 计算误差
            abs_diff = np.abs(pytorch_array - trt_array)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)
            
            # 余弦相似度
            cosine_sim = np.dot(pytorch_array, trt_array) / (
                np.linalg.norm(pytorch_array) * np.linalg.norm(trt_array) + 1e-8
            )
            
            results[key] = {
                'max_abs_diff': float(max_abs_diff),
                'mean_abs_diff': float(mean_abs_diff),
                'cosine_similarity': float(cosine_sim)
            }
            
            print(f"\n  {key}:")
            print(f"    最大绝对误差: {max_abs_diff:.6f}")
            print(f"    平均绝对误差: {mean_abs_diff:.6f}")
            print(f"    余弦相似度: {cosine_sim:.10f}")
            
            if cosine_sim > 0.9999:
                print(f"    ✓ 精度优秀")
            elif cosine_sim > 0.999:
                print(f"    ✓ 精度良好")
            else:
                print(f"    ⚠ 精度一般")
    
    return results


def save_results(export_dir, args, pytorch_time, trt_time, accuracy_results):
    """保存转换结果和配置"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'gpu_info': {
            'name': get_gpu_info()[0],
            'cuda_version': get_gpu_info()[1],
        },
        'model_config': {
            'row': args.row,
            'col': args.col,
            'hidden_dim': args.hidden_dim,
            'num_classes': args.num_classes,
            'input_size': [args.batch_size, 3, args.height, args.width],
        },
        'performance': {
            'pytorch_time_ms': float(pytorch_time),
            'tensorrt_time_ms': float(trt_time),
            'speedup': float(pytorch_time / trt_time),
        },
        'accuracy': accuracy_results,
        'precision': args.precision,
    }
    
    result_file = export_dir / 'conversion_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  结果已保存到: {result_file}")
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DETR模型TensorRT转换和验证工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--checkpoint', type=str, default=None,
                            help='模型checkpoint路径')
    model_group.add_argument('--row', type=int, default=2,
                            help='Anchor points行数')
    model_group.add_argument('--col', type=int, default=2,
                            help='Anchor points列数')
    model_group.add_argument('--hidden-dim', type=int, default=256,
                            help='隐藏层维度')
    model_group.add_argument('--num-classes', type=int, default=10,
                            help='类别数量')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout率')
    model_group.add_argument('--nheads', type=int, default=8,
                            help='注意力头数')
    model_group.add_argument('--dim-feedforward', type=int, default=2048,
                            help='FFN维度')
    model_group.add_argument('--enc-layers', type=int, default=6,
                            help='Encoder层数')
    model_group.add_argument('--dec-layers', type=int, default=6,
                            help='Decoder层数')
    model_group.add_argument('--backbone', type=str, default='resnet50',
                            choices=['resnet50', 'resnet101'],
                            help='Backbone网络')
    
    # 输入配置
    input_group = parser.add_argument_group('输入配置')
    input_group.add_argument('--batch-size', type=int, default=1,
                            help='批次大小')
    input_group.add_argument('--height', type=int, default=800,
                            help='输入图像高度')
    input_group.add_argument('--width', type=int, default=1066,
                            help='输入图像宽度')
    
    # 导出配置
    export_group = parser.add_argument_group('导出配置')
    export_group.add_argument('--precision', type=str, default='fp16',
                             choices=['fp16', 'fp32'],
                             help='推理精度')
    export_group.add_argument('--output-dir', type=str, default='exported',
                             help='导出目录基础路径')
    export_group.add_argument('--opset-version', type=int, default=17,
                             help='ONNX opset版本')
    export_group.add_argument('--force-rebuild', action='store_true',
                             help='强制重新构建引擎（即使已存在）')
    
    # 测试配置
    test_group = parser.add_argument_group('测试配置')
    test_group.add_argument('--num-runs', type=int, default=100,
                           help='性能测试运行次数')
    test_group.add_argument('--warmup', type=int, default=10,
                           help='预热次数')
    test_group.add_argument('--skip-validation', action='store_true',
                           help='跳过精度验证')
    
    args = parser.parse_args()
    
    # 打印配置
    print(f"\n{'='*70}")
    print("DETR模型TensorRT转换工具")
    print(f"{'='*70}")
    print(f"\n配置信息:")
    print(f"  模型: DETR (row={args.row}, col={args.col})")
    print(f"  输入尺寸: [{args.batch_size}, 3, {args.height}, {args.width}]")
    print(f"  精度: {args.precision.upper()}")
    print(f"  类别数: {args.num_classes}")
    
    # 创建导出目录
    export_dir = create_export_dir(args.output_dir)
    print(f"  导出目录: {export_dir}")
    
    # 构建模型配置
    model_args = argparse.Namespace()
    model_args.row = args.row
    model_args.col = args.col
    model_args.hidden_dim = args.hidden_dim
    model_args.num_classes = args.num_classes
    model_args.dropout = args.dropout
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.enc_layers = args.enc_layers
    model_args.dec_layers = args.dec_layers
    model_args.pre_norm = False
    model_args.position_embedding = 'sine'
    model_args.model = 'detr'
    model_args.backbone = args.backbone
    
    # 构建模型
    print(f"\n正在构建PyTorch模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_args, num_classes=args.num_classes)
    
    # 加载checkpoint
    checkpoint_loaded = False
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  加载checkpoint: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            checkpoint_loaded = True
            print(f"  ✓ Checkpoint加载成功")
        except Exception as e:
            print(f"  ⚠ 警告: Checkpoint加载失败: {e}")
            print(f"  继续使用未加载权重的模型")
    
    model = model.to(device)
    model.eval()
    print(f"  ✓ 模型构建完成")
    
    # 创建测试输入
    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width).to(device)
    
    # 定义输出路径 - 使用checkpoint的basename作为文件名
    if args.checkpoint and os.path.exists(args.checkpoint):
        # 使用checkpoint文件名（不含扩展名）
        checkpoint_basename = Path(args.checkpoint).stem
        model_name = checkpoint_basename
    else:
        # 使用默认名称
        model_name = "detr_model"
    
    onnx_path = export_dir / f"{model_name}.onnx"
    engine_path = export_dir / f"{model_name}_{args.precision}.trt"
    
    # 导出ONNX
    export_onnx(model, dummy_input, str(onnx_path))
    
    # 构建TensorRT引擎
    build_tensorrt_engine(str(onnx_path), str(engine_path), args.precision)
    
    # 性能测试
    if not args.skip_validation:
        print(f"\n{'='*70}")
        print("步骤4: 性能测试")
        print(f"{'='*70}")
        
        # PyTorch性能
        print(f"\n  测试PyTorch性能...")
        pytorch_time, pytorch_output = benchmark_pytorch(
            model, dummy_input, args.num_runs, args.warmup
        )
        print(f"    平均推理时间: {pytorch_time:.2f} ms")
        print(f"    FPS: {1000/pytorch_time:.2f}")
        
        # TensorRT性能
        print(f"\n  测试TensorRT性能...")
        trt_engine = TRTInference(str(engine_path))
        input_numpy = dummy_input.cpu().numpy()
        trt_time, trt_output = benchmark_tensorrt(
            trt_engine, input_numpy, args.num_runs, args.warmup
        )
        print(f"    平均推理时间: {trt_time:.2f} ms")
        print(f"    FPS: {1000/trt_time:.2f}")
        print(f"\n  速度提升: {pytorch_time/trt_time:.2f}x")
        
        # 精度验证
        accuracy_results = compare_outputs(pytorch_output, trt_output, trt_engine)
        
        # 保存结果
        save_results(export_dir, args, pytorch_time, trt_time, accuracy_results)
    
    # 最终总结
    print(f"\n{'='*70}")
    print("转换完成!")
    print(f"{'='*70}")
    print(f"\n生成的文件:")
    print(f"  ONNX模型: {onnx_path}")
    print(f"  TensorRT引擎: {engine_path}")
    if not args.skip_validation:
        print(f"  结果报告: {export_dir / 'conversion_results.json'}")
    print(f"\n✓ 所有文件已保存到: {export_dir}")


if __name__ == '__main__':
    main()

