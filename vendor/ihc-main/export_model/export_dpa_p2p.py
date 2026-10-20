#!/usr/bin/env python3
"""
DPA模型TensorRT转换和验证工具

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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.dpa_p2p.dpa_p2pnet import build_dpa_model

# 尝试导入 TensorRT（可选）
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    print("警告: TensorRT 未安装，将跳过 TensorRT 转换")

# 导入 ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    print("错误: ONNX Runtime 未安装，请运行: pip install onnxruntime-gpu")


# ============================================================================
# ONNX Runtime 推理类
# ============================================================================

class ONNXRuntimeInference:
    """ONNX Runtime 推理引擎"""
    def __init__(self, onnx_path, use_gpu=True, gpu_id=0):
        self.onnx_path = onnx_path
        
        # 配置执行提供者
        if use_gpu:
            # 配置 CUDA 提供者，尝试多种内存策略以支持 1024x1024
            cuda_provider_options = {
                'device_id': gpu_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',  # 更激进的内存增长策略
                'gpu_mem_limit': 11 * 1024 * 1024 * 1024,  # 11GB GPU 内存限制
                'cudnn_conv_algo_search': 'EXHAUSTIVE',  # 寻找最佳算法
                'tunable_op_enable': '1',  # 启用可调操作
                'tunable_op_tuning_enable': '1',  # 启用调优
            }
            providers = [
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 针对大输入的内存优化策略
        # 禁用内存模式可能会帮助大内存分配
        sess_options.enable_mem_pattern = False  # 禁用内存模式识别
        sess_options.enable_mem_reuse = False     # 禁用内存重用
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 顺序执行可能更稳定
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 获取输入输出信息
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入形状
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"  ONNX Runtime 会话创建成功")
        print(f"  执行提供者: {self.session.get_providers()}")
        print(f"  输入: {self.input_names}")
        print(f"  输出: {self.output_names}")
    
    def get_input_shape(self):
        """获取输入形状"""
        return tuple(self.input_shape)
    
    def infer(self, input_data):
        """执行推理"""
        # 确保输入是 numpy 数组
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        # 运行推理
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_data}
        )
        
        return outputs


# ============================================================================
# TensorRT相关类
# ============================================================================

if HAS_TENSORRT:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
else:
    TRT_LOGGER = None


class TRTInference:
    """TensorRT推理引擎 (TensorRT 10+ API)"""
    def __init__(self, engine_path):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT 未安装，无法使用 TRT 推理")
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
    
    # 导出ONNX - 使用优化配置以支持 1024x1024 GPU 推理
    # 禁用常量折叠可以减少内存峰值使用
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,  # 禁用常量折叠，可能减少内存峰值
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
    if not HAS_TENSORRT:
        print(f"\n{'='*70}")
        print(f"跳过: TensorRT引擎构建 (TensorRT未安装)")
        print(f"{'='*70}")
        return None
    
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


def benchmark_onnxruntime(onnx_engine, input_data, num_runs=100, warmup=10):
    """测试ONNX Runtime模型的推理速度"""
    # 预热
    for _ in range(warmup):
        _ = onnx_engine.infer(input_data)
    
    # 计时
    start_time = datetime.now()
    for _ in range(num_runs):
        output = onnx_engine.infer(input_data)
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


def compare_outputs(pytorch_output, inference_output, output_names, engine_type="TensorRT"):
    """对比PyTorch和推理引擎输出的精度"""
    print(f"\n{'='*70}")
    print(f"步骤3: 精度验证 (PyTorch vs {engine_type})")
    print(f"{'='*70}")
    
    # 将推理输出转换为字典
    if isinstance(inference_output, dict):
        inference_output_dict = inference_output
    elif isinstance(inference_output, (list, tuple)):
        # ONNX Runtime 或 TensorRT 的输出是列表
        inference_output_dict = {}
        for i, name in enumerate(output_names):
            inference_output_dict[name] = inference_output[i]
    else:
        # 单个输出
        inference_output_dict = {output_names[0]: inference_output}
    
    # 对比每个输出
    results = {}
    for key in pytorch_output.keys():
        if key in inference_output_dict:
            pytorch_array = pytorch_output[key].cpu().numpy().flatten()
            inference_array = np.array(inference_output_dict[key]).flatten()
            
            # 计算误差
            abs_diff = np.abs(pytorch_array - inference_array)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)
            
            # 计算相对误差
            relative_diff = abs_diff / (np.abs(pytorch_array) + 1e-8)
            max_relative_diff = np.max(relative_diff)
            mean_relative_diff = np.mean(relative_diff)
            
            # 余弦相似度
            cosine_sim = np.dot(pytorch_array, inference_array) / (
                np.linalg.norm(pytorch_array) * np.linalg.norm(inference_array) + 1e-8
            )
            
            results[key] = {
                'max_abs_diff': float(max_abs_diff),
                'mean_abs_diff': float(mean_abs_diff),
                'max_relative_diff': float(max_relative_diff),
                'mean_relative_diff': float(mean_relative_diff),
                'cosine_similarity': float(cosine_sim)
            }
            
            print(f"\n  {key}:")
            print(f"    最大绝对误差: {max_abs_diff:.6f}")
            print(f"    平均绝对误差: {mean_abs_diff:.6f}")
            print(f"    最大相对误差: {max_relative_diff:.6f}")
            print(f"    平均相对误差: {mean_relative_diff:.6f}")
            print(f"    余弦相似度: {cosine_sim:.10f}")
            
            if cosine_sim > 0.9999:
                print(f"    ✓ 精度优秀")
            elif cosine_sim > 0.999:
                print(f"    ✓ 精度良好")
            elif cosine_sim > 0.99:
                print(f"    ⚠ 精度一般")
            else:
                print(f"    ✗ 精度较差")
    
    return results


def save_results(export_dir, args, pytorch_time, inference_time, accuracy_results, 
                 engine_type="ONNX Runtime", tensorrt_status=None):
    """保存转换结果和配置"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'dpa_p2p',
        'inference_engine': engine_type,
        'gpu_info': {
            'name': get_gpu_info()[0],
            'cuda_version': get_gpu_info()[1],
        },
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'num_classes': args.num_classes,
            'num_levels': args.num_levels,
            'space': args.space,
            'cell_backbone': args.cell_backbone,
            'input_size': [args.batch_size, 3, args.height, args.width],
        },
        'performance': {
            'pytorch_time_ms': float(pytorch_time),
            f'{engine_type.lower().replace(" ", "_")}_time_ms': float(inference_time),
            'speedup': float(pytorch_time / inference_time),
            'fps_pytorch': float(1000 / pytorch_time),
            'fps_inference': float(1000 / inference_time),
        },
        'accuracy': accuracy_results,
        'precision': args.precision if engine_type == "TensorRT" else "fp32",
    }
    
    # 如果有 TensorRT 状态信息，添加进去
    if tensorrt_status:
        results['tensorrt_status'] = tensorrt_status
    
    result_file = export_dir / 'dpa_onnx_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  结果已保存到: {result_file}")
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DPA模型TensorRT转换和验证工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--checkpoint', type=str, default=None,
                            help='模型checkpoint路径')
    model_group.add_argument('--hidden-dim', type=int, default=256,
                            help='隐藏层维度')
    model_group.add_argument('--num-classes', type=int, default=10,
                            help='类别数量')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout率')
    model_group.add_argument('--num-levels', type=int, default=3,
                            help='多尺度特征层数')
    model_group.add_argument('--space', type=int, default=8,
                            help='Anchor points间隔')
    model_group.add_argument('--cell-backbone', type=str, default='vit_adapter',
                            choices=['vit_adapter', 'resnet50', 'resnet101'],
                            help='Backbone网络类型')
    model_group.add_argument('--add-tf-encoder', action='store_true',
                            help='是否添加Transformer Encoder')
    model_group.add_argument('--num-fov', type=int, default=0,
                            help='Field of view数量')
    model_group.add_argument('--ssl-type', type=str, default='',
                            help='SSL预训练类型')
    model_group.add_argument('--ssl-path', type=str, default='',
                            help='SSL预训练权重路径')
    
    # 输入配置
    input_group = parser.add_argument_group('输入配置')
    input_group.add_argument('--batch-size', type=int, default=1,
                            help='批次大小')
    input_group.add_argument('--height', type=int, default=512,
                            help='输入图像高度 (ViT Adapter要求能被32整除)')
    input_group.add_argument('--width', type=int, default=512,
                            help='输入图像宽度 (ViT Adapter要求能被32整除)')
    
    # 导出配置
    export_group = parser.add_argument_group('导出配置')
    export_group.add_argument('--precision', type=str, default='fp16',
                             choices=['fp16', 'fp32'],
                             help='推理精度')
    export_group.add_argument('--output-dir', type=str, default='exported',
                             help='导出目录基础路径')
    export_group.add_argument('--opset-version', type=int, default=18,
                             help='ONNX opset版本 (DPA需要18以支持col2im)')
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
    test_group.add_argument('--skip-tensorrt', action='store_true',
                           help='跳过TensorRT转换，直接使用ONNX Runtime')
    
    args = parser.parse_args()
    
    # 打印配置
    print(f"\n{'='*70}")
    print("DPA模型TensorRT转换工具")
    print(f"{'='*70}")
    print(f"\n配置信息:")
    print(f"  模型: DPA-P2P")
    print(f"  Backbone: {args.cell_backbone}")
    print(f"  输入尺寸: [{args.batch_size}, 3, {args.height}, {args.width}]")
    print(f"  精度: {args.precision.upper()}")
    print(f"  类别数: {args.num_classes}")
    print(f"  多尺度层数: {args.num_levels}")
    print(f"  Anchor间隔: {args.space}")
    
    # 创建导出目录
    export_dir = create_export_dir(args.output_dir)
    print(f"  导出目录: {export_dir}")
    
    # 构建模型配置
    model_args = argparse.Namespace()
    model_args.hidden_dim = args.hidden_dim
    model_args.position_embedding = 'sine'
    model_args.dropout = args.dropout
    model_args.nheads = 8
    model_args.dim_feedforward = 2048
    model_args.enc_layers = 6
    model_args.dec_layers = 6
    model_args.pre_norm = False
    model_args.row = 2
    model_args.col = 2
    model_args.backbone = args.cell_backbone  # For compatibility
    model_args.cell_backbone = args.cell_backbone
    model_args.add_tf_encoder = args.add_tf_encoder
    model_args.num_levels = args.num_levels
    model_args.num_fov = args.num_fov
    model_args.ssl_type = args.ssl_type
    model_args.ssl_path = args.ssl_path
    model_args.space = args.space
    model_args.num_classes = args.num_classes
    
    # 构建模型
    print(f"\n正在构建PyTorch模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_dpa_model(model_args, num_classes=args.num_classes)
    
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
        model_name = f"dpa_{checkpoint_basename}"
    else:
        # 使用默认名称
        model_name = "dpa_model"
    
    onnx_path = export_dir / f"{model_name}.onnx"
    engine_path = export_dir / f"{model_name}_{args.precision}.trt"
    
    # 导出ONNX
    export_onnx(model, dummy_input, str(onnx_path), opset_version=args.opset_version)
    
    # 尝试构建TensorRT引擎
    tensorrt_success = False
    tensorrt_status = None
    
    if HAS_TENSORRT and not args.skip_tensorrt:
        engine = build_tensorrt_engine(str(onnx_path), str(engine_path), args.precision)
        if engine is not None:
            tensorrt_success = True
            tensorrt_status = {"status": "success", "file": str(engine_path)}
        else:
            tensorrt_status = {
                "status": "failed", 
                "reason": "TensorRT不支持模型中的某些操作（如Col2Im）",
                "recommendation": "使用ONNX Runtime进行推理"
            }
    else:
        tensorrt_status = {
            "status": "skipped",
            "reason": "TensorRT未安装或用户跳过" if not HAS_TENSORRT else "用户跳过"
        }
    
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
        
        # 选择推理引擎
        use_tensorrt = tensorrt_success and HAS_TENSORRT
        
        if use_tensorrt:
            # TensorRT性能
            print(f"\n  测试TensorRT性能...")
            trt_engine = TRTInference(str(engine_path))
            input_numpy = dummy_input.cpu().numpy()
            inference_time, inference_output = benchmark_tensorrt(
                trt_engine, input_numpy, args.num_runs, args.warmup
            )
            engine_type = "TensorRT"
            output_names = trt_engine.output_names
        else:
            # ONNX Runtime性能
            if not HAS_ONNXRUNTIME:
                print(f"\n  错误: ONNX Runtime 未安装，无法进行推理测试")
                print(f"  请运行: pip install onnxruntime-gpu")
                return
            
            print(f"\n  使用ONNX Runtime进行推理...")
            print(f"  (TensorRT转换失败，退回到ONNX Runtime)")
            
            onnx_engine = ONNXRuntimeInference(str(onnx_path), use_gpu=True, gpu_id=0)
            input_numpy = dummy_input.cpu().numpy()
            inference_time, inference_output = benchmark_onnxruntime(
                onnx_engine, input_numpy, args.num_runs, args.warmup
            )
            engine_type = "ONNX Runtime"
            output_names = onnx_engine.output_names
        
        print(f"    平均推理时间: {inference_time:.2f} ms")
        print(f"    FPS: {1000/inference_time:.2f}")
        print(f"\n  速度提升: {pytorch_time/inference_time:.2f}x")
        
        # 精度验证
        accuracy_results = compare_outputs(
            pytorch_output, inference_output, output_names, engine_type
        )
        
        # 保存结果
        save_results(
            export_dir, args, pytorch_time, inference_time, 
            accuracy_results, engine_type, tensorrt_status
        )
    
    # 最终总结
    print(f"\n{'='*70}")
    print("转换完成!")
    print(f"{'='*70}")
    print(f"\n生成的文件:")
    print(f"  ONNX模型: {onnx_path}")
    if tensorrt_success:
        print(f"  TensorRT引擎: {engine_path}")
    else:
        print(f"  TensorRT引擎: 未生成（转换失败或跳过）")
    if not args.skip_validation:
        print(f"  结果报告: {export_dir / 'dpa_onnx_results.json'}")
    
    # 显示推荐
    print(f"\n推理引擎选择:")
    if tensorrt_success:
        print(f"  ✓ 推荐使用: TensorRT (已成功转换)")
    else:
        print(f"  ✓ 推荐使用: ONNX Runtime (TensorRT不支持)")
        print(f"  ℹ 原因: {tensorrt_status.get('reason', '未知')}")
    
    print(f"\n✓ 所有文件已保存到: {export_dir}")


if __name__ == '__main__':
    main()
