#!/usr/bin/env python
"""
临时脚本：绕过 CUDA 版本检查来编译 MultiScaleDeformableAttention
"""
import sys
import torch.utils.cpp_extension

# 保存原始的 _check_cuda_version 函数
original_check_cuda_version = torch.utils.cpp_extension._check_cuda_version

# 创建一个不做检查的版本
def patched_check_cuda_version(compiler_name, compiler_version):
    print(f"WARNING: Bypassing CUDA version check for {compiler_name} {compiler_version}")
    return

# 替换函数
torch.utils.cpp_extension._check_cuda_version = patched_check_cuda_version

# 现在导入并运行 setup
import setup as setup_module

print("Successfully compiled with patched CUDA version check")

