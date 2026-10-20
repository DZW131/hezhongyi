#!/usr/bin/env python
"""
编译 MultiScaleDeformableAttention CUDA 扩展（绕过 CUDA 版本检查）
"""
import sys
import os
import torch.utils.cpp_extension

print("Patching CUDA version check...")

# 修补 CUDA 版本检查
original_check = torch.utils.cpp_extension._check_cuda_version

def patched_check_cuda_version(compiler_name, compiler_version):
    print(f"INFO: Bypassing CUDA version check (Detected CUDA version may differ from PyTorch CUDA version)")
    return

torch.utils.cpp_extension._check_cuda_version = patched_check_cuda_version

print("Running setup.py...")

# 修改 sys.argv 来传递构建命令
sys.argv = ['setup.py', 'build', 'install']

# 导入并执行 setup
exec(open('setup.py').read())

