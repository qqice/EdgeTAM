# -*- coding: utf-8 -*-
"""
ONNX 到 RKNN 转换 - Decoder 特殊版本

将 Attention 相关操作指定到 CPU 执行，避免 NPU 精度问题。

使用方法:
    python convert_decoder_cpu_attention.py

需要在 x86 Linux 服务器上运行 (需要完整的 rknn-toolkit2)

作者: EdgeTAM RKNN Debug
日期: 2024-12-03
"""

import os
import sys
import argparse
from pathlib import Path

# 检查 RKNN Toolkit 是否可用
try:
    from rknn.api import RKNN
    print(f"RKNN Toolkit 版本: 已加载")
except ImportError:
    print("错误: 未找到 rknn-toolkit2")
    print("请在 x86 服务器上安装: pip install rknn-toolkit2")
    sys.exit(1)


def convert_decoder_with_cpu_ops(
    onnx_path: str,
    rknn_path: str,
    target_platform: str = "rk3576",
    cpu_ops: list = None,
):
    """
    转换 Decoder，将指定操作放到 CPU 执行
    
    Args:
        onnx_path: ONNX 模型路径
        rknn_path: 输出 RKNN 模型路径
        target_platform: 目标平台
        cpu_ops: 需要在 CPU 上执行的操作类型列表
    """
    
    print(f"\n{'='*60}")
    print(f"转换 Decoder (CPU Attention 版本)")
    print(f"{'='*60}")
    
    if cpu_ops is None:
        # 默认将 Attention 核心操作放到 CPU
        cpu_ops = ['MatMul', 'Softmax']
    
    rknn = RKNN(verbose=True)
    
    # ==================== 配置 ====================
    print("\n[1/4] 配置 RKNN...")
    
    # 构建 op_target 字典
    op_target = {op: 'cpu' for op in cpu_ops}
    
    config_params = {
        'target_platform': target_platform,
        'float_dtype': 'float32',
        'optimization_level': 0,
        'op_target': op_target,
        'disable_rules': [
            'fuse_matmul_softmax_matmul_to_sdpa',
        ],
    }
    
    print(f"  目标平台: {target_platform}")
    print(f"  精度: float32")
    print(f"  优化级别: 0")
    print(f"  CPU 操作: {cpu_ops}")
    
    ret = rknn.config(**config_params)
    if ret != 0:
        print(f"错误: 配置失败, 返回码 {ret}")
        return False
    
    # ==================== 加载模型 ====================
    print("\n[2/4] 加载 ONNX 模型...")
    
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"错误: 加载 ONNX 失败, 返回码 {ret}")
        return False
    
    print(f"  成功加载: {onnx_path}")
    
    # ==================== 构建模型 ====================
    print("\n[3/4] 构建 RKNN 模型 (不量化)...")
    
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"错误: 构建失败, 返回码 {ret}")
        return False
    
    print("  构建成功!")
    
    # ==================== 导出模型 ====================
    print("\n[4/4] 导出 RKNN 模型...")
    
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"错误: 导出失败, 返回码 {ret}")
        return False
    
    file_size = os.path.getsize(rknn_path) / 1024 / 1024
    print(f"  导出成功: {rknn_path}")
    print(f"  文件大小: {file_size:.2f} MB")
    
    rknn.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Decoder RKNN 转换 - CPU Attention 版本")
    
    parser.add_argument("--onnx_path", type=str, 
                        default="./onnx_models/edgetam_decoder.onnx",
                        help="Decoder ONNX 模型路径")
    parser.add_argument("--output_path", type=str, 
                        default="./rknn_models/no_opt/edgetam_decoder_cpu_attn.rknn",
                        help="输出 RKNN 模型路径")
    parser.add_argument("--target", type=str, default="rk3576",
                        help="目标平台: rk3576, rk3588, etc.")
    parser.add_argument("--cpu_ops", type=str, nargs='+',
                        default=['MatMul', 'Softmax'],
                        help="在 CPU 上执行的操作类型")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Decoder RKNN 转换 - CPU Attention 版本")
    print("=" * 60)
    print(f"\nONNX: {args.onnx_path}")
    print(f"输出: {args.output_path}")
    print(f"目标平台: {args.target}")
    print(f"CPU 操作: {args.cpu_ops}")
    
    # 创建输出目录
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(args.onnx_path):
        print(f"\n错误: 找不到 {args.onnx_path}")
        return
    
    success = convert_decoder_with_cpu_ops(
        onnx_path=args.onnx_path,
        rknn_path=args.output_path,
        target_platform=args.target,
        cpu_ops=args.cpu_ops,
    )
    
    if success:
        print("\n" + "=" * 60)
        print("转换成功!")
        print("=" * 60)
        print("\n使用说明:")
        print("  1. 将 .rknn 文件复制到开发板")
        print("  2. 使用 test_rknn_no_opt.py 测试精度")
        print("\n注意:")
        print("  - MatMul 和 Softmax 在 CPU 执行会降低性能")
        print("  - 但可以保证 Attention 计算的精度")
    else:
        print("\n转换失败!")


if __name__ == "__main__":
    main()
