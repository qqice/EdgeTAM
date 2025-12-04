# -*- coding: utf-8 -*-
"""
ONNX 到 RKNN 简单转换脚本 (无优化)

此脚本用于将 EdgeTAM 的 ONNX 模型转换为 RKNN 格式。
设计目标: 最大精度，禁用所有优化。

使用方法:
    python convert_onnx_to_rknn_simple.py

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
    print(f"RKNN Toolkit 版本: {RKNN().get_sdk_version()}")
except ImportError:
    print("错误: 未找到 rknn-toolkit2")
    print("请在 x86 服务器上安装: pip install rknn-toolkit2")
    sys.exit(1)


def convert_model(
    onnx_path: str,
    rknn_path: str,
    target_platform: str = "rk3576",
    use_float32: bool = True,
    opt_level: int = 0,
    mean_values: list = None,
    std_values: list = None,
):
    """
    将单个 ONNX 模型转换为 RKNN
    
    Args:
        onnx_path: ONNX 模型路径
        rknn_path: 输出 RKNN 模型路径
        target_platform: 目标平台 (rk3576, rk3588, etc.)
        use_float32: 使用 float32 而非 float16
        opt_level: 优化级别 (0=无优化, 1=基础优化, 2=中等优化, 3=最大优化)
        mean_values: 归一化均值 (仅用于 encoder)
        std_values: 归一化标准差 (仅用于 encoder)
    """
    
    print(f"\n{'='*60}")
    print(f"转换: {os.path.basename(onnx_path)}")
    print(f"{'='*60}")
    
    rknn = RKNN(verbose=True)
    
    # ==================== 配置 ====================
    print("\n[1/4] 配置 RKNN...")
    
    # 禁用所有可能影响精度的优化
    disable_rules = [
        'fuse_matmul_softmax_matmul_to_sdpa',      # 禁用 SDPA 融合
        'convert_matmul_gemm',                     # 禁用 MatMul 到 GEMM 转换
        'fuse_layer_norm',                         # 禁用 LayerNorm 融合
        'fuse_layer_norm_v2',                      # 禁用 LayerNorm v2 融合
        'fuse_gelu',                               # 禁用 GELU 融合
    ]
    
    config_params = {
        'target_platform': target_platform,
        'optimization_level': opt_level,
        'disable_rules': disable_rules,
    }
    
    # 设置精度
    if use_float32:
        config_params['float_dtype'] = 'float32'
        print(f"  精度: float32")
    else:
        config_params['float_dtype'] = 'float16'
        print(f"  精度: float16")
    
    print(f"  目标平台: {target_platform}")
    print(f"  优化级别: {opt_level}")
    print(f"  禁用规则: {disable_rules}")
    
    # 如果提供了归一化参数 (仅用于 encoder)
    if mean_values and std_values:
        config_params['mean_values'] = [mean_values]
        config_params['std_values'] = [std_values]
        print(f"  均值: {mean_values}")
        print(f"  标准差: {std_values}")
    
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
    
    # 不使用量化
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
    parser = argparse.ArgumentParser(description="ONNX 到 RKNN 简单转换 (无优化)")
    
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models",
                        help="ONNX 模型目录")
    parser.add_argument("--output_dir", type=str, default="./rknn_models/no_opt",
                        help="输出 RKNN 模型目录")
    parser.add_argument("--target", type=str, default="rk3576",
                        help="目标平台: rk3576, rk3588, rk3568, etc.")
    parser.add_argument("--float32", action="store_true", default=True,
                        help="使用 float32 (默认)")
    parser.add_argument("--float16", action="store_true",
                        help="使用 float16")
    parser.add_argument("--opt_level", type=int, default=0, choices=[0, 1, 2, 3],
                        help="优化级别: 0=无优化(默认), 1=基础, 2=中等, 3=最大")
    parser.add_argument("--encoder_only", action="store_true",
                        help="仅转换 encoder")
    parser.add_argument("--decoder_only", action="store_true",
                        help="仅转换 decoder")
    parser.add_argument("--memory_encoder_only", action="store_true",
                        help="仅转换 memory_encoder")
    
    args = parser.parse_args()
    
    # 确定精度
    use_float32 = not args.float16
    
    print("=" * 60)
    print("ONNX 到 RKNN 转换 (无优化版本)")
    print("=" * 60)
    print(f"\nONNX 目录: {args.onnx_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标平台: {args.target}")
    print(f"精度: {'float32' if use_float32 else 'float16'}")
    print(f"优化级别: {args.opt_level}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ImageNet 归一化参数 (仅用于 encoder)
    # 注意: 输入应该是 0-255 范围的 uint8
    imagenet_mean = [123.675, 116.28, 103.53]
    imagenet_std = [58.395, 57.12, 57.375]
    
    # 定义要转换的模型
    models = []
    
    if args.encoder_only or (not args.decoder_only and not args.memory_encoder_only):
        models.append({
            'name': 'encoder',
            'onnx': 'edgetam_encoder.onnx',
            'rknn': 'edgetam_encoder_no_opt.rknn',
            'mean': imagenet_mean,
            'std': imagenet_std,
        })
    
    if args.decoder_only or (not args.encoder_only and not args.memory_encoder_only):
        models.append({
            'name': 'decoder',
            'onnx': 'edgetam_decoder.onnx',
            'rknn': 'edgetam_decoder_no_opt.rknn',
            'mean': None,
            'std': None,
        })
    
    if args.memory_encoder_only or (not args.encoder_only and not args.decoder_only):
        models.append({
            'name': 'memory_encoder',
            'onnx': 'edgetam_memory_encoder.onnx',
            'rknn': 'edgetam_memory_encoder_no_opt.rknn',
            'mean': None,
            'std': None,
        })
    
    # 转换每个模型
    results = {}
    for model in models:
        onnx_path = os.path.join(args.onnx_dir, model['onnx'])
        rknn_path = os.path.join(args.output_dir, model['rknn'])
        
        if not os.path.exists(onnx_path):
            print(f"\n警告: 找不到 {onnx_path}, 跳过")
            results[model['name']] = False
            continue
        
        success = convert_model(
            onnx_path=onnx_path,
            rknn_path=rknn_path,
            target_platform=args.target,
            use_float32=use_float32,
            opt_level=args.opt_level,
            mean_values=model['mean'],
            std_values=model['std'],
        )
        results[model['name']] = success
    
    # 打印总结
    print("\n" + "=" * 60)
    print("转换结果")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    # 使用说明
    print("\n使用说明:")
    print("  1. 将转换后的 .rknn 文件复制到 RK3576 开发板")
    print("  2. 放置到: /home/radxa/EdgeTAM/RKNN_converter/rknn_models/no_opt/")
    print("  3. 运行测试脚本验证精度")
    print("\n注意:")
    print("  - 使用 float32 会增加模型大小和推理时间")
    print("  - opt_level=0 禁用所有优化，可能影响性能但保证精度")
    print("  - 如果精度仍有问题，可能是 RKNN Runtime 本身的问题")


if __name__ == "__main__":
    main()
