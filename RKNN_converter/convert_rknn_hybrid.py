#!/usr/bin/env python
# coding: utf-8
"""
EdgeTAM ONNX 模型转换为 RKNN 格式 - 混合量化版本

根据精度分析结果，encoder 的后半部分在量化后精度损失严重：
- stages_2/blocks.9 到 blocks.13 的 COS 下降到 0.78-0.95
- neck 部分的 COS 下降到 0.85-0.96

解决方案：
1. 使用混合量化，将精度损失严重的层保持为 FP16
2. 或者整个模型使用 FP16（不量化）

对于 EdgeTAM 这类视觉 Transformer 模型，建议：
- encoder: 使用 FP16（量化损失太大）
- decoder: 可以尝试量化或 FP16
- memory_encoder: 可以尝试量化或 FP16

使用方法:
    # 全部使用 FP16（推荐，精度最好）
    python convert_rknn_hybrid.py edgetam --mode fp16
    
    # 使用混合量化（平衡精度和速度）
    python convert_rknn_hybrid.py edgetam --mode hybrid
    
    # 全部量化（速度最快，但精度差）
    python convert_rknn_hybrid.py edgetam --mode quantize
"""

import datetime
import argparse
from rknn.api import RKNN
from sys import exit
import os
import onnxslim


# 根据精度分析结果，这些层在量化后精度损失严重，需要保持 FP16
# COS < 0.95 的层应该用 FP16
ENCODER_HYBRID_LAYERS = """
# stages_2 后半部分 - 精度损失严重
/image_encoder/trunk/body/stages_2/blocks/blocks.9/*
/image_encoder/trunk/body/stages_2/blocks/blocks.10/*
/image_encoder/trunk/body/stages_2/blocks/blocks.11/*
/image_encoder/trunk/body/stages_2/blocks/blocks.12/*
/image_encoder/trunk/body/stages_2/blocks/blocks.13/*
# stages_3 - 部分层精度下降
/image_encoder/trunk/body/stages_3/downsample/*
/image_encoder/trunk/body/stages_3/blocks/*
# neck 部分 - 精度下降严重
/image_encoder/neck/*
/conv_s0/*
/conv_s1/*
"""


def create_hybrid_config(model_part: str, output_dir: str) -> str:
    """创建混合量化配置文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_part == 'encoder':
        config_content = ENCODER_HYBRID_LAYERS
    else:
        # decoder 和 memory_encoder 暂时不需要混合量化配置
        return None
    
    config_path = os.path.join(output_dir, f'{model_part}_hybrid_config.txt')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


def convert_to_rknn(onnx_model: str, model_part: str, output_rknn: str = None,
                    dataset: str = None, mode: str = 'fp16',
                    calibration_dir: str = None, target_platform: str = 'rk3576',
                    hybrid_config: str = None):
    """
    转换单个 ONNX 模型到 RKNN 格式
    
    Args:
        onnx_model: ONNX 模型路径
        model_part: 模型类型 ("encoder", "decoder", "memory_encoder")
        output_rknn: 输出 RKNN 文件路径
        dataset: 量化校准数据集路径
        mode: 转换模式 ("fp16", "quantize", "hybrid")
        calibration_dir: 校准数据目录
        target_platform: 目标平台
        hybrid_config: 混合量化配置文件路径
    """
    # 自动查找校准数据
    if mode in ('quantize', 'hybrid') and dataset is None and calibration_dir is not None:
        calib_files = {
            'encoder': 'encoder_calibration_npy.txt',
            'decoder': 'decoder_calibration.txt',
            'memory_encoder': 'memory_encoder_calibration.txt',
        }
        calib_file = os.path.join(calibration_dir, calib_files.get(model_part, ''))
        if os.path.exists(calib_file):
            dataset = calib_file
            print(f"使用校准数据: {dataset}")
    
    if output_rknn is None:
        suffix = f"_{mode}" if mode != 'fp16' else ""
        output_rknn = onnx_model.replace(".onnx", f"{suffix}.rknn")
    
    timedate_iso = datetime.datetime.now().isoformat()
    
    print(f"\n{'='*60}")
    print(f"开始转换: {onnx_model}")
    print(f"输出到: {output_rknn}")
    print(f"模型类型: {model_part}")
    print(f"模式: {mode}")
    if mode == 'hybrid' and hybrid_config:
        print(f"混合量化配置: {hybrid_config}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=True)
    
    # 配置标准化参数
    if model_part == "encoder":
        mean_values = [[123.675, 116.28, 103.53]]
        std_values = [[58.395, 57.12, 57.375]]
    else:
        mean_values = None
        std_values = None
    
    # 禁用规则
    disable_rules = []
    if model_part == "memory_encoder":
        disable_rules = ['fuse_matmul_softmax_matmul_to_sdpa']
    elif model_part == "decoder":
        disable_rules = ['fuse_matmul_softmax_matmul_to_sdpa']
    
    # 量化设置
    do_quantization = mode in ('quantize', 'hybrid')
    
    # 混合量化级别
    # 0: 不使用混合量化
    # 1: 使用配置文件指定的层
    # 2: 自动混合量化（根据精度分析结果自动选择）
    if mode == 'hybrid':
        hybrid_level = 1 if hybrid_config else 2
    else:
        hybrid_level = 0
    
    rknn.config(
        dynamic_input=None,
        mean_values=mean_values,
        std_values=std_values,
        disable_rules=disable_rules,
        quantized_dtype='w8a8',
        quantized_algorithm='normal',  # normal 速度快
        quantized_method='channel',    # 逐通道量化精度更好
        quantized_hybrid_level=hybrid_level,
        target_platform=target_platform,
        quant_img_RGB2BGR=False,
        float_dtype='float16',
        optimization_level=3,
        custom_string=f"EdgeTAM {model_part}, mode={mode}, converted at {timedate_iso}",
        remove_weight=False,
        compress_weight=False,
        single_core_mode=False,
        model_pruning=False,
        enable_flash_attention=False,
    )

    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print(f"错误: 加载 ONNX 模型失败")
        return False
    
    # 构建模型
    build_kwargs = {
        'do_quantization': do_quantization,
        'dataset': dataset if do_quantization else None,
    }
    
    ret = rknn.build(**build_kwargs)
    if ret != 0:
        print(f"错误: 构建 RKNN 模型失败")
        return False
        
    ret = rknn.export_rknn(output_rknn)
    if ret != 0:
        print(f"错误: 导出 RKNN 模型失败")
        return False
    
    rknn.release()
    print(f"\n✓ 完成转换: {output_rknn}\n")
    return True


def convert_with_slim(onnx_model: str, model_part: str, output_rknn: str = None,
                      mode: str = 'fp16', calibration_dir: str = None,
                      target_platform: str = 'rk3576', hybrid_config: str = None):
    """先用 onnxslim 优化 ONNX 模型，然后转换为 RKNN"""
    if output_rknn is None:
        suffix = f"_{mode}" if mode != 'fp16' else ""
        output_rknn = onnx_model.replace(".onnx", f"{suffix}.rknn")
    
    slim_onnx = onnx_model.replace(".onnx", "_slim.onnx")
    
    print(f"\n正在优化 ONNX 模型: {onnx_model} -> {slim_onnx}")
    try:
        onnxslim.slim(onnx_model, output_model=slim_onnx, 
                      skip_fusion_patterns=["EliminationSlice"])
    except Exception as e:
        print(f"警告: onnxslim 优化失败: {e}")
        print("尝试直接转换...")
        return convert_to_rknn(onnx_model, model_part, output_rknn,
                               mode=mode, calibration_dir=calibration_dir,
                               target_platform=target_platform,
                               hybrid_config=hybrid_config)
    
    success = convert_to_rknn(slim_onnx, model_part, output_rknn,
                              mode=mode, calibration_dir=calibration_dir,
                              target_platform=target_platform,
                              hybrid_config=hybrid_config)
    
    if os.path.exists(slim_onnx):
        os.remove(slim_onnx)
        print(f"已删除临时文件: {slim_onnx}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='转换 EdgeTAM ONNX 模型到 RKNN 格式 (混合量化版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  fp16:     全部使用 FP16，精度最好，速度较慢
  quantize: 全部使用 INT8 量化，速度最快，但精度差
  hybrid:   混合量化，精度损失大的层用 FP16，其他用 INT8

推荐配置:
  - 追求精度: --mode fp16
  - 追求速度: --mode quantize (可能需要微调)
  - 平衡方案: --mode hybrid

示例:
  python convert_rknn_hybrid.py edgetam --mode fp16
  python convert_rknn_hybrid.py edgetam --mode fp16 --only encoder
  python convert_rknn_hybrid.py edgetam --mode hybrid --calibration_dir calibration_data
        """
    )
    
    parser.add_argument('model_name', type=str,
                        help='模型名称前缀 (如 edgetam)')
    parser.add_argument('--input_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录 (默认: onnx_models)')
    parser.add_argument('--output_dir', type=str, default='rknn_models',
                        help='RKNN 模型输出目录 (默认: rknn_models)')
    parser.add_argument('--only', type=str, 
                        choices=['encoder', 'decoder', 'memory_encoder'],
                        help='只转换指定的模型')
    parser.add_argument('--mode', type=str, default='fp16',
                        choices=['fp16', 'quantize', 'hybrid'],
                        help='转换模式 (默认: fp16)')
    parser.add_argument('--calibration_dir', type=str, default='calibration_data',
                        help='校准数据目录 (默认: calibration_data)')
    parser.add_argument('--target', type=str, default='rk3576',
                        choices=['rk3576', 'rk3588'],
                        help='目标平台 (默认: rk3576)')
    parser.add_argument('--no-slim', action='store_true',
                        help='跳过 onnxslim 优化')
    
    args = parser.parse_args()
    
    # 检查模式配置
    if args.mode in ('quantize', 'hybrid'):
        if not os.path.exists(args.calibration_dir):
            print(f"警告: 校准数据目录不存在: {args.calibration_dir}")
            if args.mode == 'quantize':
                print("量化模式需要校准数据，请先运行 prepare_calibration_data.py")
                exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义模型
    models = {
        'encoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_encoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_encoder.rknn"),
            'need_slim': True,
        },
        'decoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_decoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_decoder.rknn"),
            'need_slim': False,
        },
        'memory_encoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_memory_encoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_memory_encoder.rknn"),
            'need_slim': False,
        },
    }
    
    # 确定要转换的模型
    if args.only:
        models_to_convert = [args.only]
    else:
        models_to_convert = ['encoder', 'decoder', 'memory_encoder']
    
    # 检查文件
    for model_part in models_to_convert:
        onnx_path = models[model_part]['onnx']
        if not os.path.exists(onnx_path):
            print(f"错误: 找不到 {onnx_path}")
            exit(1)
    
    # 创建混合量化配置
    hybrid_configs = {}
    if args.mode == 'hybrid':
        for model_part in models_to_convert:
            config = create_hybrid_config(model_part, args.output_dir)
            if config:
                hybrid_configs[model_part] = config
    
    # 转换模型
    results = {}
    for model_part in models_to_convert:
        onnx_path = models[model_part]['onnx']
        rknn_path = models[model_part]['rknn']
        need_slim = models[model_part]['need_slim'] and not args.no_slim
        hybrid_config = hybrid_configs.get(model_part)
        
        print(f"\n{'#'*60}")
        print(f"# 转换 {model_part.upper()} (模式: {args.mode})")
        print(f"{'#'*60}")
        
        if need_slim:
            success = convert_with_slim(
                onnx_path, model_part, rknn_path,
                mode=args.mode,
                calibration_dir=args.calibration_dir,
                target_platform=args.target,
                hybrid_config=hybrid_config
            )
        else:
            success = convert_to_rknn(
                onnx_path, model_part, rknn_path,
                mode=args.mode,
                calibration_dir=args.calibration_dir,
                target_platform=args.target,
                hybrid_config=hybrid_config
            )
        
        results[model_part] = success
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"转换结果摘要 (模式: {args.mode}):")
    print(f"{'='*60}")
    for model_part, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        rknn_path = models[model_part]['rknn']
        print(f"  {model_part:20s}: {status}  -> {rknn_path}")
    
    all_success = all(results.values())
    if all_success:
        print(f"\n✓ 所有模型转换完成!")
        
        if args.mode == 'fp16':
            print("\n提示: FP16 模式精度最好，但推理速度较慢")
            print("如果需要更快的速度，可以尝试:")
            print(f"  python convert_rknn_hybrid.py {args.model_name} --mode hybrid")
    else:
        print(f"\n✗ 部分模型转换失败")
        exit(1)


if __name__ == "__main__":
    main()
