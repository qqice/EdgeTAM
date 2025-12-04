#!/usr/bin/env python
# coding: utf-8
"""
EdgeTAM ONNX 模型转换为 RKNN 格式

支持的模型:
1. encoder: 图像编码器 - 输入 [1, 3, 1024, 1024]，需要 ImageNet 标准化
2. decoder: Mask 解码器 - 输入 image_embed, high_res_feats, point_coords 等
3. memory_encoder: 记忆编码器 - 输入 pix_feat, pred_mask, object_score_logits

使用方法:
    python convert_rknn.py edgetam                    # 转换所有模型
    python convert_rknn.py edgetam --only encoder    # 只转换 encoder
    python convert_rknn.py edgetam --only decoder    # 只转换 decoder  
    python convert_rknn.py edgetam --only memory_encoder  # 只转换 memory_encoder
    python convert_rknn.py edgetam --quantize        # 启用量化
"""

import datetime
import argparse
from rknn.api import RKNN
from sys import exit
import os
import onnxslim


def convert_to_rknn(onnx_model, model_part, output_rknn=None, 
                    dataset=None, 
                    quantize=False,
                    calibration_dir=None,
                    target_platform='rk3576',
                    use_float32=False,
                    cpu_ops=None):
    """
    转换单个 ONNX 模型到 RKNN 格式
    
    Args:
        onnx_model: ONNX 模型路径
        model_part: 模型类型 ("encoder", "decoder", "memory_encoder")
        output_rknn: 输出 RKNN 文件路径，默认与 ONNX 同名
        dataset: 量化校准数据集路径 (直接指定)
        quantize: 是否启用量化
        calibration_dir: 校准数据目录 (由 prepare_calibration_data.py 生成)
        target_platform: 目标平台 (rk3576/rk3588，默认 rk3576)
        use_float32: 使用 float32 而非 float16 (更慢但更精确)
        cpu_ops: 强制在 CPU 执行的算子列表，如 ['Where', 'ScatterND']
    """
    # 自动查找校准数据
    if quantize and dataset is None and calibration_dir is not None:
        calib_files = {
            'encoder': 'encoder_calibration_npy.txt',
            'decoder': 'decoder_calibration.txt',
            'memory_encoder': 'memory_encoder_calibration.txt',
        }
        calib_file = os.path.join(calibration_dir, calib_files.get(model_part, ''))
        if os.path.exists(calib_file):
            dataset = calib_file
            print(f"使用校准数据: {dataset}")
        else:
            print(f"警告: 未找到 {model_part} 的校准数据 {calib_file}")
            print("将使用默认校准 (可能影响精度)")
            dataset = None
    if output_rknn is None:
        output_rknn = onnx_model.replace(".onnx", ".rknn")
    timedate_iso = datetime.datetime.now().isoformat()
    
    print(f"\n{'='*60}")
    print(f"开始转换: {onnx_model}")
    print(f"输出到: {output_rknn}")
    print(f"模型类型: {model_part}")
    print(f"量化: {quantize}")
    print(f"精度: {'float32' if use_float32 else 'float16'}")
    if cpu_ops:
        print(f"CPU 算子: {cpu_ops}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=True)
    
    # 根据模型类型配置不同的参数
    # ImageNet 标准化参数 (SAM2/EdgeTAM 使用)
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    # RKNN 的 mean_values 和 std_values 需要按像素值范围 [0, 255] 计算:
    # mean_values = mean * 255 = [123.675, 116.28, 103.53]
    # std_values = std * 255 = [58.395, 57.12, 57.375]
    
    if model_part == "encoder":
        # Encoder 接受 [0, 255] 范围的图像，使用 RKNN 内置标准化
        mean_values = [[123.675, 116.28, 103.53]]
        std_values = [[58.395, 57.12, 57.375]]
    else:
        # Decoder 和 Memory Encoder 不需要图像标准化
        mean_values = None
        std_values = None
    
    # Memory Encoder 中的 Spatial Perceiver 使用特殊的注意力模式，
    # 需要禁用某些优化规则来避免 RKNN 的 bug
    # Decoder 也需要禁用 SDPA 融合，因为包含复杂的注意力运算
    disable_rules = []
    if model_part == "memory_encoder":
        disable_rules = ['fuse_matmul_softmax_matmul_to_sdpa']
    elif model_part == "decoder":
        # Decoder 包含 SAM 的 mask decoder 和 transformer，禁用可能有问题的优化
        disable_rules = [
            'fuse_matmul_softmax_matmul_to_sdpa',  # 禁用 SDPA 融合
        ]
    
    # Decoder 使用较低的优化级别以避免精度问题
    opt_level = 2 if model_part == "decoder" else 3
    
    # 配置浮点精度
    float_dtype = 'float32' if use_float32 else 'float16'
    
    # 配置 CPU 执行的算子
    op_target_config = None
    if cpu_ops:
        op_target_config = {op: 'cpu' for op in cpu_ops}
    
    rknn.config(
        dynamic_input=None,
        mean_values=mean_values,
        std_values=std_values,
        disable_rules=disable_rules,
        quantized_dtype='w8a8',
        quantized_algorithm='normal',
        quantized_method='channel',
        quantized_hybrid_level=0,
        target_platform=target_platform,
        quant_img_RGB2BGR=False,
        float_dtype=float_dtype,
        optimization_level=opt_level,
        custom_string=f"EdgeTAM {model_part}, converted at {timedate_iso}",
        remove_weight=False,
        compress_weight=False,
        inputs_yuv_fmt=None,
        single_core_mode=False,
        model_pruning=False,
        op_target=op_target_config,
        quantize_weight=False,
        remove_reshape=False,
        sparse_infer=False,
        enable_flash_attention=False,
    )

    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print(f"错误: 加载 ONNX 模型失败")
        return False
        
    ret = rknn.build(do_quantization=quantize, dataset=dataset, rknn_batch_size=None)
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


def convert_with_slim(onnx_model, model_part, output_rknn=None, quantize=False, 
                      calibration_dir=None, target_platform='rk3576',
                      use_float32=False, cpu_ops=None):
    """
    先用 onnxslim 优化 ONNX 模型，然后转换为 RKNN
    
    某些模型需要先优化才能成功转换
    """
    if output_rknn is None:
        output_rknn = onnx_model.replace(".onnx", ".rknn")
    
    # 创建临时文件名
    slim_onnx = onnx_model.replace(".onnx", "_slim.onnx")
    
    print(f"\n正在优化 ONNX 模型: {onnx_model} -> {slim_onnx}")
    try:
        onnxslim.slim(onnx_model, output_model=slim_onnx, skip_fusion_patterns=["EliminationSlice"])
    except Exception as e:
        print(f"警告: onnxslim 优化失败: {e}")
        print("尝试直接转换...")
        return convert_to_rknn(onnx_model, model_part, output_rknn, 
                               quantize=quantize, calibration_dir=calibration_dir,
                               target_platform=target_platform,
                               use_float32=use_float32, cpu_ops=cpu_ops)
    
    # 转换优化后的模型
    success = convert_to_rknn(slim_onnx, model_part, output_rknn, 
                              quantize=quantize, calibration_dir=calibration_dir,
                              target_platform=target_platform,
                              use_float32=use_float32, cpu_ops=cpu_ops)
    
    # 清理临时文件
    if os.path.exists(slim_onnx):
        os.remove(slim_onnx)
        print(f"已删除临时文件: {slim_onnx}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='转换 EdgeTAM/SAM2 ONNX 模型到 RKNN 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_rknn.py edgetam                      # 从 onnx_models/ 转换所有模型
  python convert_rknn.py edgetam --only encoder       # 只转换 encoder
  python convert_rknn.py edgetam --only decoder       # 只转换 decoder
  python convert_rknn.py edgetam --only memory_encoder # 只转换 memory_encoder
  python convert_rknn.py edgetam --quantize           # 启用量化
  python convert_rknn.py edgetam --only decoder --float32  # decoder 使用 float32 提高精度
  python convert_rknn.py edgetam --cpu-ops Where,ScatterND # 指定算子在 CPU 执行

模型输入输出:
  encoder:
    输入: image [1, 3, 1024, 1024] (RGB, 0-255)
    输出: high_res_feats_0 [1, 32, 256, 256]
          high_res_feats_1 [1, 64, 128, 128]
          image_embed [1, 256, 64, 64]

  decoder:
    输入: image_embed [1, 256, 64, 64]
          high_res_feats_0 [1, 32, 256, 256]
          high_res_feats_1 [1, 64, 128, 128]
          point_coords [1, N, 2]
          point_labels [1, N]
          mask_input [1, 1, 256, 256]
          has_mask_input [1]
    输出: masks [1, 4, 256, 256]
          iou_predictions [1, 4]
          obj_ptr [1, 64]
          object_score_logits [1, 1]

  memory_encoder:
    输入: pix_feat [1, 256, 64, 64]
          pred_mask [1, 1, 1024, 1024]
          object_score_logits [1, 1]
    输出: maskmem_features [1, 64, 16, 16]
          maskmem_pos_enc [1, 64, 16, 16]
        """
    )
    parser.add_argument('model_name', type=str, 
                        help='模型名称前缀，例如: edgetam (会查找 edgetam_encoder.onnx 等)')
    parser.add_argument('--input_dir', type=str, default='onnx_models',
                        help='ONNX 模型所在目录 (默认: onnx_models)')
    parser.add_argument('--output_dir', type=str, default='rknn_models',
                        help='RKNN 模型输出目录 (默认: rknn_models)')
    parser.add_argument('--only', type=str, choices=['encoder', 'decoder', 'memory_encoder'],
                        help='只转换指定的模型')
    parser.add_argument('--quantize', action='store_true',
                        help='启用 INT8 量化')
    parser.add_argument('--calibration_dir', type=str, default=None,
                        help='校准数据目录 (由 prepare_calibration_data.py 生成)')
    parser.add_argument('--target', type=str, default='rk3576',
                        choices=['rk3576', 'rk3588'],
                        help='目标平台 (默认: rk3576)')
    parser.add_argument('--no-slim', action='store_true',
                        help='跳过 onnxslim 优化步骤')
    parser.add_argument('--float32', action='store_true',
                        help='使用 float32 精度 (更慢但更精确，推荐 decoder 使用)')
    parser.add_argument('--cpu-ops', type=str, default=None,
                        help='强制在 CPU 执行的算子，逗号分隔，如 "Where,ScatterND"')
    args = parser.parse_args()
    
    # 解析 CPU 算子列表
    cpu_ops = None
    if args.cpu_ops:
        cpu_ops = [op.strip() for op in args.cpu_ops.split(',')]
        print(f"将以下算子强制在 CPU 执行: {cpu_ops}")
    
    # 检查量化配置
    if args.quantize and args.calibration_dir is None:
        print("警告: 启用量化但未指定校准数据目录")
        print("建议先运行: python prepare_calibration_data.py --image_dir <图像目录>")
        print("然后使用: python convert_rknn.py edgetam --quantize --calibration_dir calibration_data")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义模型文件
    models = {
        'encoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_encoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_encoder.rknn"),
            'need_slim': True,  # encoder 需要 onnxslim 优化
        },
        'decoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_decoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_decoder.rknn"),
            'need_slim': False,  # decoder 通常不需要
        },
        'memory_encoder': {
            'onnx': os.path.join(args.input_dir, f"{args.model_name}_memory_encoder.onnx"),
            'rknn': os.path.join(args.output_dir, f"{args.model_name}_memory_encoder.rknn"),
            'need_slim': False,  # memory_encoder 通常不需要
        },
    }
    
    # 确定要转换的模型
    if args.only:
        models_to_convert = [args.only]
    else:
        models_to_convert = ['encoder', 'decoder', 'memory_encoder']
    
    # 检查文件是否存在
    for model_part in models_to_convert:
        onnx_path = models[model_part]['onnx']
        if not os.path.exists(onnx_path):
            print(f"错误: 找不到文件 {onnx_path}")
            exit(1)
    
    # 转换模型
    results = {}
    for model_part in models_to_convert:
        onnx_path = models[model_part]['onnx']
        rknn_path = models[model_part]['rknn']
        need_slim = models[model_part]['need_slim'] and not args.no_slim
        
        print(f"\n{'#'*60}")
        print(f"# 转换 {model_part.upper()}")
        print(f"{'#'*60}")
        
        if need_slim:
            success = convert_with_slim(onnx_path, model_part, rknn_path, 
                                        quantize=args.quantize, 
                                        calibration_dir=args.calibration_dir,
                                        target_platform=args.target,
                                        use_float32=args.float32,
                                        cpu_ops=cpu_ops)
        else:
            success = convert_to_rknn(onnx_path, model_part, rknn_path, 
                                      quantize=args.quantize,
                                      calibration_dir=args.calibration_dir,
                                      target_platform=args.target,
                                      use_float32=args.float32,
                                      cpu_ops=cpu_ops)
        
        results[model_part] = success
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print("转换结果摘要:")
    print(f"{'='*60}")
    for model_part, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        rknn_path = models[model_part]['rknn']
        print(f"  {model_part:20s}: {status}  -> {rknn_path}")
    
    all_success = all(results.values())
    if all_success:
        print(f"\n✓ 所有模型转换完成!")
    else:
        print(f"\n✗ 部分模型转换失败")
        exit(1)


if __name__ == "__main__":
    main()
