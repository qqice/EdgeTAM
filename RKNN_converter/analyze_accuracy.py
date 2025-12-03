#!/usr/bin/env python
# coding: utf-8
"""
RKNN 模型精度分析工具

按照 RKNN SDK 文档建议的分阶段排查流程:
1. 模拟器 FP16 精度 - 确认浮点精度是否正常
2. 模拟器量化精度 - 检查量化后精度下降情况
3. Runtime (板端) 精度 - 连板对比

分析结果将保存在 output_dir 中，包含:
- snapshot_*.csv: 每层的 COS/Euc 指标
- 各层的中间结果对比

使用方法:
    # 分析 encoder
    python analyze_accuracy.py edgetam --model encoder --stage fp16
    python analyze_accuracy.py edgetam --model encoder --stage quantize
    
    # 分析所有模型
    python analyze_accuracy.py edgetam --stage fp16
    
    # 尝试不同的量化算法
    python analyze_accuracy.py edgetam --model encoder --stage quantize --quant_algo mmse
"""

import argparse
import os
import datetime
import numpy as np
from PIL import Image
import cv2
import glob
import shutil

try:
    from rknn.api import RKNN
except ImportError:
    print("错误: 请安装 rknn-toolkit2")
    print("pip install rknn-toolkit2")
    exit(1)


# ImageNet 标准化参数
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def prepare_encoder_input(image_path: str) -> np.ndarray:
    """准备 encoder 输入 - 返回 uint8 图像 [1, 3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
    img_np = np.array(img).astype(np.uint8)  # [H, W, 3]
    img_np = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
    img_np = np.expand_dims(img_np, axis=0)  # [1, 3, H, W]
    return img_np


def prepare_decoder_input(calib_dir: str) -> list:
    """准备 decoder 输入 - 从校准数据加载"""
    files = [
        'image_embed_0000.npy',
        'high_res_feats_0_0000.npy',
        'high_res_feats_1_0000.npy',
        'point_coords_0000.npy',
        'point_labels_0000.npy',
        'mask_input_0000.npy',
        'has_mask_input_0000.npy',
    ]
    
    inputs = []
    for f in files:
        path = os.path.join(calib_dir, 'decoder_calibration', f)
        if os.path.exists(path):
            data = np.load(path)
            inputs.append(data)
        else:
            print(f"警告: 找不到 {path}")
            return None
    return inputs


def prepare_memory_encoder_input(calib_dir: str) -> list:
    """准备 memory_encoder 输入 - 从校准数据加载"""
    files = [
        'pix_feat_0000.npy',
        'pred_mask_0000.npy',
        # 注意: ONNX 模型可能只有 2 个输入
    ]
    
    inputs = []
    for f in files:
        path = os.path.join(calib_dir, 'memory_encoder_calibration', f)
        if os.path.exists(path):
            data = np.load(path)
            inputs.append(data)
        else:
            print(f"警告: 找不到 {path}")
            return None
    return inputs


def analyze_model(onnx_path: str, model_part: str, output_dir: str,
                  stage: str = 'fp16', quant_algo: str = 'normal',
                  quant_method: str = 'channel', dataset: str = None,
                  input_data=None, target_platform: str = 'rk3576',
                  target: str = None):
    """
    分析单个模型的精度
    
    Args:
        onnx_path: ONNX 模型路径
        model_part: 模型类型 (encoder/decoder/memory_encoder)
        output_dir: 分析结果输出目录
        stage: 分析阶段 (fp16/quantize)
        quant_algo: 量化算法 (normal/kl_divergence/mmse)
        quant_method: 量化方式 (layer/channel)
        dataset: 量化校准数据集
        input_data: 用于精度分析的输入数据 (numpy 数组或文件路径列表)
        target_platform: 目标平台
        target: 连板目标 (None 表示使用模拟器)
    """
    print(f"\n{'='*60}")
    print(f"精度分析: {model_part}")
    print(f"阶段: {stage}")
    print(f"ONNX: {onnx_path}")
    print(f"输出目录: {output_dir}")
    if stage == 'quantize':
        print(f"量化算法: {quant_algo}")
        print(f"量化方式: {quant_method}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    rknn = RKNN(verbose=True)
    
    # 配置
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
    
    rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        disable_rules=disable_rules,
        quantized_dtype='w8a8',
        quantized_algorithm=quant_algo,
        quantized_method=quant_method,
        target_platform=target_platform,
        quant_img_RGB2BGR=False,
        float_dtype='float16',
        optimization_level=3,
    )
    
    # 加载模型
    print("\n加载 ONNX 模型...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("错误: 加载 ONNX 失败")
        rknn.release()
        return False
    
    # 构建模型
    do_quantization = (stage == 'quantize')
    print(f"\n构建模型 (量化={do_quantization})...")
    
    if do_quantization and dataset is None:
        print("警告: 量化模式但没有提供 dataset，将使用默认校准")
    
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
    if ret != 0:
        print("错误: 构建模型失败")
        rknn.release()
        return False
    
    # 精度分析
    print("\n开始精度分析...")
    
    # 准备输入
    if input_data is None:
        print("错误: 未提供输入数据")
        rknn.release()
        return False
    
    # 如果是文件路径列表，直接传递
    if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
        inputs_for_analysis = input_data
    # 如果是 numpy 数组列表，需要保存为临时文件
    elif isinstance(input_data, list) and all(isinstance(x, np.ndarray) for x in input_data):
        # 保存为临时 npy 文件
        temp_dir = os.path.join(output_dir, 'temp_inputs')
        os.makedirs(temp_dir, exist_ok=True)
        inputs_for_analysis = []
        for i, arr in enumerate(input_data):
            temp_path = os.path.join(temp_dir, f'input_{i}.npy')
            np.save(temp_path, arr)
            inputs_for_analysis.append(temp_path)
    else:
        inputs_for_analysis = input_data
    
    try:
        ret = rknn.accuracy_analysis(
            inputs=inputs_for_analysis,
            output_dir=output_dir,
            target=target,  # None = 模拟器, 'rk3576' = 板端
        )
        if ret != 0:
            print("警告: 精度分析返回非零状态")
    except Exception as e:
        print(f"精度分析异常: {e}")
        rknn.release()
        return False
    
    rknn.release()
    
    # 解析结果
    print(f"\n分析完成! 结果保存在: {output_dir}")
    
    # 查找并显示 snapshot 文件
    snapshot_files = glob.glob(os.path.join(output_dir, 'snapshot*.csv'))
    if snapshot_files:
        print("\n" + "="*60)
        print("精度分析摘要:")
        print("="*60)
        
        for snapshot_file in snapshot_files:
            print(f"\n文件: {os.path.basename(snapshot_file)}")
            parse_snapshot_csv(snapshot_file)
    
    return True


def parse_snapshot_csv(csv_path: str, show_top_n: int = 20):
    """解析 snapshot CSV 文件并显示精度最差的层"""
    try:
        import csv
        
        layers = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer_name = row.get('layer_name', row.get('name', 'unknown'))
                cos = row.get('cos', row.get('COS', '1.0'))
                euc = row.get('euc', row.get('Euc', '0.0'))
                
                try:
                    cos_val = float(cos) if cos else 1.0
                    euc_val = float(euc) if euc else 0.0
                except:
                    cos_val = 1.0
                    euc_val = 0.0
                
                layers.append({
                    'name': layer_name,
                    'cos': cos_val,
                    'euc': euc_val,
                })
        
        # 按 COS 排序 (升序，找出最差的层)
        layers_sorted = sorted(layers, key=lambda x: x['cos'])
        
        print(f"\n精度最差的 {min(show_top_n, len(layers_sorted))} 层 (按 COS 升序):")
        print("-" * 70)
        print(f"{'层名称':50s} {'COS':>10s} {'Euc':>10s}")
        print("-" * 70)
        
        bad_layers = 0
        for layer in layers_sorted[:show_top_n]:
            cos = layer['cos']
            # 标记 COS < 0.98 的层
            marker = "❌" if cos < 0.98 else ("⚠️" if cos < 0.99 else "")
            print(f"{layer['name'][:50]:50s} {cos:10.6f} {layer['euc']:10.4f} {marker}")
            if cos < 0.98:
                bad_layers += 1
        
        print("-" * 70)
        
        # 统计
        total = len(layers)
        cos_below_98 = sum(1 for l in layers if l['cos'] < 0.98)
        cos_below_99 = sum(1 for l in layers if l['cos'] < 0.99)
        min_cos = min(l['cos'] for l in layers) if layers else 1.0
        avg_cos = sum(l['cos'] for l in layers) / len(layers) if layers else 1.0
        
        print(f"\n统计:")
        print(f"  总层数: {total}")
        print(f"  COS < 0.99 的层: {cos_below_99} ({100*cos_below_99/total:.1f}%)")
        print(f"  COS < 0.98 的层: {cos_below_98} ({100*cos_below_98/total:.1f}%)")
        print(f"  最小 COS: {min_cos:.6f}")
        print(f"  平均 COS: {avg_cos:.6f}")
        
        if min_cos < 0.95:
            print("\n⚠️ 警告: 存在严重精度损失 (COS < 0.95)")
            print("  建议: 考虑使用混合量化或调整量化算法")
        elif min_cos < 0.98:
            print("\n⚠️ 注意: 存在明显精度损失 (COS < 0.98)")
            print("  建议: 尝试 mmse 量化算法或 channel 量化方式")
        
    except Exception as e:
        print(f"解析 CSV 失败: {e}")


def find_test_image(search_dirs: list) -> str:
    """查找测试图像"""
    for dir_path in search_dirs:
        if os.path.isdir(dir_path):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                files = glob.glob(os.path.join(dir_path, ext))
                if files:
                    return files[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='RKNN 模型精度分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 第一阶段: FP16 精度分析 (检查模型本身是否有问题)
  python analyze_accuracy.py edgetam --model encoder --stage fp16
  
  # 第二阶段: 量化精度分析 (默认 normal 算法)
  python analyze_accuracy.py edgetam --model encoder --stage quantize
  
  # 尝试不同的量化算法
  python analyze_accuracy.py edgetam --model encoder --stage quantize --quant_algo mmse
  python analyze_accuracy.py edgetam --model encoder --stage quantize --quant_algo kl_divergence
  
  # 使用逐通道量化
  python analyze_accuracy.py edgetam --model encoder --stage quantize --quant_method channel
  
  # 分析所有模型
  python analyze_accuracy.py edgetam --stage fp16

分析指标说明:
  COS (余弦相似度): 越接近 1.0 越好，< 0.98 表示有明显误差
  Euc (欧氏距离): 越接近 0 越好
        """
    )
    
    parser.add_argument('model_name', type=str,
                        help='模型名称前缀 (如 edgetam)')
    parser.add_argument('--model', type=str, 
                        choices=['encoder', 'decoder', 'memory_encoder', 'all'],
                        default='all',
                        help='要分析的模型 (默认: all)')
    parser.add_argument('--stage', type=str,
                        choices=['fp16', 'quantize'],
                        default='fp16',
                        help='分析阶段: fp16=浮点, quantize=量化 (默认: fp16)')
    parser.add_argument('--quant_algo', type=str,
                        choices=['normal', 'kl_divergence', 'mmse'],
                        default='normal',
                        help='量化算法 (默认: normal)')
    parser.add_argument('--quant_method', type=str,
                        choices=['layer', 'channel'],
                        default='channel',
                        help='量化方式 (默认: channel)')
    parser.add_argument('--input_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录 (默认: onnx_models)')
    parser.add_argument('--output_dir', type=str, default='accuracy_analysis',
                        help='分析结果输出目录 (默认: accuracy_analysis)')
    parser.add_argument('--calibration_dir', type=str, default='calibration_data',
                        help='校准数据目录 (默认: calibration_data)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='用于 encoder 分析的测试图像')
    parser.add_argument('--target', type=str, default='rk3576',
                        help='目标平台 (默认: rk3576)')
    parser.add_argument('--board', action='store_true',
                        help='连板分析 (需要连接开发板)')
    
    args = parser.parse_args()
    
    # 确定要分析的模型
    if args.model == 'all':
        models = ['encoder', 'decoder', 'memory_encoder']
    else:
        models = [args.model]
    
    # 查找测试图像
    test_image = args.test_image
    if test_image is None:
        search_dirs = [
            '../notebooks/images',
            '../examples',
            'calibration_data',
        ]
        test_image = find_test_image(search_dirs)
        if test_image:
            print(f"使用测试图像: {test_image}")
    
    # 分析每个模型
    results = {}
    for model_part in models:
        onnx_path = os.path.join(args.input_dir, f"{args.model_name}_{model_part}.onnx")
        
        if not os.path.exists(onnx_path):
            print(f"跳过 {model_part}: 找不到 {onnx_path}")
            continue
        
        # 设置输出目录
        output_subdir = os.path.join(
            args.output_dir, 
            f"{model_part}_{args.stage}_{args.quant_algo if args.stage == 'quantize' else 'fp16'}"
        )
        
        # 准备输入数据
        input_data = None
        dataset = None
        
        if model_part == 'encoder':
            # encoder 使用图像
            if test_image and os.path.exists(test_image):
                input_data = [test_image]
            else:
                # 使用 npy 校准数据
                calib_txt = os.path.join(args.calibration_dir, 'encoder_calibration_npy.txt')
                if os.path.exists(calib_txt):
                    with open(calib_txt, 'r') as f:
                        lines = f.readlines()
                    if lines:
                        input_data = [lines[0].strip()]
            
            # 量化数据集
            if args.stage == 'quantize':
                calib_txt = os.path.join(args.calibration_dir, 'encoder_calibration_npy.txt')
                if os.path.exists(calib_txt):
                    dataset = calib_txt
                    
        elif model_part == 'decoder':
            # decoder 使用预先准备的输入
            input_data = prepare_decoder_input(args.calibration_dir)
            
            if args.stage == 'quantize':
                calib_txt = os.path.join(args.calibration_dir, 'decoder_calibration.txt')
                if os.path.exists(calib_txt):
                    dataset = calib_txt
                    
        elif model_part == 'memory_encoder':
            # memory_encoder 使用预先准备的输入
            input_data = prepare_memory_encoder_input(args.calibration_dir)
            
            if args.stage == 'quantize':
                calib_txt = os.path.join(args.calibration_dir, 'memory_encoder_calibration.txt')
                if os.path.exists(calib_txt):
                    dataset = calib_txt
        
        if input_data is None:
            print(f"跳过 {model_part}: 无法准备输入数据")
            continue
        
        # 运行分析
        target = args.target if args.board else None
        success = analyze_model(
            onnx_path=onnx_path,
            model_part=model_part,
            output_dir=output_subdir,
            stage=args.stage,
            quant_algo=args.quant_algo,
            quant_method=args.quant_method,
            dataset=dataset,
            input_data=input_data,
            target_platform=args.target,
            target=target,
        )
        
        results[model_part] = success
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("分析摘要:")
    print(f"{'='*60}")
    for model_part, success in results.items():
        status = "✓ 完成" if success else "✗ 失败"
        print(f"  {model_part:20s}: {status}")
    
    if all(results.values()):
        print(f"\n分析结果保存在: {args.output_dir}/")
        print("\n下一步:")
        if args.stage == 'fp16':
            print("  1. 检查上面的 COS 值，确保 FP16 精度正常 (COS > 0.99)")
            print("  2. 如果 FP16 正常，运行量化分析:")
            print(f"     python analyze_accuracy.py {args.model_name} --stage quantize")
        else:
            print("  1. 检查上面的 COS 值")
            print("  2. 如果精度差，尝试:")
            print(f"     python analyze_accuracy.py {args.model_name} --stage quantize --quant_algo mmse")
            print(f"     python analyze_accuracy.py {args.model_name} --stage quantize --quant_algo kl_divergence")


if __name__ == "__main__":
    main()
