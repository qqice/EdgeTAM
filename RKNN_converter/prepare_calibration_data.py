#!/usr/bin/env python
# coding: utf-8
"""
EdgeTAM 量化校准数据准备脚本

为 RKNN 量化转换准备校准数据集:
1. encoder: 图像列表文件 (txt)
2. decoder: numpy 格式的校准数据
3. memory_encoder: numpy 格式的校准数据

使用方法:
    python prepare_calibration_data.py --image_dir ../notebooks/videos/bedroom --num_samples 50
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

# 添加上级目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


IMG_SIZE = 1024


def prepare_encoder_calibration(image_dir: str, output_dir: str, num_samples: int = 50):
    """
    准备 encoder 校准数据
    
    RKNN encoder 使用内置标准化，校准数据只需要图像路径列表
    """
    print("\n" + "=" * 60)
    print("准备 Encoder 校准数据")
    print("=" * 60)
    
    # 查找所有图像
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
    image_files.extend(sorted(glob(os.path.join(image_dir, "*.png"))))
    
    if len(image_files) == 0:
        print(f"错误: 在 {image_dir} 中未找到图像")
        return None
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 选择均匀分布的样本
    if len(image_files) > num_samples:
        step = len(image_files) // num_samples
        selected = image_files[::step][:num_samples]
    else:
        selected = image_files
    
    print(f"选择 {len(selected)} 张用于校准")
    
    # 写入图像列表文件
    list_file = os.path.join(output_dir, "encoder_calibration.txt")
    with open(list_file, 'w') as f:
        for img_path in selected:
            # RKNN 需要绝对路径
            abs_path = os.path.abspath(img_path)
            f.write(abs_path + "\n")
    
    print(f"已保存图像列表: {list_file}")
    
    # 同时生成预处理后的 npy 文件（用于精确校准）
    npy_dir = os.path.join(output_dir, "encoder_npy")
    os.makedirs(npy_dir, exist_ok=True)
    
    npy_list_file = os.path.join(output_dir, "encoder_calibration_npy.txt")
    with open(npy_list_file, 'w') as f:
        for i, img_path in enumerate(tqdm(selected, desc="生成 encoder npy")):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            img_np = np.array(img).astype(np.uint8)  # [H, W, 3], uint8
            img_np = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
            img_np = np.expand_dims(img_np, axis=0)  # [1, 3, H, W]
            
            npy_path = os.path.join(npy_dir, f"sample_{i:04d}.npy")
            np.save(npy_path, img_np)
            f.write(os.path.abspath(npy_path) + "\n")
    
    print(f"已保存 npy 列表: {npy_list_file}")
    
    return list_file, selected


def prepare_decoder_calibration(image_files: list, output_dir: str, 
                                 onnx_dir: str = "onnx_models"):
    """
    准备 decoder 校准数据
    
    使用 ONNX encoder 生成 image_embed 和 high_res_feats
    """
    print("\n" + "=" * 60)
    print("准备 Decoder 校准数据")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("错误: 需要 onnxruntime")
        return None
    
    # 加载 encoder
    encoder_path = os.path.join(onnx_dir, "edgetam_encoder.onnx")
    if not os.path.exists(encoder_path):
        print(f"错误: 未找到 {encoder_path}")
        return None
    
    print(f"加载 encoder: {encoder_path}")
    encoder = ort.InferenceSession(encoder_path)
    
    # ImageNet 标准化参数
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    
    # 生成校准数据
    calibration_data = {
        'image_embed': [],
        'high_res_feats_0': [],
        'high_res_feats_1': [],
        'point_coords': [],
        'point_labels': [],
        'mask_input': [],
        'has_mask_input': [],
    }
    
    for img_path in tqdm(image_files, desc="生成 decoder 校准数据"):
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, :]  # [1, 3, H, W]
        
        # 标准化
        img_normalized = (img_np / 255.0 - mean) / std
        
        # 运行 encoder
        high_res_0, high_res_1, image_embed = encoder.run(None, {
            'image': img_normalized.astype(np.float32)
        })
        
        # 生成随机点击点 (模拟用户交互)
        # 在图像中心区域随机采样
        cx = np.random.randint(orig_w // 4, 3 * orig_w // 4)
        cy = np.random.randint(orig_h // 4, 3 * orig_h // 4)
        
        # 坐标归一化到 1024x1024
        point_x = cx * IMG_SIZE / orig_w
        point_y = cy * IMG_SIZE / orig_h
        point_coords = np.array([[[point_x, point_y]]]).astype(np.float32)
        point_labels = np.array([[1]]).astype(np.float32)
        
        # 无先前 mask
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        
        calibration_data['image_embed'].append(image_embed)
        calibration_data['high_res_feats_0'].append(high_res_0)
        calibration_data['high_res_feats_1'].append(high_res_1)
        calibration_data['point_coords'].append(point_coords)
        calibration_data['point_labels'].append(point_labels)
        calibration_data['mask_input'].append(mask_input)
        calibration_data['has_mask_input'].append(has_mask_input)
    
    # 保存校准数据
    decoder_calib_dir = os.path.join(output_dir, "decoder_calibration")
    os.makedirs(decoder_calib_dir, exist_ok=True)
    
    list_file = os.path.join(output_dir, "decoder_calibration.txt")
    with open(list_file, 'w') as f:
        for i in range(len(image_files)):
            sample_files = []
            for key in ['image_embed', 'high_res_feats_0', 'high_res_feats_1', 
                        'point_coords', 'point_labels', 'mask_input', 'has_mask_input']:
                npy_path = os.path.join(decoder_calib_dir, f"{key}_{i:04d}.npy")
                np.save(npy_path, calibration_data[key][i])
                sample_files.append(os.path.abspath(npy_path))
            f.write(" ".join(sample_files) + "\n")
    
    print(f"已保存 decoder 校准数据: {decoder_calib_dir}")
    print(f"已保存校准列表: {list_file}")
    
    return list_file


def prepare_memory_encoder_calibration(image_files: list, output_dir: str,
                                        onnx_dir: str = "onnx_models"):
    """
    准备 memory_encoder 校准数据
    
    使用 ONNX encoder + decoder 生成校准数据
    """
    print("\n" + "=" * 60)
    print("准备 Memory Encoder 校准数据")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        import cv2
    except ImportError:
        print("错误: 需要 onnxruntime 和 opencv-python")
        return None
    
    # 加载模型
    encoder_path = os.path.join(onnx_dir, "edgetam_encoder.onnx")
    decoder_path = os.path.join(onnx_dir, "edgetam_decoder.onnx")
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"错误: 未找到 encoder 或 decoder")
        return None
    
    print(f"加载 encoder: {encoder_path}")
    encoder = ort.InferenceSession(encoder_path)
    print(f"加载 decoder: {decoder_path}")
    decoder = ort.InferenceSession(decoder_path)
    
    # ImageNet 标准化参数
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    
    # 生成校准数据
    calibration_data = {
        'pix_feat': [],
        'pred_mask': [],
    }
    
    for img_path in tqdm(image_files, desc="生成 memory_encoder 校准数据"):
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, :]
        
        # 标准化
        img_normalized = (img_np / 255.0 - mean) / std
        
        # 运行 encoder
        high_res_0, high_res_1, image_embed = encoder.run(None, {
            'image': img_normalized.astype(np.float32)
        })
        
        # 生成随机点
        cx = np.random.randint(orig_w // 4, 3 * orig_w // 4)
        cy = np.random.randint(orig_h // 4, 3 * orig_h // 4)
        point_x = cx * IMG_SIZE / orig_w
        point_y = cy * IMG_SIZE / orig_h
        point_coords = np.array([[[point_x, point_y]]]).astype(np.float32)
        point_labels = np.array([[1]]).astype(np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        
        # 运行 decoder
        masks, iou_pred, obj_ptr, obj_score = decoder.run(None, {
            'image_embed': image_embed,
            'high_res_feats_0': high_res_0,
            'high_res_feats_1': high_res_1,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
        })
        
        # 选择最佳 mask
        best_idx = np.argmax(iou_pred[0])
        mask_256 = masks[0, best_idx]  # [256, 256]
        
        # 上采样到 1024x1024
        pred_mask = cv2.resize(mask_256, (IMG_SIZE, IMG_SIZE))
        pred_mask = np.where(pred_mask > 0, 10.0, -10.0).astype(np.float32)
        pred_mask = pred_mask[np.newaxis, np.newaxis, :, :]  # [1, 1, 1024, 1024]
        
        calibration_data['pix_feat'].append(image_embed)  # [1, 256, 64, 64]
        calibration_data['pred_mask'].append(pred_mask)   # [1, 1, 1024, 1024]
    
    # 保存校准数据
    mem_enc_calib_dir = os.path.join(output_dir, "memory_encoder_calibration")
    os.makedirs(mem_enc_calib_dir, exist_ok=True)
    
    list_file = os.path.join(output_dir, "memory_encoder_calibration.txt")
    with open(list_file, 'w') as f:
        for i in range(len(image_files)):
            sample_files = []
            for key in ['pix_feat', 'pred_mask']:
                npy_path = os.path.join(mem_enc_calib_dir, f"{key}_{i:04d}.npy")
                np.save(npy_path, calibration_data[key][i])
                sample_files.append(os.path.abspath(npy_path))
            f.write(" ".join(sample_files) + "\n")
    
    print(f"已保存 memory_encoder 校准数据: {mem_enc_calib_dir}")
    print(f"已保存校准列表: {list_file}")
    
    return list_file


def main():
    parser = argparse.ArgumentParser(description='准备 EdgeTAM 量化校准数据')
    parser.add_argument('--image_dir', type=str, 
                        default='../notebooks/videos/bedroom',
                        help='图像目录')
    parser.add_argument('--output_dir', type=str, default='calibration_data',
                        help='输出目录')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='校准样本数量')
    parser.add_argument('--only', type=str, 
                        choices=['encoder', 'decoder', 'memory_encoder'],
                        help='只准备指定模型的校准数据')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EdgeTAM 量化校准数据准备")
    print("=" * 60)
    print(f"图像目录: {args.image_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"样本数量: {args.num_samples}")
    
    results = {}
    
    # 1. Encoder 校准数据
    if args.only is None or args.only == 'encoder':
        encoder_list, selected_images = prepare_encoder_calibration(
            args.image_dir, args.output_dir, args.num_samples
        )
        results['encoder'] = encoder_list
    else:
        # 获取图像列表供其他模型使用
        image_files = sorted(glob(os.path.join(args.image_dir, "*.jpg")))
        image_files.extend(sorted(glob(os.path.join(args.image_dir, "*.png"))))
        if len(image_files) > args.num_samples:
            step = len(image_files) // args.num_samples
            selected_images = image_files[::step][:args.num_samples]
        else:
            selected_images = image_files
    
    # 2. Decoder 校准数据
    if args.only is None or args.only == 'decoder':
        decoder_list = prepare_decoder_calibration(
            selected_images, args.output_dir, args.onnx_dir
        )
        results['decoder'] = decoder_list
    
    # 3. Memory Encoder 校准数据
    if args.only is None or args.only == 'memory_encoder':
        mem_enc_list = prepare_memory_encoder_calibration(
            selected_images, args.output_dir, args.onnx_dir
        )
        results['memory_encoder'] = mem_enc_list
    
    # 打印结果
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    for model, path in results.items():
        if path:
            print(f"{model}: {path}")
    
    print("\n使用方法:")
    print("  在主机上运行 convert_rknn.py 时指定校准数据:")
    print("  python convert_rknn.py edgetam --quantize --calibration_dir calibration_data")


if __name__ == "__main__":
    main()
