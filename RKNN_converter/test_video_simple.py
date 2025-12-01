# -*- coding: utf-8 -*-
"""
简化版视频推理 - 纯 ONNX 实现（无 memory，逐帧预测）

这是一个简化版本的视频推理，使用 ONNX encoder + decoder 逐帧预测。
由于不使用 memory 模块，跟踪效果不如完整的 SAM2 视频推理，
但可以完全在 ONNX/RKNN 上运行。

适用场景:
- 需要在没有 PyTorch 的设备上运行（如嵌入式设备）
- 目标物体在视频中移动较小
- 对跟踪精度要求不高的场景

使用方法:
    python test_video_simple.py --video_dir <视频帧目录或视频文件> --point <x,y> --output <输出目录>
    
支持的输入:
    - 视频帧目录（包含 jpg/png 图片）
    - 视频文件（mp4/avi/mov/mkv 等）
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def is_video_file(path):
    """判断是否为视频文件"""
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    _, ext = os.path.splitext(path)
    return ext.lower() in video_exts


def load_video_file(video_path):
    """从视频文件加载帧"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    
    frames = []
    frame_names = []
    frame_idx = 0
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        frames.append(img)
        frame_names.append(f"{frame_idx:05d}")
        frame_idx += 1
    
    cap.release()
    return frames, frame_names


def load_video_frames(video_dir):
    """加载视频帧目录中的所有图片"""
    supported_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    frame_files = []
    for f in os.listdir(video_dir):
        if any(f.endswith(ext) for ext in supported_exts):
            frame_files.append(f)
    
    # 按文件名排序
    frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    
    frames = []
    frame_names = []
    for f in frame_files:
        path = os.path.join(video_dir, f)
        img = Image.open(path).convert("RGB")
        frames.append(img)
        frame_names.append(os.path.splitext(f)[0])
    
    return frames, frame_names


def load_input(input_path):
    """
    加载输入（支持视频文件或帧目录）
    
    Args:
        input_path: 视频文件路径或帧目录路径
    
    Returns:
        frames: PIL Image 列表
        frame_names: 帧名称列表
    """
    if os.path.isfile(input_path):
        if is_video_file(input_path):
            print(f"检测到视频文件: {input_path}")
            return load_video_file(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {input_path}")
    elif os.path.isdir(input_path):
        print(f"检测到帧目录: {input_path}")
        return load_video_frames(input_path)
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


def preprocess_image(image, target_size=1024):
    """预处理图像 - 与 SAM2Transforms 保持一致"""
    orig_w, orig_h = image.size
    
    # 直接 resize 到 target_size x target_size
    resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    # 转换为 numpy 并归一化
    img_np = np.array(resized).astype(np.float32) / 255.0
    
    # ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # HWC -> CHW, 添加 batch 维度
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np, (orig_w, orig_h)


def transform_point(point, orig_size, target_size=1024):
    """将原始图像坐标转换为模型坐标"""
    orig_w, orig_h = orig_size
    x, y = point
    model_x = x / orig_w * target_size
    model_y = y / orig_h * target_size
    return model_x, model_y


def get_stability_scores(mask_logits, threshold=0.0, stability_delta=1.0):
    """计算 mask 的稳定性分数"""
    high_thresh_mask = mask_logits > (threshold + stability_delta)
    low_thresh_mask = mask_logits > (threshold - stability_delta)
    
    intersections = np.sum(high_thresh_mask & low_thresh_mask, axis=(-2, -1))
    unions = np.sum(high_thresh_mask | low_thresh_mask, axis=(-2, -1))
    
    stability_scores = np.where(unions > 0, intersections / unions, 1.0)
    return stability_scores


def select_best_mask(all_masks, all_iou_scores, stability_thresh=0.95):
    """
    从 4 个 mask 中选择最佳的一个
    使用稳定性选择逻辑
    """
    batch_size = all_masks.shape[0]
    
    # 从多 mask 输出 (1-3) 中选择 IoU 最高的
    multimask_logits = all_masks[:, 1:, :, :]
    multimask_iou_scores = all_iou_scores[:, 1:]
    best_indices = np.argmax(multimask_iou_scores, axis=-1)
    
    best_multimask_logits = np.array([
        multimask_logits[b, best_indices[b]] for b in range(batch_size)
    ])
    best_multimask_iou = np.array([
        multimask_iou_scores[b, best_indices[b]] for b in range(batch_size)
    ])
    
    # 单 mask 输出及其稳定性分数
    singlemask_logits = all_masks[:, 0, :, :]
    singlemask_iou = all_iou_scores[:, 0]
    stability_scores = get_stability_scores(singlemask_logits[:, np.newaxis, :, :])
    
    # 根据稳定性选择
    is_stable = stability_scores.squeeze() >= stability_thresh
    
    mask_out = np.where(
        is_stable[..., np.newaxis, np.newaxis],
        singlemask_logits,
        best_multimask_logits
    )
    iou_out = np.where(is_stable, singlemask_iou, best_multimask_iou)
    
    return mask_out, iou_out


def find_mask_center(mask):
    """找到 mask 的质心作为下一帧的跟踪点"""
    if mask.sum() == 0:
        return None
    
    # 找到 mask 的边界框和质心
    y_indices, x_indices = np.where(mask > 0)
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    
    return (center_x, center_y)


def postprocess_mask(mask_logits, orig_size):
    """后处理 mask: resize 到原始尺寸并二值化"""
    orig_w, orig_h = orig_size
    
    # Resize 到原始尺寸
    mask_resized = cv2.resize(
        mask_logits,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 二值化
    mask_binary = mask_resized > 0.0
    
    return mask_binary


def save_visualization(frame, mask, point, output_path, frame_idx, iou, lost=False):
    """保存可视化结果"""
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    
    # 叠加 mask
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask] = [1.0, 0.0, 0.0, 0.5]  # 红色半透明
    plt.imshow(colored_mask)
    
    # 绘制跟踪点
    if point is not None:
        plt.plot(point[0], point[1], 'rx', markersize=15, markeredgewidth=3)
    
    status = "LOST" if lost else f"IoU: {iou:.3f}"
    plt.title(f'Frame {frame_idx} | {status}')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='简化版视频推理 (纯 ONNX)')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频帧目录路径或视频文件路径（支持 mp4/avi/mov 等）')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置，格式为 x,y')
    parser.add_argument('--output', type=str, default='output_simple',
                        help='输出目录')
    parser.add_argument('--encoder', type=str, default='edgetam_encoder.onnx',
                        help='ONNX encoder 模型路径')
    parser.add_argument('--decoder', type=str, default='edgetam_decoder.onnx',
                        help='ONNX decoder 模型路径')
    parser.add_argument('--use_prev_mask', action='store_true',
                        help='使用上一帧的 mask 作为输入提示')
    parser.add_argument('--track_point', action='store_true',
                        help='跟踪 mask 质心作为下一帧的点击位置')
    parser.add_argument('--visualize', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='最大处理帧数（-1 表示全部）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.encoder):
        print(f"错误: 找不到 encoder 模型 {args.encoder}")
        print("请先运行: python export_onnx.py --model_type edgetam --checkpoint ../checkpoints/edgetam.pt --output_encoder edgetam_encoder.onnx --output_decoder edgetam_decoder.onnx")
        return
    if not os.path.exists(args.decoder):
        print(f"错误: 找不到 decoder 模型 {args.decoder}")
        return
    
    # 解析初始点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output, 'vis'), exist_ok=True)
    
    # 加载视频帧（支持视频文件或帧目录）
    print(f"加载输入: {args.video_dir}")
    frames, frame_names = load_input(args.video_dir)
    num_frames = len(frames)
    print(f"共 {num_frames} 帧")
    
    if num_frames == 0:
        print("错误: 未找到任何图片帧")
        return
    
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)
        print(f"处理前 {num_frames} 帧")
    
    # 获取原始图像尺寸
    orig_size = frames[0].size
    print(f"原始图像尺寸: {orig_size[0]}x{orig_size[1]}")
    
    # 初始化 ONNX sessions
    print("加载 ONNX 模型...")
    encoder_session = onnxruntime.InferenceSession(args.encoder)
    decoder_session = onnxruntime.InferenceSession(args.decoder)
    
    # 准备 mask 输入缓冲区
    prev_mask_logits = None
    
    # 处理每一帧
    print("\n开始处理视频...")
    start_time = time.time()
    
    all_masks = []
    all_ious = []
    
    # 计算最小 mask 面积阈值（像素数）
    min_area_pixels = orig_size[0] * orig_size[1] * args.min_mask_area
    object_lost = False  # 物体是否已丢失
    lost_frame_count = 0  # 连续丢失帧数
    
    for i in tqdm(range(num_frames)):
        frame = frames[i]
        frame_name = frame_names[i]
        
        # 预处理图像
        img_np, _ = preprocess_image(frame)
        
        # 运行 encoder
        encoder_outputs = encoder_session.run(None, {'image': img_np})
        high_res_feats_0, high_res_feats_1, image_embed = encoder_outputs
        
        # 准备 decoder 输入
        # 转换点坐标
        model_x, model_y = transform_point(current_point, orig_size)
        point_coords = np.array([[[model_x, model_y]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)  # 前景点
        
        # 准备 mask 输入
        if args.use_prev_mask and prev_mask_logits is not None:
            # 使用上一帧的 mask 作为提示
            mask_input = prev_mask_logits[np.newaxis, np.newaxis, :, :]
            # Resize 到 256x256
            mask_input = cv2.resize(prev_mask_logits, (256, 256))[np.newaxis, np.newaxis, :, :]
            mask_input = mask_input.astype(np.float32)
            has_mask_input = np.array([1], dtype=np.float32)
        else:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        
        # 检查物体是否已丢失
        if object_lost:
            # 物体已丢失，只使用 prev_mask 尝试恢复（不使用点击）
            if args.use_prev_mask and prev_mask_logits is not None:
                # 只用 mask 提示
                mask_input = cv2.resize(prev_mask_logits, (256, 256))[np.newaxis, np.newaxis, :, :]
                mask_input = mask_input.astype(np.float32)
                has_mask_input = np.array([1], dtype=np.float32)
                # 使用图像中心作为虚拟点（但设置为背景点来减少影响）
                point_coords = np.array([[[512, 512]]], dtype=np.float32)  # 中心点
                point_labels = np.array([[0]], dtype=np.float32)  # 背景点
            else:
                # 没有 mask 可用，输出空 mask
                final_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)
                all_masks.append(final_mask)
                all_ious.append(0.0)
                mask_uint8 = (final_mask * 255).astype(np.uint8)
                Image.fromarray(mask_uint8).save(os.path.join(args.output, f'{frame_name}.png'))
                if args.visualize:
                    save_visualization(
                        frame, final_mask, None,
                        os.path.join(args.output, 'vis', f'{frame_name}.png'),
                        i, 0.0, lost=True
                    )
                continue
        
        # 运行 decoder
        decoder_inputs = {
            'image_embed': image_embed,
            'high_res_feats_0': high_res_feats_0,
            'high_res_feats_1': high_res_feats_1,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
        }
        
        low_res_masks, iou_predictions = decoder_session.run(None, decoder_inputs)
        
        # 选择最佳 mask
        best_mask, best_iou = select_best_mask(low_res_masks, iou_predictions)
        
        # 保存当前 mask 用于下一帧
        prev_mask_logits = best_mask[0]
        
        # 后处理
        final_mask = postprocess_mask(best_mask[0], orig_size)
        
        # 计算 mask 面积
        mask_area = final_mask.sum()
        
        # 检测物体是否消失
        if mask_area < min_area_pixels or best_iou[0] < args.min_iou:
            lost_frame_count += 1
            if lost_frame_count >= 3:  # 连续3帧丢失才认为物体真的消失
                if not object_lost:
                    print(f"\n[Frame {i}] 物体丢失! (面积={mask_area:.0f}, IoU={best_iou[0]:.3f})")
                object_lost = True
                current_point = None  # 移除追踪点
        else:
            # 物体恢复
            if object_lost:
                print(f"\n[Frame {i}] 物体恢复! (面积={mask_area:.0f}, IoU={best_iou[0]:.3f})")
                # 恢复追踪点
                new_point = find_mask_center(final_mask)
                if new_point is not None:
                    current_point = new_point
            object_lost = False
            lost_frame_count = 0
        
        all_masks.append(final_mask)
        all_ious.append(best_iou[0])
        
        # 更新跟踪点（使用 mask 质心）
        if args.track_point and not object_lost:
            new_point = find_mask_center(final_mask)
            if new_point is not None:
                current_point = new_point
        
        # 保存 mask
        mask_uint8 = (final_mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(os.path.join(args.output, f'{frame_name}.png'))
        
        # 可视化
        if args.visualize:
            save_visualization(
                frame, final_mask, current_point if not object_lost else None,
                os.path.join(args.output, 'vis', f'{frame_name}.png'),
                i, best_iou[0], lost=object_lost
            )
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    # 统计信息
    print(f"\n完成!")
    print(f"处理帧数: {num_frames}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {fps:.2f}")
    print(f"平均 IoU 预测: {np.mean(all_ious):.3f}")
    print(f"输出目录: {args.output}")
    
    if args.use_prev_mask:
        print("\n提示: 使用了 --use_prev_mask，上一帧的 mask 被用作当前帧的输入提示")
    if args.track_point:
        print("提示: 使用了 --track_point，跟踪点会随 mask 质心移动")


if __name__ == "__main__":
    main()
