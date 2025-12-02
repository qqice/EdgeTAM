# -*- coding: utf-8 -*-
"""
使用原始 PyTorch 模型的视频推理脚本

这个脚本调用原始的 EdgeTAM PT 模型，实现与 test_video_standalone.py 完全等效的操作。
主要用于对比分析和调试，观察原始实现在遮挡等情况下的 IoU、Area、Score 等数据。

使用方法:
    python test_video_original.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --checkpoint ../checkpoints/edgetam.pt \
        --visualize --save_video
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# 不要改变工作目录，保持用户的当前目录

import argparse
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from sam2.build_sam import build_sam2_video_predictor


# ==================== 常量 ====================

IMG_SIZE = 1024
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


# ==================== 辅助函数 ====================

def is_video_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def load_video_file(video_path: str, max_frames: int = None) -> Tuple[List[Image.Image], List[str], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    frames = []
    frame_names = []
    idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_names.append(f"{idx:05d}")
        idx += 1
        
        if max_frames and idx >= max_frames:
            break
    
    cap.release()
    return frames, frame_names, fps


def load_frame_dir(dir_path: str) -> Tuple[List[Image.Image], List[str], float]:
    exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    files = [f for f in os.listdir(dir_path) if any(f.endswith(e) for e in exts)]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    
    frames = []
    frame_names = []
    for f in files:
        img = Image.open(os.path.join(dir_path, f)).convert("RGB")
        frames.append(img)
        frame_names.append(os.path.splitext(f)[0])
    
    return frames, frame_names, 30.0  # 默认 30 FPS


def find_mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """计算 mask 质心"""
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    binary_mask = (mask > 0.5).astype(np.uint8)
    if binary_mask.sum() == 0:
        return None
    
    center = ndimage.center_of_mass(binary_mask)
    return int(center[1]), int(center[0])


def save_mask(mask: np.ndarray, output_path: str):
    """保存 mask 图像"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_uint8)


def create_visualization(
    frame: Image.Image,
    mask: np.ndarray,
    point: Tuple[int, int],
    frame_idx: int,
    iou: float,
    obj_score: float,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    alpha: float = 0.5,
) -> np.ndarray:
    """创建可视化图像
    
    Args:
        frame: 原始帧 (PIL Image)
        mask: 预测的 mask
        point: 点击位置
        frame_idx: 帧索引
        iou: IoU 预测分数
        obj_score: 物体存在分数
        color: mask 颜色 (RGB, 0-1)
        alpha: 透明度
        
    Returns:
        可视化图像 (numpy array, BGR)
    """
    # 转换为 numpy
    frame_np = np.array(frame)
    h, w = frame_np.shape[:2]
    
    # 确保 mask 尺寸匹配
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.float32), (w, h))
    
    # 创建彩色 mask
    mask_binary = (mask > 0.5).astype(np.float32)
    colored_mask = np.zeros_like(frame_np, dtype=np.float32)
    colored_mask[:, :, 0] = color[0] * 255  # R
    colored_mask[:, :, 1] = color[1] * 255  # G
    colored_mask[:, :, 2] = color[2] * 255  # B
    
    # 混合原图和 mask
    result = frame_np.astype(np.float32).copy()
    mask_3d = np.stack([mask_binary] * 3, axis=-1)
    result = result * (1 - mask_3d * alpha) + colored_mask * mask_3d * alpha
    result = result.astype(np.uint8)
    
    # 绘制 mask 边界
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    
    # 绘制点击位置
    cv2.drawMarker(result, point, (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
    cv2.circle(result, point, 8, (255, 0, 0), 2)
    
    # 添加文字信息
    text_bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # 帧信息
    info_text = f"Frame: {frame_idx}"
    (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    cv2.rectangle(result, (10, 10), (20 + tw, 20 + th + 10), text_bg_color, -1)
    cv2.putText(result, info_text, (15, 15 + th), font, font_scale, text_color, thickness)
    
    # IoU 和 Score
    score_text = f"Conf: {iou:.2f} | Score: {obj_score:.2f}"
    (tw, th), _ = cv2.getTextSize(score_text, font, font_scale, thickness)
    cv2.rectangle(result, (10, 50), (20 + tw, 60 + th + 10), text_bg_color, -1)
    cv2.putText(result, score_text, (15, 55 + th), font, font_scale, text_color, thickness)
    
    # Mask 面积
    area = int(np.sum(mask_binary))
    area_text = f"Area: {area} px"
    (tw, th), _ = cv2.getTextSize(area_text, font, font_scale, thickness)
    cv2.rectangle(result, (10, 90), (20 + tw, 100 + th + 10), text_bg_color, -1)
    cv2.putText(result, area_text, (15, 95 + th), font, font_scale, text_color, thickness)
    
    # 转换为 BGR (OpenCV 格式)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result_bgr


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0,
):
    """从可视化帧创建视频
    
    Args:
        frame_paths: 可视化帧的路径列表
        output_path: 输出视频路径
        fps: 帧率
    """
    if not frame_paths:
        print("  没有帧可用于创建视频")
        return
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"  无法读取帧: {frame_paths[0]}")
        return
    
    h, w = first_frame.shape[:2]
    
    # 先使用 OpenCV 生成临时视频
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("  警告: 无法创建视频写入器")
        return
    
    for path in tqdm(frame_paths, desc="  生成视频"):
        frame = cv2.imread(path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    
    # 使用 ffmpeg 转换为 H.264 格式（VSCode 可预览）
    import subprocess
    
    # 尝试不同的 H.264 编码器
    encoders = ['h264_nvenc', 'libx264', 'h264']
    converted = False
    
    for encoder in encoders:
        try:
            cmd = [
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', encoder,
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                converted = True
                os.remove(temp_path)
                print(f"  视频已使用 {encoder} 编码保存")
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if not converted:
        # 如果转换失败，使用原始文件
        os.rename(temp_path, output_path)
        print(f"  视频保存到: {output_path} (MPEG4 格式，可能无法在 VSCode 中预览)")
    else:
        print(f"  视频保存到: {output_path}")


def compare_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个 mask 的 IoU"""
    mask1_bin = (mask1 > 0.5).astype(bool)
    mask2_bin = (mask2 > 0.5).astype(bool)
    
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='使用原始 PT 模型的视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/edgetam.pt',
                        help='模型检查点路径')
    parser.add_argument('--model_cfg', type=str, default='configs/edgetam.yaml',
                        help='模型配置文件')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: ./output/{视频名}_{X}_{Y}_original)')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--track_point', action='store_true',
                        help='根据 mask 质心更新点击位置')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='与另一个输出目录进行对比')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图像')
    parser.add_argument('--save_video', action='store_true',
                        help='生成可视化视频')
    parser.add_argument('--video_fps', type=float, default=None,
                        help='输出视频帧率 (默认使用源视频帧率)')
    parser.add_argument('--mask_color', type=str, default='blue',
                        choices=['blue', 'green', 'red', 'yellow', 'cyan', 'magenta'],
                        help='mask 颜色')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    initial_point = (point_x, point_y)
    current_point = initial_point
    
    # 颜色映射
    color_map = {
        'blue': (0.2, 0.6, 1.0),
        'green': (0.2, 0.8, 0.2),
        'red': (1.0, 0.2, 0.2),
        'yellow': (1.0, 0.9, 0.1),
        'cyan': (0.1, 0.9, 0.9),
        'magenta': (0.9, 0.2, 0.9),
    }
    mask_color = color_map.get(args.mask_color, (0.2, 0.6, 1.0))
    
    # 自动生成输出目录名称
    if args.output is None:
        video_name = os.path.basename(args.video_dir)
        if os.path.isfile(args.video_dir):
            video_name = os.path.splitext(video_name)[0]
        args.output = os.path.join('output', f'{video_name}_{point_x}_{point_y}_original')
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建 mask 子目录
    mask_dir = os.path.join(args.output, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    # 创建可视化子目录
    vis_dir = None
    if args.visualize or args.save_video:
        vis_dir = os.path.join(args.output, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    # 加载视频
    print(f"\n加载视频: {args.video_dir}")
    if is_video_file(args.video_dir):
        frames, frame_names, video_fps = load_video_file(args.video_dir, args.max_frames)
    else:
        frames, frame_names, video_fps = load_frame_dir(args.video_dir)
        if args.max_frames:
            frames = frames[:args.max_frames]
            frame_names = frame_names[:args.max_frames]
    
    # 使用用户指定的帧率或源视频帧率
    output_fps = args.video_fps if args.video_fps else video_fps
    
    print(f"共 {len(frames)} 帧, 图像尺寸: {frames[0].size}")
    
    # 保存帧到临时目录供 video predictor 使用
    temp_frames_dir = os.path.join(args.output, '.temp_frames')
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    print("\n保存临时帧...")
    for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames))):
        frame.save(os.path.join(temp_frames_dir, f"{name}.jpg"))
    
    # 初始化模型
    print("\n" + "=" * 60)
    print("初始化原始 PT 模型")
    print("=" * 60)
    
    print(f"加载模型: {args.model_cfg}")
    print(f"检查点: {args.checkpoint}")
    
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)
    
    # 初始化推理状态
    print("\n初始化推理状态...")
    inference_state = predictor.init_state(video_path=temp_frames_dir)
    
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]
    print(f"  视频尺寸: {video_width}x{video_height}")
    
    # 在第一帧添加点击
    print(f"\n在第一帧添加点击: ({point_x}, {point_y})")
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[point_x, point_y]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    
    # 处理视频并收集结果
    print("\n开始推理...")
    all_masks = []
    all_frames = []
    iou_scores = []
    obj_scores = []
    vis_paths = []
    point_history = [current_point]
    
    # 逐帧推理
    for frame_idx, obj_ids, video_res_masks in tqdm(
        predictor.propagate_in_video(inference_state),
        total=len(frames),
        desc="处理帧"
    ):
        # video_res_masks: [num_objects, 1, H, W] 的 logits
        mask_logits = video_res_masks[0, 0].cpu().numpy()  # 取第一个物体
        mask = (mask_logits > 0).astype(np.float32)
        
        all_masks.append(mask)
        all_frames.append(frames[frame_idx])
        
        # 获取该帧的详细输出信息 (IoU, object_score_logits)
        # 从 output_dict 获取
        output_dict = inference_state["output_dict"]
        
        # 先查找 cond_frame_outputs
        frame_out = output_dict["cond_frame_outputs"].get(frame_idx, None)
        if frame_out is None:
            frame_out = output_dict["non_cond_frame_outputs"].get(frame_idx, None)
        
        if frame_out is not None:
            obj_score = float(frame_out["object_score_logits"][0, 0].cpu().numpy())
        else:
            obj_score = 0.0
        
        # 注意: SAM2 的 propagate_in_video 不直接返回 IoU 预测值
        # 我们使用 sigmoid(object_score_logits) 作为物体存在的置信度
        # 这与 IoU 预测不同，但可以反映物体是否存在
        confidence = 1.0 / (1.0 + np.exp(-obj_score))  # sigmoid(obj_score)
        
        # 对于调试，我们需要的是这些原始值：
        # - obj_score: 物体存在分数 (logits)
        # - confidence: sigmoid(obj_score) 物体存在置信度
        # - area: mask 面积
        # - center: mask 质心
        
        iou_scores.append(confidence)  # 使用 confidence 作为 IoU 的代理
        obj_scores.append(obj_score)
        
        # 保存 mask 到 mask 子目录
        name = frame_names[frame_idx]
        mask_path = os.path.join(mask_dir, f"{name}.png")
        save_mask(mask, mask_path)
        
        # 生成可视化图像
        if args.visualize or args.save_video:
            vis_img = create_visualization(
                frames[frame_idx], mask, current_point, frame_idx, confidence, obj_score,
                color=mask_color, alpha=0.5
            )
            vis_path = os.path.join(vis_dir, f"{name}.jpg")
            cv2.imwrite(vis_path, vis_img)
            vis_paths.append(vis_path)
        
        # 更新点击位置
        if args.track_point:
            new_point = find_mask_center(mask)
            if new_point is not None:
                current_point = new_point
        point_history.append(current_point)
    
    # 清理临时帧目录
    import shutil
    shutil.rmtree(temp_frames_dir)
    
    # 生成视频
    if args.save_video and vis_paths:
        print("\n生成可视化视频...")
        video_path = os.path.join(args.output, 'output_video.mp4')
        create_video_from_frames(vis_paths, video_path, fps=output_fps)
    
    # 打印统计
    print(f"\n处理完成!")
    print(f"  总帧数: {len(frames)}")
    print(f"  平均置信度 (sigmoid): {np.mean(iou_scores):.3f}")
    print(f"  平均物体分数: {np.mean(obj_scores):.3f}")
    print(f"  输出目录: {args.output}")
    print(f"  Mask 文件: {mask_dir}")
    if args.visualize or args.save_video:
        print(f"  可视化图像: {vis_dir}")
    if args.save_video:
        print(f"  可视化视频: {video_path}")
    
    # 打印每帧的详细信息
    print("\n帧详细信息:")
    print(f"{'Frame':>6} | {'Conf':>8} | {'Score':>8} | {'Area':>10} | {'Center':>15}")
    print("-" * 55)
    for i, (mask, conf, score) in enumerate(zip(all_masks, iou_scores, obj_scores)):
        area = int(np.sum(mask > 0.5))
        center = find_mask_center(mask)
        center_str = f"({center[0]:4d}, {center[1]:4d})" if center else "None"
        print(f"{i:>6} | {conf:>8.3f} | {score:>8.3f} | {area:>10} | {center_str:>15}")
    
    # 与参考结果对比
    if args.compare_with and os.path.exists(args.compare_with):
        print(f"\n与参考结果对比: {args.compare_with}")
        
        iou_vs_ref = []
        for i, name in enumerate(frame_names):
            if i >= len(all_masks):
                break
            
            # 尝试多种命名格式
            ref_candidates = [
                os.path.join(args.compare_with, 'mask', f"{name}.png"),
                os.path.join(args.compare_with, f"{name}.png"),
                os.path.join(args.compare_with, f"{name}_mask.png"),
            ]
            
            ref_path = None
            for candidate in ref_candidates:
                if os.path.exists(candidate):
                    ref_path = candidate
                    break
            
            if ref_path:
                ref_mask = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE) / 255.0
                iou = compare_masks(all_masks[i], ref_mask)
                iou_vs_ref.append(iou)
                
                if i % 20 == 0 or i < 5:
                    area_ours = np.sum(all_masks[i] > 0.5)
                    area_ref = np.sum(ref_mask > 0.5)
                    print(f"  帧 {i:3d}: IoU={iou:.3f}, 面积(ours)={area_ours:.0f}, 面积(ref)={area_ref:.0f}")
        
        if iou_vs_ref:
            print(f"\n  与参考的平均 IoU: {np.mean(iou_vs_ref):.3f}")
            print(f"  最小 IoU: {np.min(iou_vs_ref):.3f}")
            print(f"  最大 IoU: {np.max(iou_vs_ref):.3f}")


if __name__ == "__main__":
    main()
