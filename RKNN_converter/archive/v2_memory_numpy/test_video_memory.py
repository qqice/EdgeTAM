# -*- coding: utf-8 -*-
"""
完整的视频推理脚本 - 使用完整 PyTorch 模型的记忆机制

这个脚本实现了完整的 SAM2/EdgeTAM 视频推理流程，包括:
- 完整的 Memory Bank 记忆管理
- 物体指针 (Object Pointers) 追踪
- RoPE 注意力记忆融合

由于 Memory Attention 使用 RoPE 注意力无法导出到 ONNX，
这个脚本使用 PyTorch 实现完整的记忆机制，用于验证物体消失重现后的追踪能力。

如果需要部署，可以:
1. 使用 ONNX encoder/decoder/memory_encoder 进行边缘推理
2. Memory Attention 使用 PyTorch 或等效的 C++ 实现

使用方法:
    python test_video_memory.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 100,200 \
        --output output_memory \
        --visualize
        
    python test_video_memory.py \
        --video_dir /path/to/video.mp4 \
        --point 500,300 \
        --output output_memory
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import tempfile
import shutil
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

from sam2.build_sam import build_sam2_video_predictor


# 支持的视频格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def is_video_file(path: str) -> bool:
    """检查是否是视频文件"""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def load_video_file(video_path: str, max_frames: int = None) -> Tuple[List[Image.Image], List[str]]:
    """加载视频文件"""
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
    return frames, frame_names


def load_frame_dir(dir_path: str) -> Tuple[List[Image.Image], List[str]]:
    """加载帧目录"""
    exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    files = [f for f in os.listdir(dir_path) if any(f.endswith(e) for e in exts)]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    
    frames = []
    frame_names = []
    for f in files:
        img = Image.open(os.path.join(dir_path, f)).convert("RGB")
        frames.append(img)
        frame_names.append(os.path.splitext(f)[0])
    
    return frames, frame_names


def find_mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """找到 mask 质心"""
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    binary_mask = (mask > 0.5).astype(np.uint8)
    if binary_mask.sum() == 0:
        return None
    
    center = ndimage.center_of_mass(binary_mask)
    return int(center[1]), int(center[0])


def save_visualization(frame: Image.Image, mask: np.ndarray, point: Optional[Tuple[int, int]],
                       output_path: str, frame_idx: int, obj_score: float = None,
                       lost: bool = False):
    """保存可视化结果"""
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    
    # 叠加 mask
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    colored_mask = np.zeros((*mask.shape, 4))
    if not lost:
        color = [0.2, 0.6, 1.0, 0.5]  # 蓝色
    else:
        color = [1.0, 0.2, 0.2, 0.3]  # 红色
    colored_mask[mask > 0] = color
    plt.imshow(colored_mask)
    
    # 绘制点击位置
    if point is not None:
        plt.plot(point[0], point[1], 'rx', markersize=15, markeredgewidth=3)
    
    if obj_score is not None:
        status = "LOST" if lost else f"Score={obj_score:.2f}"
    else:
        status = "LOST" if lost else "OK"
    plt.title(f"Frame {frame_idx} | {status}")
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


class VideoFrameDataset:
    """简单的视频帧数据集，模拟 SAM2 视频预测器需要的接口"""
    
    def __init__(self, frames: List[Image.Image]):
        self.frames = frames
        self.video_name = "video"
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return np.array(self.frames[idx])


def main():
    parser = argparse.ArgumentParser(description='使用完整记忆机制的视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default='output_memory',
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/edgetam.pt')
    parser.add_argument('--model_cfg', type=str, default='configs/edgetam.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--min_mask_area', type=float, default=0.001,
                        help='最小 mask 面积比例')
    parser.add_argument('--min_score', type=float, default=-2.0,
                        help='最小物体存在分数 (logits)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output, 'vis'), exist_ok=True)
    
    # 检查输入路径
    video_path = args.video_dir
    temp_dir = None
    
    if is_video_file(video_path):
        # 如果是视频文件，提取帧到临时目录
        print(f"加载视频: {video_path}")
        temp_dir = tempfile.mkdtemp()
        print(f"提取帧到临时目录: {temp_dir}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(os.path.join(temp_dir, f"{idx:05d}.jpg"))
            idx += 1
            
            if args.max_frames and idx >= args.max_frames:
                break
        
        cap.release()
        video_path = temp_dir
        print(f"提取完成，共 {idx} 帧")
    else:
        print(f"加载帧目录: {video_path}")
    
    # 初始化模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print("初始化 SAM2 视频预测器...")
    
    predictor = build_sam2_video_predictor(
        config_file=args.model_cfg,
        ckpt_path=args.checkpoint,
        device=device,
    )
    
    # 初始化推理状态
    print("\n初始化推理状态...")
    
    with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
        inference_state = predictor.init_state(
            video_path=video_path, 
            offload_video_to_cpu=True
        )
        
        num_frames = inference_state["num_frames"]
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        
        print(f"帧数: {num_frames}, 尺寸: {video_width}x{video_height}")
        
        # 在第一帧添加点击
        print(f"在第一帧添加点击: ({point_x}, {point_y})")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[point_x, point_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        
        # 计算最小 mask 面积阈值
        min_area_pixels = video_width * video_height * args.min_mask_area
        
        # 传播到所有帧
        print("\n开始视频推理...")
        start_time = time.time()
        
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, obj_id in enumerate(out_obj_ids)
            }
        
        end_time = time.time()
    
    total_time = end_time - start_time
    fps_result = num_frames / total_time
    
    # 加载原始帧用于可视化
    frames, frame_names = load_frame_dir(video_path)
    
    # 保存结果
    print("\n保存结果...")
    current_point = (point_x, point_y)
    
    for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames), desc="保存")):
        if i in video_segments and 1 in video_segments[i]:
            mask = video_segments[i][1].astype(np.float32)
        else:
            mask = np.zeros((video_height, video_width), dtype=np.float32)
        
        # 计算 mask 面积
        mask_area = mask.sum()
        
        # 检测物体是否丢失
        is_lost = mask_area < min_area_pixels
        
        # 更新追踪点
        if not is_lost:
            center = find_mask_center(mask)
            if center is not None:
                current_point = center
        
        # 保存 mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(os.path.join(args.output, f'{name}.png'))
        
        # 可视化
        if args.visualize:
            save_visualization(
                frame, mask,
                current_point if not is_lost else None,
                os.path.join(args.output, 'vis', f'{name}.png'),
                i, None, lost=is_lost
            )
    
    # 清理临时目录
    if temp_dir is not None:
        shutil.rmtree(temp_dir)
    
    print(f"\n完成!")
    print(f"处理帧数: {num_frames}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {fps_result:.2f}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
