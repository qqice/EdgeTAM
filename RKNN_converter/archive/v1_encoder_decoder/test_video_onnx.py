# -*- coding: utf-8 -*-
"""视频推理测试脚本 - 使用 PyTorch 视频预测器

本脚本使用 SAM2/EdgeTAM 的完整视频预测器进行视频目标分割。
支持功能：
- 支持视频文件和帧目录两种输入
- 支持追踪物体（自动检测最佳点击位置）
- 支持使用前一帧 mask 作为提示

使用方法:
    # 使用帧目录
    python test_video_onnx.py --video_dir ../notebooks/videos/bedroom --point 300,200 --output output_video
    
    # 使用视频文件
    python test_video_onnx.py --video_dir ../examples/03_blocks.mp4 --point 966,271 --output output_video
    
    # 启用追踪和前一帧 mask
    python test_video_onnx.py --video_dir video.mp4 --point 100,200 --track_point --use_prev_mask --output output_video
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import tempfile
import shutil
import numpy as np
import torch
import cv2
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

from sam2.build_sam import build_sam2_video_predictor

# 支持的视频格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def is_video_file(path):
    """检查路径是否为视频文件"""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def load_video_file(video_path, max_frames=None):
    """从视频文件加载帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    frame_names = []
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)
        frame_names.append(f"{frame_idx:05d}.jpg")
        frame_idx += 1
        
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    return frames, frame_names


def load_video_frames(video_dir):
    """加载视频帧目录中的所有图片"""
    supported_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    frame_files = []
    for f in os.listdir(video_dir):
        if any(f.endswith(ext) for ext in supported_exts):
            frame_files.append(f)
    
    # 按文件名排序（假设文件名包含帧号）
    frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    
    frames = []
    frame_paths = []
    for f in frame_files:
        path = os.path.join(video_dir, f)
        img = Image.open(path).convert("RGB")
        frames.append(img)
        frame_paths.append(f)
    
    return frames, frame_paths


def load_input(input_path, max_frames=None):
    """统一的输入加载接口，支持视频文件和帧目录"""
    if is_video_file(input_path):
        print(f"检测到视频文件: {input_path}")
        return load_video_file(input_path, max_frames)
    elif os.path.isdir(input_path):
        print(f"检测到帧目录: {input_path}")
        return load_video_frames(input_path)
    else:
        raise ValueError(f"不支持的输入: {input_path}（需要视频文件或帧目录）")


def find_mask_center(mask):
    """找到 mask 的质心位置"""
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # 二值化
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return None
    
    # 使用 scipy 计算质心
    center_of_mass = ndimage.center_of_mass(binary_mask)
    y, x = center_of_mass
    
    return int(x), int(y)


def preprocess_image_for_onnx(image, target_size=1024):
    """预处理图像用于 ONNX encoder - 与 SAM2Transforms 保持一致"""
    orig_w, orig_h = image.size
    
    # 直接 resize 到 target_size x target_size（不保持长宽比）
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


def visualize_masks(frame, masks, point=None, output_path=None, title=""):
    """可视化 mask 结果"""
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    
    # 叠加所有对象的 mask
    for obj_id, mask in masks.items():
        # 确保 mask 是 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        # 使用不同颜色显示不同对象
        color = plt.cm.tab10(obj_id % 10)[:3]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = [*color, 0.5]
        plt.imshow(colored_mask)
    
    if point is not None:
        plt.plot(point[0], point[1], 'rx', markersize=15, markeredgewidth=3)
    
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def save_mask(mask, output_path):
    """保存 mask 为 PNG 文件"""
    # 确保 mask 是 2D 数组
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(output_path)


def main():
    parser = argparse.ArgumentParser(description='使用 PyTorch 视频预测器进行视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频帧目录路径或视频文件路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置，格式为 x,y')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='添加点击的帧索引（默认为第一帧）')
    parser.add_argument('--output', type=str, default='output_video',
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/edgetam.pt',
                        help='模型检查点路径')
    parser.add_argument('--model_cfg', type=str, default='configs/edgetam.yaml',
                        help='模型配置文件')
    parser.add_argument('--encoder_onnx', type=str, default='edgetam_encoder.onnx',
                        help='ONNX encoder 模型路径（可选，用于对比）')
    parser.add_argument('--use_onnx_encoder', action='store_true',
                        help='是否使用 ONNX encoder（实验性）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否保存可视化结果')
    parser.add_argument('--track_point', action='store_true',
                        help='是否追踪物体（根据前一帧 mask 更新点击位置）')
    parser.add_argument('--use_prev_mask', action='store_true',
                        help='是否使用前一帧的 mask 作为提示')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最大处理帧数（用于测试）')
    parser.add_argument('--output_video', action='store_true',
                        help='是否输出视频文件')
    parser.add_argument('--min_mask_area', type=float, default=0.001,
                        help='最小 mask 面积比例（相对于图像面积），低于此值认为物体消失')
    parser.add_argument('--min_score', type=float, default=0.5,
                        help='最小预测分数，低于此值认为物体消失')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    point = np.array([[point_x, point_y]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)  # 1 = 前景点
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output, 'vis'), exist_ok=True)
    
    # 加载视频帧
    print(f"加载输入: {args.video_dir}")
    frames, frame_names = load_input(args.video_dir, args.max_frames)
    print(f"共 {len(frames)} 帧")
    
    if len(frames) == 0:
        print("错误: 未找到任何图片帧")
        return
    
    # 获取原始图像尺寸
    orig_w, orig_h = frames[0].size
    print(f"原始图像尺寸: {orig_w}x{orig_h}")
    
    # 如果输入是视频文件，需要先保存为帧目录（SAM2 video predictor 需要）
    temp_dir = None
    video_path = args.video_dir
    if is_video_file(args.video_dir):
        temp_dir = tempfile.mkdtemp(prefix='sam2_frames_')
        print(f"将视频帧保存到临时目录: {temp_dir}")
        for i, frame in enumerate(tqdm(frames, desc="保存帧")):
            frame.save(os.path.join(temp_dir, f"{i:05d}.jpg"))
        video_path = temp_dir
    
    # 初始化 PyTorch 视频预测器
    print("初始化视频预测器...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device)
    
    # 初始化推理状态
    print("初始化推理状态...")
    inference_state = predictor.init_state(video_path=video_path)
    
    # 添加初始点击
    print(f"在第 {args.frame_idx} 帧添加点击: ({point_x}, {point_y})")
    _, obj_ids, video_res_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=args.frame_idx,
        obj_id=1,  # 对象 ID
        points=point,
        labels=label,
    )
    
    # 保存初始帧的 mask
    init_mask = (video_res_masks[0] > 0.0).cpu().numpy()
    save_mask(init_mask, os.path.join(args.output, f'{args.frame_idx:05d}.png'))
    
    if args.visualize:
        visualize_masks(
            frames[args.frame_idx], 
            {1: init_mask},
            point=(point_x, point_y),
            output_path=os.path.join(args.output, 'vis', f'{args.frame_idx:05d}.png'),
            title=f'Frame {args.frame_idx} (Initial)'
        )
    
    # 传播到所有帧
    print("开始视频传播...")
    start_time = time.time()
    
    # 根据模式选择传播方式
    if args.track_point or args.use_prev_mask:
        # 逐帧追踪模式：每帧重新初始化状态并使用新的点击位置
        # 这种方式适合需要动态追踪的场景
        print("使用逐帧追踪模式...")
        frame_count = 0
        prev_mask = init_mask
        current_point = (point_x, point_y)
        
        # 使用 image predictor 进行逐帧推理（更适合动态追踪）
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        image_predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=device))
        
        # 计算最小 mask 面积阈值（像素数）
        min_area_pixels = orig_w * orig_h * args.min_mask_area
        object_lost = False  # 物体是否已丢失
        lost_frame_count = 0  # 连续丢失帧数
        
        for frame_idx in tqdm(range(1, len(frames)), desc="处理帧"):
            frame = frames[frame_idx]
            
            # 设置当前帧图像
            image_predictor.set_image(frame)
            
            # 检查物体是否已丢失
            if object_lost:
                # 物体已丢失，只使用 prev_mask 尝试恢复（不使用点击）
                if args.use_prev_mask and prev_mask is not None:
                    mask_squeezed = prev_mask.squeeze()
                    mask_resized = cv2.resize(mask_squeezed.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
                    mask_input = mask_resized[None, :, :]  # (1, 256, 256)
                    
                    # 只使用 mask 提示，不使用点击
                    masks, scores, logits = image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        mask_input=mask_input,
                        multimask_output=True,
                    )
                else:
                    # 没有 mask 可用，输出空 mask
                    mask = np.zeros((orig_h, orig_w), dtype=bool)
                    save_mask(mask, os.path.join(args.output, f'{frame_idx:05d}.png'))
                    if args.visualize:
                        visualize_masks(
                            frame,
                            {1: mask},
                            point=None,
                            output_path=os.path.join(args.output, 'vis', f'{frame_idx:05d}.png'),
                            title=f'Frame {frame_idx} (LOST)'
                        )
                    frame_count += 1
                    continue
            else:
                # 物体未丢失，正常追踪
                # 如果启用追踪，根据前一帧 mask 更新点击位置
                if args.track_point and prev_mask is not None:
                    center = find_mask_center(prev_mask)
                    if center is not None:
                        current_point = center
                
                # 准备输入
                new_point = np.array([[current_point[0], current_point[1]]], dtype=np.float32)
                new_label = np.array([1], dtype=np.int32)
                
                # 使用 mask 作为额外提示（如果启用）
                mask_input = None
                if args.use_prev_mask and prev_mask is not None:
                    # 将 mask resize 到 256x256（SAM2 的 mask 输入尺寸）
                    mask_squeezed = prev_mask.squeeze()
                    mask_resized = cv2.resize(mask_squeezed.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
                    mask_input = mask_resized[None, :, :]  # (1, 256, 256)
                
                # 预测
                masks, scores, logits = image_predictor.predict(
                    point_coords=new_point,
                    point_labels=new_label,
                    mask_input=mask_input,
                    multimask_output=True,
                )
            
            # 选择最佳 mask
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            mask = masks[best_idx]
            
            # 计算 mask 面积
            mask_area = mask.sum()
            
            # 检测物体是否消失
            if mask_area < min_area_pixels or best_score < args.min_score:
                lost_frame_count += 1
                if lost_frame_count >= 3:  # 连续3帧丢失才认为物体真的消失
                    if not object_lost:
                        print(f"\n[Frame {frame_idx}] 物体丢失! (面积={mask_area:.0f}, 分数={best_score:.3f})")
                    object_lost = True
                    current_point = None  # 移除追踪点
            else:
                # 物体恢复
                if object_lost:
                    print(f"\n[Frame {frame_idx}] 物体恢复! (面积={mask_area:.0f}, 分数={best_score:.3f})")
                    # 恢复追踪点
                    center = find_mask_center(mask)
                    if center is not None:
                        current_point = center
                object_lost = False
                lost_frame_count = 0
            
            prev_mask = mask
            
            # 保存 mask
            save_mask(mask, os.path.join(args.output, f'{frame_idx:05d}.png'))
            
            # 可视化
            if args.visualize:
                status = "LOST" if object_lost else f"score={best_score:.3f}"
                visualize_masks(
                    frame,
                    {1: mask},
                    point=current_point if not object_lost else None,
                    output_path=os.path.join(args.output, 'vis', f'{frame_idx:05d}.png'),
                    title=f'Frame {frame_idx} ({status})'
                )
            
            frame_count += 1
    else:
        # 标准传播模式
        frame_count = 0
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
            # 跳过初始帧（已经处理过）
            if frame_idx == args.frame_idx:
                continue
            
            # 转换 mask
            mask = (video_res_masks[0] > 0.0).cpu().numpy()
            
            # 保存 mask
            save_mask(mask, os.path.join(args.output, f'{frame_idx:05d}.png'))
            
            # 可视化
            if args.visualize:
                visualize_masks(
                    frames[frame_idx],
                    {1: mask},
                    output_path=os.path.join(args.output, 'vis', f'{frame_idx:05d}.png'),
                    title=f'Frame {frame_idx}'
                )
            
            frame_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n完成!")
    print(f"处理帧数: {frame_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {fps:.2f}")
    print(f"输出目录: {args.output}")
    
    # 输出视频文件
    if args.output_video and args.visualize:
        output_video_path = os.path.join(args.output, 'output.mp4')
        print(f"\n生成输出视频: {output_video_path}")
        
        vis_dir = os.path.join(args.output, 'vis')
        vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
        
        if len(vis_files) > 0:
            first_img = cv2.imread(os.path.join(vis_dir, vis_files[0]))
            h, w = first_img.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
            
            for f in tqdm(vis_files, desc="写入视频"):
                img = cv2.imread(os.path.join(vis_dir, f))
                out.write(img)
            
            out.release()
            print(f"视频已保存: {output_video_path}")
    
    # 清理临时目录
    if temp_dir and os.path.exists(temp_dir):
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)


def compare_with_onnx_encoder():
    """
    比较 PyTorch encoder 和 ONNX encoder 的输出
    这是一个实验性功能，用于验证 ONNX encoder 的正确性
    """
    print("\n=== 对比 PyTorch vs ONNX Encoder ===")
    
    # 加载测试图像
    test_image = Image.open("dog.jpg").convert("RGB") if os.path.exists("dog.jpg") else None
    if test_image is None:
        print("未找到测试图像 dog.jpg，跳过对比")
        return
    
    # 预处理
    img_np, (orig_w, orig_h) = preprocess_image_for_onnx(test_image)
    
    # ONNX 推理
    encoder_path = "edgetam_encoder.onnx"
    if not os.path.exists(encoder_path):
        print(f"未找到 ONNX 模型 {encoder_path}，跳过对比")
        return
    
    print("运行 ONNX encoder...")
    encoder_session = onnxruntime.InferenceSession(encoder_path)
    onnx_outputs = encoder_session.run(None, {'image': img_np})
    
    print(f"ONNX 输出形状:")
    for i, out in enumerate(onnx_outputs):
        print(f"  Output {i}: {out.shape}")
    
    # PyTorch 推理
    print("\n运行 PyTorch encoder...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    predictor = SAM2ImagePredictor(build_sam2("configs/edgetam.yaml", "../checkpoints/edgetam.pt"))
    predictor.set_image(test_image)
    
    # 获取 PyTorch 特征
    pytorch_feats = predictor._features
    print(f"PyTorch 输出形状:")
    print(f"  high_res_feats[0]: {pytorch_feats['high_res_feats'][0].shape}")
    print(f"  high_res_feats[1]: {pytorch_feats['high_res_feats'][1].shape}")
    print(f"  image_embed: {pytorch_feats['image_embed'].shape}")
    
    # 计算差异
    onnx_feats = [torch.from_numpy(x) for x in onnx_outputs]
    pytorch_feats_list = [
        pytorch_feats['high_res_feats'][0].cpu(),
        pytorch_feats['high_res_feats'][1].cpu(),
        pytorch_feats['image_embed'].cpu(),
    ]
    
    print("\n特征差异 (L2 距离):")
    for i, (onnx_f, pt_f) in enumerate(zip(onnx_feats, pytorch_feats_list)):
        diff = torch.norm(onnx_f.float() - pt_f.float()).item()
        print(f"  Feature {i}: {diff:.6f}")


if __name__ == "__main__":
    main()
