#!/usr/bin/env python3
"""
深入调试 memory attention 后的特征差异
比较原始 PT 模型和 standalone 实现在相同记忆输入下的输出
"""

import sys
import os
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from sam2.build_sam import build_sam2_video_predictor

IMG_SIZE = 1024


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="../examples/01_dog_2.mp4")
    parser.add_argument("--point", type=str, default="633,444")
    parser.add_argument("--start_frame", type=int, default=70)
    parser.add_argument("--end_frame", type=int, default=110)
    args = parser.parse_args()
    
    point = tuple(map(int, args.point.split(',')))
    
    print(f"\n=== 深入调试 Memory Attention ===")
    print(f"视频: {args.video}")
    print(f"点: {point}")
    print(f"帧范围: {args.start_frame} - {args.end_frame}")
    
    # 创建临时目录保存视频帧
    temp_dir = Path("/tmp/edgetam_debug_frames")
    temp_dir.mkdir(exist_ok=True)
    
    # 提取视频帧
    print("\n提取视频帧...")
    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = temp_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
    cap.release()
    print(f"提取了 {frame_idx} 帧")
    
    # 加载原始 PT 模型
    print("\n加载原始 PT 模型...")
    model_cfg = "configs/edgetam.yaml"
    checkpoint = str(parent_dir / "checkpoints" / "edgetam.pt")
    
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=torch.device("cpu"))
    
    # 初始化推理状态
    print("初始化推理状态...")
    inference_state = predictor.init_state(video_path=str(temp_dir))
    
    # 在第一帧添加点击
    print(f"在第一帧添加点击: {point}")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[point[0], point[1]]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    
    print("\n开始推理并收集中间结果...")
    print("\n帧     Obj Score    Mask Area    Mask Center")
    print("-" * 55)
    
    # 逐帧推理
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
        if frame_idx < args.start_frame:
            continue
        if frame_idx > args.end_frame:
            break
            
        # 获取 mask
        mask_logits = video_res_masks[0, 0].cpu().numpy()
        mask = (mask_logits > 0).astype(np.float32)
        area = int(mask.sum())
        
        # 获取 object_score_logits
        output_dict = inference_state["output_dict"]
        frame_out = output_dict["cond_frame_outputs"].get(frame_idx, None)
        if frame_out is None:
            frame_out = output_dict["non_cond_frame_outputs"].get(frame_idx, None)
        
        if frame_out is not None:
            obj_score = float(frame_out["object_score_logits"][0, 0].cpu().numpy())
        else:
            obj_score = 0.0
        
        # 计算质心
        if area > 0:
            from scipy import ndimage
            center = ndimage.center_of_mass(mask)
            center_str = f"({int(center[1]):4d}, {int(center[0]):4d})"
        else:
            center_str = "       None"
        
        print(f"{frame_idx:4d}    {obj_score:8.3f}    {area:8d}    {center_str}")
        
        # 关键：检查这个帧的 memory attention 输入和输出
        if frame_idx >= args.start_frame and frame_idx <= args.end_frame:
            # 查看 pix_feat_with_mem 的统计信息
            if "pix_feat_with_mem" in frame_out:
                pix_feat = frame_out["pix_feat_with_mem"]
                print(f"        pix_feat_with_mem: mean={pix_feat.mean():.4f}, std={pix_feat.std():.4f}, "
                      f"min={pix_feat.min():.4f}, max={pix_feat.max():.4f}")
    
    print("-" * 55)
    
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir)
    print("\n调试完成")


if __name__ == "__main__":
    main()
