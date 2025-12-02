#!/usr/bin/env python3
"""简化的调试脚本 - 对比 PT 和 ONNX decoder 的 object_score_logits"""

import sys
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import cv2
import os

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

IMG_SIZE = 1024


def preprocess_image(frame: Image.Image):
    """预处理图像"""
    frame = frame.convert("RGB")
    frame_resized = frame.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(frame_resized)
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_normalized = (img_np.astype(np.float32) / 255.0 - mean) / std
    img_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, ...]
    
    return img_tensor.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="../examples/01_dog_2.mp4")
    parser.add_argument("--point", type=str, default="633,444")
    parser.add_argument("--frames", type=str, default="0,70,75,78,80,85,90,95,100,103,104,110")
    args = parser.parse_args()
    
    point = tuple(map(int, args.point.split(',')))
    frame_indices = list(map(int, args.frames.split(',')))
    
    print(f"\n=== 调试 Decoder 的 object_score_logits ===")
    print(f"视频: {args.video}")
    print(f"点: {point}")
    print(f"帧: {frame_indices}")
    
    # 加载 PT 模型
    model_cfg = "configs/edgetam.yaml"
    checkpoint = str(parent_dir / "checkpoints" / "edgetam.pt")
    
    sam2_model: SAM2Base = build_sam2(model_cfg, checkpoint, device=torch.device("cpu"))
    sam2_model.eval()
    
    # 加载 ONNX 模型
    onnx_dir = current_dir / "onnx_models"
    encoder_session = ort.InferenceSession(str(onnx_dir / "edgetam_encoder.onnx"))
    decoder_session = ort.InferenceSession(str(onnx_dir / "edgetam_decoder.onnx"))
    
    # 打开视频
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数: {total_frames}")
    
    print("\n帧      ONNX Score   PT Score     差异")
    print("-" * 50)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"{frame_idx:3d}   读取失败")
            continue
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame_rgb)
        orig_size = frame.size  # (W, H)
        
        # 预处理
        img_input = preprocess_image(frame)
        img_tensor = torch.from_numpy(img_input)
        
        # ========== ONNX 推理 ==========
        onnx_enc_out = encoder_session.run(None, {'image': img_input})
        onnx_high_res_0, onnx_high_res_1, onnx_image_embed = onnx_enc_out
        
        # 准备 decoder 输入
        scale_x = IMG_SIZE / orig_size[0]
        scale_y = IMG_SIZE / orig_size[1]
        model_x = point[0] * scale_x
        model_y = point[1] * scale_y
        
        point_coords = np.array([[[model_x, model_y]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        
        onnx_dec_out = decoder_session.run(None, {
            'image_embed': onnx_image_embed,
            'high_res_feats_0': onnx_high_res_0,
            'high_res_feats_1': onnx_high_res_1,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
        })
        onnx_masks, onnx_iou, onnx_obj_ptr, onnx_obj_score = onnx_dec_out
        onnx_score = float(onnx_obj_score[0, 0])
        
        # ========== PT 推理 ==========
        with torch.no_grad():
            # Encoder
            backbone_out = sam2_model.forward_image(img_tensor)
            _, pt_vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
            
            if sam2_model.directly_add_no_mem_embed:
                pt_vision_feats[-1] = pt_vision_feats[-1] + sam2_model.no_mem_embed
            
            # 使用固定的 feat sizes
            bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
            
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(pt_vision_feats[::-1], bb_feat_sizes[::-1])
            ][::-1]
            
            pt_image_embed = feats[-1]
            pt_high_res_0 = feats[0]
            pt_high_res_1 = feats[1]
            
            # Decoder
            pt_point_coords = torch.tensor([[[model_x, model_y]]], dtype=torch.float32)
            pt_point_labels = torch.tensor([[1]], dtype=torch.int32)
            
            sparse_embeddings, dense_embeddings = sam2_model.sam_prompt_encoder(
                points=(pt_point_coords, pt_point_labels),
                boxes=None,
                masks=None,
            )
            
            pt_masks, pt_iou, pt_tokens, pt_obj_score_tensor = sam2_model.sam_mask_decoder.predict_masks(
                image_embeddings=pt_image_embed,
                image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                repeat_image=False,
                high_res_features=[pt_high_res_0, pt_high_res_1],
            )
            pt_score = float(pt_obj_score_tensor[0, 0])
        
        diff = abs(onnx_score - pt_score)
        print(f"{frame_idx:3d}     {onnx_score:8.3f}    {pt_score:8.3f}    {diff:8.3f}")
    
    cap.release()
    print("-" * 50)


if __name__ == "__main__":
    main()
