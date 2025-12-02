# -*- coding: utf-8 -*-
"""
完全独立的 EdgeTAM 视频推理脚本

这个脚本不依赖 sam2 库，只使用:
1. ONNX 模型: encoder, decoder, memory_encoder
2. 独立 PyTorch 模块: memory_attention_standalone.py

通过与 test_video_memory_onnx.py (使用 sam2) 的结果对比，验证实现的正确性。

使用方法:
    python test_video_standalone.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --output output_standalone \
        --compare_with output_memory_onnx
"""

import os
import sys
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

# 导入独立的 Memory Attention 模块
from memory_attention_standalone import build_memory_attention


# ==================== 常量 ====================

IMG_SIZE = 1024
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


# ==================== 记忆库 ====================

class MemoryBank:
    """记忆库 - 存储历史帧的记忆特征和物体指针"""
    
    def __init__(self, num_maskmem: int = 7, mem_dim: int = 64, hidden_dim: int = 256,
                 max_obj_ptrs: int = 16, maskmem_tpos_enc: np.ndarray = None):
        self.num_maskmem = num_maskmem
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.max_obj_ptrs = max_obj_ptrs
        # 时间位置编码: [num_maskmem, 1, 1, mem_dim]
        self.maskmem_tpos_enc = maskmem_tpos_enc
        
        self.cond_frame_outputs: Dict[int, dict] = OrderedDict()
        self.non_cond_frame_outputs: Dict[int, dict] = OrderedDict()
        
    def add_memory(self, frame_idx: int, maskmem_features: np.ndarray, 
                   maskmem_pos_enc: np.ndarray, obj_ptr: np.ndarray,
                   object_score_logits: float, is_cond_frame: bool = False):
        """添加一帧的记忆"""
        output = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        
        if is_cond_frame:
            self.cond_frame_outputs[frame_idx] = output
        else:
            self.non_cond_frame_outputs[frame_idx] = output
            while len(self.non_cond_frame_outputs) > self.num_maskmem * 2:
                oldest_key = next(iter(self.non_cond_frame_outputs))
                del self.non_cond_frame_outputs[oldest_key]
    
    def get_memories_for_frame(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
        """获取用于当前帧的记忆
        
        关键：需要为每个记忆帧添加时间位置编码 maskmem_tpos_enc
        - 条件帧 t_pos=0
        - 非条件帧 t_pos=1 到 num_maskmem-1
        """
        to_cat_memory = []
        to_cat_memory_pos = []
        
        # 1. 添加条件帧记忆 (t_pos=0)
        for t, out in self.cond_frame_outputs.items():
            if t >= frame_idx:
                continue
            feats = out["maskmem_features"]
            pos = out["maskmem_pos_enc"]
            feats = np.transpose(feats, (1, 0, 2))  # [B, HW, C] -> [HW, B, C]
            pos = np.transpose(pos, (1, 0, 2))      # [B, HW, C] -> [HW, B, C]
            
            # 添加时间位置编码 t_pos=0 -> maskmem_tpos_enc[num_maskmem - 0 - 1] = [6]
            if self.maskmem_tpos_enc is not None:
                t_pos = 0
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]  # [1, 1, mem_dim]
                tpos_enc = tpos_enc.reshape(1, 1, -1)  # [1, 1, mem_dim]
                pos = pos + tpos_enc  # broadcast add
            
            to_cat_memory.append(feats)
            to_cat_memory_pos.append(pos)
        
        # 2. 添加最近的非条件帧记忆 (t_pos=1 到 num_maskmem-1)
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                prev_frame_idx = frame_idx - 1
            else:
                prev_frame_idx = frame_idx - t_rel
            
            if prev_frame_idx < 0:
                continue
            
            out = self.non_cond_frame_outputs.get(prev_frame_idx, None)
            if out is None:
                continue
            
            feats = out["maskmem_features"]
            pos = out["maskmem_pos_enc"]
            feats = np.transpose(feats, (1, 0, 2))  # [B, HW, C] -> [HW, B, C]
            pos = np.transpose(pos, (1, 0, 2))      # [B, HW, C] -> [HW, B, C]
            
            # 添加时间位置编码
            if self.maskmem_tpos_enc is not None:
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]  # [1, 1, mem_dim]
                tpos_enc = tpos_enc.reshape(1, 1, -1)  # [1, 1, mem_dim]
                pos = pos + tpos_enc  # broadcast add
            
            to_cat_memory.append(feats)
            to_cat_memory_pos.append(pos)
        
        num_spatial_mem = len(to_cat_memory)
        
        # 3. 添加物体指针
        num_obj_ptr_tokens = 0
        obj_ptrs_list = []
        
        for t, out in self.cond_frame_outputs.items():
            if t < frame_idx:
                obj_ptr = out["obj_ptr"]
                obj_ptrs_list.append(obj_ptr)
        
        for t_diff in range(1, self.max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = self.non_cond_frame_outputs.get(t, None)
            if out is not None:
                obj_ptr = out["obj_ptr"]
                obj_ptrs_list.append(obj_ptr)
            
            if len(obj_ptrs_list) >= self.max_obj_ptrs:
                break
        
        if len(obj_ptrs_list) > 0:
            obj_ptrs = np.stack(obj_ptrs_list, axis=0)
            
            if self.mem_dim < self.hidden_dim:
                num_ptrs = obj_ptrs.shape[0]
                B = obj_ptrs.shape[1]
                split_factor = self.hidden_dim // self.mem_dim
                obj_ptrs = obj_ptrs.reshape(num_ptrs, B, split_factor, self.mem_dim)
                obj_ptrs = np.transpose(obj_ptrs, (0, 2, 1, 3)).reshape(-1, B, self.mem_dim)
            
            obj_ptr_pos = np.zeros_like(obj_ptrs)
            
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_ptr_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]
        
        if len(to_cat_memory) == 0:
            return None, None, 0, 0
        
        memory = np.concatenate(to_cat_memory, axis=0)
        memory_pos = np.concatenate(to_cat_memory_pos, axis=0)
        
        return memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem
    
    def clear(self):
        """清空记忆库"""
        self.cond_frame_outputs.clear()
        self.non_cond_frame_outputs.clear()


# ==================== 独立推理引擎 ====================

class StandaloneVideoInference:
    """完全独立的视频推理引擎 - 不依赖 sam2 库"""
    
    def __init__(self, onnx_dir: str, device: str = "cpu"):
        self.device = device
        self.onnx_dir = onnx_dir
        
        print("=" * 60)
        print("初始化独立视频推理引擎")
        print("=" * 60)
        
        # 加载 ONNX 模型
        print("加载 ONNX 模型...")
        providers = ['CPUExecutionProvider']
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.encoder_session = ort.InferenceSession(
            os.path.join(onnx_dir, "edgetam_encoder.onnx"),
            providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            os.path.join(onnx_dir, "edgetam_decoder.onnx"),
            providers=providers
        )
        self.memory_encoder_session = ort.InferenceSession(
            os.path.join(onnx_dir, "edgetam_memory_encoder.onnx"),
            providers=providers
        )
        
        # 加载独立的 Memory Attention 模块
        print("加载独立 Memory Attention 模块...")
        weights_path = os.path.join(onnx_dir, "memory_attention.pt")
        self.memory_attention, self.extras, self.config = build_memory_attention(
            weights_path, device=device
        )
        
        # 获取配置
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.mem_dim = self.config.get('mem_dim', 64)
        self.num_maskmem = self.config.get('num_maskmem', 7)
        self.directly_add_no_mem_embed = self.config.get('directly_add_no_mem_embed', True)
        
        # 获取关键参数
        self.no_mem_embed = self.extras.get('no_mem_embed', torch.zeros(1, 1, 256))
        
        # 获取时间位置编码 maskmem_tpos_enc: [num_maskmem, 1, 1, mem_dim]
        maskmem_tpos_enc = self.extras.get('maskmem_tpos_enc', None)
        if maskmem_tpos_enc is not None:
            maskmem_tpos_enc = maskmem_tpos_enc.cpu().numpy()
            print(f"  maskmem_tpos_enc shape: {maskmem_tpos_enc.shape}")
        else:
            print("  警告: maskmem_tpos_enc 未找到!")
        
        # 记忆库 (传入时间位置编码)
        self.memory_bank = MemoryBank(
            num_maskmem=self.num_maskmem,
            mem_dim=self.mem_dim,
            hidden_dim=self.hidden_dim,
            maskmem_tpos_enc=maskmem_tpos_enc
        )
        
        # 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 缓存位置编码
        self._cached_pos_enc = None
        
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  mem_dim: {self.mem_dim}")
        print(f"  num_maskmem: {self.num_maskmem}")
        print("初始化完成")
        print("=" * 60)
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """预处理图像"""
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np
    
    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码图像"""
        img_np = self.preprocess_image(image)
        outputs = self.encoder_session.run(None, {'image': img_np})
        return outputs[0], outputs[1], outputs[2]  # high_res_0, high_res_1, image_embed
    
    def get_position_encoding(self, H: int = 64, W: int = 64) -> torch.Tensor:
        """获取位置编码 (正弦位置编码)"""
        if self._cached_pos_enc is not None:
            return self._cached_pos_enc
        
        # 使用 SAM 的正弦位置编码
        dim = self.hidden_dim
        scale = 2 * np.pi
        
        y_embed = np.arange(1, H + 1, dtype=np.float32).reshape(H, 1)
        y_embed = np.tile(y_embed, (1, W))
        x_embed = np.arange(1, W + 1, dtype=np.float32).reshape(1, W)
        x_embed = np.tile(x_embed, (H, 1))
        
        y_embed = y_embed / (H + 1e-6) * scale
        x_embed = x_embed / (W + 1e-6) * scale
        
        dim_t = np.arange(dim // 2, dtype=np.float32)
        dim_t = 10000.0 ** (2 * (dim_t // 2) / (dim // 2))
        
        pos_x = x_embed[:, :, np.newaxis] / dim_t
        pos_y = y_embed[:, :, np.newaxis] / dim_t
        
        pos_x = np.stack([np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])], axis=3).reshape(H, W, dim // 2)
        pos_y = np.stack([np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])], axis=3).reshape(H, W, dim // 2)
        
        pos = np.concatenate([pos_y, pos_x], axis=2)  # [H, W, dim]
        pos = pos.reshape(H * W, dim)  # [HW, dim]
        pos = pos[np.newaxis, :, :]  # [1, HW, dim]
        pos = np.transpose(pos, (1, 0, 2))  # [HW, 1, dim]
        
        self._cached_pos_enc = torch.from_numpy(pos.astype(np.float32)).to(self.device)
        return self._cached_pos_enc
    
    @torch.inference_mode()
    def apply_memory_attention(self, image_embed: np.ndarray, frame_idx: int) -> np.ndarray:
        """应用记忆注意力"""
        memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem = self.memory_bank.get_memories_for_frame(frame_idx)
        
        # 转换 image_embed 为 PyTorch 格式
        # [1, 256, 64, 64] -> [4096, 1, 256]
        image_embed_torch = torch.from_numpy(image_embed).to(self.device, dtype=torch.float32)
        B, C, H, W = image_embed_torch.shape
        curr_feat = image_embed_torch.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # 获取位置编码
        curr_pos = self.get_position_encoding(H, W)  # [HW, 1, C]
        
        if memory is None:
            # 没有记忆，添加 no_mem_embed
            pix_feat_with_mem = curr_feat + self.no_mem_embed.view(1, 1, -1)
        else:
            # 转换记忆为 PyTorch 格式
            memory_torch = torch.from_numpy(memory).to(self.device, dtype=torch.float32)
            memory_pos_torch = torch.from_numpy(memory_pos).to(self.device, dtype=torch.float32)
            
            # 运行 Memory Attention
            pix_feat_with_mem = self.memory_attention(
                curr=curr_feat,
                memory=memory_torch,
                curr_pos=curr_pos,
                memory_pos=memory_pos_torch,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                num_spatial_mem=num_spatial_mem,
            )
        
        # 转换回 [1, 256, 64, 64]
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        
        return pix_feat_with_mem.cpu().numpy()
    
    def decode_mask(self, image_embed: np.ndarray, high_res_0: np.ndarray, high_res_1: np.ndarray,
                    point: Tuple[int, int], orig_size: Tuple[int, int],
                    mask_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """解码 mask"""
        scale_x = IMG_SIZE / orig_size[0]
        scale_y = IMG_SIZE / orig_size[1]
        model_x = point[0] * scale_x
        model_y = point[1] * scale_y
        
        point_coords = np.array([[[model_x, model_y]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)
        
        if mask_input is not None:
            mask_input_resized = cv2.resize(mask_input, (256, 256))[np.newaxis, np.newaxis, :, :]
            mask_input_resized = mask_input_resized.astype(np.float32)
            has_mask_input = np.array([1], dtype=np.float32)
        else:
            mask_input_resized = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        
        outputs = self.decoder_session.run(None, {
            'image_embed': image_embed,
            'high_res_feats_0': high_res_0,
            'high_res_feats_1': high_res_1,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input_resized,
            'has_mask_input': has_mask_input,
        })
        
        masks, iou_predictions, obj_ptr, object_score_logits = outputs
        
        obj_score = float(object_score_logits[0, 0])
        
        # 关键：当 object_score_logits <= 0 时，物体不存在，返回空 mask
        if obj_score <= 0:
            # 物体不存在，返回空 mask
            empty_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)
            return empty_mask, 0.0, obj_ptr, obj_score
        
        # 处理 masks 的形状：可能是 [1, 4, H, W] 或 [1, 1, 4, H, W]
        if masks.ndim == 5:
            masks = masks[0]  # [1, 4, H, W]
        
        # 选择最佳 mask (使用 mask 1-3，跳过 mask 0)
        masks_for_multimask = masks[0, 1:4]  # [3, H, W]
        iou_for_multimask = iou_predictions[0, 1:4]  # [3]
        
        stability_scores = self._get_stability_scores(masks_for_multimask)
        combined_scores = stability_scores * iou_for_multimask
        best_idx = np.argmax(combined_scores)
        best_mask = masks_for_multimask[best_idx]
        best_iou = iou_for_multimask[best_idx]
        
        mask_resized = cv2.resize(best_mask, (orig_size[0], orig_size[1]))
        final_mask = (mask_resized > 0).astype(np.float32)
        
        return final_mask, float(best_iou), obj_ptr, obj_score
    
    def decode_mask_no_prompt(self, image_embed: np.ndarray, high_res_0: np.ndarray, 
                               high_res_1: np.ndarray, orig_size: Tuple[int, int]
                               ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """解码 mask - 无 prompt 模式
        
        用于视频跟踪时的非条件帧，只依赖 memory attention 后的特征
        使用 label=-1 的空点来表示没有 prompt
        """
        # 使用 label=-1 的空点（与 SAM2 行为一致）
        point_coords = np.zeros((1, 1, 2), dtype=np.float32)
        point_labels = np.array([[-1]], dtype=np.float32)  # -1 表示 padding/无 prompt
        
        mask_input_resized = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        
        outputs = self.decoder_session.run(None, {
            'image_embed': image_embed,
            'high_res_feats_0': high_res_0,
            'high_res_feats_1': high_res_1,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input_resized,
            'has_mask_input': has_mask_input,
        })
        
        masks, iou_predictions, obj_ptr, object_score_logits = outputs
        
        obj_score = float(object_score_logits[0, 0])
        
        # 关键：当 object_score_logits <= 0 时，物体不存在，返回空 mask
        if obj_score <= 0:
            empty_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)
            return empty_mask, 0.0, obj_ptr, obj_score
        
        # 处理 masks 的形状
        if masks.ndim == 5:
            masks = masks[0]
        
        # 非 multimask 模式：使用 mask 0（单 mask 输出）
        # 在 SAM2 中，_use_multimask 返回 False 表示非初始条件帧
        best_mask = masks[0, 0]  # [H, W]
        best_iou = iou_predictions[0, 0]
        
        mask_resized = cv2.resize(best_mask, (orig_size[0], orig_size[1]))
        final_mask = (mask_resized > 0).astype(np.float32)
        
        return final_mask, float(best_iou), obj_ptr, obj_score
    
    def _get_stability_scores(self, masks: np.ndarray, threshold_offset: float = 0.0) -> np.ndarray:
        """计算 mask 稳定性分数"""
        high_threshold = 0.0 + threshold_offset
        low_threshold = 0.0 - threshold_offset
        
        intersections = np.sum((masks > high_threshold).astype(np.float32), axis=(1, 2))
        unions = np.sum((masks > low_threshold).astype(np.float32), axis=(1, 2))
        
        stability_scores = np.where(unions > 0, intersections / unions, 1.0)
        return stability_scores
    
    def encode_memory(self, image_embed: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """编码记忆"""
        image_embed = image_embed.astype(np.float32)
        
        if mask.shape[-1] != IMG_SIZE or mask.shape[-2] != IMG_SIZE:
            mask_resized = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        else:
            mask_resized = mask.astype(np.float32)
        
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mask_logits = np.where(mask_input > 0.5, 10.0, -10.0).astype(np.float32)
        
        outputs = self.memory_encoder_session.run(None, {
            'pix_feat': image_embed,
            'pred_mask': mask_logits,
        })
        
        return outputs[0], outputs[1]  # maskmem_features, maskmem_pos_enc
    
    def process_frame(self, frame: Image.Image, frame_idx: int, 
                      point: Tuple[int, int],
                      prev_mask: Optional[np.ndarray] = None,
                      is_cond_frame: bool = False) -> Tuple[np.ndarray, float, float]:
        """处理单帧
        
        关键：在非条件帧时，不使用 point 和 mask 输入，只依赖 memory attention
        这与原始 SAM2 的 propagate_in_video 行为一致
        """
        orig_size = frame.size
        
        # 1. 编码图像
        high_res_0, high_res_1, image_embed = self.encode_image(frame)
        
        # 2. 应用记忆注意力
        if frame_idx > 0 and not is_cond_frame:
            image_embed_with_mem = self.apply_memory_attention(image_embed, frame_idx)
        else:
            image_embed_with_mem = image_embed
        
        # 3. 准备输入
        # 关键：只有条件帧使用 point 和 mask 输入
        # 非条件帧只依赖 memory attention，不使用 point/mask
        if is_cond_frame:
            # 条件帧：使用 point 输入
            mask_input = None
            if prev_mask is not None:
                mask_input = np.where(prev_mask > 0.5, 10.0, -10.0).astype(np.float32)
            
            # 4. 解码 mask (使用 point)
            mask, iou, obj_ptr, obj_score = self.decode_mask(
                image_embed_with_mem, high_res_0, high_res_1,
                point, orig_size, mask_input
            )
        else:
            # 非条件帧：不使用 point 和 mask 输入
            # 需要一个特殊的 decode 方法，不传递 point
            mask, iou, obj_ptr, obj_score = self.decode_mask_no_prompt(
                image_embed_with_mem, high_res_0, high_res_1, orig_size
            )
        
        # 5. 编码记忆
        maskmem_features, maskmem_pos_enc = self.encode_memory(image_embed, mask)
        
        # 6. 添加到记忆库
        self.memory_bank.add_memory(
            frame_idx,
            maskmem_features,
            maskmem_pos_enc,
            obj_ptr,
            obj_score,
            is_cond_frame=is_cond_frame
        )
        
        return mask, iou, obj_score
    
    def reset(self):
        """重置状态"""
        self.memory_bank.clear()
        self._cached_pos_enc = None


# ==================== 辅助函数 ====================

def is_video_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def load_video_file(video_path: str, max_frames: int = None) -> Tuple[List[Image.Image], List[str]]:
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
    score_text = f"IoU: {iou:.2f} | Score: {obj_score:.2f}"
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
    parser = argparse.ArgumentParser(description='完全独立的 EdgeTAM 视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: ./output/{视频名}_{X}_{Y})')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--track_point', action='store_true',
                        help='根据 mask 质心更新点击位置')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='与另一个输出目录进行对比')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图像')
    parser.add_argument('--save_video', action='store_true',
                        help='生成可视化视频')
    parser.add_argument('--video_fps', type=float, default=30.0,
                        help='输出视频帧率')
    parser.add_argument('--mask_color', type=str, default='blue',
                        choices=['blue', 'green', 'red', 'yellow', 'cyan', 'magenta'],
                        help='mask 颜色')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    initial_point = current_point  # 保存初始点位置
    
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
        # 获取视频名称（不带后缀）
        video_name = os.path.basename(args.video_dir)
        if os.path.isfile(args.video_dir):
            video_name = os.path.splitext(video_name)[0]
        # 生成默认输出目录: ./output/{视频名}_{X}_{Y}
        args.output = os.path.join('output', f'{video_name}_{point_x}_{point_y}')
    
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
        frames, frame_names = load_video_file(args.video_dir, args.max_frames)
    else:
        frames, frame_names = load_frame_dir(args.video_dir)
        if args.max_frames:
            frames = frames[:args.max_frames]
            frame_names = frame_names[:args.max_frames]
    
    print(f"共 {len(frames)} 帧, 图像尺寸: {frames[0].size}")
    
    # 初始化推理引擎
    engine = StandaloneVideoInference(args.onnx_dir, device=args.device)
    
    # 处理视频
    print("\n开始处理视频...")
    all_masks = []
    all_frames = []
    iou_scores = []
    obj_scores = []
    vis_paths = []
    point_history = [current_point]
    
    prev_mask = None
    for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames))):
        is_cond_frame = (i == 0)
        
        mask, iou, obj_score = engine.process_frame(
            frame, i, current_point,
            prev_mask=prev_mask,
            is_cond_frame=is_cond_frame
        )
        
        all_masks.append(mask)
        all_frames.append(frame)
        iou_scores.append(iou)
        obj_scores.append(obj_score)
        prev_mask = mask
        
        # 保存 mask 到 mask 子目录
        mask_path = os.path.join(mask_dir, f"{name}.png")
        save_mask(mask, mask_path)
        
        # 生成可视化图像
        if args.visualize or args.save_video:
            vis_img = create_visualization(
                frame, mask, current_point, i, iou, obj_score,
                color=mask_color, alpha=0.5
            )
            vis_path = os.path.join(vis_dir, f"{name}.jpg")
            cv2.imwrite(vis_path, vis_img)
            vis_paths.append(vis_path)
        
        # 更新点击位置
        if args.track_point and i > 0:
            new_point = find_mask_center(mask)
            if new_point is not None:
                current_point = new_point
        point_history.append(current_point)
    
    # 生成视频
    if args.save_video and vis_paths:
        print("\n生成可视化视频...")
        video_path = os.path.join(args.output, 'output_video.mp4')
        create_video_from_frames(vis_paths, video_path, fps=args.video_fps)
    
    # 打印统计
    print(f"\n处理完成!")
    print(f"  总帧数: {len(frames)}")
    print(f"  平均 IoU 预测: {np.mean(iou_scores):.3f}")
    print(f"  平均物体分数: {np.mean(obj_scores):.3f}")
    print(f"  输出目录: {args.output}")
    print(f"  Mask 文件: {mask_dir}")
    if args.visualize or args.save_video:
        print(f"  可视化图像: {vis_dir}")
    if args.save_video:
        print(f"  可视化视频: {video_path}")
    
    # 打印每帧的详细信息
    print("\n帧详细信息:")
    print(f"{'Frame':>6} | {'IoU':>8} | {'Score':>8} | {'Area':>10} | {'Center':>15}")
    print("-" * 55)
    for i, (mask, iou, score) in enumerate(zip(all_masks, iou_scores, obj_scores)):
        area = int(np.sum(mask > 0.5))
        center = find_mask_center(mask)
        center_str = f"({center[0]:4d}, {center[1]:4d})" if center else "None"
        print(f"{i:>6} | {iou:>8.3f} | {score:>8.3f} | {area:>10} | {center_str:>15}")
    
    # 与参考结果对比
    if args.compare_with and os.path.exists(args.compare_with):
        print(f"\n与参考结果对比: {args.compare_with}")
        
        iou_vs_ref = []
        for i, name in enumerate(frame_names):
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
                    print(f"  帧 {i:3d}: IoU={iou:.3f}, 面积(ours)={area_ours}, 面积(ref)={area_ref}")
        
        if iou_vs_ref:
            print(f"\n  与参考的平均 IoU: {np.mean(iou_vs_ref):.3f}")
            print(f"  最小 IoU: {np.min(iou_vs_ref):.3f}")
            print(f"  最大 IoU: {np.max(iou_vs_ref):.3f}")


if __name__ == "__main__":
    main()
