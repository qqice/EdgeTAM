# -*- coding: utf-8 -*-
"""
完整的视频推理脚本 - 使用 ONNX 模块 + PyTorch Memory Attention

这个脚本实现了完整的 SAM2/EdgeTAM 视频推理流程:
- ONNX: image_encoder, mask_decoder, memory_encoder
- PyTorch: memory_attention (因为 RoPE 注意力无法导出到 ONNX)

与 test_video_memory.py 功能等效，但使用 ONNX 进行大部分计算。

使用方法:
    python test_video_memory_onnx.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --output output_memory_onnx \
        --visualize
        
    python test_video_memory_onnx.py \
        --video_dir ../examples/01_dog.mp4 \
        --point 633,444 \
        --output output_memory_onnx
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

from sam2.build_sam import build_sam2


# 支持的视频格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


class MemoryBank:
    """记忆库 - 存储历史帧的记忆特征和物体指针
    
    SAM2/EdgeTAM 的记忆机制包含:
    1. 条件帧输出 (cond_frame_outputs): 用户交互的帧，作为"锚点"
    2. 非条件帧输出 (non_cond_frame_outputs): 自动传播的帧
    3. 物体指针 (obj_ptr): 256维向量，编码物体身份信息
    
    每帧记忆包含:
    - maskmem_features: [B, num_latents, mem_dim] 空间记忆特征
    - maskmem_pos_enc: [B, num_latents, mem_dim] 空间位置编码
    - obj_ptr: [B, hidden_dim] 物体指针
    """
    
    def __init__(self, num_maskmem: int = 7, mem_dim: int = 64, hidden_dim: int = 256,
                 max_obj_ptrs: int = 16):
        self.num_maskmem = num_maskmem  # 最多保留的记忆帧数
        self.mem_dim = mem_dim          # 记忆特征维度 (64)
        self.hidden_dim = hidden_dim    # 隐藏维度 (256)
        self.max_obj_ptrs = max_obj_ptrs  # 最大物体指针数量
        
        # 条件帧输出 (用户交互的帧)
        self.cond_frame_outputs: Dict[int, dict] = OrderedDict()
        # 非条件帧输出 (自动传播的帧)
        self.non_cond_frame_outputs: Dict[int, dict] = OrderedDict()
        
    def add_memory(self, frame_idx: int, maskmem_features: np.ndarray, 
                   maskmem_pos_enc: np.ndarray, obj_ptr: np.ndarray,
                   object_score_logits: float, is_cond_frame: bool = False):
        """添加一帧的记忆
        
        Args:
            frame_idx: 帧索引
            maskmem_features: [1, num_latents, mem_dim] 记忆特征
            maskmem_pos_enc: [1, num_latents, mem_dim] 位置编码
            obj_ptr: [1, hidden_dim] 物体指针
            object_score_logits: 物体存在分数 (logits)
            is_cond_frame: 是否是条件帧
        """
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
            # 限制非条件帧数量，保留最近的帧
            while len(self.non_cond_frame_outputs) > self.num_maskmem * 2:
                oldest_key = next(iter(self.non_cond_frame_outputs))
                del self.non_cond_frame_outputs[oldest_key]
    
    def get_memories_for_frame(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """获取用于当前帧的记忆
        
        实现 SAM2 的记忆选择策略:
        1. 所有条件帧 (t_pos=0)
        2. 最近 (num_maskmem - 1) 帧的非条件帧记忆
        3. 物体指针 tokens
        
        Returns:
            memory: [N, 1, mem_dim] 拼接的记忆特征
            memory_pos: [N, 1, mem_dim] 记忆位置编码
            num_obj_ptr_tokens: 物体指针 token 数量
        """
        to_cat_memory = []
        to_cat_memory_pos = []
        
        # 1. 添加条件帧记忆 (t_pos=0)
        for t, out in self.cond_frame_outputs.items():
            if t >= frame_idx:
                continue  # 只使用当前帧之前的条件帧
            
            feats = out["maskmem_features"]  # [1, num_latents, mem_dim]
            pos = out["maskmem_pos_enc"]
            
            # [B, N, C] -> [N, B, C]
            feats = np.transpose(feats, (1, 0, 2))
            pos = np.transpose(pos, (1, 0, 2))
            
            to_cat_memory.append(feats)
            to_cat_memory_pos.append(pos)
        
        # 2. 添加最近 (num_maskmem - 1) 帧的非条件帧记忆
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
            
            feats = np.transpose(feats, (1, 0, 2))
            pos = np.transpose(pos, (1, 0, 2))
            
            to_cat_memory.append(feats)
            to_cat_memory_pos.append(pos)
        
        num_spatial_mem = len(to_cat_memory)
        
        # 3. 添加物体指针
        num_obj_ptr_tokens = 0
        obj_ptrs_list = []
        
        # 从条件帧收集 obj_ptr
        for t, out in self.cond_frame_outputs.items():
            if t < frame_idx:
                obj_ptr = out["obj_ptr"]  # [1, hidden_dim]
                obj_ptrs_list.append(obj_ptr)
        
        # 从最近的非条件帧收集 obj_ptr
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
            # obj_ptrs: [num_ptrs, 1, hidden_dim]
            obj_ptrs = np.stack(obj_ptrs_list, axis=0)
            
            # 如果 mem_dim < hidden_dim，需要拆分
            # hidden_dim=256, mem_dim=64, 所以每个 obj_ptr 拆分为 4 个 tokens
            if self.mem_dim < self.hidden_dim:
                num_ptrs = obj_ptrs.shape[0]
                B = obj_ptrs.shape[1]
                split_factor = self.hidden_dim // self.mem_dim
                # [num_ptrs, B, hidden_dim] -> [num_ptrs, B, split_factor, mem_dim]
                obj_ptrs = obj_ptrs.reshape(num_ptrs, B, split_factor, self.mem_dim)
                # -> [num_ptrs * split_factor, B, mem_dim]
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


class VideoInferenceEngineONNX:
    """视频推理引擎 - 使用 ONNX + PyTorch 混合推理
    
    ONNX 模块:
    - encoder: 图像编码
    - decoder: Mask 解码
    - memory_encoder: 记忆编码
    
    PyTorch 模块:
    - memory_attention: 记忆注意力 (因为 RoPE 无法导出)
    """
    
    def __init__(self, 
                 encoder_path: str,
                 decoder_path: str, 
                 memory_encoder_path: str,
                 model_cfg: str,
                 checkpoint: str,
                 device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载 ONNX 模型
        print("加载 ONNX 模型...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        
        self.encoder_session = onnxruntime.InferenceSession(encoder_path, providers=providers)
        self.decoder_session = onnxruntime.InferenceSession(decoder_path, providers=providers)
        self.memory_encoder_session = onnxruntime.InferenceSession(memory_encoder_path, providers=providers)
        
        print(f"  - Encoder: {encoder_path}")
        print(f"  - Decoder: {decoder_path}")
        print(f"  - Memory Encoder: {memory_encoder_path}")
        
        # 加载 PyTorch 模型用于 Memory Attention
        print("加载 PyTorch 模型用于 Memory Attention...")
        self.sam_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.sam_model.eval()
        
        # 关键模块
        self.memory_attention = self.sam_model.memory_attention
        self.no_mem_embed = self.sam_model.no_mem_embed
        self.no_mem_pos_enc = self.sam_model.no_mem_pos_enc
        self.maskmem_tpos_enc = self.sam_model.maskmem_tpos_enc
        
        # 获取位置编码生成器
        self.image_encoder = self.sam_model.image_encoder
        
        # 模型配置
        self.image_size = 1024
        self.hidden_dim = self.sam_model.hidden_dim  # 256
        self.mem_dim = self.sam_model.mem_dim  # 64
        self.num_maskmem = self.sam_model.num_maskmem  # 7
        
        print(f"  - hidden_dim: {self.hidden_dim}")
        print(f"  - mem_dim: {self.mem_dim}")
        print(f"  - num_maskmem: {self.num_maskmem}")
        
        # 记忆库
        self.memory_bank = MemoryBank(
            num_maskmem=self.num_maskmem,
            mem_dim=self.mem_dim,
            hidden_dim=self.hidden_dim
        )
        
        # ImageNet 标准化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 缓存位置编码
        self._cached_pos_enc = None
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """预处理图像"""
        # Resize 到 1024x1024
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
        # 转换为 numpy 并归一化
        img_np = np.array(resized).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        
        # HWC -> CHW, 添加 batch 维度
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        
        return img_np
    
    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用 ONNX 编码图像
        
        Returns:
            high_res_feats_0: [1, 32, 256, 256]
            high_res_feats_1: [1, 64, 128, 128]
            image_embed: [1, 256, 64, 64]
        """
        img_np = self.preprocess_image(image)
        outputs = self.encoder_session.run(None, {'image': img_np})
        return outputs[0], outputs[1], outputs[2]
    
    def get_position_encoding(self, image_embed: np.ndarray) -> torch.Tensor:
        """获取位置编码
        
        使用 PyTorch 模型的 position_encoding 模块
        """
        if self._cached_pos_enc is not None:
            return self._cached_pos_enc
        
        # 创建与 image_embed 相同形状的 tensor
        B, C, H, W = image_embed.shape
        dummy_input = torch.zeros(B, C, H, W, device=self.device)
        
        # 使用模型的 position_encoding
        pos_enc = self.image_encoder.neck.position_encoding(dummy_input)
        self._cached_pos_enc = pos_enc
        
        return pos_enc
    
    @torch.inference_mode()
    def apply_memory_attention(self, image_embed: np.ndarray, frame_idx: int) -> np.ndarray:
        """应用记忆注意力 (使用 PyTorch)
        
        将当前帧特征与历史记忆融合
        
        Args:
            image_embed: [1, 256, 64, 64] 当前帧的图像特征
            frame_idx: 当前帧索引
            
        Returns:
            pix_feat_with_mem: [1, 256, 64, 64] 融合记忆后的特征
        """
        # 获取历史记忆
        memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem = self.memory_bank.get_memories_for_frame(frame_idx)
        
        # 转换 image_embed 为 PyTorch 格式
        # [1, 256, 64, 64] -> [4096, 1, 256]
        image_embed_torch = torch.from_numpy(image_embed).to(self.device, dtype=torch.float32)
        B, C, H, W = image_embed_torch.shape
        curr_feat = image_embed_torch.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # 获取位置编码
        pos_enc = self.get_position_encoding(image_embed)
        curr_pos = pos_enc.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        if memory is None:
            # 第一帧: 直接添加 no_mem_embed
            pix_feat_with_mem = curr_feat + self.no_mem_embed
        else:
            # 转换记忆为 PyTorch 格式
            memory_torch = torch.from_numpy(memory).to(self.device, dtype=torch.float32)
            memory_pos_torch = torch.from_numpy(memory_pos).to(self.device, dtype=torch.float32)
            
            # 应用记忆注意力
            pix_feat_with_mem = self.memory_attention(
                curr=curr_feat,
                curr_pos=curr_pos,
                memory=memory_torch,
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
        """使用 ONNX 解码 mask
        
        Returns:
            mask: 二值化的 mask (原始尺寸)
            iou: IoU 预测分数
            obj_ptr: [1, 256] 物体指针
            obj_score: 物体存在分数 (logits)
        """
        # 转换点坐标到模型坐标系
        scale_x = self.image_size / orig_size[0]
        scale_y = self.image_size / orig_size[1]
        model_x = point[0] * scale_x
        model_y = point[1] * scale_y
        
        point_coords = np.array([[[model_x, model_y]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)
        
        # 准备 mask 输入
        if mask_input is not None:
            mask_input_resized = cv2.resize(mask_input, (256, 256))[np.newaxis, np.newaxis, :, :]
            mask_input_resized = mask_input_resized.astype(np.float32)
            has_mask_input = np.array([1], dtype=np.float32)
        else:
            mask_input_resized = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        
        # 运行 ONNX decoder
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
        
        # 选择最佳 mask (使用多 mask 策略)
        masks_for_multimask = masks[0, 1:4]  # 多 mask tokens
        iou_for_multimask = iou_predictions[0, 1:4]
        
        # 计算稳定性分数并选择最佳
        stability_scores = self._get_stability_scores(masks_for_multimask)
        combined_scores = stability_scores * iou_for_multimask
        best_idx = np.argmax(combined_scores)
        best_mask = masks_for_multimask[best_idx]
        best_iou = iou_for_multimask[best_idx]
        
        # 后处理: resize 到原始尺寸并二值化
        mask_resized = cv2.resize(best_mask, (orig_size[0], orig_size[1]))
        final_mask = (mask_resized > 0).astype(np.float32)
        
        return final_mask, float(best_iou), obj_ptr, float(object_score_logits[0, 0])
    
    def _get_stability_scores(self, masks: np.ndarray, threshold_offset: float = 0.0) -> np.ndarray:
        """计算 mask 稳定性分数"""
        high_threshold = 0.0 + threshold_offset
        low_threshold = 0.0 - threshold_offset
        
        intersections = np.sum((masks > high_threshold).astype(np.float32), axis=(1, 2))
        unions = np.sum((masks > low_threshold).astype(np.float32), axis=(1, 2))
        
        stability_scores = np.where(unions > 0, intersections / unions, 1.0)
        return stability_scores
    
    def encode_memory(self, image_embed: np.ndarray, mask: np.ndarray, 
                      object_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """使用 ONNX 编码记忆
        
        Args:
            image_embed: [1, 256, 64, 64] 图像特征
            mask: 二值化 mask (原始尺寸或 1024x1024)
            object_score: 物体存在分数 (未使用，因为当前 ONNX 模型不需要)
            
        Returns:
            maskmem_features: [1, 512, 64] 记忆特征
            maskmem_pos_enc: [1, 512, 64] 位置编码
        """
        # 确保 image_embed 是 float32
        image_embed = image_embed.astype(np.float32)
        
        # mask 需要是 1024x1024
        if mask.shape[-1] != self.image_size or mask.shape[-2] != self.image_size:
            mask_resized = cv2.resize(mask.astype(np.float32), (self.image_size, self.image_size))
        else:
            mask_resized = mask.astype(np.float32)
        
        # 转换为 logits 格式 [1, 1, 1024, 1024]
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mask_logits = np.where(mask_input > 0.5, 10.0, -10.0).astype(np.float32)
        
        # 检查 ONNX 模型的输入名称
        input_names = [inp.name for inp in self.memory_encoder_session.get_inputs()]
        
        feed_dict = {
            'pix_feat': image_embed,
            'pred_mask': mask_logits,
        }
        
        # 如果模型需要 object_score_logits，添加它
        if 'object_score_logits' in input_names:
            feed_dict['object_score_logits'] = np.array([[object_score]], dtype=np.float32)
        
        # 运行 ONNX memory encoder
        outputs = self.memory_encoder_session.run(None, feed_dict)
        
        return outputs[0], outputs[1]  # maskmem_features, maskmem_pos_enc
    
    def process_frame(self, frame: Image.Image, frame_idx: int, 
                      point: Tuple[int, int],
                      prev_mask: Optional[np.ndarray] = None,
                      is_cond_frame: bool = False) -> Tuple[np.ndarray, float, float]:
        """处理单帧
        
        完整的帧处理流程:
        1. ONNX 图像编码
        2. PyTorch 记忆注意力融合 (如果有历史记忆)
        3. ONNX Mask 解码
        4. ONNX 记忆编码
        5. 更新记忆库
        
        Args:
            frame: PIL 图像
            frame_idx: 帧索引
            point: 点击位置 (x, y)
            prev_mask: 前一帧的 mask (用于 mask 提示)
            is_cond_frame: 是否是条件帧
            
        Returns:
            mask: 二值化的 mask
            iou: IoU 分数
            obj_score: 物体存在分数 (logits)
        """
        orig_size = frame.size
        
        # 1. ONNX 编码图像
        high_res_0, high_res_1, image_embed = self.encode_image(frame)
        
        # 2. 应用记忆注意力 (PyTorch)
        if frame_idx > 0 and not is_cond_frame:
            image_embed_with_mem = self.apply_memory_attention(image_embed, frame_idx)
        else:
            image_embed_with_mem = image_embed
        
        # 3. 准备 mask 输入
        mask_input = None
        if prev_mask is not None:
            # 将 mask 转换为 logits 格式
            mask_input = np.where(prev_mask > 0.5, 10.0, -10.0).astype(np.float32)
        
        # 4. ONNX 解码 mask
        mask, iou, obj_ptr, obj_score = self.decode_mask(
            image_embed_with_mem, high_res_0, high_res_1, 
            point, orig_size, mask_input
        )
        
        # 5. ONNX 编码记忆
        maskmem_features, maskmem_pos_enc = self.encode_memory(
            image_embed, mask, obj_score
        )
        
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
        """重置推理状态"""
        self.memory_bank.clear()
        self._cached_pos_enc = None


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
                       output_path: str, frame_idx: int, iou: float, obj_score: float,
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
        color = [1.0, 0.2, 0.2, 0.3]  # 红色 (丢失状态)
    colored_mask[mask > 0] = color
    plt.imshow(colored_mask)
    
    # 绘制点击位置
    if point is not None:
        plt.plot(point[0], point[1], 'rx', markersize=15, markeredgewidth=3)
    
    status = "LOST" if lost else f"IoU={iou:.2f}, Score={obj_score:.2f}"
    plt.title(f"Frame {frame_idx} | {status}")
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='使用 ONNX + Memory Bank 的完整视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default='output_memory_onnx',
                        help='输出目录')
    parser.add_argument('--encoder', type=str, default='onnx_models/edgetam_encoder.onnx')
    parser.add_argument('--decoder', type=str, default='onnx_models/edgetam_decoder.onnx')
    parser.add_argument('--memory_encoder', type=str, default='onnx_models/edgetam_memory_encoder.onnx')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/edgetam.pt')
    parser.add_argument('--model_cfg', type=str, default='configs/edgetam.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--min_mask_area', type=float, default=0.001,
                        help='最小 mask 面积比例')
    parser.add_argument('--min_score', type=float, default=-2.0,
                        help='最小物体存在分数 (logits)')
    parser.add_argument('--track_point', action='store_true',
                        help='根据 mask 质心更新点击位置')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output, 'vis'), exist_ok=True)
    
    # 加载视频
    print(f"加载输入: {args.video_dir}")
    if is_video_file(args.video_dir):
        frames, frame_names = load_video_file(args.video_dir, args.max_frames)
    else:
        frames, frame_names = load_frame_dir(args.video_dir)
    
    print(f"共 {len(frames)} 帧")
    if len(frames) == 0:
        print("错误: 未找到帧")
        return
    
    orig_size = frames[0].size
    print(f"原始尺寸: {orig_size[0]}x{orig_size[1]}")
    
    # 初始化推理引擎
    print("\n初始化推理引擎...")
    engine = VideoInferenceEngineONNX(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        memory_encoder_path=args.memory_encoder,
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    
    # 计算最小 mask 面积阈值
    min_area_pixels = orig_size[0] * orig_size[1] * args.min_mask_area
    
    # 开始推理
    print("\n开始视频推理...")
    start_time = time.time()
    
    prev_mask = None
    object_lost = False
    lost_frame_count = 0
    
    for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames), desc="处理帧")):
        is_cond_frame = (i == 0)  # 第一帧是条件帧
        
        # 处理帧
        mask, iou, obj_score = engine.process_frame(
            frame, i,
            point=current_point,
            prev_mask=prev_mask if not is_cond_frame else None,
            is_cond_frame=is_cond_frame
        )
        
        # 计算 mask 面积
        mask_area = mask.sum()
        
        # 检测物体是否丢失
        if mask_area < min_area_pixels or obj_score < args.min_score:
            lost_frame_count += 1
            if lost_frame_count >= 3:
                if not object_lost:
                    print(f"\n[Frame {i}] 物体丢失! (面积={mask_area:.0f}, Score={obj_score:.3f})")
                object_lost = True
        else:
            if object_lost:
                print(f"\n[Frame {i}] 物体恢复! (面积={mask_area:.0f}, Score={obj_score:.3f})")
                # 恢复追踪点
                center = find_mask_center(mask)
                if center is not None:
                    current_point = center
            object_lost = False
            lost_frame_count = 0
        
        # 更新追踪点
        if args.track_point and not object_lost:
            center = find_mask_center(mask)
            if center is not None:
                current_point = center
        
        prev_mask = mask
        
        # 保存 mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(os.path.join(args.output, f'{name}.png'))
        
        # 可视化
        if args.visualize:
            save_visualization(
                frame, mask,
                current_point if not object_lost else None,
                os.path.join(args.output, 'vis', f'{name}.png'),
                i, iou, obj_score, lost=object_lost
            )
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(frames) / total_time
    
    print(f"\n完成!")
    print(f"处理帧数: {len(frames)}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {fps:.2f}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
