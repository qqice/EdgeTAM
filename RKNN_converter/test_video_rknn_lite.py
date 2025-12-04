# -*- coding: utf-8 -*-
"""
RKNN Lite 版本的 EdgeTAM 视频推理脚本 - 适用于 RK3576/RK3588 等开发板

这个脚本专为在 Rockchip 开发板上本地运行而设计，使用 rknn-toolkit-lite2。
与 test_video_rknn.py 的区别：
- 使用 rknnlite.api.RKNNLite 替代 rknn.api.RKNN
- 不需要 ADB 连接，直接在开发板上运行
- 优化内存使用

使用方法:
    # 在开发板上运行
    python3 test_video_rknn_lite.py \
        --video_dir ./videos/bedroom \
        --point 450,250 \
        --output output_rknn

    # 指定 NPU 核心 (可选，默认自动选择)
    python3 test_video_rknn_lite.py \
        --video_dir ./videos/bedroom \
        --point 450,250 \
        --core_mask 1  # 0: 自动, 1: NPU0, 2: NPU1, 4: NPU2, 7: NPU0+1+2

依赖安装:
    pip3 install numpy opencv-python pillow tqdm scipy torch
    pip3 install rknn_toolkit_lite2-x.x.x-cpXX-cpXX-linux_aarch64.whl

需要的文件:
    - rknn_models/edgetam_encoder.rknn
    - rknn_models/edgetam_decoder.rknn
    - rknn_models/edgetam_memory_encoder.rknn
    - rknn_models/maskmem_pos_enc.npy
    - onnx_models/memory_attention.pt (或 pt_dir 指定的位置)
    - memory_attention_standalone.py
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
from PIL import Image
from tqdm import tqdm

try:
    from scipy import ndimage
except ImportError:
    # 如果 scipy 不可用，使用简单的质心计算
    ndimage = None

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


# ==================== RKNN Lite 后端 ====================

class RKNNLiteBackend:
    """RKNN Lite 后端 - 用于开发板本地推理"""
    
    def __init__(self, rknn_dir: str, core_mask: int = 0):
        """
        Args:
            rknn_dir: RKNN 模型目录
            core_mask: NPU 核心选择
                - 0: 自动选择
                - 1: NPU0
                - 2: NPU1  
                - 4: NPU2
                - 7: NPU0 + NPU1 + NPU2 (三核并行)
        """
        from rknnlite.api import RKNNLite
        self.RKNNLite = RKNNLite
        self.core_mask = core_mask
        
        print(f"  使用 RKNN Lite 后端 (开发板本地模式)")
        print(f"  NPU 核心: {self._core_mask_to_str(core_mask)}")
        print(f"  RKNN 模型目录: {rknn_dir}")
        
        # 加载 RKNN 模型
        self.encoder = self._load_rknn_model(
            os.path.join(rknn_dir, "edgetam_encoder.rknn"), "encoder"
        )
        self.decoder = self._load_rknn_model(
            os.path.join(rknn_dir, "edgetam_decoder.rknn"), "decoder"
        )
        self.memory_encoder = self._load_rknn_model(
            os.path.join(rknn_dir, "edgetam_memory_encoder.rknn"), "memory_encoder"
        )
    
    def _core_mask_to_str(self, core_mask: int) -> str:
        """将 core_mask 转换为可读字符串"""
        if core_mask == 0:
            return "自动"
        cores = []
        if core_mask & 1:
            cores.append("NPU0")
        if core_mask & 2:
            cores.append("NPU1")
        if core_mask & 4:
            cores.append("NPU2")
        return "+".join(cores) if cores else "未知"
    
    def _load_rknn_model(self, model_path: str, name: str):
        """加载 RKNN 模型并初始化运行时"""
        from rknnlite.api import RKNNLite
        
        print(f"  加载 {name}: {model_path}")
        
        rknn = RKNNLite(verbose=False)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败: {model_path}")
        
        # 初始化运行时 - 本地模式，可指定 NPU 核心
        if self.core_mask > 0:
            ret = rknn.init_runtime(core_mask=self.core_mask)
        else:
            ret = rknn.init_runtime()
        
        if ret != 0:
            raise RuntimeError(f"初始化 RKNN 运行时失败: {model_path}")
        
        return rknn
    
    def run_encoder(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """运行 encoder - RKNN 版本已内置标准化，期望 uint8 [0, 255]"""
        outputs = self.encoder.inference(inputs=[image])
        return outputs[0], outputs[1], outputs[2]
    
    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """运行 decoder"""
        return self.decoder.inference(inputs=inputs)
    
    def run_memory_encoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """运行 memory_encoder"""
        return self.memory_encoder.inference(inputs=inputs)
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'encoder') and self.encoder:
            self.encoder.release()
        if hasattr(self, 'decoder') and self.decoder:
            self.decoder.release()
        if hasattr(self, 'memory_encoder') and self.memory_encoder:
            self.memory_encoder.release()


# ==================== 视频推理引擎 ====================

class RKNNLiteVideoInference:
    """RKNN Lite 版本的视频推理引擎 - 用于开发板"""
    
    def __init__(self, rknn_dir: str = "rknn_models", pt_dir: str = "onnx_models",
                 core_mask: int = 0, device: str = "cpu"):
        """
        Args:
            rknn_dir: RKNN 模型目录
            pt_dir: PyTorch 模型目录 (memory_attention.pt)
            core_mask: NPU 核心选择 (0=自动, 1=NPU0, 2=NPU1, 4=NPU2, 7=全部)
            device: PyTorch 设备 (用于 memory attention，开发板建议用 cpu)
        """
        self.device = device
        self.rknn_dir = rknn_dir
        self.pt_dir = pt_dir
        
        print("=" * 60)
        print("初始化 RKNN Lite 视频推理引擎 (开发板模式)")
        print("=" * 60)
        
        # 创建后端
        print("\n加载 RKNN 模型...")
        self.backend = RKNNLiteBackend(rknn_dir, core_mask)
        
        # 加载独立的 Memory Attention 模块 (PyTorch)
        print("\n加载 Memory Attention 模块 (PyTorch)...")
        weights_path = os.path.join(pt_dir, "memory_attention.pt")
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
        
        # 获取物体不存在时的参数 (用于推理时的条件判断)
        self.no_obj_ptr = self.extras.get('no_obj_ptr', None)
        self.no_obj_embed_spatial = self.extras.get('no_obj_embed_spatial', None)
        self.soft_no_obj_ptr = self.extras.get('soft_no_obj_ptr', False)
        self.fixed_no_obj_ptr = self.extras.get('fixed_no_obj_ptr', False)
        self.pred_obj_scores = self.extras.get('pred_obj_scores', True)
        
        if self.no_obj_ptr is not None:
            self.no_obj_ptr = self.no_obj_ptr.cpu().numpy()
            print(f"  no_obj_ptr shape: {self.no_obj_ptr.shape}")
        if self.no_obj_embed_spatial is not None:
            self.no_obj_embed_spatial = self.no_obj_embed_spatial.cpu().numpy()
            print(f"  no_obj_embed_spatial shape: {self.no_obj_embed_spatial.shape}")
        
        # 获取时间位置编码
        maskmem_tpos_enc = self.extras.get('maskmem_tpos_enc', None)
        if maskmem_tpos_enc is not None:
            maskmem_tpos_enc = maskmem_tpos_enc.cpu().numpy()
            print(f"  maskmem_tpos_enc shape: {maskmem_tpos_enc.shape}")
        
        # 记忆库
        self.memory_bank = MemoryBank(
            num_maskmem=self.num_maskmem,
            mem_dim=self.mem_dim,
            hidden_dim=self.hidden_dim,
            maskmem_tpos_enc=maskmem_tpos_enc
        )
        
        # 缓存位置编码
        self._cached_pos_enc = None
        
        print(f"\n配置:")
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  mem_dim: {self.mem_dim}")
        print(f"  num_maskmem: {self.num_maskmem}")
        print("\n初始化完成")
        print("=" * 60)
    
    def preprocess_image_for_encoder(self, image: Image.Image) -> np.ndarray:
        """预处理图像用于 encoder
        
        RKNN encoder 已配置内置标准化，输入应为 [0, 255] 的 uint8
        """
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.uint8)  # [H, W, 3], uint8, 0-255
        img_np = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
        img_np = np.expand_dims(img_np, axis=0)  # [1, 3, H, W]
        return img_np
    
    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码图像"""
        img_np = self.preprocess_image_for_encoder(image)
        high_res_0, high_res_1, image_embed = self.backend.run_encoder(img_np)
        return high_res_0, high_res_1, image_embed
    
    def get_position_encoding(self, H: int = 64, W: int = 64) -> torch.Tensor:
        """获取位置编码 - 与 standalone 版本一致"""
        if self._cached_pos_enc is not None:
            return self._cached_pos_enc
        
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
    def run_memory_attention(self, image_embed: np.ndarray, memory: np.ndarray,
                             memory_pos: np.ndarray, num_obj_ptr_tokens: int,
                             num_spatial_mem: int) -> np.ndarray:
        """运行 Memory Attention (PyTorch) - 与 standalone 版本一致"""
        # 转换 image_embed 为 PyTorch 格式
        # [1, 256, 64, 64] -> [4096, 1, 256]
        image_embed_torch = torch.from_numpy(image_embed).to(self.device, dtype=torch.float32)
        B, C, H, W = image_embed_torch.shape
        curr_feat = image_embed_torch.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # 获取位置编码
        curr_pos = self.get_position_encoding(H, W)  # [HW, 1, C]
        
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
    
    def decode_mask_with_prompt(self, image_embed: np.ndarray, high_res_0: np.ndarray,
                                high_res_1: np.ndarray, point: Tuple[int, int],
                                orig_size: Tuple[int, int], mask_input: np.ndarray = None
                                ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """解码 mask - 使用 prompt"""
        scale_x = IMG_SIZE / orig_size[0]
        scale_y = IMG_SIZE / orig_size[1]
        scaled_point = (point[0] * scale_x, point[1] * scale_y)
        
        point_coords = np.array([[[scaled_point[0], scaled_point[1]]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)
        
        if mask_input is not None:
            mask_input_resized = cv2.resize(mask_input, (256, 256))[np.newaxis, np.newaxis, :, :]
            mask_input_resized = mask_input_resized.astype(np.float32)
            has_mask_input = np.array([1], dtype=np.float32)
        else:
            mask_input_resized = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        
        outputs = self.backend.run_decoder([
            image_embed.astype(np.float32),
            high_res_0.astype(np.float32),
            high_res_1.astype(np.float32),
            point_coords,
            point_labels,
            mask_input_resized,
            has_mask_input,
        ])
        
        masks, iou_predictions, obj_ptr, object_score_logits = outputs
        
        obj_score = float(object_score_logits.flatten()[0])
        
        # 处理 obj_ptr 的 no_obj_ptr 混合 (原本在 decoder 导出中的逻辑)
        obj_ptr = self._apply_no_obj_ptr(obj_ptr, obj_score)
        
        if obj_score <= 0:
            empty_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)
            return empty_mask, 0.0, obj_ptr, obj_score
        
        if masks.ndim == 5:
            masks = masks[0]
        
        masks_for_multimask = masks[0, 1:4]
        iou_for_multimask = iou_predictions[0, 1:4]
        
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
        """解码 mask - 无 prompt 模式"""
        point_coords = np.zeros((1, 1, 2), dtype=np.float32)
        point_labels = np.array([[-1]], dtype=np.float32)
        
        mask_input_resized = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        
        outputs = self.backend.run_decoder([
            image_embed.astype(np.float32),
            high_res_0.astype(np.float32),
            high_res_1.astype(np.float32),
            point_coords,
            point_labels,
            mask_input_resized,
            has_mask_input,
        ])
        
        masks, iou_predictions, obj_ptr, object_score_logits = outputs
        
        obj_score = float(object_score_logits.flatten()[0])
        
        # 处理 obj_ptr 的 no_obj_ptr 混合 (原本在 decoder 导出中的逻辑)
        obj_ptr = self._apply_no_obj_ptr(obj_ptr, obj_score)
        
        if obj_score <= 0:
            empty_mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.float32)
            return empty_mask, 0.0, obj_ptr, obj_score
        
        if masks.ndim == 5:
            masks = masks[0]
        
        best_mask = masks[0, 0]
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
    
    def _apply_no_obj_ptr(self, obj_ptr: np.ndarray, obj_score: float) -> np.ndarray:
        """处理 obj_ptr 的 no_obj_ptr 混合
        
        原本在 decoder 导出中的逻辑，现在移到推理代码中：
        - 当物体存在时 (obj_score > 0)：使用原始 obj_ptr
        - 当物体不存在时：混合或替换为 no_obj_ptr
        """
        if self.no_obj_ptr is None:
            return obj_ptr
        
        is_obj_appearing = obj_score > 0
        
        if self.soft_no_obj_ptr:
            # 使用 sigmoid 软混合
            lambda_is_obj_appearing = 1.0 / (1.0 + np.exp(-obj_score))
        else:
            lambda_is_obj_appearing = 1.0 if is_obj_appearing else 0.0
        
        if self.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        
        obj_ptr = obj_ptr + (1.0 - lambda_is_obj_appearing) * self.no_obj_ptr
        
        return obj_ptr
    
    def encode_memory(self, image_embed: np.ndarray, mask: np.ndarray, 
                      object_score_logits: float) -> Tuple[np.ndarray, np.ndarray]:
        """编码记忆
        
        memory_encoder 输出 maskmem_features 和 maskmem_pos_enc
        no_obj_embed_spatial 的添加逻辑移到这里处理（如果物体不存在）
        """
        image_embed = image_embed.astype(np.float32)
        
        if mask.shape[-1] != IMG_SIZE or mask.shape[-2] != IMG_SIZE:
            mask_resized = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        else:
            mask_resized = mask.astype(np.float32)
        
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mask_logits = np.where(mask_input > 0.5, 10.0, -10.0).astype(np.float32)
        
        outputs = self.backend.run_memory_encoder([
            image_embed,
            mask_logits,
        ])
        
        maskmem_features = outputs[0]  # [B, HW, C]
        maskmem_pos_enc = outputs[1]   # [B, HW, C]
        
        # 添加 no_obj_embed_spatial (当物体不存在时)
        # 原始逻辑在 SAM2MemoryEncoder 中：
        #   if obj_score <= 0: maskmem_features += no_obj_embed_spatial
        if self.no_obj_embed_spatial is not None and object_score_logits <= 0:
            # no_obj_embed_spatial: [C] -> broadcast to [B, HW, C]
            maskmem_features = maskmem_features + self.no_obj_embed_spatial
        
        return maskmem_features, maskmem_pos_enc
    
    def process_frame(self, image: Image.Image, frame_idx: int, 
                      initial_point: Tuple[int, int],
                      prev_mask: np.ndarray = None,
                      is_cond_frame: bool = False) -> Tuple[np.ndarray, float, float]:
        """处理单帧"""
        orig_size = image.size
        
        # 1. 编码图像
        high_res_0, high_res_1, image_embed = self.encode_image(image)
        
        # 2. 获取记忆并进行 memory attention
        memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem = \
            self.memory_bank.get_memories_for_frame(frame_idx)
        
        if memory is not None:
            pix_feat = self.run_memory_attention(
                image_embed, memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem
            )
        else:
            pix_feat = image_embed.copy()
            if self.directly_add_no_mem_embed:
                no_mem = self.no_mem_embed.view(1, -1, 1, 1).cpu().numpy()
                pix_feat = pix_feat + no_mem
        
        # 3. 解码 mask
        if is_cond_frame:
            mask, iou, obj_ptr, obj_score = self.decode_mask_with_prompt(
                pix_feat, high_res_0, high_res_1, initial_point, orig_size, None
            )
        else:
            mask, iou, obj_ptr, obj_score = self.decode_mask_no_prompt(
                pix_feat, high_res_0, high_res_1, orig_size
            )
            
            # 遮挡检测与重新定位
            if obj_score <= 0 or np.sum(mask > 0.5) < 100:
                if prev_mask is not None:
                    center = find_mask_center(prev_mask)
                    if center is not None:
                        mask, iou, obj_ptr, obj_score = self.decode_mask_with_prompt(
                            pix_feat, high_res_0, high_res_1, center, orig_size, prev_mask
                        )
        
        # 4. 编码记忆
        if obj_score > 0:
            maskmem_features, maskmem_pos_enc = self.encode_memory(
                image_embed, mask, obj_score
            )
            
            self.memory_bank.add_memory(
                frame_idx=frame_idx,
                maskmem_features=maskmem_features,
                maskmem_pos_enc=maskmem_pos_enc,
                obj_ptr=obj_ptr,
                object_score_logits=obj_score,
                is_cond_frame=is_cond_frame
            )
        
        return mask, iou, obj_score
    
    def reset(self):
        """重置状态"""
        self.memory_bank.clear()
        self._cached_pos_enc = None
    
    def release(self):
        """释放资源"""
        print("\n释放 RKNN 资源...")
        if hasattr(self, 'backend'):
            self.backend.release()


# ==================== 辅助函数 ====================

def find_mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """计算 mask 的质心"""
    if mask is None:
        return None
    
    binary_mask = (mask > 0.5).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return None
    
    if ndimage is not None:
        center_of_mass = ndimage.center_of_mass(binary_mask)
        if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
            return (int(center_of_mass[1]), int(center_of_mass[0]))
    else:
        # 简单的质心计算
        y_indices, x_indices = np.where(binary_mask > 0)
        if len(x_indices) > 0:
            return (int(np.mean(x_indices)), int(np.mean(y_indices)))
    
    return None


def is_video_file(path: str) -> bool:
    """判断是否为视频文件"""
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def load_video_file(video_path: str, max_frames: int = None) -> Tuple[List[Image.Image], List[str]]:
    """加载视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    frames = []
    frame_names = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_names.append(f"{frame_idx:05d}")
        frame_idx += 1
        
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    return frames, frame_names


def load_frame_dir(frame_dir: str) -> Tuple[List[Image.Image], List[str]]:
    """加载帧目录"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    files = sorted([
        f for f in os.listdir(frame_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    frames = []
    frame_names = []
    
    for f in files:
        path = os.path.join(frame_dir, f)
        frames.append(Image.open(path).convert('RGB'))
        frame_names.append(os.path.splitext(f)[0])
    
    return frames, frame_names


def save_mask(mask: np.ndarray, path: str):
    """保存 mask"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask_uint8)


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
    parser = argparse.ArgumentParser(description='RKNN Lite 版本的 EdgeTAM 视频推理 (开发板)')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--rknn_dir', type=str, default='rknn_models',
                        help='RKNN 模型目录')
    parser.add_argument('--pt_dir', type=str, default='onnx_models',
                        help='PyTorch 模型目录 (memory_attention.pt)')
    parser.add_argument('--core_mask', type=int, default=0,
                        help='NPU 核心选择: 0=自动, 1=NPU0, 2=NPU1, 4=NPU2, 7=全部')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最大处理帧数')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='与另一个输出目录进行对比')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    
    # 自动生成输出目录名称
    if args.output is None:
        video_name = os.path.basename(args.video_dir)
        if os.path.isfile(args.video_dir):
            video_name = os.path.splitext(video_name)[0]
        args.output = os.path.join('output', f'{video_name}_{point_x}_{point_y}_rknn_lite')
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    mask_dir = os.path.join(args.output, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
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
    engine = RKNNLiteVideoInference(
        rknn_dir=args.rknn_dir,
        pt_dir=args.pt_dir,
        core_mask=args.core_mask,
        device="cpu"  # 开发板建议使用 CPU
    )
    
    try:
        # 处理视频
        print("\n开始处理视频...")
        all_masks = []
        iou_scores = []
        obj_scores = []
        
        prev_mask = None
        for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames))):
            is_cond_frame = (i == 0)
            
            mask, iou, obj_score = engine.process_frame(
                frame, i, current_point,
                prev_mask=prev_mask,
                is_cond_frame=is_cond_frame
            )
            
            all_masks.append(mask)
            iou_scores.append(iou)
            obj_scores.append(obj_score)
            prev_mask = mask
            
            # 保存 mask
            mask_path = os.path.join(mask_dir, f"{name}.png")
            save_mask(mask, mask_path)
        
        # 打印统计
        print(f"\n处理完成!")
        print(f"  总帧数: {len(frames)}")
        print(f"  平均 IoU 预测: {np.mean(iou_scores):.3f}")
        print(f"  平均物体分数: {np.mean(obj_scores):.3f}")
        print(f"  输出目录: {args.output}")
        print(f"  Mask 文件: {mask_dir}")
        
        # 打印每帧详情
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
                ref_candidates = [
                    os.path.join(args.compare_with, 'mask', f"{name}.png"),
                    os.path.join(args.compare_with, f"{name}.png"),
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
                        print(f"  帧 {i:3d}: IoU={iou:.3f}, 面积(本机)={int(area_ours)}, 面积(参考)={int(area_ref)}")
            
            if iou_vs_ref:
                print(f"\n  与参考的平均 IoU: {np.mean(iou_vs_ref):.3f}")
                print(f"  最小 IoU: {np.min(iou_vs_ref):.3f}")
                print(f"  最大 IoU: {np.max(iou_vs_ref):.3f}")
    
    finally:
        # 释放资源
        engine.release()


if __name__ == "__main__":
    main()
