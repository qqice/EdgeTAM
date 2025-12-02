# -*- coding: utf-8 -*-
"""
RKNN 版本的 EdgeTAM 视频推理脚本

这个脚本使用 RKNN 模型进行推理:
1. RKNN 模型: encoder, decoder, memory_encoder
2. 独立 PyTorch 模块: memory_attention_standalone.py

支持两种模式:
- 真机模式: --target rk3588 (需要 ADB 连接到 RK3588 设备)
- ONNX 模拟模式: --simulate (使用 ONNX 模型在 PC 上模拟，验证流程正确性)

注意: RKNN 的 load_rknn() 加载的模型不支持 PC 模拟器，必须连接真机。
如果要在 PC 上测试，请使用 --simulate 参数使用 ONNX 模型模拟。

使用方法:
    # 真机模式 (需要 ADB 连接)
    python test_video_rknn.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --target rk3588

    # ONNX 模拟模式 (PC 上运行)
    python test_video_rknn.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --simulate

    # 与 standalone 结果对比
    python test_video_rknn.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 420,270 \
        --simulate \
        --compare_with output_standalone
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
from scipy import ndimage

# 导入独立的 Memory Attention 模块
from memory_attention_standalone import build_memory_attention


# ==================== 常量 ====================

IMG_SIZE = 1024
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

# ImageNet 标准化参数
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ==================== 记忆库 (与 standalone 版本相同) ====================

class MemoryBank:
    """记忆库 - 存储历史帧的记忆特征和物体指针"""
    
    def __init__(self, num_maskmem: int = 7, mem_dim: int = 64, hidden_dim: int = 256,
                 max_obj_ptrs: int = 16, maskmem_tpos_enc: np.ndarray = None):
        self.num_maskmem = num_maskmem
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.max_obj_ptrs = max_obj_ptrs
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
        """获取用于当前帧的记忆"""
        to_cat_memory = []
        to_cat_memory_pos = []
        
        # 1. 添加条件帧记忆 (t_pos=0)
        for t, out in self.cond_frame_outputs.items():
            if t >= frame_idx:
                continue
            feats = out["maskmem_features"]
            pos = out["maskmem_pos_enc"]
            feats = np.transpose(feats, (1, 0, 2))
            pos = np.transpose(pos, (1, 0, 2))
            
            if self.maskmem_tpos_enc is not None:
                t_pos = 0
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                tpos_enc = tpos_enc.reshape(1, 1, -1)
                pos = pos + tpos_enc
            
            to_cat_memory.append(feats)
            to_cat_memory_pos.append(pos)
        
        # 2. 添加最近的非条件帧记忆
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
            
            if self.maskmem_tpos_enc is not None:
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                tpos_enc = tpos_enc.reshape(1, 1, -1)
                pos = pos + tpos_enc
            
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


# ==================== 推理引擎抽象 ====================

class BaseInferenceBackend:
    """推理后端基类"""
    
    def run_encoder(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError
    
    def run_memory_encoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError
    
    def release(self):
        pass


class ONNXBackend(BaseInferenceBackend):
    """ONNX Runtime 后端 (用于 PC 模拟)"""
    
    def __init__(self, onnx_dir: str):
        import onnxruntime as ort
        self.ort = ort
        
        print(f"  使用 ONNX Runtime 后端")
        print(f"  ONNX 模型目录: {onnx_dir}")
        
        # 创建 session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 加载 ONNX 模型
        encoder_path = os.path.join(onnx_dir, "edgetam_encoder.onnx")
        decoder_path = os.path.join(onnx_dir, "edgetam_decoder.onnx")
        mem_encoder_path = os.path.join(onnx_dir, "edgetam_memory_encoder.onnx")
        
        print(f"  加载 encoder: {encoder_path}")
        self.encoder = ort.InferenceSession(encoder_path, sess_opts, providers=['CPUExecutionProvider'])
        
        print(f"  加载 decoder: {decoder_path}")
        self.decoder = ort.InferenceSession(decoder_path, sess_opts, providers=['CPUExecutionProvider'])
        
        print(f"  加载 memory_encoder: {mem_encoder_path}")
        self.memory_encoder = ort.InferenceSession(mem_encoder_path, sess_opts, providers=['CPUExecutionProvider'])
    
    def run_encoder(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ONNX 版本需要 float32，并手动标准化
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            # 从 NCHW [0, 255] 转换为标准化后的 NCHW
            # image shape: [1, 3, H, W]
            for c in range(3):
                image[0, c, :, :] = (image[0, c, :, :] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        
        outputs = self.encoder.run(None, {"image": image})
        return outputs[0], outputs[1], outputs[2]
    
    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        input_names = [inp.name for inp in self.decoder.get_inputs()]
        feed_dict = {name: val for name, val in zip(input_names, inputs)}
        return self.decoder.run(None, feed_dict)
    
    def run_memory_encoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        input_names = [inp.name for inp in self.memory_encoder.get_inputs()]
        feed_dict = {name: val for name, val in zip(input_names, inputs)}
        return self.memory_encoder.run(None, feed_dict)


class RKNNBackend(BaseInferenceBackend):
    """RKNN 后端 (用于真机)"""
    
    def __init__(self, rknn_dir: str, target: str, device_id: str = None):
        from rknn.api import RKNN
        self.RKNN = RKNN
        
        self.target = target
        self.device_id = device_id
        
        print(f"  使用 RKNN 后端")
        print(f"  目标平台: {target}")
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
    
    def _load_rknn_model(self, model_path: str, name: str):
        """加载 RKNN 模型并初始化运行时"""
        from rknn.api import RKNN
        
        print(f"  加载 {name}: {model_path}")
        
        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败: {model_path}")
        
        # 初始化运行时 - 真机模式
        ret = rknn.init_runtime(target=self.target, device_id=self.device_id)
        if ret != 0:
            raise RuntimeError(f"初始化 RKNN 运行时失败: {model_path}")
        
        return rknn
    
    def run_encoder(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # RKNN 版本已内置标准化，期望 uint8 [0, 255]
        outputs = self.encoder.inference(inputs=[image])
        return outputs[0], outputs[1], outputs[2]
    
    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        return self.decoder.inference(inputs=inputs)
    
    def run_memory_encoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        return self.memory_encoder.inference(inputs=inputs)
    
    def release(self):
        if hasattr(self, 'encoder'):
            self.encoder.release()
        if hasattr(self, 'decoder'):
            self.decoder.release()
        if hasattr(self, 'memory_encoder'):
            self.memory_encoder.release()


# ==================== 视频推理引擎 ====================

class RKNNVideoInference:
    """RKNN/ONNX 版本的视频推理引擎"""
    
    def __init__(self, rknn_dir: str = None, onnx_dir: str = None, pt_dir: str = None, 
                 target: str = None, device_id: str = None, device: str = "cpu",
                 simulate: bool = False):
        """
        Args:
            rknn_dir: RKNN 模型目录 (真机模式)
            onnx_dir: ONNX 模型目录 (ONNX 模拟模式)
            pt_dir: PyTorch 模型目录 (memory_attention.pt)
            target: 目标平台 (真机模式必须指定，如 'rk3588')
            device_id: ADB 设备 ID (多设备时使用)
            device: PyTorch 设备 (用于 memory attention)
            simulate: 是否使用 ONNX 模拟模式
        """
        self.device = device
        self.simulate = simulate
        self.target = target
        self.device_id = device_id
        
        # 确定模型目录
        if simulate:
            self.onnx_dir = onnx_dir or 'onnx_models'
            self.rknn_dir = rknn_dir or 'rknn_models'
            self.pt_dir = pt_dir or self.onnx_dir
        else:
            if target is None:
                raise ValueError("真机模式必须指定 target (如 'rk3588')。"
                               "如果要在 PC 上测试，请使用 --simulate 参数。")
            self.rknn_dir = rknn_dir or 'rknn_models'
            self.onnx_dir = onnx_dir or 'onnx_models'
            self.pt_dir = pt_dir or self.onnx_dir
        
        print("=" * 60)
        print("初始化视频推理引擎")
        if simulate:
            print(f"  模式: ONNX 模拟 (PC)")
        else:
            print(f"  模式: RKNN 真机 ({target})")
        print("=" * 60)
        
        # 创建后端
        print("\n加载模型...")
        if simulate:
            self.backend = ONNXBackend(self.onnx_dir)
        else:
            self.backend = RKNNBackend(self.rknn_dir, target, device_id)
        
        # 加载 memory_encoder 的常量输出 maskmem_pos_enc
        pos_enc_path = os.path.join(self.rknn_dir, "maskmem_pos_enc.npy")
        if os.path.exists(pos_enc_path):
            self.maskmem_pos_enc_const = np.load(pos_enc_path)
            print(f"  加载常量 maskmem_pos_enc: {self.maskmem_pos_enc_const.shape}")
        else:
            print(f"  警告: 未找到 maskmem_pos_enc.npy，将使用 memory_encoder 输出")
            self.maskmem_pos_enc_const = None
        
        # 加载独立的 Memory Attention 模块 (PyTorch)
        print("\n加载 Memory Attention 模块 (PyTorch)...")
        weights_path = os.path.join(self.pt_dir, "memory_attention.pt")
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
        
        注意: RKNN encoder 已配置内置标准化，输入应为 [0, 255] 的 uint8
        ONNX 模式会在后端中自动处理标准化
        """
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.uint8)  # [H, W, 3], uint8, 0-255
        img_np = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
        img_np = np.expand_dims(img_np, axis=0)  # [1, 3, H, W]
        return img_np
    
    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码图像"""
        img_np = self.preprocess_image_for_encoder(image)
        
        # 使用后端推理
        high_res_0, high_res_1, image_embed = self.backend.run_encoder(img_np)
        return high_res_0, high_res_1, image_embed
    
    def get_position_encoding(self, H: int = 64, W: int = 64) -> torch.Tensor:
        """获取位置编码"""
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
        
        pos = np.concatenate([pos_y, pos_x], axis=2)
        pos = pos.reshape(H * W, dim)
        pos = pos[np.newaxis, :, :]
        pos = np.transpose(pos, (1, 0, 2))
        
        self._cached_pos_enc = torch.from_numpy(pos.astype(np.float32)).to(self.device)
        return self._cached_pos_enc
    
    @torch.inference_mode()
    def apply_memory_attention(self, image_embed: np.ndarray, frame_idx: int) -> np.ndarray:
        """应用记忆注意力 (使用 PyTorch)"""
        memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem = self.memory_bank.get_memories_for_frame(frame_idx)
        
        # 转换 image_embed 为 PyTorch 格式
        image_embed_torch = torch.from_numpy(image_embed).to(self.device, dtype=torch.float32)
        B, C, H, W = image_embed_torch.shape
        curr_feat = image_embed_torch.flatten(2).permute(2, 0, 1)
        
        curr_pos = self.get_position_encoding(H, W)
        
        if memory is None:
            pix_feat_with_mem = curr_feat + self.no_mem_embed.view(1, 1, -1)
        else:
            memory_torch = torch.from_numpy(memory).to(self.device, dtype=torch.float32)
            memory_pos_torch = torch.from_numpy(memory_pos).to(self.device, dtype=torch.float32)
            
            pix_feat_with_mem = self.memory_attention(
                curr=curr_feat,
                memory=memory_torch,
                curr_pos=curr_pos,
                memory_pos=memory_pos_torch,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                num_spatial_mem=num_spatial_mem,
            )
        
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
        
        # 使用后端推理
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
        
        # 使用后端推理
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
    
    def encode_memory(self, image_embed: np.ndarray, mask: np.ndarray, 
                      object_score_logits: float) -> Tuple[np.ndarray, np.ndarray]:
        """编码记忆"""
        image_embed = image_embed.astype(np.float32)
        
        if mask.shape[-1] != IMG_SIZE or mask.shape[-2] != IMG_SIZE:
            mask_resized = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        else:
            mask_resized = mask.astype(np.float32)
        
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mask_logits = np.where(mask_input > 0.5, 10.0, -10.0).astype(np.float32)
        
        obj_score_input = np.array([[object_score_logits]], dtype=np.float32)
        
        # 使用后端推理 memory_encoder
        outputs = self.backend.run_memory_encoder([
            image_embed,
            mask_logits,
            obj_score_input,
        ])
        
        maskmem_features = outputs[0]
        
        # 使用预加载的常量 pos_enc 或 memory_encoder 输出
        if self.maskmem_pos_enc_const is not None:
            maskmem_pos_enc = self.maskmem_pos_enc_const
        else:
            maskmem_pos_enc = outputs[1] if len(outputs) > 1 else np.zeros_like(maskmem_features)
        
        return maskmem_features, maskmem_pos_enc
    
    def process_frame(self, frame: Image.Image, frame_idx: int, 
                      point: Tuple[int, int],
                      prev_mask: Optional[np.ndarray] = None,
                      is_cond_frame: bool = False) -> Tuple[np.ndarray, float, float]:
        """处理单帧"""
        orig_size = frame.size
        
        # 1. 编码图像
        high_res_0, high_res_1, image_embed = self.encode_image(frame)
        
        # 2. 应用记忆注意力
        if frame_idx > 0 and not is_cond_frame:
            image_embed_with_mem = self.apply_memory_attention(image_embed, frame_idx)
        else:
            image_embed_with_mem = image_embed
        
        # 3. 解码 mask
        if is_cond_frame:
            mask_input = None
            if prev_mask is not None:
                mask_input = np.where(prev_mask > 0.5, 10.0, -10.0).astype(np.float32)
            
            mask, iou, obj_ptr, obj_score = self.decode_mask(
                image_embed_with_mem, high_res_0, high_res_1,
                point, orig_size, mask_input
            )
        else:
            mask, iou, obj_ptr, obj_score = self.decode_mask_no_prompt(
                image_embed_with_mem, high_res_0, high_res_1, orig_size
            )
        
        # 4. 编码记忆
        maskmem_features, maskmem_pos_enc = self.encode_memory(image_embed, mask, obj_score)
        
        # 5. 添加到记忆库
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
    
    def release(self):
        """释放资源"""
        print("\n释放资源...")
        if hasattr(self, 'backend'):
            self.backend.release()


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
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_uint8)


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
    parser = argparse.ArgumentParser(description='RKNN/ONNX 版本的 EdgeTAM 视频推理')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--rknn_dir', type=str, default='rknn_models',
                        help='RKNN 模型目录')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录')
    parser.add_argument('--pt_dir', type=str, default='onnx_models',
                        help='PyTorch 模型目录 (memory_attention.pt)')
    parser.add_argument('--simulate', action='store_true',
                        help='使用 ONNX 模型在 PC 上模拟 (不需要真机)')
    parser.add_argument('--target', type=str, default=None,
                        choices=[None, 'rk3588', 'rk3576', 'rk3568', 'rk3566', 'rk3562'],
                        help='RKNN 目标平台 (真机模式必须指定)')
    parser.add_argument('--device_id', type=str, default=None,
                        help='ADB 设备 ID (多设备时使用)')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--compare_with', type=str, default=None,
                        help='与另一个输出目录进行对比')
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch 设备 (用于 memory attention)')
    
    args = parser.parse_args()
    
    # 检查模式
    if not args.simulate and args.target is None:
        print("=" * 60)
        print("错误: 必须指定 --simulate 或 --target")
        print("")
        print("用法:")
        print("  1. ONNX 模拟模式 (PC 上运行):")
        print("     python test_video_rknn.py --video_dir ... --point ... --simulate")
        print("")
        print("  2. RKNN 真机模式 (需要 ADB 连接 RK3588):")
        print("     python test_video_rknn.py --video_dir ... --point ... --target rk3588")
        print("")
        print("注意: RKNN 的 load_rknn() 加载的模型不支持 PC 模拟器。")
        print("      如果要在 PC 上验证模型正确性，请使用 --simulate 参数。")
        print("=" * 60)
        return
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    
    # 自动生成输出目录名称
    if args.output is None:
        video_name = os.path.basename(args.video_dir)
        if os.path.isfile(args.video_dir):
            video_name = os.path.splitext(video_name)[0]
        suffix = 'onnx_sim' if args.simulate else 'rknn'
        args.output = os.path.join('output', f'{video_name}_{point_x}_{point_y}_{suffix}')
    
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
    engine = RKNNVideoInference(
        rknn_dir=args.rknn_dir,
        onnx_dir=args.onnx_dir,
        pt_dir=args.pt_dir,
        target=args.target,
        device_id=args.device_id,
        device=args.device,
        simulate=args.simulate
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
                        print(f"  帧 {i:3d}: IoU={iou:.3f}, 面积(rknn)={area_ours}, 面积(ref)={area_ref}")
            
            if iou_vs_ref:
                print(f"\n  与参考的平均 IoU: {np.mean(iou_vs_ref):.3f}")
                print(f"  最小 IoU: {np.min(iou_vs_ref):.3f}")
                print(f"  最大 IoU: {np.max(iou_vs_ref):.3f}")
    
    finally:
        # 释放资源
        engine.release()


if __name__ == "__main__":
    main()
