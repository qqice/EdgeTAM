# -*- coding: utf-8 -*-
"""
混合推理脚本 - RKNN Encoder + ONNX Decoder

此脚本支持灵活配置各个模块的推理后端：
- Encoder: RKNN (NPU) 或 ONNX (CPU)
- Decoder: RKNN (NPU) 或 ONNX (CPU)  
- Memory Encoder: RKNN (NPU) 或 ONNX (CPU)
- Memory Attention: PyTorch (CPU/GPU)

每个阶段的耗时会被记录，便于性能分析。

使用方法:
    # 默认: RKNN encoder + ONNX decoder (推荐配置)
    python test_video_hybrid.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 633,444

    # 全部使用 ONNX (参考)
    python test_video_hybrid.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 633,444 \
        --encoder onnx --decoder onnx --memory_encoder onnx

    # 全部使用 RKNN (可能精度有问题)
    python test_video_hybrid.py \
        --video_dir ../notebooks/videos/bedroom \
        --point 633,444 \
        --encoder rknn --decoder rknn --memory_encoder rknn

作者: EdgeTAM RKNN Debug
日期: 2024-12-03
"""

import os
import sys
import argparse
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

# 尝试导入 RKNNLite
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
    print("RKNNLite 可用")
except ImportError:
    try:
        from rknn.api import RKNN as RKNNLite
        RKNN_AVAILABLE = True
        print("RKNN 可用")
    except ImportError:
        RKNN_AVAILABLE = False
        print("警告: RKNN/RKNNLite 不可用，将只能使用 ONNX")

# 导入独立的 Memory Attention 模块
from memory_attention_standalone import build_memory_attention


# ==================== 全局可变状态（用于实时/交互式调用） ====================

# 是否允许基于当前 include point 进行推理
allow_inference: bool = False

# 当前 include point（x, y），仅在 allow_inference 为 True 时有效
_current_point: Optional[Tuple[int, int]] = None

# 混合推理引擎及其配置缓存
_global_engine: Optional["HybridVideoInference"] = None
_global_engine_config: Optional[Dict[str, str]] = None

# 连续帧计数与上一帧的 mask，用于增量推理
_global_frame_idx: int = 0
_prev_mask: Optional[np.ndarray] = None


# ==================== 常量 ====================

IMG_SIZE = 1024
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


# ==================== 性能统计 ====================

@dataclass
class TimingStats:
    """各阶段耗时统计"""
    encoder_times: List[float] = field(default_factory=list)
    decoder_times: List[float] = field(default_factory=list)
    memory_encoder_times: List[float] = field(default_factory=list)
    memory_attention_times: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    
    def add_frame(self, encoder_ms: float, decoder_ms: float, 
                  memory_encoder_ms: float, memory_attention_ms: float, total_ms: float):
        self.encoder_times.append(encoder_ms)
        self.decoder_times.append(decoder_ms)
        self.memory_encoder_times.append(memory_encoder_ms)
        self.memory_attention_times.append(memory_attention_ms)
        self.total_times.append(total_ms)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取统计摘要"""
        def calc_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"mean": 0, "min": 0, "max": 0, "std": 0}
            return {
                "mean": np.mean(times),
                "min": np.min(times),
                "max": np.max(times),
                "std": np.std(times),
            }
        
        return {
            "encoder": calc_stats(self.encoder_times),
            "decoder": calc_stats(self.decoder_times),
            "memory_encoder": calc_stats(self.memory_encoder_times),
            "memory_attention": calc_stats(self.memory_attention_times),
            "total": calc_stats(self.total_times),
        }
    
    def print_summary(self, config: Dict[str, str]):
        """打印统计摘要"""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("性能统计 (单位: 毫秒)")
        print("=" * 70)
        print(f"{'模块':<20} | {'后端':<10} | {'设备':<6} | {'平均':>8} | {'最小':>8} | {'最大':>8} | {'标准差':>8}")
        print("-" * 70)
        
        modules = [
            ("Encoder", "encoder"),
            ("Decoder", "decoder"),
            ("Memory Encoder", "memory_encoder"),
            ("Memory Attention", "memory_attention"),
            ("总计", "total"),
        ]
        
        for name, key in modules:
            stats = summary[key]
            if key == "total":
                backend = "-"
                device = "-"
            elif key == "memory_attention":
                backend = "PyTorch"
                device = config.get("device", "CPU")
            else:
                backend = config.get(key, "ONNX").upper()
                device = "NPU" if backend == "RKNN" else "CPU"
            
            print(f"{name:<20} | {backend:<10} | {device:<6} | "
                  f"{stats['mean']:>8.2f} | {stats['min']:>8.2f} | "
                  f"{stats['max']:>8.2f} | {stats['std']:>8.2f}")
        
        print("=" * 70)
        
        # 计算 FPS
        if summary["total"]["mean"] > 0:
            fps = 1000.0 / summary["total"]["mean"]
            print(f"平均帧率: {fps:.2f} FPS")
        
        # 各阶段占比
        total_mean = summary["total"]["mean"]
        if total_mean > 0:
            print("\n各阶段耗时占比:")
            for name, key in modules[:-1]:  # 排除总计
                pct = summary[key]["mean"] / total_mean * 100
                print(f"  {name}: {pct:.1f}%")


# ==================== 记忆库 ====================

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
        to_cat_memory = []
        to_cat_memory_pos = []
        
        # 1. 添加条件帧记忆
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
        
        # 2. 添加非条件帧记忆
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
        self.cond_frame_outputs.clear()
        self.non_cond_frame_outputs.clear()


# ==================== 混合推理引擎 ====================

class HybridVideoInference:
    """混合推理引擎 - 支持 RKNN 和 ONNX 混合使用"""
    
    def __init__(
        self, 
        onnx_dir: str,
        rknn_dir: str = None,
        encoder_backend: str = "rknn",
        decoder_backend: str = "onnx",
        memory_encoder_backend: str = "onnx",
        device: str = "cpu"
    ):
        self.device = device
        self.onnx_dir = onnx_dir
        self.rknn_dir = rknn_dir or os.path.join(os.path.dirname(onnx_dir), "rknn_models", "no_opt")
        
        # 后端配置
        self.encoder_backend = encoder_backend.lower()
        self.decoder_backend = decoder_backend.lower()
        self.memory_encoder_backend = memory_encoder_backend.lower()
        
        # 检查 RKNN 可用性
        if not RKNN_AVAILABLE:
            if self.encoder_backend == "rknn":
                print("警告: RKNN 不可用，Encoder 将使用 ONNX")
                self.encoder_backend = "onnx"
            if self.decoder_backend == "rknn":
                print("警告: RKNN 不可用，Decoder 将使用 ONNX")
                self.decoder_backend = "onnx"
            if self.memory_encoder_backend == "rknn":
                print("警告: RKNN 不可用，Memory Encoder 将使用 ONNX")
                self.memory_encoder_backend = "onnx"
        
        print("=" * 60)
        print("初始化混合推理引擎")
        print("=" * 60)
        print(f"  Encoder: {self.encoder_backend.upper()} ({'NPU' if self.encoder_backend == 'rknn' else 'CPU'})")
        print(f"  Decoder: {self.decoder_backend.upper()} ({'NPU' if self.decoder_backend == 'rknn' else 'CPU'})")
        print(f"  Memory Encoder: {self.memory_encoder_backend.upper()} ({'NPU' if self.memory_encoder_backend == 'rknn' else 'CPU'})")
        print(f"  Memory Attention: PyTorch ({device.upper()})")
        print("=" * 60)
        
        # ONNX 配置
        providers = ['CPUExecutionProvider']
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 加载 Encoder
        self._load_encoder(providers)
        
        # 加载 Decoder
        self._load_decoder(providers)
        
        # 加载 Memory Encoder
        self._load_memory_encoder(providers)
        
        # 加载 Memory Attention (始终使用 PyTorch)
        print("加载 Memory Attention (PyTorch)...")
        weights_path = os.path.join(onnx_dir, "memory_attention.pt")
        self.memory_attention, self.extras, self.config = build_memory_attention(
            weights_path, device=device
        )
        
        # 获取配置
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.mem_dim = self.config.get('mem_dim', 64)
        self.num_maskmem = self.config.get('num_maskmem', 7)
        
        # 获取关键参数
        self.no_mem_embed = self.extras.get('no_mem_embed', torch.zeros(1, 1, 256))
        self.no_obj_ptr = self.extras.get('no_obj_ptr', None)
        self.no_obj_embed_spatial = self.extras.get('no_obj_embed_spatial', None)
        self.soft_no_obj_ptr = self.extras.get('soft_no_obj_ptr', False)
        self.fixed_no_obj_ptr = self.extras.get('fixed_no_obj_ptr', False)
        
        if self.no_obj_ptr is not None:
            self.no_obj_ptr = self.no_obj_ptr.cpu().numpy()
        if self.no_obj_embed_spatial is not None:
            self.no_obj_embed_spatial = self.no_obj_embed_spatial.cpu().numpy()
        
        # 时间位置编码
        maskmem_tpos_enc = self.extras.get('maskmem_tpos_enc', None)
        if maskmem_tpos_enc is not None:
            maskmem_tpos_enc = maskmem_tpos_enc.cpu().numpy()
        
        # 记忆库
        self.memory_bank = MemoryBank(
            num_maskmem=self.num_maskmem,
            mem_dim=self.mem_dim,
            hidden_dim=self.hidden_dim,
            maskmem_tpos_enc=maskmem_tpos_enc
        )
        
        # 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # ImageNet 归一化 (用于 RKNN)
        self.imagenet_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.imagenet_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        # 缓存
        self._cached_pos_enc = None
        
        # 性能统计
        self.timing_stats = TimingStats()
        
        print("初始化完成")
        print("=" * 60)
    
    def _load_encoder(self, providers):
        """加载 Encoder"""
        if self.encoder_backend == "rknn":
            print("加载 Encoder (RKNN)...")
            rknn_path = os.path.join(self.rknn_dir, "edgetam_encoder_no_opt.rknn")
            if not os.path.exists(rknn_path):
                raise FileNotFoundError(f"找不到 RKNN encoder: {rknn_path}")
            
            self.encoder_rknn = RKNNLite()
            ret = self.encoder_rknn.load_rknn(rknn_path)
            if ret != 0:
                raise RuntimeError(f"加载 RKNN encoder 失败: {ret}")
            ret = self.encoder_rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN encoder 运行时失败: {ret}")
            print(f"  成功加载: {rknn_path}")
        else:
            print("加载 Encoder (ONNX)...")
            onnx_path = os.path.join(self.onnx_dir, "edgetam_encoder.onnx")
            self.encoder_onnx = ort.InferenceSession(onnx_path, providers=providers)
            print(f"  成功加载: {onnx_path}")
    
    def _load_decoder(self, providers):
        """加载 Decoder"""
        if self.decoder_backend == "rknn":
            print("加载 Decoder (RKNN)...")
            rknn_path = os.path.join(self.rknn_dir, "edgetam_decoder_no_opt.rknn")
            if not os.path.exists(rknn_path):
                raise FileNotFoundError(f"找不到 RKNN decoder: {rknn_path}")
            
            self.decoder_rknn = RKNNLite()
            ret = self.decoder_rknn.load_rknn(rknn_path)
            if ret != 0:
                raise RuntimeError(f"加载 RKNN decoder 失败: {ret}")
            ret = self.decoder_rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN decoder 运行时失败: {ret}")
            print(f"  成功加载: {rknn_path}")
        else:
            print("加载 Decoder (ONNX)...")
            onnx_path = os.path.join(self.onnx_dir, "edgetam_decoder.onnx")
            self.decoder_onnx = ort.InferenceSession(onnx_path, providers=providers)
            print(f"  成功加载: {onnx_path}")
    
    def _load_memory_encoder(self, providers):
        """加载 Memory Encoder"""
        if self.memory_encoder_backend == "rknn":
            print("加载 Memory Encoder (RKNN)...")
            rknn_path = os.path.join(self.rknn_dir, "edgetam_memory_encoder_no_opt.rknn")
            if not os.path.exists(rknn_path):
                raise FileNotFoundError(f"找不到 RKNN memory encoder: {rknn_path}")
            
            self.memory_encoder_rknn = RKNNLite()
            ret = self.memory_encoder_rknn.load_rknn(rknn_path)
            if ret != 0:
                raise RuntimeError(f"加载 RKNN memory encoder 失败: {ret}")
            ret = self.memory_encoder_rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN memory encoder 运行时失败: {ret}")
            print(f"  成功加载: {rknn_path}")
            
            # 加载常量 maskmem_pos_enc (RKNN 转换时检测到是常量，保存为 npy)
            # 尝试多个可能的路径
            pos_enc_candidates = [
                os.path.join(self.rknn_dir, "maskmem_pos_enc.npy"),
                os.path.join(os.path.dirname(self.rknn_dir), "maskmem_pos_enc.npy"),
                os.path.join(os.path.dirname(self.rknn_dir), "rknn_models", "maskmem_pos_enc.npy"),
            ]
            self.maskmem_pos_enc_const = None
            for pos_enc_path in pos_enc_candidates:
                if os.path.exists(pos_enc_path):
                    self.maskmem_pos_enc_const = np.load(pos_enc_path)
                    print(f"  加载常量 maskmem_pos_enc: {pos_enc_path}, 形状: {self.maskmem_pos_enc_const.shape}")
                    break
            
            if self.maskmem_pos_enc_const is None:
                print("  警告: 未找到 maskmem_pos_enc.npy，将尝试从 ONNX 获取")
                # 从 ONNX 模型运行一次获取常量
                onnx_path = os.path.join(self.onnx_dir, "edgetam_memory_encoder.onnx")
                if os.path.exists(onnx_path):
                    temp_session = ort.InferenceSession(onnx_path, providers=providers)
                    dummy_pix_feat = np.zeros((1, 256, 64, 64), dtype=np.float32)
                    dummy_mask = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
                    outputs = temp_session.run(None, {'pix_feat': dummy_pix_feat, 'pred_mask': dummy_mask})
                    self.maskmem_pos_enc_const = outputs[1]
                    print(f"  从 ONNX 获取 maskmem_pos_enc 常量: {self.maskmem_pos_enc_const.shape}")
        else:
            print("加载 Memory Encoder (ONNX)...")
            onnx_path = os.path.join(self.onnx_dir, "edgetam_memory_encoder.onnx")
            self.memory_encoder_onnx = ort.InferenceSession(onnx_path, providers=providers)
            print(f"  成功加载: {onnx_path}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """预处理图像 (ONNX 格式: 归一化后的 float32)"""
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np
    
    def preprocess_image_rknn(self, image: Image.Image) -> np.ndarray:
        """预处理图像 (RKNN 格式: 0-255 uint8 NHWC)"""
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.uint8)  # [H, W, 3], 0-255
        img_np = np.expand_dims(img_np, axis=0)      # [1, H, W, 3]
        return img_np
    
    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码图像"""
        if self.encoder_backend == "rknn":
            img_np = self.preprocess_image_rknn(image)
            outputs = self.encoder_rknn.inference(inputs=[img_np])
            return outputs[0], outputs[1], outputs[2]
        else:
            img_np = self.preprocess_image(image)
            outputs = self.encoder_onnx.run(None, {'image': img_np})
            return outputs[0], outputs[1], outputs[2]
    
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
    def apply_memory_attention(self, image_embed: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, float]:
        """应用记忆注意力，返回处理后的特征和耗时"""
        start_time = time.perf_counter()
        
        memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem = self.memory_bank.get_memories_for_frame(frame_idx)
        
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
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return pix_feat_with_mem.cpu().numpy(), elapsed_ms
    
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
        
        if self.decoder_backend == "rknn":
            outputs = self.decoder_rknn.inference(inputs=[
                image_embed,
                high_res_0,
                high_res_1,
                point_coords,
                point_labels,
                mask_input_resized,
                has_mask_input,
            ])
        else:
            outputs = self.decoder_onnx.run(None, {
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
        
        if self.decoder_backend == "rknn":
            outputs = self.decoder_rknn.inference(inputs=[
                image_embed,
                high_res_0,
                high_res_1,
                point_coords,
                point_labels,
                mask_input_resized,
                has_mask_input,
            ])
        else:
            outputs = self.decoder_onnx.run(None, {
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
        high_threshold = 0.0 + threshold_offset
        low_threshold = 0.0 - threshold_offset
        
        intersections = np.sum((masks > high_threshold).astype(np.float32), axis=(1, 2))
        unions = np.sum((masks > low_threshold).astype(np.float32), axis=(1, 2))
        
        stability_scores = np.where(unions > 0, intersections / unions, 1.0)
        return stability_scores
    
    def _apply_no_obj_ptr(self, obj_ptr: np.ndarray, obj_score: float) -> np.ndarray:
        if self.no_obj_ptr is None:
            return obj_ptr
        
        is_obj_appearing = obj_score > 0
        
        if self.soft_no_obj_ptr:
            lambda_is_obj_appearing = 1.0 / (1.0 + np.exp(-obj_score))
        else:
            lambda_is_obj_appearing = 1.0 if is_obj_appearing else 0.0
        
        if self.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        
        obj_ptr = obj_ptr + (1.0 - lambda_is_obj_appearing) * self.no_obj_ptr
        
        return obj_ptr
    
    def encode_memory(self, image_embed: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """编码记忆"""
        image_embed = image_embed.astype(np.float32)
        
        if mask.shape[-1] != IMG_SIZE or mask.shape[-2] != IMG_SIZE:
            mask_resized = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        else:
            mask_resized = mask.astype(np.float32)
        
        mask_input = mask_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mask_logits = np.where(mask_input > 0.5, 10.0, -10.0).astype(np.float32)
        
        if self.memory_encoder_backend == "rknn":
            outputs = self.memory_encoder_rknn.inference(inputs=[
                image_embed,
                mask_logits,
            ])
            # RKNN 只返回 maskmem_features，maskmem_pos_enc 是常量
            maskmem_features = outputs[0]
            maskmem_pos_enc = self.maskmem_pos_enc_const
            return maskmem_features, maskmem_pos_enc
        else:
            outputs = self.memory_encoder_onnx.run(None, {
                'pix_feat': image_embed,
                'pred_mask': mask_logits,
            })
        
        return outputs[0], outputs[1]
    
    def process_frame(self, frame: Image.Image, frame_idx: int, 
                      point: Tuple[int, int],
                      prev_mask: Optional[np.ndarray] = None,
                      is_cond_frame: bool = False) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
        """处理单帧，返回 mask、IoU、obj_score 和各阶段耗时"""
        orig_size = frame.size
        timings = {}
        
        frame_start = time.perf_counter()
        
        # 1. 编码图像
        enc_start = time.perf_counter()
        high_res_0, high_res_1, image_embed = self.encode_image(frame)
        timings['encoder'] = (time.perf_counter() - enc_start) * 1000
        
        # 2. 应用记忆注意力
        if frame_idx > 0 and not is_cond_frame:
            image_embed_with_mem, mem_attn_time = self.apply_memory_attention(image_embed, frame_idx)
            timings['memory_attention'] = mem_attn_time
        else:
            image_embed_with_mem = image_embed
            timings['memory_attention'] = 0.0
        
        # 3. 解码 mask
        dec_start = time.perf_counter()
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
        timings['decoder'] = (time.perf_counter() - dec_start) * 1000
        
        # 4. 编码记忆
        mem_enc_start = time.perf_counter()
        maskmem_features, maskmem_pos_enc = self.encode_memory(image_embed, mask)
        timings['memory_encoder'] = (time.perf_counter() - mem_enc_start) * 1000
        
        # 5. 添加到记忆库
        self.memory_bank.add_memory(
            frame_idx,
            maskmem_features,
            maskmem_pos_enc,
            obj_ptr,
            obj_score,
            is_cond_frame=is_cond_frame
        )
        
        timings['total'] = (time.perf_counter() - frame_start) * 1000
        
        # 记录到统计
        self.timing_stats.add_frame(
            timings['encoder'],
            timings['decoder'],
            timings['memory_encoder'],
            timings['memory_attention'],
            timings['total']
        )
        
        return mask, iou, obj_score, timings
    
    def reset(self):
        """重置状态"""
        self.memory_bank.clear()
        self._cached_pos_enc = None
        self.timing_stats = TimingStats()
    
    def get_config(self) -> Dict[str, str]:
        """获取当前配置"""
        return {
            "encoder": self.encoder_backend,
            "decoder": self.decoder_backend,
            "memory_encoder": self.memory_encoder_backend,
            "memory_attention": "pytorch",
            "device": self.device,
        }
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'encoder_rknn'):
            self.encoder_rknn.release()
        if hasattr(self, 'decoder_rknn'):
            self.decoder_rknn.release()
        if hasattr(self, 'memory_encoder_rknn'):
            self.memory_encoder_rknn.release()


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


# ==================== 实时/交互式封装函数 ====================

def _validate_point(point: Tuple[int, int], frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """校验输入点，确保为非负整数，必要时检查是否在帧内。"""
    if (not isinstance(point, tuple)) or len(point) != 2:
        raise ValueError("point 必须是长度为 2 的元组 (x, y)")
    x, y = point
    if not (isinstance(x, int) and isinstance(y, int)):
        raise ValueError("point 的 x/y 必须是整数")
    if x < 0 or y < 0:
        raise ValueError("point 的 x/y 不能为负")
    if frame_shape is not None:
        h, w = frame_shape
        if x >= w or y >= h:
            raise ValueError(f"point 超出图像范围: ({x}, {y}) 不在 [0,{w})x[0,{h})")
    return x, y


def _ensure_engine(
    onnx_dir: str = 'onnx_models',
    rknn_dir: Optional[str] = None,
    encoder: str = 'rknn',
    decoder: str = 'onnx',
    memory_encoder: str = 'onnx',
    device: str = 'cpu',
) -> HybridVideoInference:
    """懒加载/复用混合推理引擎，避免重复初始化。"""
    global _global_engine, _global_engine_config
    cfg = {
        'onnx_dir': onnx_dir,
        'rknn_dir': rknn_dir,
        'encoder': encoder,
        'decoder': decoder,
        'memory_encoder': memory_encoder,
        'device': device,
    }
    if _global_engine is None or _global_engine_config != cfg:
        if _global_engine is not None:
            _global_engine.release()
        _global_engine = HybridVideoInference(
            onnx_dir=onnx_dir,
            rknn_dir=rknn_dir,
            encoder_backend=encoder,
            decoder_backend=decoder,
            memory_encoder_backend=memory_encoder,
            device=device,
        )
        _global_engine_config = cfg
    return _global_engine


def reset_inference():
    """重置推理状态并关闭允许标记。"""
    global allow_inference, _global_frame_idx, _prev_mask, _current_point
    if _global_engine is not None:
        _global_engine.reset()
    _global_frame_idx = 0
    _prev_mask = None
    _current_point = None
    allow_inference = False


def set_include_point(point: Tuple[int, int]):
    """设置 include point 并开启推理开关。"""
    global allow_inference, _current_point
    _validate_point(point)
    reset_inference()  # 重置状态以避免旧记忆影响新点
    _current_point = point
    allow_inference = True


def inference_single_frame(
    frame_bgr: np.ndarray,
    *,
    onnx_dir: str = 'onnx_models',
    rknn_dir: Optional[str] = None,
    encoder: str = 'rknn',
    decoder: str = 'onnx',
    memory_encoder: str = 'onnx',
    device: str = 'cpu',
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    对单帧执行推理：
      - allow_inference=True 时，返回 (mask, 可视化 BGR 图像)
      - allow_inference=False 时，返回 (None, 原始帧)
    输入帧应为 OpenCV BGR 格式的 np.ndarray。
    """
    global allow_inference, _global_frame_idx, _prev_mask
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        raise ValueError("frame_bgr 必须是形状为 HxWx3 的 np.ndarray (BGR)")
    h, w, c = frame_bgr.shape
    if c != 3:
        raise ValueError("frame_bgr 必须有 3 个通道 (BGR)")

    if not allow_inference:
        # 未允许推理时直接回传原图
        return None, frame_bgr

    if _current_point is None:
        raise ValueError("尚未设置 include point，先调用 set_include_point")

    # 点位校验（在当前帧尺寸下）
    point = _validate_point(_current_point, frame_shape=(h, w))

    engine = _ensure_engine(
        onnx_dir=onnx_dir,
        rknn_dir=rknn_dir,
        encoder=encoder,
        decoder=decoder,
        memory_encoder=memory_encoder,
        device=device,
    )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    is_cond_frame = (_global_frame_idx == 0)
    mask, iou, obj_score, timings = engine.process_frame(
        frame_pil,
        _global_frame_idx,
        point,
        prev_mask=_prev_mask,
        is_cond_frame=is_cond_frame,
    )

    _prev_mask = mask
    _global_frame_idx += 1

    vis = create_visualization(
        frame_pil,
        mask,
        point,
        _global_frame_idx - 1,
        iou,
        obj_score,
        timings,
        engine.get_config(),
        color=(0.2, 0.6, 1.0),
        alpha=0.5,
    )

    return mask, vis


def release_global_engine():
    """释放全局引擎资源并清空缓存。"""
    global _global_engine, _global_engine_config
    if _global_engine is not None:
        _global_engine.release()
    _global_engine = None
    _global_engine_config = None


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


def create_visualization(
    frame: Image.Image,
    mask: np.ndarray,
    point: Tuple[int, int],
    frame_idx: int,
    iou: float,
    obj_score: float,
    timings: Dict[str, float],
    config: Dict[str, str],
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    alpha: float = 0.5,
) -> np.ndarray:
    """创建可视化图像"""
    frame_np = np.array(frame)
    h, w = frame_np.shape[:2]
    
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.float32), (w, h))
    
    mask_binary = (mask > 0.5).astype(np.float32)
    colored_mask = np.zeros_like(frame_np, dtype=np.float32)
    colored_mask[:, :, 0] = color[0] * 255
    colored_mask[:, :, 1] = color[1] * 255
    colored_mask[:, :, 2] = color[2] * 255
    
    result = frame_np.astype(np.float32).copy()
    mask_3d = np.stack([mask_binary] * 3, axis=-1)
    result = result * (1 - mask_3d * alpha) + colored_mask * mask_3d * alpha
    result = result.astype(np.uint8)
    
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    
    cv2.drawMarker(result, point, (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
    cv2.circle(result, point, 8, (255, 0, 0), 2)
    
    text_bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    y_offset = 20
    line_height = 25
    
    # 帧信息
    info_text = f"Frame: {frame_idx} | IoU: {iou:.2f} | Score: {obj_score:.2f}"
    (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    cv2.rectangle(result, (5, y_offset - th - 5), (15 + tw, y_offset + 5), text_bg_color, -1)
    cv2.putText(result, info_text, (10, y_offset), font, font_scale, text_color, thickness)
    y_offset += line_height
    
    # 性能信息
    timing_text = f"Total: {timings['total']:.1f}ms | FPS: {1000/timings['total']:.1f}"
    (tw, th), _ = cv2.getTextSize(timing_text, font, font_scale, thickness)
    cv2.rectangle(result, (5, y_offset - th - 5), (15 + tw, y_offset + 5), text_bg_color, -1)
    cv2.putText(result, timing_text, (10, y_offset), font, font_scale, text_color, thickness)
    y_offset += line_height
    
    # 各模块耗时
    modules = [
        ("Enc", "encoder", config.get("encoder", "onnx").upper()),
        ("Dec", "decoder", config.get("decoder", "onnx").upper()),
        ("MemEnc", "memory_encoder", config.get("memory_encoder", "onnx").upper()),
        ("MemAttn", "memory_attention", "PT"),
    ]
    
    detail_text = " | ".join([f"{name}({backend}): {timings.get(key, 0):.1f}ms" for name, key, backend in modules])
    (tw, th), _ = cv2.getTextSize(detail_text, font, 0.5, thickness)
    cv2.rectangle(result, (5, y_offset - th - 5), (15 + tw, y_offset + 5), text_bg_color, -1)
    cv2.putText(result, detail_text, (10, y_offset), font, 0.5, text_color, thickness)
    
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result_bgr


def compare_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bin = (mask1 > 0.5).astype(bool)
    mask2_bin = (mask2 > 0.5).astype(bool)
    
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='混合推理 - RKNN + ONNX')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件或帧目录路径')
    parser.add_argument('--point', type=str, required=True,
                        help='初始点击位置, 格式: x,y')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='ONNX 模型目录')
    parser.add_argument('--rknn_dir', type=str, default=None,
                        help='RKNN 模型目录 (默认: rknn_models/no_opt)')
    
    # 后端选择
    parser.add_argument('--encoder', type=str, default='rknn', choices=['rknn', 'onnx'],
                        help='Encoder 后端')
    parser.add_argument('--decoder', type=str, default='onnx', choices=['rknn', 'onnx'],
                        help='Decoder 后端')
    parser.add_argument('--memory_encoder', type=str, default='onnx', choices=['rknn', 'onnx'],
                        help='Memory Encoder 后端')
    
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='生成可视化图像')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='与参考输出对比')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 解析点击位置
    point_x, point_y = map(int, args.point.split(','))
    current_point = (point_x, point_y)
    
    # 生成输出目录名
    if args.output is None:
        video_name = os.path.basename(args.video_dir)
        if os.path.isfile(args.video_dir):
            video_name = os.path.splitext(video_name)[0]
        backend_suffix = f"enc_{args.encoder}_dec_{args.decoder}"
        args.output = os.path.join('output', f'{video_name}_{point_x}_{point_y}_{backend_suffix}')
    
    os.makedirs(args.output, exist_ok=True)
    mask_dir = os.path.join(args.output, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    vis_dir = None
    if args.visualize:
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
    engine = HybridVideoInference(
        onnx_dir=args.onnx_dir,
        rknn_dir=args.rknn_dir,
        encoder_backend=args.encoder,
        decoder_backend=args.decoder,
        memory_encoder_backend=args.memory_encoder,
        device=args.device,
    )
    
    config = engine.get_config()
    
    # 处理视频
    print("\n开始处理视频...")
    all_masks = []
    iou_scores = []
    obj_scores = []
    
    prev_mask = None
    for i, (frame, name) in enumerate(tqdm(zip(frames, frame_names), total=len(frames))):
        is_cond_frame = (i == 0)
        
        mask, iou, obj_score, timings = engine.process_frame(
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
        
        # 生成可视化
        if args.visualize:
            vis_img = create_visualization(
                frame, mask, current_point, i, iou, obj_score,
                timings, config,
                color=(0.2, 0.6, 1.0), alpha=0.5
            )
            vis_path = os.path.join(vis_dir, f"{name}.jpg")
            cv2.imwrite(vis_path, vis_img)
    
    # 打印性能统计
    engine.timing_stats.print_summary(config)
    
    # 打印结果统计
    print(f"\n处理完成!")
    print(f"  总帧数: {len(frames)}")
    print(f"  平均 IoU: {np.mean(iou_scores):.3f}")
    print(f"  平均物体分数: {np.mean(obj_scores):.3f}")
    print(f"  输出目录: {args.output}")
    
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
        
        if iou_vs_ref:
            print(f"  与参考的平均 IoU: {np.mean(iou_vs_ref):.3f}")
            print(f"  最小 IoU: {np.min(iou_vs_ref):.3f}")
            print(f"  最大 IoU: {np.max(iou_vs_ref):.3f}")
    
    # 释放资源
    engine.release()


if __name__ == "__main__":
    main()
