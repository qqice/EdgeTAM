# -*- coding: utf-8 -*-
"""
完整的 EdgeTAM/SAM2 ONNX 导出脚本

导出以下模块用于视频推理：
1. image_encoder: 图像编码器 (backbone + FPN)
2. mask_decoder: Mask 解码器 (prompt encoder + mask decoder)  
3. memory_encoder: 记忆编码器 (将 mask 编码为记忆特征)
4. memory_attention: 记忆注意力 (融合当前帧与历史记忆)

使用方法:
    python export_onnx_full.py \
        --checkpoint ../checkpoints/edgetam.pt \
        --model_type edgetam \
        --output_dir ./onnx_models
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from typing import Any, Optional, List, Tuple
import argparse
import pathlib

import torch
from torch import nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base


class SAM2ImageEncoder(nn.Module):
    """图像编码器 - 提取多尺度特征"""
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.image_encoder(x)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        return feats[0], feats[1], feats[2]


class SAM2ImageDecoder(nn.Module):
    """Mask 解码器 - 生成分割 mask 和物体指针"""
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.img_size = sam_model.image_size
        self.hidden_dim = sam_model.hidden_dim
        
        # 物体指针投影
        self.obj_ptr_proj = sam_model.obj_ptr_proj
        self.pred_obj_scores = sam_model.pred_obj_scores
        if self.pred_obj_scores:
            self.no_obj_ptr = sam_model.no_obj_ptr
            self.fixed_no_obj_ptr = sam_model.fixed_no_obj_ptr
            self.soft_no_obj_ptr = sam_model.soft_no_obj_ptr

    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            masks: [B, 4, H, W] 所有 4 个 mask 输出
            iou_predictions: [B, 4] IoU 预测
            obj_ptr: [B, C] 物体指针
            object_score_logits: [B, 1] 物体存在分数
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]

        # 获取所有输出
        low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = \
            self.mask_decoder.predict_masks(
                image_embeddings=image_embed,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding,
                repeat_image=False,
                high_res_features=high_res_feats,
            )

        masks = torch.clamp(low_res_masks, -32.0, 32.0)
        
        # 提取物体指针
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        
        # 遮挡处理
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return masks, iou_predictions, obj_ptr, object_score_logits

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5

        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device)
        padding_label = -torch.ones((point_labels.shape[0], 1), device=point_labels.device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


class SAM2MemoryEncoder(nn.Module):
    """记忆编码器 - 将 mask 和视觉特征编码为记忆特征"""
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.memory_encoder = sam_model.memory_encoder
        self.spatial_perceiver = sam_model.spatial_perceiver
        self.no_obj_embed_spatial = sam_model.no_obj_embed_spatial
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc

    @torch.no_grad()
    def forward(
        self,
        pix_feat: torch.Tensor,  # [B, C, H, W] 视觉特征 (image_embed)
        pred_mask: torch.Tensor,  # [B, 1, H_mask, W_mask] 预测的 mask (高分辨率)
        object_score_logits: torch.Tensor,  # [B, 1] 物体存在分数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pix_feat: [B, 256, 64, 64] 视觉特征
            pred_mask: [B, 1, 1024, 1024] 高分辨率 mask
            object_score_logits: [B, 1] 物体存在分数
            
        Returns:
            maskmem_features: [B, 256, 64] 或 [B, 64, 64, 64] 记忆特征
            maskmem_pos_enc: [B, 256, 64] 或 [B, 64, 64, 64] 位置编码
        """
        # 应用 sigmoid 和缩放
        mask_for_mem = torch.sigmoid(pred_mask)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        
        # 运行记忆编码器
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"][0]
        
        # 添加 no-object embedding
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features = maskmem_features + (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)
        
        # 空间感知器降维
        if self.spatial_perceiver is not None:
            maskmem_features, maskmem_pos_enc = self.spatial_perceiver(maskmem_features, maskmem_pos_enc)
        
        return maskmem_features, maskmem_pos_enc


class SAM2MemoryAttention(nn.Module):
    """记忆注意力 - 融合当前帧特征与历史记忆"""
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.memory_attention = sam_model.memory_attention
        self.hidden_dim = sam_model.hidden_dim
        self.num_maskmem = sam_model.num_maskmem
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.mem_dim = sam_model.mem_dim

    @torch.no_grad()
    def forward(
        self,
        curr_feat: torch.Tensor,  # [HW, B, C] 当前帧特征
        curr_pos: torch.Tensor,   # [HW, B, C] 当前帧位置编码
        memory: torch.Tensor,     # [N, B, C] 记忆特征
        memory_pos: torch.Tensor, # [N, B, C] 记忆位置编码
        num_obj_ptr_tokens: torch.Tensor,  # 物体指针数量 (标量)
    ) -> torch.Tensor:
        """
        Args:
            curr_feat: [4096, 1, 256] 当前帧特征 (64*64, batch=1, 256)
            curr_pos: [4096, 1, 256] 当前帧位置编码
            memory: [N, 1, 64] 拼接的记忆特征 (N 个 token)
            memory_pos: [N, 1, 64] 记忆位置编码
            num_obj_ptr_tokens: 物体指针 token 数量
            
        Returns:
            pix_feat_with_mem: [4096, 1, 256] 融合记忆后的特征
        """
        # 计算空间记忆数量
        num_obj_ptr = int(num_obj_ptr_tokens.item()) if isinstance(num_obj_ptr_tokens, torch.Tensor) else int(num_obj_ptr_tokens)
        total_mem = memory.shape[0]
        num_spatial_mem = total_mem - num_obj_ptr
        
        pix_feat_with_mem = self.memory_attention(
            curr=[curr_feat],
            curr_pos=[curr_pos],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr,
            num_spatial_mem=num_spatial_mem,
        )
        
        return pix_feat_with_mem


def export_models(args):
    """导出所有 ONNX 模型"""
    
    input_size = (1024, 1024)
    model_type = args.model_type
    
    # 选择配置文件
    if model_type == "sam2.1_hiera_tiny":
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif model_type == "sam2.1_hiera_small":
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_type == "sam2.1_hiera_base_plus":
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif model_type == "sam2.1_hiera_large":
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif model_type == "edgetam":
        model_cfg = "configs/edgetam.yaml"
    else:
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"加载模型: {model_cfg}")
    sam2_model = build_sam2(model_cfg, args.checkpoint, device="cpu")
    
    # 创建输出目录
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 1. 导出 Image Encoder ====================
    print("\n[1/4] 导出 Image Encoder...")
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu().eval()
    
    img = torch.randn(1, 3, input_size[0], input_size[1])
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)
    
    encoder_path = output_dir / f"{model_type}_encoder.onnx"
    torch.onnx.export(
        sam2_encoder,
        img,
        str(encoder_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
    )
    print(f"  保存到: {encoder_path}")
    print(f"  输出形状: high_res_feats_0={high_res_feats_0.shape}, high_res_feats_1={high_res_feats_1.shape}, image_embed={image_embed.shape}")
    
    # ==================== 2. 导出 Mask Decoder ====================
    print("\n[2/4] 导出 Mask Decoder...")
    sam2_decoder = SAM2ImageDecoder(sam2_model).cpu().eval()
    
    embed_dim = sam2_model.sam_prompt_encoder.embed_dim
    embed_size = (sam2_model.image_size // sam2_model.backbone_stride,
                  sam2_model.image_size // sam2_model.backbone_stride)
    mask_input_size = [4 * x for x in embed_size]
    
    num_points = 1
    point_coords = torch.randint(low=0, high=input_size[1], size=(1, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(1, num_points), dtype=torch.float)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    
    decoder_path = output_dir / f"{model_type}_decoder.onnx"
    torch.onnx.export(
        sam2_decoder,
        (image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input),
        str(decoder_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "image_embed", "high_res_feats_0", "high_res_feats_1",
            "point_coords", "point_labels", "mask_input", "has_mask_input"
        ],
        output_names=["masks", "iou_predictions", "obj_ptr", "object_score_logits"],
    )
    print(f"  保存到: {decoder_path}")
    
    # 测试 decoder 输出
    masks, iou_preds, obj_ptr, obj_score = sam2_decoder(
        image_embed, high_res_feats_0, high_res_feats_1,
        point_coords, point_labels, mask_input, has_mask_input
    )
    print(f"  输出形状: masks={masks.shape}, iou_predictions={iou_preds.shape}, obj_ptr={obj_ptr.shape}, object_score_logits={obj_score.shape}")
    
    # ==================== 3. 导出 Memory Encoder ====================
    print("\n[3/4] 导出 Memory Encoder...")
    sam2_mem_encoder = SAM2MemoryEncoder(sam2_model).cpu().eval()
    
    # 准备输入
    # pix_feat: 从 image_embed 转换来 (需要 reshape)
    B, C, H, W = 1, 256, 64, 64
    pix_feat = image_embed  # [1, 256, 64, 64]
    pred_mask_high_res = torch.randn(1, 1, 1024, 1024)  # 高分辨率 mask
    object_score_logits = torch.tensor([[5.0]])  # 物体存在
    
    mem_encoder_path = output_dir / f"{model_type}_memory_encoder.onnx"
    torch.onnx.export(
        sam2_mem_encoder,
        (pix_feat, pred_mask_high_res, object_score_logits),
        str(mem_encoder_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["pix_feat", "pred_mask", "object_score_logits"],
        output_names=["maskmem_features", "maskmem_pos_enc"],
    )
    print(f"  保存到: {mem_encoder_path}")
    
    # 测试输出
    maskmem_features, maskmem_pos_enc = sam2_mem_encoder(pix_feat, pred_mask_high_res, object_score_logits)
    print(f"  输出形状: maskmem_features={maskmem_features.shape}, maskmem_pos_enc={maskmem_pos_enc.shape}")
    
    # ==================== 4. 导出 Memory Attention ====================
    print("\n[4/4] 导出 Memory Attention...")
    sam2_mem_attention = SAM2MemoryAttention(sam2_model).cpu().eval()
    
    # 准备输入
    # curr_feat: [HW, B, C] = [4096, 1, 256]
    # 从 image_embed [1, 256, 64, 64] 转换
    curr_feat = image_embed.flatten(2).permute(2, 0, 1)  # [4096, 1, 256]
    
    # 获取位置编码
    backbone_out = sam2_model.image_encoder(img)
    vision_pos_embeds = backbone_out["vision_pos_enc"][-sam2_model.num_feature_levels:]
    curr_pos = vision_pos_embeds[-1].flatten(2).permute(2, 0, 1)  # [4096, 1, 256]
    
    # 记忆输入: 固定 7 个记忆槽 + 最多 16 个物体指针
    # 使用 spatial perceiver 后: [B, 256, 64] => [256, B, 64]
    num_mem_slots = 7  # num_maskmem
    mem_dim = sam2_model.mem_dim  # 64
    hidden_dim = sam2_model.hidden_dim  # 256
    
    # 每个记忆特征: [256, 1, 64] (spatial perceiver 输出)
    # 总记忆: 7 * 256 = 1792 个 spatial tokens
    # 加上最多 16 个 obj_ptr tokens (每个分成 256/64=4 个 tokens) = 16 * 4 = 64
    # 总共最多: 1792 + 64 = 1856 tokens (但我们用固定数量简化)
    
    # 简化: 使用固定的 memory 大小
    # 1 个 cond frame (256 tokens) + 6 个 non-cond frames (6*256=1536 tokens) = 1792
    # 加上 4 个 obj_ptr (4*4=16 tokens) = 1808
    num_spatial_tokens = 256 * num_mem_slots  # 1792
    num_obj_ptr_tokens = 16  # 4 个物体指针，每个分成 4 个 token
    total_mem_tokens = num_spatial_tokens + num_obj_ptr_tokens
    
    memory = torch.randn(total_mem_tokens, 1, mem_dim)  # [1808, 1, 64]
    memory_pos = torch.randn(total_mem_tokens, 1, mem_dim)  # [1808, 1, 64]
    num_obj_ptr = torch.tensor([num_obj_ptr_tokens], dtype=torch.int64)
    
    mem_attention_path = output_dir / f"{model_type}_memory_attention.onnx"
    
    # 注意: Memory Attention 有动态序列长度，ONNX 导出可能有限制
    # 我们尝试导出，如果失败则提供说明
    try:
        torch.onnx.export(
            sam2_mem_attention,
            (curr_feat, curr_pos, memory, memory_pos, num_obj_ptr),
            str(mem_attention_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["curr_feat", "curr_pos", "memory", "memory_pos", "num_obj_ptr_tokens"],
            output_names=["pix_feat_with_mem"],
            dynamic_axes={
                "memory": {0: "num_mem_tokens"},
                "memory_pos": {0: "num_mem_tokens"},
            }
        )
        print(f"  保存到: {mem_attention_path}")
        
        # 测试输出
        pix_feat_with_mem = sam2_mem_attention(curr_feat, curr_pos, memory, memory_pos, num_obj_ptr)
        print(f"  输出形状: pix_feat_with_mem={pix_feat_with_mem.shape}")
    except Exception as e:
        print(f"  警告: Memory Attention 导出失败: {e}")
        print(f"  这是因为 Memory Attention 使用了动态序列长度和 RoPE 注意力")
        print(f"  将在推理脚本中使用 PyTorch 模块替代")
    
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print(f"\n导出的模型:")
    print(f"  1. {encoder_path}")
    print(f"  2. {decoder_path}")
    print(f"  3. {mem_encoder_path}")
    if mem_attention_path.exists():
        print(f"  4. {mem_attention_path}")
    else:
        print(f"  4. Memory Attention 将使用 PyTorch 模块")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出完整的 SAM2/EdgeTAM ONNX 模型")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_type", type=str, required=True,
                        help="模型类型: edgetam, sam2.1_hiera_tiny/small/base_plus/large")
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
                        help="输出目录")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本")
    
    args = parser.parse_args()
    export_models(args)
