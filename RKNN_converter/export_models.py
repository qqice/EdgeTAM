# -*- coding: utf-8 -*-
"""
EdgeTAM/SAM2 模型导出脚本

导出以下模块用于视频推理：
1. image_encoder.onnx: 图像编码器 (backbone + FPN)
2. mask_decoder.onnx: Mask 解码器 (prompt encoder + mask decoder)  
3. memory_encoder.onnx: 记忆编码器 (将 mask 编码为记忆特征)
4. memory_attention.pt: 记忆注意力 (PyTorch 模块，包含 RoPE 注意力)

使用方法:
    python export_models.py
    python export_models.py --checkpoint ../checkpoints/edgetam.pt --output_dir ./onnx_models
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(current_dir)

from typing import Tuple
import argparse
import pathlib

import torch
from torch import nn
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base


# ============================================================================
# ONNX 导出模块
# ============================================================================

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
        
        # 注意: 遮挡处理 (object_score_logits 条件判断) 移到推理代码中执行
        # RKNN 对 torch.where 的动态条件支持有问题，会导致条件被错误优化
        # 因此我们在 ONNX 导出时不做这个处理，而是在推理时根据 object_score_logits 判断
        #
        # 物体指针的 no_obj_ptr 混合也移到推理代码中
        # 原始逻辑:
        #   if obj_score > 0: use obj_ptr
        #   else: use no_obj_ptr (or mixed based on soft_no_obj_ptr)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pix_feat: [B, 256, 64, 64] 视觉特征
            pred_mask: [B, 1, 1024, 1024] 高分辨率 mask
            
        Returns:
            maskmem_features: [B, HW, C] 记忆特征 (经过 spatial perceiver)
            maskmem_pos_enc: [B, HW, C] 位置编码
            
        注意: object_score_logits 相关的 no_obj_embed_spatial 逻辑
        移到推理代码中处理，因为 RKNN 对动态条件支持有问题
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
        
        # 注意: no-object embedding 的条件判断移到推理代码中
        # RKNN 对 torch.where 和动态条件支持有问题
        # 原始逻辑:
        #   if obj_score > 0: use maskmem_features
        #   else: maskmem_features += no_obj_embed_spatial
        
        # 空间感知器降维
        if self.spatial_perceiver is not None:
            maskmem_features, maskmem_pos_enc = self.spatial_perceiver(maskmem_features, maskmem_pos_enc)
        
        return maskmem_features, maskmem_pos_enc


# ============================================================================
# 导出函数
# ============================================================================

def export_onnx_models(sam2_model: SAM2Base, output_dir: pathlib.Path, model_type: str, opset: int):
    """导出 ONNX 模型 (encoder, decoder, memory_encoder)"""
    
    input_size = (1024, 1024)
    
    # ==================== 1. 导出 Image Encoder ====================
    print("\n[1/3] 导出 Image Encoder...")
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu().eval()
    
    img = torch.randn(1, 3, input_size[0], input_size[1])
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)
    
    encoder_path = output_dir / f"{model_type}_encoder.onnx"
    torch.onnx.export(
        sam2_encoder,
        img,
        str(encoder_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
    )
    print(f"  保存到: {encoder_path}")
    print(f"  输出形状: high_res_feats_0={high_res_feats_0.shape}, high_res_feats_1={high_res_feats_1.shape}, image_embed={image_embed.shape}")
    
    # ==================== 2. 导出 Mask Decoder ====================
    print("\n[2/3] 导出 Mask Decoder...")
    sam2_decoder = SAM2ImageDecoder(sam2_model).cpu().eval()
    
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
        opset_version=opset,
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
    print("\n[3/3] 导出 Memory Encoder...")
    sam2_mem_encoder = SAM2MemoryEncoder(sam2_model).cpu().eval()
    
    # 准备输入 (不再需要 object_score_logits)
    pix_feat = image_embed  # [1, 256, 64, 64]
    pred_mask_high_res = torch.randn(1, 1, 1024, 1024)  # 高分辨率 mask
    
    mem_encoder_path = output_dir / f"{model_type}_memory_encoder.onnx"
    torch.onnx.export(
        sam2_mem_encoder,
        (pix_feat, pred_mask_high_res),
        str(mem_encoder_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["pix_feat", "pred_mask"],
        output_names=["maskmem_features", "maskmem_pos_enc"],
    )
    print(f"  保存到: {mem_encoder_path}")
    
    # 测试输出
    maskmem_features, maskmem_pos_enc = sam2_mem_encoder(pix_feat, pred_mask_high_res)
    print(f"  输出形状: maskmem_features={maskmem_features.shape}, maskmem_pos_enc={maskmem_pos_enc.shape}")
    
    return encoder_path, decoder_path, mem_encoder_path


def export_memory_attention_pt(sam2_model: SAM2Base, output_dir: pathlib.Path):
    """导出 Memory Attention 为 PyTorch 模块 (.pt 文件)"""
    
    print("\n[额外] 导出 Memory Attention (PyTorch)...")
    
    mem_attn = sam2_model.memory_attention
    
    # 保存 state_dict
    state_dict = mem_attn.state_dict()
    
    # 收集额外的关键参数
    extras = {}
    
    # no_mem_embed: 当没有记忆时添加到特征上
    if hasattr(sam2_model, 'no_mem_embed'):
        extras['no_mem_embed'] = sam2_model.no_mem_embed.detach().cpu()
        print(f"  no_mem_embed: {extras['no_mem_embed'].shape}")
    
    # no_mem_pos_enc: 用于位置编码
    if hasattr(sam2_model, 'no_mem_pos_enc'):
        extras['no_mem_pos_enc'] = sam2_model.no_mem_pos_enc.detach().cpu()
        print(f"  no_mem_pos_enc: {extras['no_mem_pos_enc'].shape}")
    
    # maskmem_tpos_enc: 时间位置编码 [num_maskmem, 1, 1, mem_dim]
    if hasattr(sam2_model, 'maskmem_tpos_enc'):
        extras['maskmem_tpos_enc'] = sam2_model.maskmem_tpos_enc.detach().cpu()
        print(f"  maskmem_tpos_enc: {extras['maskmem_tpos_enc'].shape}")
    
    # no_obj_embed_spatial: 当物体不存在时添加到记忆特征上
    if hasattr(sam2_model, 'no_obj_embed_spatial') and sam2_model.no_obj_embed_spatial is not None:
        extras['no_obj_embed_spatial'] = sam2_model.no_obj_embed_spatial.detach().cpu()
        print(f"  no_obj_embed_spatial: {extras['no_obj_embed_spatial'].shape}")
    
    # no_obj_ptr: 当物体不存在时使用的物体指针
    if hasattr(sam2_model, 'no_obj_ptr') and sam2_model.no_obj_ptr is not None:
        extras['no_obj_ptr'] = sam2_model.no_obj_ptr.detach().cpu()
        print(f"  no_obj_ptr: {extras['no_obj_ptr'].shape}")
    
    # soft_no_obj_ptr 和 fixed_no_obj_ptr 配置
    extras['soft_no_obj_ptr'] = getattr(sam2_model, 'soft_no_obj_ptr', False)
    extras['fixed_no_obj_ptr'] = getattr(sam2_model, 'fixed_no_obj_ptr', False)
    extras['pred_obj_scores'] = getattr(sam2_model, 'pred_obj_scores', True)
    
    output_path = output_dir / "memory_attention.pt"
    torch.save({
        'state_dict': state_dict,
        'extras': extras,
        'config': {
            'd_model': mem_attn.d_model,
            'num_layers': mem_attn.num_layers,
            'pos_enc_at_input': mem_attn.pos_enc_at_input,
            'batch_first': mem_attn.batch_first,
            'directly_add_no_mem_embed': getattr(sam2_model, 'directly_add_no_mem_embed', True),
            'num_maskmem': getattr(sam2_model, 'num_maskmem', 7),
            'mem_dim': getattr(sam2_model, 'mem_dim', 64),
            'hidden_dim': getattr(sam2_model, 'hidden_dim', 256),
        }
    }, str(output_path))
    print(f"  保存到: {output_path}")
    
    # 打印配置信息
    print(f"\n  配置:")
    print(f"    d_model: {mem_attn.d_model}")
    print(f"    num_layers: {mem_attn.num_layers}")
    print(f"    pos_enc_at_input: {mem_attn.pos_enc_at_input}")
    print(f"    batch_first: {mem_attn.batch_first}")
    
    # 打印各层信息
    for i, layer in enumerate(mem_attn.layers):
        print(f"\n  Layer {i}:")
        print(f"    self_attn type: {type(layer.self_attn).__name__}")
        print(f"    cross_attn type: {type(layer.cross_attn_image).__name__}")
        print(f"    pos_enc_at_attn: {layer.pos_enc_at_attn}")
        print(f"    pos_enc_at_cross_attn_keys: {layer.pos_enc_at_cross_attn_keys}")
        print(f"    pos_enc_at_cross_attn_queries: {layer.pos_enc_at_cross_attn_queries}")
        
        # 打印 RoPE 配置
        if hasattr(layer.self_attn, 'freqs_cis'):
            print(f"    self_attn.freqs_cis: {layer.self_attn.freqs_cis.shape}")
        if hasattr(layer.cross_attn_image, 'freqs_cis_q'):
            print(f"    cross_attn.freqs_cis_q: {layer.cross_attn_image.freqs_cis_q.shape}")
        if hasattr(layer.cross_attn_image, 'freqs_cis_k'):
            print(f"    cross_attn.freqs_cis_k: {layer.cross_attn_image.freqs_cis_k.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in mem_attn.parameters())
    print(f"\n  总参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    return output_path


def get_model_config(model_type: str) -> str:
    """获取模型配置文件路径"""
    config_map = {
        "edgetam": "configs/edgetam.yaml",
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    return config_map.get(model_type, "configs/edgetam.yaml")


def main():
    parser = argparse.ArgumentParser(description="导出 EdgeTAM/SAM2 模型")
    
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/edgetam.pt",
                        help="模型检查点路径")
    parser.add_argument("--model_type", type=str, default="edgetam",
                        help="模型类型: edgetam, sam2.1_hiera_tiny/small/base_plus/large")
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
                        help="输出目录")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EdgeTAM/SAM2 模型导出")
    print("=" * 60)
    
    # 检查 checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到 checkpoint: {args.checkpoint}")
        return
    
    # 获取配置文件
    model_cfg = get_model_config(args.model_type)
    print(f"模型类型: {args.model_type}")
    print(f"配置文件: {model_cfg}")
    print(f"检查点: {args.checkpoint}")
    
    # 加载模型
    print(f"\n加载模型...")
    sam2_model = build_sam2(model_cfg, args.checkpoint, device="cpu")
    sam2_model.eval()
    
    # 创建输出目录
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 导出 ONNX 模型
    encoder_path, decoder_path, mem_encoder_path = export_onnx_models(
        sam2_model, output_dir, args.model_type, args.opset
    )
    
    # 导出 Memory Attention (PyTorch)
    mem_attn_path = export_memory_attention_pt(sam2_model, output_dir)
    
    # 总结
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print(f"\n导出的模型文件:")
    print(f"  1. {encoder_path}")
    print(f"  2. {decoder_path}")
    print(f"  3. {mem_encoder_path}")
    print(f"  4. {mem_attn_path}")
    
    # 文件大小
    print(f"\n文件大小:")
    for path in [encoder_path, decoder_path, mem_encoder_path, mem_attn_path]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path.name}: {size_mb:.2f} MB")
    
    print(f"\n使用说明:")
    print(f"  - ONNX 模型可用于 ONNX Runtime 或转换为 RKNN 等格式")
    print(f"  - Memory Attention 使用 PyTorch 模块 (因为包含 RoPE 注意力)")
    print(f"  - 运行推理: python test_video_standalone.py --video_dir <视频路径> --point X,Y")


if __name__ == "__main__":
    main()
