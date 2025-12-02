# -*- coding: utf-8 -*-
"""
导出 Memory Attention 权重为 NumPy 格式

由于 Memory Attention 使用 RoPE，无法直接导出为 ONNX，
这个脚本将权重导出为 NumPy .npz 文件，供 test_video_memory_rope.py 使用。

使用方法:
    python export_memory_attention_weights.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor


def export_memory_attention_weights():
    """导出 Memory Attention 的权重为 NumPy 格式"""
    
    print("=" * 60)
    print("导出 Memory Attention 权重")
    print("=" * 60)
    
    # 加载模型
    model_cfg = "configs/edgetam.yaml"
    checkpoint = "../checkpoints/edgetam.pt"
    
    if not os.path.exists(checkpoint):
        print(f"错误: 找不到 checkpoint: {checkpoint}")
        return
    
    print(f"加载模型: {model_cfg}")
    model = build_sam2(model_cfg, checkpoint, device="cpu")
    model.eval()
    
    # 获取 Memory Attention 模块
    mem_attn = model.memory_attention
    print(f"\nMemory Attention 结构:")
    print(f"  - num_layers: {len(mem_attn.layers)}")
    print(f"  - d_model: {mem_attn.d_model}")
    print(f"  - pos_enc_at_input: {mem_attn.pos_enc_at_input}")
    
    # 收集权重
    weights = {}
    
    # 1. 最终 LayerNorm
    weights['norm.weight'] = mem_attn.norm.weight.detach().cpu().numpy()
    weights['norm.bias'] = mem_attn.norm.bias.detach().cpu().numpy()
    
    # 2. 各层权重
    for layer_idx, layer in enumerate(mem_attn.layers):
        prefix = f'layer{layer_idx}'
        
        # Self-attention
        self_attn = layer.self_attn
        weights[f'{prefix}.self_attn.q_proj.weight'] = self_attn.q_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.q_proj.bias'] = self_attn.q_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.k_proj.weight'] = self_attn.k_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.k_proj.bias'] = self_attn.k_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.v_proj.weight'] = self_attn.v_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.v_proj.bias'] = self_attn.v_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.out_proj.weight'] = self_attn.out_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.self_attn.out_proj.bias'] = self_attn.out_proj.bias.detach().cpu().numpy()
        
        # Cross-attention
        cross_attn = layer.cross_attn_image
        weights[f'{prefix}.cross_attn.q_proj.weight'] = cross_attn.q_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.q_proj.bias'] = cross_attn.q_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.k_proj.weight'] = cross_attn.k_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.k_proj.bias'] = cross_attn.k_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.v_proj.weight'] = cross_attn.v_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.v_proj.bias'] = cross_attn.v_proj.bias.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.out_proj.weight'] = cross_attn.out_proj.weight.detach().cpu().numpy()
        weights[f'{prefix}.cross_attn.out_proj.bias'] = cross_attn.out_proj.bias.detach().cpu().numpy()
        
        # LayerNorms
        weights[f'{prefix}.norm1.weight'] = layer.norm1.weight.detach().cpu().numpy()
        weights[f'{prefix}.norm1.bias'] = layer.norm1.bias.detach().cpu().numpy()
        weights[f'{prefix}.norm2.weight'] = layer.norm2.weight.detach().cpu().numpy()
        weights[f'{prefix}.norm2.bias'] = layer.norm2.bias.detach().cpu().numpy()
        weights[f'{prefix}.norm3.weight'] = layer.norm3.weight.detach().cpu().numpy()
        weights[f'{prefix}.norm3.bias'] = layer.norm3.bias.detach().cpu().numpy()
        
        # MLP
        mlp = layer.linear1
        weights[f'{prefix}.mlp.fc1.weight'] = mlp.weight.detach().cpu().numpy()
        weights[f'{prefix}.mlp.fc1.bias'] = mlp.bias.detach().cpu().numpy()
        mlp2 = layer.linear2
        weights[f'{prefix}.mlp.fc2.weight'] = mlp2.weight.detach().cpu().numpy()
        weights[f'{prefix}.mlp.fc2.bias'] = mlp2.bias.detach().cpu().numpy()
        
        print(f"  Layer {layer_idx}: self_attn, cross_attn, norm1, norm2, norm3, mlp")
    
    # 3. 获取 RoPE 参数 (如果有)
    if hasattr(mem_attn.layers[0].self_attn, 'freqs_cis'):
        freqs = mem_attn.layers[0].self_attn.freqs_cis.detach().cpu().numpy()
        weights['rope_freqs_cis'] = np.stack([freqs.real, freqs.imag], axis=-1)
        print(f"  RoPE freqs_cis shape: {freqs.shape}")
    
    # 4. 获取 no_mem_embed (如果有)
    if hasattr(mem_attn, 'no_mem_embed'):
        weights['no_mem_embed'] = mem_attn.no_mem_embed.detach().cpu().numpy()
        print(f"  no_mem_embed shape: {weights['no_mem_embed'].shape}")
    
    # 5. 获取配置参数
    config = {
        'd_model': mem_attn.d_model,
        'num_layers': len(mem_attn.layers),
        'pos_enc_at_input': mem_attn.pos_enc_at_input,
    }
    
    # 获取第一层的更多参数
    first_layer = mem_attn.layers[0]
    if hasattr(first_layer.self_attn, 'num_heads'):
        config['num_heads'] = first_layer.self_attn.num_heads
    if hasattr(first_layer.self_attn, 'internal_dim'):
        config['internal_dim'] = first_layer.self_attn.internal_dim
    if hasattr(first_layer.cross_attn_image, 'kv_in_dim'):
        config['d_mem'] = first_layer.cross_attn_image.kv_in_dim
    
    print(f"\n配置参数: {config}")
    
    # 保存权重
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "memory_attention_weights.npz")
    
    np.savez(output_path, config=config, **weights)
    print(f"\n权重已保存到: {output_path}")
    
    # 统计
    total_params = sum(w.size for w in weights.values() if isinstance(w, np.ndarray))
    print(f"总参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    print("\n权重列表:")
    for name, arr in weights.items():
        if isinstance(arr, np.ndarray):
            print(f"  {name}: {arr.shape}")
    
    return output_path


if __name__ == "__main__":
    export_memory_attention_weights()
