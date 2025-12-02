# V2: Memory Encoder ONNX + NumPy Memory Attention 版本

这是第二个实现方案，添加了 Memory Encoder 的 ONNX 导出，并尝试用 NumPy 重新实现 Memory Attention。

## 方案内容

1. **ONNX 模型**: Image Encoder, Mask Decoder, Memory Encoder
2. **NumPy 实现**: Memory Attention（包含 RoPE 注意力机制）

## 局限性

- **RoPE 实现困难**: Memory Attention 中的 RoPE (Rotary Position Embedding) 涉及复数运算
- **RoPEv2 更复杂**: Cross-Attention 使用的 RoPEAttentionv2 需要为 Q 和 K 使用不同的位置编码
- **精度问题**: NumPy 实现难以完全匹配 PyTorch 的计算精度，导致跟踪质量下降

## 文件说明

| 文件 | 说明 |
|------|------|
| `export_onnx_full.py` | 导出完整的 ONNX 模型（encoder, decoder, memory_encoder） |
| `export_memory_attention_weights.py` | 导出 Memory Attention 权重为 NPZ 格式 |
| `test_video_memory.py` | 使用 NumPy Memory Attention 的视频推理 |
| `memory_attention_weights.npz` | Memory Attention 权重文件 |

## 问题分析

测试发现 mask 面积随时间不断增大，IoU 急剧下降，说明 RoPE 注意力的 NumPy 实现存在错误。
最终决定放弃纯 NumPy 实现，改用 PyTorch 保留 Memory Attention。
