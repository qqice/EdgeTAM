# V1: Encoder + Decoder ONNX 版本

这是最初的实现方案，只导出了 Image Encoder 和 Mask Decoder 到 ONNX/RKNN。

## 局限性

- **无视频跟踪能力**: 缺少 Memory Encoder 和 Memory Attention，无法进行视频物体跟踪
- **仅支持单帧推理**: 每帧独立处理，无法利用历史信息

## 文件说明

| 文件 | 说明 |
|------|------|
| `export_onnx.py` | 导出 encoder 和 decoder 到 ONNX |
| `test_onnx.py` | 测试 ONNX 模型的单帧推理 |
| `test_video_simple.py` | 简单的视频推理（无记忆） |
| `test_video_onnx.py` | 使用 ONNX 的视频推理（无记忆） |
| `edgetam_encoder.onnx` | Image Encoder ONNX 模型 |
| `edgetam_decoder.onnx` | Mask Decoder ONNX 模型 |
| `*.rknn` | 转换后的 RKNN 模型 |

## 使用方法

```bash
# 导出 ONNX
python export_onnx.py

# 测试单帧
python test_onnx.py --image path/to/image.jpg --point 420,270
```
