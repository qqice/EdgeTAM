import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import onnxruntime
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_image(url):
    """加载并预处理图片"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Original image size: {image.size}")
    
    # 计算resize后的尺寸,保持长宽比
    target_size = (1024, 1024)
    w, h = image.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    print(f"Scale factor: {scale}")
    print(f"Resized dimensions: {new_w}x{new_h}")
    
    # resize图片
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 创建1024x1024的黑色背景
    processed_image = Image.new("RGB", target_size, (0, 0, 0))
    # 将resized图片粘贴到中心位置
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    print(f"Paste position: ({paste_x}, {paste_y})")
    processed_image.paste(resized_image, (paste_x, paste_y))
    
    # 保存处理后的图片用于检查
    processed_image.save("debug_processed_image.png")
    
    # 转换为numpy数组并归一化到[0,1]
    img_np = np.array(processed_image).astype(np.float32) / 255.0
    # 调整维度顺序从HWC到CHW
    img_np = img_np.transpose(2, 0, 1)
    # 添加batch维度
    img_np = np.expand_dims(img_np, axis=0)
    
    print(f"Final input tensor shape: {img_np.shape}")
    
    return image, img_np, (scale, paste_x, paste_y)

def prepare_point_input(point_coords, point_labels, image_size=(1024, 1024)):
    """准备点击输入数据"""
    point_coords = np.array(point_coords, dtype=np.float32)
    point_labels = np.array(point_labels, dtype=np.float32)
    
    # 添加batch维度
    point_coords = np.expand_dims(point_coords, axis=0)
    point_labels = np.expand_dims(point_labels, axis=0)
    
    # 准备mask输入
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    orig_im_size = np.array(image_size, dtype=np.int32)
    
    return point_coords, point_labels, mask_input, has_mask_input, orig_im_size

def main():
    # 1. 加载原始图片
    url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
    orig_image, input_image, (scale, offset_x, offset_y) = load_image(url)
    
    # 2. 准备输入点 - 需要根据scale和offset调整点击坐标
    input_point_orig = [[750, 400]]
    input_point = [[
        int(x * scale + offset_x), 
        int(y * scale + offset_y)
    ] for x, y in input_point_orig]
    print(f"Original point: {input_point_orig}")
    print(f"Transformed point: {input_point}")
    input_label = [1]
    
    # 3. 运行PyTorch模型
    print("Running PyTorch model...")
    checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    with torch.inference_mode():
        predictor.set_image(orig_image)
        masks_pt, iou_scores_pt, low_res_masks_pt = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True
        )
    
    # 4. 运行ONNX模型
    print("Running ONNX model...")
    encoder_path = "sam2.1_hiera_tiny_encoder.s.onnx"
    decoder_path = "sam2.1_hiera_tiny_decoder.onnx"
    
    # 创建ONNX Runtime会话
    encoder_session = onnxruntime.InferenceSession(encoder_path)
    decoder_session = onnxruntime.InferenceSession(decoder_path)
    
    # 运行encoder
    encoder_inputs = {'image': input_image}
    high_res_feats_0, high_res_feats_1, image_embed = encoder_session.run(None, encoder_inputs)
    
    # 准备decoder输入
    point_coords, point_labels, mask_input, has_mask_input, orig_im_size = prepare_point_input(
        input_point, input_label, orig_image.size[::-1]
    )
    
    # 运行decoder
    decoder_inputs = {
        'image_embed': image_embed,
        'high_res_feats_0': high_res_feats_0,
        'high_res_feats_1': high_res_feats_1,
        'point_coords': point_coords,
        'point_labels': point_labels,
        # 'orig_im_size': orig_im_size,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
    }
    
    low_res_masks, iou_predictions = decoder_session.run(None, decoder_inputs)
    
    # 后处理: 将low_res_masks缩放到原始图片尺寸
    w, h = orig_image.size
    
    # 1. 首先将mask缩放到1024x1024
    masks_1024 = torch.nn.functional.interpolate(
        torch.from_numpy(low_res_masks),
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False
    )
    
    # 2. 去除padding
    new_h = int(h * scale)
    new_w = int(w * scale)
    start_h = (1024 - new_h) // 2
    start_w = (1024 - new_w) // 2
    masks_no_pad = masks_1024[..., start_h:start_h+new_h, start_w:start_w+new_w]
    
    # 3. 缩放到原始图片尺寸
    masks_onnx = torch.nn.functional.interpolate(
        masks_no_pad,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    
    # 4. 二值化
    masks_onnx = masks_onnx > 0.0
    masks_onnx = masks_onnx.numpy()
    
    # 在运行ONNX模型后,打印输出的shape
    print(f"\nOutput shapes:")
    print(f"PyTorch masks shape: {masks_pt.shape}")
    print(f"ONNX masks shape: {masks_onnx.shape}")
    
    # 修改可视化部分,暂时注释掉差异图
    plt.figure(figsize=(10, 5))
    
    # PyTorch结果
    plt.subplot(121)
    plt.imshow(orig_image)
    plt.imshow(masks_pt[0], alpha=0.5)
    plt.plot(input_point_orig[0][0], input_point_orig[0][1], 'rx')
    plt.title('PyTorch Output')
    plt.axis('off')
    
    # ONNX结果
    plt.subplot(122)
    plt.imshow(orig_image)
    plt.imshow(masks_onnx[0,0], alpha=0.5)
    plt.plot(input_point_orig[0][0], input_point_orig[0][1], 'rx')
    plt.title('ONNX Output')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 6. 打印一些统计信息
    print("\nStatistics:")
    print(f"PyTorch IoU scores: {iou_scores_pt}")
    print(f"ONNX IoU predictions: {iou_predictions}")

if __name__ == "__main__":
    main() 