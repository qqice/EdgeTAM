import os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnxruntime
from rknnlite.api import RKNNLite
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_image(path):
    """加载并预处理图片"""
    image = Image.open(path).convert("RGB")
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
    
    # 转换为numpy数组并归一化到[0,1] # 归一化整合到模型了
    img_np = np.array(processed_image).astype(np.float32) # / 255.0
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
    path = "dog.jpg"
    orig_image, input_image, (scale, offset_x, offset_y) = load_image(path)
    # RKNN 和 ONNX 模型路径
    # 对于 EdgeTAM: 先运行 export_onnx.py 生成 ONNX，再运行 convert_rknn.py 转换为 RKNN
    # python export_onnx.py --model_type edgetam --checkpoint ../checkpoints/edgetam.pt --output_encoder edgetam_encoder.onnx --output_decoder edgetam_decoder.onnx
    # python convert_rknn.py edgetam
    decoder_path = "edgetam_decoder.onnx"
    encoder_path = "edgetam_encoder.rknn"

    # 2. 准备输入点
    # input_point_orig = [[750, 400]]
    input_point_orig = [[189, 394]]
    input_point = [[
        int(x * scale + offset_x), 
        int(y * scale + offset_y)
    ] for x, y in input_point_orig]
    input_label = [1]
    
    # 3. 运行RKNN encoder
    print("Running RKNN encoder...")
    rknn_lite = RKNNLite(verbose=False)
    
    ret = rknn_lite.load_rknn(encoder_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    start_time = time.time()
    encoder_outputs = rknn_lite.inference(inputs=[input_image], data_format="nchw")
    end_time = time.time()
    print(f"RKNN encoder time: {end_time - start_time} seconds")
    high_res_feats_0, high_res_feats_1, image_embed = encoder_outputs
    rknn_lite.release()
    
    # 4. 运行ONNX decoder
    print("Running ONNX decoder...")
    decoder_session = onnxruntime.InferenceSession(decoder_path)
    
    point_coords, point_labels, mask_input, has_mask_input, orig_im_size = prepare_point_input(
        input_point, input_label, orig_image.size[::-1]
    )
    
    decoder_inputs = {
        'image_embed': image_embed,
        'high_res_feats_0': high_res_feats_0,
        'high_res_feats_1': high_res_feats_1,
        'point_coords': point_coords,
        'point_labels': point_labels,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
    }
    start_time = time.time()
    low_res_masks, iou_predictions = decoder_session.run(None, decoder_inputs)
    end_time = time.time()
    print(f"ONNX decoder time: {end_time - start_time} seconds")
    print(low_res_masks.shape)
    # 5. 后处理
    w, h = orig_image.size
    masks_rknn = []
    
    # 处理所有3个mask
    for i in range(low_res_masks.shape[1]):
        # 将mask缩放到1024x1024
        masks_1024 = cv2.resize(
            low_res_masks[0,i],
            (1024, 1024),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 去除padding
        new_h = int(h * scale)
        new_w = int(w * scale)
        start_h = (1024 - new_h) // 2
        start_w = (1024 - new_w) // 2
        masks_no_pad = masks_1024[start_h:start_h+new_h, start_w:start_w+new_w]
        
        # 缩放到原始图片尺寸
        mask = cv2.resize(
            masks_no_pad,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 二值化
        mask = mask > 0.0
        masks_rknn.append(mask)
    
    # 6. 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 获取IoU分数排序的索引
    sorted_indices = np.argsort(iou_predictions[0])[::-1]  # 降序排序
    
    for idx, mask_idx in enumerate(sorted_indices):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(orig_image)
        plt.imshow(masks_rknn[mask_idx], alpha=0.5)
        plt.plot(input_point_orig[0][0], input_point_orig[0][1], 'rx')
        plt.title(f'Mask {mask_idx+1}\nIoU: {iou_predictions[0][mask_idx]:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("result.png")
    
    print(f"\nIoU predictions: {iou_predictions}")

if __name__ == "__main__":
    main() 