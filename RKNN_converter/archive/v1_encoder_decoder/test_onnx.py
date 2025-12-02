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
    """加载并预处理图片 - 与 SAM2ImagePredictor 保持一致"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Original image size: {image.size}")
    orig_w, orig_h = image.size
    
    # SAM2ImagePredictor 使用 torchvision.transforms.Resize 直接 resize 到 1024x1024
    # 不保持长宽比！
    target_size = (1024, 1024)
    resized_image = image.resize(target_size, Image.Resampling.BILINEAR)
    
    # 转换为 numpy 数组
    img_np = np.array(resized_image).astype(np.float32) / 255.0
    
    # 使用与 SAM2Transforms 相同的 ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # 调整维度顺序从 HWC 到 CHW
    img_np = img_np.transpose(2, 0, 1)
    # 添加 batch 维度
    img_np = np.expand_dims(img_np, axis=0)
    
    print(f"Final input tensor shape: {img_np.shape}")
    
    return image, img_np, (orig_w, orig_h)

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
    
    return point_coords, point_labels, mask_input, has_mask_input


def get_stability_scores(mask_logits, threshold=0.0, stability_delta=1.0):
    """
    计算 mask 的稳定性分数。
    稳定性分数 = IoU(高阈值二值化mask, 低阈值二值化mask)
    """
    high_thresh_mask = mask_logits > (threshold + stability_delta)
    low_thresh_mask = mask_logits > (threshold - stability_delta)
    
    # 计算交集和并集
    intersections = np.sum(high_thresh_mask & low_thresh_mask, axis=(-2, -1))
    unions = np.sum(high_thresh_mask | low_thresh_mask, axis=(-2, -1))
    
    # 避免除以零
    stability_scores = np.where(unions > 0, intersections / unions, 1.0)
    return stability_scores


def select_mask_by_stability(all_masks, all_iou_scores, multimask_output=False, stability_thresh=0.95):
    """
    根据稳定性选择 mask，复现 _dynamic_multimask_via_stability 的逻辑。
    
    Args:
        all_masks: 所有 4 个 mask 输出 [batch, 4, H, W]
        all_iou_scores: 所有 4 个 IoU 分数 [batch, 4]
        multimask_output: 是否返回多个 mask
        stability_thresh: 稳定性阈值
    
    Returns:
        选择后的 masks 和 iou_scores
    """
    if multimask_output:
        # 多 mask 输出模式：返回 mask 1-3（跳过 mask 0）
        return all_masks[:, 1:, :, :], all_iou_scores[:, 1:]
    
    # 单 mask 输出模式：根据稳定性动态选择
    batch_size = all_masks.shape[0]
    
    # 从多 mask 输出 (1-3) 中选择 IoU 最高的
    multimask_logits = all_masks[:, 1:, :, :]  # [batch, 3, H, W]
    multimask_iou_scores = all_iou_scores[:, 1:]  # [batch, 3]
    best_indices = np.argmax(multimask_iou_scores, axis=-1)  # [batch]
    
    # 获取最佳多 mask
    best_multimask_logits = np.array([
        multimask_logits[b, best_indices[b]] for b in range(batch_size)
    ])[:, np.newaxis, :, :]  # [batch, 1, H, W]
    best_multimask_iou_scores = np.array([
        multimask_iou_scores[b, best_indices[b]] for b in range(batch_size)
    ])[:, np.newaxis]  # [batch, 1]
    
    # 单 mask 输出 (mask 0) 及其稳定性分数
    singlemask_logits = all_masks[:, 0:1, :, :]
    singlemask_iou_scores = all_iou_scores[:, 0:1]
    stability_scores = get_stability_scores(singlemask_logits)
    
    # 根据稳定性选择
    is_stable = stability_scores >= stability_thresh
    
    # 选择结果
    mask_out = np.where(
        is_stable[..., np.newaxis, np.newaxis],
        singlemask_logits,
        best_multimask_logits
    )
    iou_out = np.where(
        is_stable,
        singlemask_iou_scores,
        best_multimask_iou_scores
    )
    
    return mask_out, iou_out

def main():
    # 1. 加载原始图片
    url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
    orig_image, input_image, (orig_w, orig_h) = load_image(url)
    
    # 2. 准备输入点
    # 使用原始图像坐标，后面会转换为模型坐标
    input_point_orig = [[750, 400]]
    input_label = [1]
    
    # 3. 运行PyTorch模型
    # 支持的模型类型:
    # - SAM2.1: sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large
    # - EdgeTAM: edgetam
    print("Running PyTorch model...")
    # 选择模型配置 (可修改为其他模型)
    # checkpoint = "sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "../checkpoints/edgetam.pt"  # EdgeTAM 检查点路径
    model_cfg = "configs/edgetam.yaml"  # EdgeTAM 配置
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    with torch.inference_mode():
        predictor.set_image(orig_image)
        masks_pt, iou_scores_pt, low_res_masks_pt = predictor.predict(
            point_coords=np.array(input_point_orig),
            point_labels=np.array(input_label),
            multimask_output=True
        )
    
    # 4. 运行ONNX模型
    print("Running ONNX model...")
    # ONNX 模型路径 (需要先通过 export_onnx.py 生成)
    # 对于 EdgeTAM: python export_onnx.py --model_type edgetam --checkpoint ../checkpoints/edgetam.pt --output_encoder edgetam_encoder.onnx --output_decoder edgetam_decoder.onnx
    encoder_path = "edgetam_encoder.onnx"
    decoder_path = "edgetam_decoder.onnx"
    
    # 创建ONNX Runtime会话
    encoder_session = onnxruntime.InferenceSession(encoder_path)
    decoder_session = onnxruntime.InferenceSession(decoder_path)
    
    # 运行encoder
    encoder_inputs = {'image': input_image}
    high_res_feats_0, high_res_feats_1, image_embed = encoder_session.run(None, encoder_inputs)
    
    # 将原始图像坐标转换为模型坐标（与 SAM2Transforms.transform_coords 一致）
    # 公式: coord_model = coord_orig / orig_size * 1024
    input_point_model = [[
        x / orig_w * 1024,
        y / orig_h * 1024
    ] for x, y in input_point_orig]
    print(f"Original point: {input_point_orig}")
    print(f"Model point (1024x1024): {input_point_model}")
    
    # 准备decoder输入
    point_coords, point_labels, mask_input, has_mask_input = prepare_point_input(
        input_point_model, input_label, (orig_h, orig_w)
    )
    
    # 运行decoder
    decoder_inputs = {
        'image_embed': image_embed,
        'high_res_feats_0': high_res_feats_0,
        'high_res_feats_1': high_res_feats_1,
        'point_coords': point_coords,
        'point_labels': point_labels,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
    }
    
    low_res_masks, iou_predictions = decoder_session.run(None, decoder_inputs)
    
    # 根据稳定性选择 mask（复现原始 PyTorch 模型的行为）
    # 对于单 mask 输出 (multimask_output=False)，这会根据稳定性动态选择
    # 对于多 mask 输出 (multimask_output=True)，这会返回 mask 1-3
    selected_masks, selected_iou = select_mask_by_stability(
        low_res_masks, iou_predictions, multimask_output=True
    )
    
    # 后处理: 将low_res_masks缩放到原始图片尺寸
    w, h = orig_image.size
    
    # 由于 SAM2ImagePredictor 使用直接 resize（不保持长宽比），
    # 输出的 mask 也是 1024x1024，可以直接 resize 回原始尺寸
    # 先将 256x256 的 low_res_mask 缩放到原始图片尺寸
    masks_onnx = torch.nn.functional.interpolate(
        torch.from_numpy(selected_masks),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    
    # 二值化
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
    print(f"ONNX IoU predictions (selected): {selected_iou}")
    print(f"ONNX IoU predictions (all 4): {iou_predictions}")

if __name__ == "__main__":
    main() 