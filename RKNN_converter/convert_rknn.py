#!/usr/bin/env python
# coding: utf-8

import datetime
import argparse
from rknn.api import RKNN
from sys import exit
import os
import onnxslim

num_pointss = [1]
num_labelss = [1]

# 注意: SAM2.1 和 EdgeTAM 的 decoder 输入形状是相同的，因为:
# - 两者都使用 image_size=1024, backbone_stride=16
# - FpnNeck 将所有 backbone 通道统一转换为 d_model=256
# - conv_s0 输出 32 通道 (256x256), conv_s1 输出 64 通道 (128x128)
# - image_embed 是 256 通道 (64x64)

def convert_to_rknn(onnx_model, model_part, dataset="/home/zt/rk3588-nn/rknn_model_zoo/datasets/COCO/coco_subset_20.txt", quantize=False):
    """转换单个ONNX模型到RKNN格式"""
    rknn_model = onnx_model.replace(".onnx",".rknn")
    timedate_iso = datetime.datetime.now().isoformat()
    
    print(f"\n开始转换 {onnx_model} 到 {rknn_model}")

    input_shapes = None

    if model_part == "encoder":
        input_shapes = None
    elif model_part == "decoder":
        input_shapes = [
            [
                [1, 256, 64, 64],  # image_embedding
                [1, 32, 256, 256],  # high_res_feats_0
                [1, 64, 128, 128],  # high_res_feats_1
                [num_labels, num_points, 2],  # point_coords
                [num_labels, num_points],  # point_labels
                [num_labels, 1, 256, 256],  # mask_input
                [num_labels],  # has_mask_input
            ]
            for num_labels in num_labelss
            for num_points in num_pointss
        ]
    
    rknn = RKNN(verbose=True)
    rknn.config(
        dynamic_input=input_shapes,
        std_values=[[255,255,255]] if model_part == "encoder" else None,
        quantized_dtype='w8a8',
        quantized_algorithm='normal',
        quantized_method='channel',
        quantized_hybrid_level=0,
        target_platform='rk3588',
        quant_img_RGB2BGR = False,
        float_dtype='float16',
        optimization_level=3,
        custom_string=f"converted at {timedate_iso}",
        remove_weight=False,
        compress_weight=False,
        inputs_yuv_fmt=None,
        single_core_mode=False,
        model_pruning=False,
        op_target=None,
        quantize_weight=False,
        remove_reshape=False,
        sparse_infer=False,
        enable_flash_attention=False,
    )

    ret = rknn.load_onnx(model=onnx_model)
    ret = rknn.build(do_quantization=quantize, dataset=dataset, rknn_batch_size=None)
    ret = rknn.export_rknn(rknn_model)
    print(f"完成转换 {rknn_model}\n")

def main():
    parser = argparse.ArgumentParser(description='转换SAM模型从ONNX到RKNN格式')
    parser.add_argument('model_name', type=str, help='模型名称,例如: sam2.1_hiera_tiny')
    args = parser.parse_args()
    
    # 构建encoder和decoder的文件名
    encoder_onnx = f"{args.model_name}_encoder.onnx"
    decoder_onnx = f"{args.model_name}_decoder.onnx"
    
    # 检查文件是否存在
    for model in [encoder_onnx, decoder_onnx]:
        if not os.path.exists(model):
            print(f"错误: 找不到文件 {model}")
            exit(1)
    
    # 转换encoder和decoder
    #encoder需要先跑一个onnxslim
    print("开始转换encoder...")
    onnxslim.slim(encoder_onnx, output_model="encoder_slim.onnx", skip_fusion_patterns=["EliminationSlice"])
    convert_to_rknn("encoder_slim.onnx", model_part="encoder")
    os.rename("encoder_slim.rknn", encoder_onnx.replace(".onnx", ".rknn"))
    os.remove("encoder_slim.onnx")

    # convert_to_rknn(decoder_onnx, model_part="decoder") # 坏的
    
    print("所有模型转换完成!")

if __name__ == "__main__":
    main()
