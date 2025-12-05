# -*- coding: utf-8 -*-
"""
实时/模拟摄像头推理测试脚本。

场景：
- 加载 examples/03_blocks.mp4 作为视频流（可切换为摄像头序号）
- 在第 10 帧设置 include point (966, 271)
- 在第 240 帧重置模型状态

输出：
- 将可视化帧写入 output/run_hybrid_camera_sim/<video_name>/vis
- 在控制台打印关键事件

依赖：需先安装本项目依赖，并确保 test_video_hybrid.py 中的封装函数可用。
"""

import os
import argparse
import cv2
from pathlib import Path

from test_video_hybrid import (
    set_include_point,
    reset_inference,
    inference_single_frame,
    release_global_engine,
)


def process_stream(
    source: str,
    output_dir: str,
    *,
    onnx_dir: str = 'onnx_models',
    rknn_dir: str = None,
    encoder: str = 'rknn',
    decoder: str = 'onnx',
    memory_encoder: str = 'onnx',
    device: str = 'cpu',
    set_point_frame: int = 10,
    include_point = (966, 271),
    reset_frame: int = 240,
    save_visualization: bool = True,
    show_window: bool = True,
    window_name: str = "EdgeTAM Hybrid",
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(output_dir) / 'vis'
    if save_visualization:
        vis_dir.mkdir(parents=True, exist_ok=True)

    reset_inference()

    frame_idx = 0
    try:
        last_log = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 事件：设置 include point
            if frame_idx == set_point_frame:
                set_include_point(include_point)
                print(f"[Frame {frame_idx}] 设置 include point 为 {include_point}, 开启推理")

            # 事件：重置
            if frame_idx == reset_frame:
                reset_inference()
                print(f"[Frame {frame_idx}] 重置模型状态，关闭推理")

            mask, vis = inference_single_frame(
                frame_bgr,
                onnx_dir=onnx_dir,
                rknn_dir=rknn_dir,
                encoder=encoder,
                decoder=decoder,
                memory_encoder=memory_encoder,
                device=device,
            )

            # 仅保存可视化帧（mask 保留在内存即可）
            if save_visualization:
                out_name = f"{frame_idx:05d}.jpg"
                cv2.imwrite(str(vis_dir / out_name), vis)

            # 实时展示窗口
            if show_window:
                cv2.imshow(window_name, vis)
                # 1ms 等待，支持键盘退出（q/ESC）
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print("检测到退出按键，提前结束。")
                    break

            # 命令行提示：实时速度和帧计数
            if frame_idx - last_log >= 10:
                last_log = frame_idx
                print(f"[Frame {frame_idx}] 速度提示：已处理 {frame_idx+1} 帧")

            frame_idx += 1

    finally:
        cap.release()
        release_global_engine()
        if show_window:
            cv2.destroyAllWindows()

    print(f"处理完成，共 {frame_idx} 帧，输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="混合推理摄像头模拟测试")
    parser.add_argument('--source', type=str, default='examples/03_blocks.mp4',
                        help='视频文件路径或摄像头序号（如 0）')
    parser.add_argument('--output', type=str, default='output/run_hybrid_camera_sim',
                        help='输出目录根路径')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models')
    parser.add_argument('--rknn_dir', type=str, default=None)
    parser.add_argument('--encoder', type=str, default='rknn', choices=['rknn', 'onnx'])
    parser.add_argument('--decoder', type=str, default='onnx', choices=['rknn', 'onnx'])
    parser.add_argument('--memory_encoder', type=str, default='onnx', choices=['rknn', 'onnx'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--set_point_frame', type=int, default=10)
    parser.add_argument('--include_point', type=str, default='966,271')
    parser.add_argument('--reset_frame', type=int, default=240)
    parser.add_argument('--no_save_vis', action='store_true', help='不保存可视化帧')
    parser.add_argument('--no_show', action='store_true', help='不弹出窗口显示实时可视化')

    args = parser.parse_args()

    include_point = tuple(map(int, args.include_point.split(',')))

    video_name = os.path.splitext(os.path.basename(str(args.source)))[0]
    output_dir = os.path.join(args.output, video_name)

    process_stream(
        source=args.source,
        output_dir=output_dir,
        onnx_dir=args.onnx_dir,
        rknn_dir=args.rknn_dir,
        encoder=args.encoder,
        decoder=args.decoder,
        memory_encoder=args.memory_encoder,
        device=args.device,
        set_point_frame=args.set_point_frame,
        include_point=include_point,
        reset_frame=args.reset_frame,
        save_visualization=not args.no_save_vis,
        show_window=not args.no_show,
    )


if __name__ == '__main__':
    main()
