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
import time
import threading
from pathlib import Path
import numpy as np

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
    max_flush_per_iter: int = 5,
    async_overlay: bool = False,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    # 尝试限制底层缓冲，避免堆积（并非所有后端都支持）
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(output_dir) / 'vis'
    if save_visualization:
        vis_dir.mkdir(parents=True, exist_ok=True)

    reset_inference()

    frame_idx = 0
    processed_frames = 0

    # 异步推理共享状态
    inf_lock = threading.Lock()
    inference_running = False
    last_result = None  # {'mask': np.ndarray, 'iou': float, 'obj_score': float, 'timings': dict, 'point': tuple, 'frame_idx': int}

    def draw_mask_overlay(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(0, 153, 255)) -> np.ndarray:
        if mask is None:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
        else:
            mask_resized = mask.astype(np.float32)
        mask_bin = (mask_resized > 0.5).astype(np.float32)
        overlay = np.zeros_like(frame_bgr, dtype=np.float32)
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        out = frame_bgr.astype(np.float32)
        out = out * (1 - mask_bin[..., None] * alpha) + overlay * mask_bin[..., None] * alpha
        return out.astype(np.uint8)

    def infer_async(frame_bgr_copy: np.ndarray, cur_idx: int):
        nonlocal inference_running, last_result, processed_frames
        mask, vis = inference_single_frame(
            frame_bgr_copy,
            onnx_dir=onnx_dir,
            rknn_dir=rknn_dir,
            encoder=encoder,
            decoder=decoder,
            memory_encoder=memory_encoder,
            device=device,
        )
        with inf_lock:
            last_result = {
                'mask': mask,
                'frame_idx': cur_idx,
            }
            inference_running = False
            processed_frames += 1
    try:
        last_log = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            loop_start = time.perf_counter()

            # 事件：设置 include point
            if frame_idx == set_point_frame:
                set_include_point(include_point)
                print(f"[Frame {frame_idx}] 设置 include point 为 {include_point}, 开启推理")

            # 事件：重置
            if frame_idx == reset_frame:
                reset_inference()
                print(f"[Frame {frame_idx}] 重置模型状态，关闭推理")

            if async_overlay:
                # 异步模式：不阻塞采集。若当前无推理在跑，则启动一次；否则跳过，保持最新帧显示。
                with inf_lock:
                    running = inference_running
                if not running:
                    with inf_lock:
                        inference_running = True
                    threading.Thread(
                        target=infer_async,
                        args=(frame_bgr.copy(), frame_idx),
                        daemon=True,
                    ).start()

                # 显示时，将最新完成的 mask 覆盖到当前帧上
                with inf_lock:
                    lr = last_result
                vis = frame_bgr
                if lr is not None:
                    vis = draw_mask_overlay(frame_bgr, lr['mask'])

                # 仅保存可视化帧
                if save_visualization:
                    out_name = f"{frame_idx:05d}.jpg"
                    cv2.imwrite(str(vis_dir / out_name), vis)

                if show_window:
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        print("检测到退出按键，提前结束。")
                        break

                # 实时日志：以处理过的帧数为准
                if frame_idx - last_log >= 10:
                    last_log = frame_idx
                    elapsed_ms = (time.perf_counter() - loop_start) * 1000
                    rt_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
                    print(f"[Frame {frame_idx}] 异步模式：已完成 {processed_frames} 帧推理 | 采集间隔耗时 {elapsed_ms:.1f} ms (~{rt_fps:.1f} FPS)")

                # 异步模式不主动 flush，保持平滑显示

            else:
                # 同步/丢帧模式：立即推理并可选丢弃缓冲
                mask, vis = inference_single_frame(
                    frame_bgr,
                    onnx_dir=onnx_dir,
                    rknn_dir=rknn_dir,
                    encoder=encoder,
                    decoder=decoder,
                    memory_encoder=memory_encoder,
                    device=device,
                )

                processed_frames += 1

                if save_visualization:
                    out_name = f"{frame_idx:05d}.jpg"
                    cv2.imwrite(str(vis_dir / out_name), vis)

                if show_window:
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        print("检测到退出按键，提前结束。")
                        break

                if frame_idx - last_log >= 10:
                    last_log = frame_idx
                    elapsed_ms = (time.perf_counter() - loop_start) * 1000
                    rt_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
                    print(f"[Frame {frame_idx}] 已处理 {processed_frames} 帧 | 当前推理耗时 {elapsed_ms:.1f} ms (~{rt_fps:.1f} FPS)")

                # 若模型较慢，主动丢弃缓冲中的旧帧，始终处理最新帧，避免堆积
                flushed = 0
                while flushed < max_flush_per_iter:
                    ok_flush = cap.grab()
                    if not ok_flush:
                        break
                    flushed += 1
                if flushed > 0:
                    pass

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
    parser.add_argument('--max_flush_per_iter', type=int, default=5,
                        help='每次推理后最多丢弃的旧帧数量，用于防止摄像头缓冲堆积')
    parser.add_argument('--async_overlay', action='store_true',
                        help='启用异步推理+叠加：采集不阻塞，推理完成后将上一帧mask叠加到当前帧')

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
        max_flush_per_iter=args.max_flush_per_iter,
        async_overlay=args.async_overlay,
    )


if __name__ == '__main__':
    main()
