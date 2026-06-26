import cv2
import os
import json
import csv
import time
import threading
import queue
import serial
from typing import List, Tuple, Dict, Any
import natsort
import numpy as np
from datetime import datetime
import random
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ========== 串口配置 ==========
ser = serial.Serial("COM3", 115200, timeout=0)
time.sleep(2)

def send_command_async(cmd):
    """异步发送指令，不等待反馈，直接写进串口缓冲区"""
    if ser is not None and ser.is_open:
        try:
            ser.write((cmd + "\n").encode("utf-8"))
        except:
            pass


# ========== 2. 串口线程 (重构) ==========
def run_lanes(lane_queues, sleep_time=0.01):
    """
    重构后的串口线程：
    - 不再使用 readline() 阻塞等待。
    - 只要队列有任务，立即拼接成 3 位指令发给 Arduino 队列。
    """
    print("串口分发线程启动...")
    while True:
        cmd = ""
        task_found = False

        for i in range(3):
            if not lane_queues[i].empty():
                task = lane_queues[i].get()
                if task is None:  # 收到 None 信号结束线程
                    print(f"Lane {i} 队列监听结束")
                    return
                cmd += str(task)
                task_found = True
            else:
                cmd += "0"  # 占位，表示该通道暂无新动作

        if task_found:
            send_command_async(cmd)

        time.sleep(sleep_time)


def image_saver_loop(save_queue, stop_event, save_root, save_interval=1,
                     fps_stats=None):
    """
    Save only the 5-channel detection npy data in a background thread.

    save_queue item format:
        {
            "block_id": ...,
            "timestamp": ...,
            "detection_frame": H x W x 5
        }
    """
    if save_interval < 1:
        raise ValueError("save_interval must be >= 1")

    detection_dir = os.path.join(save_root, "detection_5ch_npy")
    os.makedirs(detection_dir, exist_ok=True)

    frame_index = 0
    saved_count = 0
    fps_window_start = time.perf_counter()
    fps_window_count = 0

    while not stop_event.is_set() or not save_queue.empty():
        try:
            save_item = save_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if save_item is None:
                continue

            should_save = frame_index % save_interval == 0
            frame_index += 1
            if not should_save:
                continue

            block_id = save_item.get("block_id", "unknown")
            timestamp = save_item.get(
                "timestamp",
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            if isinstance(block_id, (int, np.integer)):
                safe_block_id = f"{int(block_id):08d}"
            else:
                safe_block_id = str(block_id).replace(os.sep, "_")
            if os.altsep:
                safe_block_id = safe_block_id.replace(os.altsep, "_")
            file_stem = f"block_{safe_block_id}_{timestamp}"

            detection_frame = save_item.get("detection_frame")
            if detection_frame is not None:
                np.save(
                    os.path.join(detection_dir, f"{file_stem}_5ch.npy"),
                    detection_frame,
                    allow_pickle=False
                )
                saved_count += 1
                fps_window_count += 1

            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                saving_fps = fps_window_count / elapsed
                if fps_stats is not None:
                    with fps_stats["lock"]:
                        fps_stats["saving_fps"] = saving_fps
                fps_window_start = now
                fps_window_count = 0
        except Exception as exc:
            print(f"Warning: npy save failed: {exc}")
        finally:
            save_queue.task_done()

    print(f"Detection data save thread ended, saved {saved_count} npy files")


def enqueue_save_task(save_queue, block_id, detection_frame,
                      display_frame=None, visualized_frame=None):
    """Submit a non-blocking npy save task; drop it if disk saving falls behind."""
    if save_queue is None:
        return

    save_item = {
        "block_id": block_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "detection_frame": detection_frame
    }

    try:
        save_queue.put_nowait(save_item)
    except queue.Full:
        pass

# def determine_size(img, contour):
#     # offset = 0.2405 * 4 * 1.1  # in/px
#     offset = 0.769 * 2  # 0.769mm/px * 2  (0.769是根据校正得到的像素尺寸，乘以2是因为送入模型的尺寸只有图片实际尺寸的一半)
#     ellipse = cv2.fitEllipse(contour)
#     center, axes, angle = ellipse
#     long_axis = max(axes)
#     short_axis = min(axes)
#     # Compute endpoints of long axis
#     long_axis_angle_rad = np.deg2rad(angle)
#     long_axis_length = long_axis / 2
#     sin_angle = np.sin(long_axis_angle_rad)
#     cos_angle = np.cos(long_axis_angle_rad)
#     pt1 = (int(center[0] - long_axis_length * sin_angle), int(center[1] + long_axis_length * cos_angle))
#     pt2 = (int(center[0] + long_axis_length * sin_angle), int(center[1] - long_axis_length * cos_angle))

#     # Compute endpoints of short axis
#     short_axis_angle_rad = np.deg2rad(angle + 90)
#     short_axis_length = short_axis / 2
#     sin_angle = np.sin(short_axis_angle_rad)
#     cos_angle = np.cos(short_axis_angle_rad)
#     pt3 = (int(center[0] - short_axis_length * sin_angle), int(center[1] + short_axis_length * cos_angle))
#     pt4 = (int(center[0] + short_axis_length * sin_angle), int(center[1] - short_axis_length * cos_angle))

#     # Draw the long and short axes
#     cv2.line(img, pt1, pt2, (0, 255, 0), 2)
#     cv2.line(img, pt3, pt4, (0, 255, 0), 2)

#     # Display the lengths of the long and short axes
#     Size_1 = long_axis_length * 2 * offset
#     Size_2 = short_axis_length * 2 * offset

#     return Size_1, Size_2, center, long_axis_length
def determine_size(img, contour):
    """
    使用面积等效直径法计算苹果尺寸。

    返回:
        equivalent_diameter_mm: 面积等效直径，最终用于分级
        long_axis_mm: 椭圆长轴，仅用于 debug / 可视化
        short_axis_mm: 椭圆短轴，仅用于 debug / 可视化
        center: 椭圆中心
        equivalent_radius_px: 等效圆半径，像素单位，仅用于可视化
    """

    # 如果你现在测出来尺寸接近真实值，就保持 0.769
    # 如果后面确认 contour 是 1024x768 尺度，而 0.769 是 2048x1536 原图标定值，再改成 0.769 * 2
    offset = 0.769  # mm / pixel

    if contour is None or len(contour) < 5:
        raise ValueError("Contour points are fewer than 5, cannot fit ellipse.")

    # =====================================================
    # 1. 面积等效直径
    # =====================================================
    area_px = cv2.contourArea(contour)

    if area_px <= 0:
        raise ValueError("Contour area is zero or negative.")

    equivalent_diameter_px = np.sqrt(4.0 * area_px / np.pi)
    equivalent_radius_px = equivalent_diameter_px / 2.0
    equivalent_diameter_mm = equivalent_diameter_px * offset

    # =====================================================
    # 2. 椭圆拟合：只作为 debug / 可视化，不作为最终直径
    # =====================================================
    ellipse = cv2.fitEllipse(contour)
    center, axes, angle = ellipse

    long_axis_px = max(axes)
    short_axis_px = min(axes)

    long_axis_mm = long_axis_px * offset
    short_axis_mm = short_axis_px * offset

    # =====================================================
    # 3. 可视化
    # =====================================================
    # cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    cx, cy = int(center[0]), int(center[1])

    # 画等效圆
    cv2.circle(
        img,
        (cx, cy),
        int(equivalent_radius_px),
        (0, 255, 255),
        2
    )

    # 显示等效直径
    cv2.putText(
        img,
        f"EqD: {equivalent_diameter_mm:.1f} mm",
        (cx - 70, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

    return (
        equivalent_diameter_mm,
        long_axis_mm,
        short_axis_mm,
        center,
        equivalent_radius_px
    )

# def final_grading(class_list, confidence_list, average_diameter):
#     # log_info = ""

#     if len(class_list) == 0 or len(confidence_list) == 0:
#         return 0, f"Empty list → class 0\n"

#     if average_diameter < 50:
#         return 2, f"Diameter {average_diameter} < 50 → Class 2\n"

#     # 按类别提取 confidence
#     prob_for_cull  = [d for c, d in zip(class_list, confidence_list) if c == 2]
#     prob_for_processing  = [d for c, d in zip(class_list, confidence_list) if c == 1]
#     prob_for_class_1 = [d for c, d in zip(class_list, confidence_list) if c == 0]

#     if 50 <= average_diameter <= 63:
#         if len(prob_for_cull ) <= 5 and np.mean(prob_for_cull ) >= 0.987:
#             return 2, f"High confidence class3, lower than 5 times for Class3. mean confidence={np.mean(prob_for_cull ):.2f}\n"
#         if 4 < len(prob_for_cull ) <= 24 and np.mean(prob_for_cull ) >= 0.981:
#             return 2, f"High confidence class3, lower than 24 times for Class3. mean confidence={np.mean(prob_for_cull ):.2f}\n"
#         if len(prob_for_cull ) > 24 and np.mean(prob_for_cull ) > 0.971:
#             return 2, f"High confidence class3, more than 24 times for Class3 (Size smaller than 65). mean confidence={np.mean(prob_for_cull ):.2f}\n"
#         else:
#             return 1, "Diameter in class 2 range\n"

#     if len(prob_for_cull ) <= 5 and np.mean(prob_for_cull ) >= 0.987:
#         return 2, f"High confidence class3, lower than 5 times for Class3. mean confidence={np.mean(prob_for_cull ):.2f}\n"
#     if 5 < len(prob_for_cull ) <= 24 and np.mean(prob_for_cull ) >= 0.981:
#         return 2, f"High confidence class3, lower than 24 times for Class3. mean confidence={np.mean(prob_for_cull ):.2f}\n"
#     if len(prob_for_cull ) > 24 and np.mean(prob_for_cull ) > 0.971:
#         return 2, f"High confidence class3, more than 24 times for Class3. mean confidence={np.mean(prob_for_cull ):.2f}\n"
#     if len(prob_for_processing) <= 7 and np.mean(prob_for_processing) >= 0.916:
#         return 1, f"High confidence class2, lower than 7 times for Class2. mean confidence={np.mean(prob_for_processing):.2f}\n"
#     if 7 < len(prob_for_processing) <= 21 and np.mean(prob_for_processing) >= 0.892:
#         return 1, f"High confidence class2, lower than 21 times for Class2. mean confidence={np.mean(prob_for_processing):.2f}\n"
#     if len(prob_for_processing) > 21 and np.mean(prob_for_processing) > 0.730:
#         return 1, f"High confidence class2, more than 21 times for Class2. mean confidence={np.mean(prob_for_processing):.2f}\n"

#     # 加权平均
#     class_weights = {0: 1.0, 1: 2.726, 2: 8.177}
#     weighted_sum = {0: 0.0, 1: 0.0, 2: 0.0}
#     total_confidence = 0.0

#     for cls, conf in zip(class_list, confidence_list):
#         weighted_conf = conf * class_weights[cls]
#         weighted_sum[cls] += weighted_conf
#         total_confidence += weighted_conf

#     weighted_avg = {cls: weighted_sum[cls] / total_confidence for cls in [0, 1, 2]}
#     final_class = max(weighted_avg, key=weighted_avg.get)

#     return final_class, f"Weighted avg: {weighted_avg}\n"


import numpy as np


def safe_mean(values, default=0.0):
    """
    安全计算平均值，避免 np.mean([]) warning。
    """
    if values is None or len(values) == 0:
        return default
    return float(np.mean(values))


def topk_mean(values, k=3, default=0.0):
    """
    计算 top-k confidence 的平均值。
    比普通 mean 更适合检测“少量但高置信度 cull 证据”。
    """
    if values is None or len(values) == 0:
        return default

    values = sorted(values, reverse=True)
    k = min(k, len(values))
    return float(np.mean(values[:k]))


def analyze_class_stats(class_list, confidence_list):
    """
    统计每个类别的 count, ratio, mean_conf, top1, top2_mean, top3_mean。
    
    class 0: fresh
    class 1: processing
    class 2: cull
    """
    total = len(class_list)

    conf_by_class = {
        0: [],
        1: [],
        2: [],
    }

    for cls, conf in zip(class_list, confidence_list):
        if cls in conf_by_class:
            conf_by_class[cls].append(float(conf))

    count = {
        cls: len(conf_by_class[cls])
        for cls in [0, 1, 2]
    }

    ratio = {
        cls: count[cls] / total if total > 0 else 0.0
        for cls in [0, 1, 2]
    }

    mean_conf = {
        cls: safe_mean(conf_by_class[cls])
        for cls in [0, 1, 2]
    }

    top1 = {
        cls: max(conf_by_class[cls]) if len(conf_by_class[cls]) > 0 else 0.0
        for cls in [0, 1, 2]
    }

    top2_mean = {
        cls: topk_mean(conf_by_class[cls], k=2)
        for cls in [0, 1, 2]
    }

    top3_mean = {
        cls: topk_mean(conf_by_class[cls], k=3)
        for cls in [0, 1, 2]
    }

    return {
        "total": total,
        "conf_by_class": conf_by_class,
        "count": count,
        "ratio": ratio,
        "mean_conf": mean_conf,
        "top1": top1,
        "top2_mean": top2_mean,
        "top3_mean": top3_mean,
    }


def safe_mean(values, default=0.0):
    """
    安全计算平均值，避免 np.mean([]) warning。
    """
    if values is None or len(values) == 0:
        return default
    return float(np.mean(values))


def topk_mean(values, k=3, default=0.0):
    """
    计算 top-k confidence 的平均值。

    用途：
    - 对 cull 来说，少量高置信度帧可能代表某一个表面严重腐烂；
    - 因此 top-k 比普通 mean 更适合保留“局部严重缺陷”的证据。
    """
    if values is None or len(values) == 0:
        return default

    values = sorted(values, reverse=True)
    k = min(k, len(values))
    return float(np.mean(values[:k]))


def analyze_class_stats(class_list, confidence_list):
    """
    统计每个类别的 count, ratio, mean_conf, top1, top2_mean, top3_mean。

    class 0: fresh
    class 1: processing
    class 2: cull
    """
    total = len(class_list)

    conf_by_class = {
        0: [],
        1: [],
        2: [],
    }

    for cls, conf in zip(class_list, confidence_list):
        cls = int(cls)
        conf = float(conf)

        if cls in conf_by_class:
            conf_by_class[cls].append(conf)

    count = {
        cls: len(conf_by_class[cls])
        for cls in [0, 1, 2]
    }

    ratio = {
        cls: count[cls] / total if total > 0 else 0.0
        for cls in [0, 1, 2]
    }

    mean_conf = {
        cls: safe_mean(conf_by_class[cls])
        for cls in [0, 1, 2]
    }

    top1 = {
        cls: max(conf_by_class[cls]) if len(conf_by_class[cls]) > 0 else 0.0
        for cls in [0, 1, 2]
    }

    top2_mean = {
        cls: topk_mean(conf_by_class[cls], k=2)
        for cls in [0, 1, 2]
    }

    top3_mean = {
        cls: topk_mean(conf_by_class[cls], k=3)
        for cls in [0, 1, 2]
    }

    return {
        "total": total,
        "conf_by_class": conf_by_class,
        "count": count,
        "ratio": ratio,
        "mean_conf": mean_conf,
        "top1": top1,
        "top2_mean": top2_mean,
        "top3_mean": top3_mean,
    }


def final_grading(
    all_class_list,
    all_confidence_list,
    average_diameter,
    stable_class_list=None,
    stable_confidence_list=None,
    debug=True,
):
    """
    Final grading algorithm v4.

    class 0: fresh
    class 1: processing
    class 2: cull

    设计目标：
    1. 保留 cull-first，因为单个表面严重腐烂也必须判为 cull；
    2. 放松 cull top-K 阈值，避免强 cull evidence 被压成 processing/fresh；
    3. corrected_diameter = median_diameter - 2.3 mm；
    4. corrected_diameter < 50 mm 仍然作为 hard cull rule；
    5. 50–65 mm 不直接 hard return processing，而是给 processing_score 加尺寸 bonus；
    6. 65–67 mm 内 processing bonus 线性衰减；
    7. 放宽 processing 判断，减少 processing → fresh；
    8. 收紧 fresh 判断，fresh 必须有足够 clean dominant 证据；
    9. fallback 前增加 defect fallback：有缺陷证据但不够 cull，则判 processing。
    """

    log_info = ""

    # ==========================================================
    # 0. 输入检查
    # ==========================================================
    if all_class_list is None or all_confidence_list is None:
        return 0, "Input is None → fresh\n"

    if len(all_class_list) == 0 or len(all_confidence_list) == 0:
        return 0, "Empty all_class_list/all_confidence_list → fresh\n"

    if len(all_class_list) != len(all_confidence_list):
        n = min(len(all_class_list), len(all_confidence_list))
        log_info += (
            f"[WARNING] all_class_list and all_confidence_list length mismatch. "
            f"Use first {n} records.\n"
        )
        all_class_list = all_class_list[:n]
        all_confidence_list = all_confidence_list[:n]

    all_class_list = [int(x) for x in all_class_list]
    all_confidence_list = [float(x) for x in all_confidence_list]

    # ==========================================================
    # 1. 参数设置
    # ==========================================================

    # ---------- 尺寸修正 ----------
    DIAMETER_BIAS_MM = 2.3

    CULL_DIAMETER_TH = 50.0
    PROCESSING_LOW = 50.0
    PROCESSING_HIGH = 65.0
    SIZE_MARGIN_MM = 2.0

    # 尺寸对 processing 的最大加权
    PROCESSING_SIZE_BONUS_MAX = 0.35

    # ---------- cull top-K 阈值 ----------
    # v4 放松 cull 触发，避免 cull 被压成 processing/fresh
    CULL_TOP1_TH = 0.985
    CULL_TOP2_MEAN_TH = 0.975
    CULL_TOP3_MEAN_TH = 0.960

    # top-K 触发时的最低比例约束
    # 防止只有极少数异常帧导致误判 cull
    CULL_TOP1_RATIO_MIN = 0.08
    CULL_TOP2_RATIO_MIN = 0.10
    CULL_TOP3_RATIO_MIN = 0.20
    CULL_TOP3_MEAN_CONF_MIN = 0.88

    # ---------- sustained cull 阈值 ----------
    SUSTAINED_CULL_RATIO_TH = 0.35
    SUSTAINED_CULL_MEAN_TH = 0.86

    # ---------- processing 阈值 ----------
    PROCESSING_CLOSE_MARGIN = 0.1
    PROCESSING_RELAXED_RATIO_TH = 0.30
    PROCESSING_RELAXED_MEAN_TH = 0.80

    # ---------- fresh 阈值 ----------
    FRESH_SCORE_MARGIN = 0.20

    # ==========================================================
    # 2. 尺寸修正
    # ==========================================================
    raw_diameter = float(average_diameter)
    corrected_diameter = raw_diameter - DIAMETER_BIAS_MM

    log_info += f"Raw diameter={raw_diameter:.2f} mm\n"
    log_info += f"Corrected diameter={corrected_diameter:.2f} mm\n"

    # ==========================================================
    # 3. 尺寸 hard cull rule
    # ==========================================================
    if corrected_diameter < CULL_DIAMETER_TH:
        return 2, log_info + "Corrected diameter < 50 mm → cull\n"

    # ==========================================================
    # 4. 统计所有帧
    # ==========================================================
    all_stats = analyze_class_stats(all_class_list, all_confidence_list)

    log_info += "\n[All frames statistics]\n"
    log_info += f"total={all_stats['total']}\n"
    log_info += f"count={all_stats['count']}\n"
    log_info += f"ratio={all_stats['ratio']}\n"
    log_info += f"mean_conf={all_stats['mean_conf']}\n"
    log_info += f"top1={all_stats['top1']}\n"
    log_info += f"top2_mean={all_stats['top2_mean']}\n"
    log_info += f"top3_mean={all_stats['top3_mean']}\n"

    fresh_count = all_stats["count"][0]
    processing_count = all_stats["count"][1]
    cull_count = all_stats["count"][2]

    fresh_ratio = all_stats["ratio"][0]
    processing_ratio = all_stats["ratio"][1]
    cull_ratio = all_stats["ratio"][2]

    fresh_mean = all_stats["mean_conf"][0]
    processing_mean = all_stats["mean_conf"][1]
    cull_mean = all_stats["mean_conf"][2]

    fresh_top3_mean = all_stats["top3_mean"][0]
    processing_top3_mean = all_stats["top3_mean"][1]

    cull_top1 = all_stats["top1"][2]
    cull_top2_mean = all_stats["top2_mean"][2]
    cull_top3_mean = all_stats["top3_mean"][2]

    # ==========================================================
    # 5. Cull-first top-K 规则
    # ==========================================================
    # 这部分保留你的设计初衷：
    # 如果某一个表面严重腐烂，即使只在少数帧中出现，
    # 只要 confidence 足够高，仍然应该判为 cull。

    if (
        cull_count >= 1
        and cull_top1 >= CULL_TOP1_TH
        and cull_ratio >= CULL_TOP1_RATIO_MIN
    ):
        return 2, log_info + (
            f"\nDecision: cull_top1 >= {CULL_TOP1_TH:.3f} "
            f"and cull_ratio >= {CULL_TOP1_RATIO_MIN:.2f} → cull\n"
        )

    if (
        cull_count >= 2
        and cull_top2_mean >= CULL_TOP2_MEAN_TH
        and cull_ratio >= CULL_TOP2_RATIO_MIN
    ):
        return 2, log_info + (
            f"\nDecision: cull_top2_mean >= {CULL_TOP2_MEAN_TH:.3f} "
            f"and cull_ratio >= {CULL_TOP2_RATIO_MIN:.2f} → cull\n"
        )

    if (
        cull_count >= 3
        and cull_top3_mean >= CULL_TOP3_MEAN_TH
        and cull_ratio >= CULL_TOP3_RATIO_MIN
        and cull_mean >= CULL_TOP3_MEAN_CONF_MIN
    ):
        return 2, log_info + (
            f"\nDecision: cull_top3_mean >= {CULL_TOP3_MEAN_TH:.3f}, "
            f"cull_ratio >= {CULL_TOP3_RATIO_MIN:.2f}, "
            f"cull_mean >= {CULL_TOP3_MEAN_CONF_MIN:.2f} → cull\n"
        )

    # sustained cull evidence:
    # 如果 cull 不是极端高置信度，但在较多帧里持续出现，也判 cull。
    if (
        cull_count >= 3
        and cull_ratio >= SUSTAINED_CULL_RATIO_TH
        and cull_mean >= SUSTAINED_CULL_MEAN_TH
    ):
        return 2, log_info + (
            "\nDecision: sustained cull evidence "
            f"ratio >= {SUSTAINED_CULL_RATIO_TH:.2f}, "
            f"mean >= {SUSTAINED_CULL_MEAN_TH:.2f} → cull\n"
        )
    total_frames = all_stats["total"]
    localized_cull_min_count = max(4, int(np.ceil(0.08 * total_frames)))

    if (
        cull_count >= localized_cull_min_count
        and cull_top3_mean >= 0.94
        and cull_mean >= 0.80
    ):
        return 2, log_info + (
            "\nDecision: localized cull evidence "
            f"cull_count >= {localized_cull_min_count}, "
            "top3 >= 0.94, mean >= 0.80 → cull\n"
        )
    if (
        processing_ratio >= 0.90
        and processing_mean >= 0.88
        and fresh_ratio <= 0.05
        and corrected_diameter >= PROCESSING_HIGH
    ):
        return 2, log_info + (
            "\nDecision: heavy processing-like defect outside processing size range "
            "→ cull\n"
        )

    # ==========================================================
    # 6. 基础 score
    # ==========================================================
    fresh_score_base = fresh_ratio * fresh_mean
    processing_score_base = processing_ratio * processing_mean
    cull_score_base = cull_ratio * cull_mean

    fresh_score = fresh_score_base
    processing_score = processing_score_base
    cull_score = cull_score_base

    # ==========================================================
    # 7. 尺寸先验：给 processing_score 加 bonus
    # ==========================================================
    size_processing_bonus = 0.0

    if PROCESSING_LOW <= corrected_diameter <= PROCESSING_HIGH:
        size_processing_bonus = PROCESSING_SIZE_BONUS_MAX

    elif PROCESSING_HIGH < corrected_diameter <= PROCESSING_HIGH + SIZE_MARGIN_MM:
        distance = corrected_diameter - PROCESSING_HIGH
        decay_ratio = 1.0 - distance / SIZE_MARGIN_MM
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        size_processing_bonus = PROCESSING_SIZE_BONUS_MAX * decay_ratio

    processing_score += size_processing_bonus

    log_info += "\n[Scores]\n"
    log_info += f"fresh_score_base={fresh_score_base:.4f}\n"
    log_info += f"processing_score_base={processing_score_base:.4f}\n"
    log_info += f"cull_score_base={cull_score_base:.4f}\n"
    log_info += f"size_processing_bonus={size_processing_bonus:.4f}\n"
    log_info += f"fresh_score={fresh_score:.4f}\n"
    log_info += f"processing_score={processing_score:.4f}\n"
    log_info += f"cull_score={cull_score:.4f}\n"

    # ==========================================================
    # 8. Processing 判断：v4 放宽
    # ==========================================================

    # 8.1 processing score 接近或超过 fresh
    # v4 放宽 close margin，减少 processing → fresh
    if (
        processing_ratio >= PROCESSING_RELAXED_RATIO_TH
        and processing_mean >= PROCESSING_RELAXED_MEAN_TH
        and processing_score >= fresh_score - PROCESSING_CLOSE_MARGIN
    ):
        return 1, log_info + (
            "\nDecision: relaxed processing score close to fresh "
            "→ processing\n"
        )

    # 8.2 processing ratio 不低，confidence 可靠
    if (
        processing_ratio >= 0.30
        and processing_mean >= 0.78
    ):
        return 1, log_info + (
            "\nDecision: relaxed processing ratio and confidence "
            "→ processing\n"
        )

    # 8.3 processing 占多数
    if (
        processing_ratio >= 0.45
        and processing_mean >= 0.72
    ):
        return 1, log_info + (
            "\nDecision: processing majority with acceptable confidence "
            "→ processing\n"
        )

    # 8.4 如果 processing top3 很强，即使 ratio 稍低，也不直接 fresh
    if (
        processing_count >= 3
        and processing_ratio >= 0.22
        and processing_top3_mean >= 0.90
        and processing_mean >= 0.76
    ):
        return 1, log_info + (
            "\nDecision: strong processing top3 evidence "
            "→ processing\n"
        )

    # ==========================================================
    # 9. Defect fallback
    # ==========================================================
    # 如果 cull 没有强到触发 cull，但 processing + cull 总缺陷证据不弱，
    # 则不应该回到 fresh，而是判 processing。
    if (
        processing_ratio + cull_ratio >= 0.38
        and fresh_ratio <= 0.62
        and max(processing_mean, cull_mean) >= 0.82
    ):
        return 1, log_info + (
            "\nDecision: defect evidence fallback "
            "→ processing\n"
        )

    # 更保守一点的 defect fallback：
    # cull ratio 不低但 confidence 不够 cull，则至少判 processing。
    if (
        cull_ratio >= 0.15
        and cull_mean >= 0.80
        and fresh_ratio <= 0.75
    ):
        return 1, log_info + (
            "\nDecision: weak cull evidence fallback "
            "→ processing\n"
        )

    # ==========================================================
    # 10. Fresh 判断：v4 收紧
    # ==========================================================
    # fresh 不能作为默认结果，必须有足够 clean evidence。

    if (
        fresh_ratio >= 0.78
        and processing_ratio <= 0.20
        and cull_ratio < 0.05
    ):
        return 0, log_info + (
            "\nDecision: strongly clean fresh dominant → fresh\n"
        )

    if (
        fresh_ratio >= 0.68
        and processing_ratio <= 0.25
        and cull_ratio < 0.05
        and fresh_score >= processing_score + FRESH_SCORE_MARGIN
    ):
        return 0, log_info + (
            f"\nDecision: fresh dominant with score margin "
            f">= {FRESH_SCORE_MARGIN:.2f} → fresh\n"
        )

    # ==========================================================
    # 11. Final fallback
    # ==========================================================
    # v4 中 fallback 不再无脑 fresh。
    # 如果存在一定缺陷证据，则 fallback processing；
    # 只有缺陷证据很弱时才 fallback fresh。

    if (
        processing_ratio + cull_ratio >= 0.25
        and max(processing_mean, cull_mean) >= 0.75
    ):
        return 1, log_info + (
            "\nDecision: final fallback with defect evidence "
            "→ processing\n"
        )

    return 0, log_info + "\nDecision: final fallback with weak defect evidence → fresh\n"


def draw_results(img, draw_tasks, colors, class_names, alpha=0.4):
    """
    优化绘制函数：一次性叠加所有掩膜，减少循环次数。
    显示内容：掩膜、边框、ID、置信度、尺寸信息。
    """
    if not draw_tasks:
        return img

    # t_start = time.perf_counter()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # h, w = img.shape[:2]
    # overlay_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # === Step 1: 一次性生成所有掩膜颜色层 ===
    # for t in draw_tasks:
    #     seg = t["seg"]
    #     color = colors[t["cls"]]
    #     overlay_mask[seg > 0] = np.maximum(
    #         overlay_mask[seg > 0],
    #         np.array(color, dtype=np.uint8)
    #     )
    #
    # # === Step 2: 一次性混合掩膜 ===
    # img = cv2.addWeighted(img, 1 - alpha, overlay_mask, alpha, 0)

    # === Step 3: 绘制外框、类别、ID、置信度 ===
    for t in draw_tasks:
        color = colors[t["cls"]]
        xmin, ymin, xmax, ymax = map(int, t["bbox"])
        label = f'ID {t["obj_id"]} | {class_names[t["cls"]]} {t["conf"]:.2f}'

        # 外框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2, lineType=cv2.LINE_AA)

        # 标签底色 + 文字
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img, (xmin, ymin - th - 4), (xmin + tw + 2, ymin), color, -1)
        cv2.putText(img, label, (xmin, ymin - 5), font, font_scale,
                    (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # === Step 4: 绘制尺寸信息 ===
    for t in draw_tasks:
        text = t["text"]
        center = t["center"]
        long_axis = t["long_axis"]
        text_pos = (int(center[0] - 10), int(center[1] + long_axis + 20))

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        bg_tl = (text_pos[0] - 2, text_pos[1] - th - 2)
        bg_br = (text_pos[0] + tw + 2, text_pos[1] + 2)
        cv2.rectangle(img, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.putText(img, text, text_pos, font, font_scale,
                    (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # t_end = time.perf_counter()
    # print(f"  绘图阶段耗时: {(t_end - t_start) * 1000:.2f} ms")

    return img

def create_static_overlay(w, h, Start_line, Grading_line, Line_1, Line_2):
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # 画线
    cv2.line(overlay, (Grading_line, 0), (Grading_line, h), (0, 100, 255), 2)
    cv2.line(overlay, (Start_line, 0), (Start_line, h), (0, 100, 255), 2)

    # 文本
    cv2.putText(overlay, "Start", (Start_line, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, "Grade", (Grading_line, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, "Lane one", (Start_line + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(overlay, "Lane two", (Start_line + 50, Line_1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(overlay, "Lane three", (Start_line + 50, Line_2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return overlay

# tracking relative
def compute_iou_matrix(boxesA, boxesB):
    """计算 IOU 矩阵"""
    iou_matrix = np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)
    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0:
                iou = 0.0
            else:
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)
            iou_matrix[i, j] = iou
    return iou_matrix
class TrackedObject:
    def __init__(self, object_id, bbox, center, prev_center=None, dt=1.0):
        self.id = object_id
        self.bbox = bbox
        self.center = center
        self.missing_frames = 0
        self.age = 0
        self.confirmed = False
        self.active = True  # 是否仍在画面中
        self.trajectory = [center]

        # === Kalman Filter ===
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05

        if prev_center is not None:
            vx = (center[0] - prev_center[0]) / dt
            vy = (center[1] - prev_center[1]) / dt
        else:
            vx, vy = 0, 0
        self.kf.statePost = np.array([center[0], center[1], vx, vy], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return (int(pred[0]), int(pred[1]))

    def update(self, center, max_speed=100):
        pred = self.predict()
        dist = np.linalg.norm(np.array(center) - np.array(pred))
        if dist > max_speed:
            center = (0.7 * np.array(center) + 0.3 * np.array(pred)).tolist()

        measurement = np.array(center, dtype=np.float32)
        self.kf.correct(measurement)

        self.center = center
        self.missing_frames = 0
        self.age += 1
        if self.age >= 3:
            self.confirmed = True

        smooth_center = (
            int(0.6 * self.center[0] + 0.4 * pred[0]),
            int(0.6 * self.center[1] + 0.4 * pred[1])
        )
        self.trajectory.append(smooth_center)


# ================= 多目标管理类 =================
class TrackerManager:
    def __init__(self, img_width, max_missing=8, iou_threshold=0.2):
        self.trackers = []
        self.next_id = 0
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.img_width = img_width

    def update(self, detections):
        """
        更新 tracker，并返回本帧 tracker 与 detection 的严格匹配关系。

        Args:
            detections: List[[x1, y1, x2, y2]]，只包含 bbox，不包含 conf/cls。

        Returns:
            matched_objects: List[(TrackedObject, det_idx)]
                每个元素表示：这个 TrackedObject 在当前帧匹配到了 detections[det_idx]。
                后续必须用 det_idx 去取 detections_full[det_idx] 和 masks_bin[det_idx]，
                不要再用 zip(self.trackers, detections, masks_bin)。
        """
        matched_objects = []

        if len(detections) == 0:
            for t in self.trackers:
                t.missing_frames += 1
            self.trackers = [t for t in self.trackers if t.missing_frames < self.max_missing and t.active]
            return matched_objects

        detection_centers = [
            ((x1 + x2) / 2, (y1 + y2) / 2)
            for (x1, y1, x2, y2) in detections
        ]

        # === 没有历史追踪对象：新建 ===
        if len(self.trackers) == 0:
            for j, c in enumerate(detection_centers):
                # 仅从左端出现的才建立追踪
                if c[0] < self.img_width * 0.25:
                    obj = TrackedObject(self.next_id, detections[j], c)
                    self.trackers.append(obj)
                    matched_objects.append((obj, j))
                    self.next_id += 1
            return matched_objects

        # === 构建代价矩阵（IOU + 中心点距离）===
        cost_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for i, tracker in enumerate(self.trackers):
            pred_center = tracker.predict()
            for j, det in enumerate(detections):
                det_center = detection_centers[j]
                iou = compute_iou_matrix([tracker.bbox], [det])[0, 0]
                dist = np.linalg.norm(np.array(pred_center) - np.array(det_center))
                dist_norm = np.clip(dist / 150, 0, 1)
                combined_score = 0.60 * iou + 0.40 * (1 - dist_norm)
                cost_matrix[i, j] = 1 - combined_score

        matched_row, matched_col = linear_sum_assignment(cost_matrix)
        matched_detection_indices = set()
        matched_tracker_indices = set()

        for i, j in zip(matched_row, matched_col):
            if cost_matrix[i, j] < (1 - self.iou_threshold):
                self.trackers[i].update(detection_centers[j])
                self.trackers[i].bbox = detections[j]
                matched_objects.append((self.trackers[i], j))
                matched_detection_indices.add(j)
                matched_tracker_indices.add(i)
            else:
                self.trackers[i].missing_frames += 1

        # === 未匹配追踪对象 ===
        unmatched_trackers = set(range(len(self.trackers))) - matched_tracker_indices
        for i in unmatched_trackers:
            self.trackers[i].missing_frames += 1

        # === 新目标仅从左边进入 ===
        unmatched_detections = set(range(len(detections))) - matched_detection_indices
        for j in unmatched_detections:
            c = detection_centers[j]
            if c[0] < self.img_width * 0.25:  # 左端进入
                obj = TrackedObject(self.next_id, detections[j], c)
                self.trackers.append(obj)
                matched_objects.append((obj, j))
                self.next_id += 1

        # === 标记右端消失的目标 ===
        for t in self.trackers:
            if t.center[0] > self.img_width * 0.99:
                t.active = False

        # 清理失效 tracker。matched_objects 保存的是对象引用，不受列表重排影响。
        self.trackers = [
            t for t in self.trackers
            if t.missing_frames < self.max_missing and t.active
        ]

        return matched_objects

# Here we define a function to initialize the experiment output directory and configuration file.

def init_experiment_output(
    output_root,
    experiment_id,
    model_path="best.pt",
    model_input_size=(1024, 768),
    original_image_size=(2048, 1536),
    conf_threshold=0.25,
    iou_threshold=0.5,
    tracker_iou_threshold=0.3,
    max_missing=15,
    start_line=100,
    grading_line=924,
    line_1=256,
    line_2=512,
    size_method="equivalent_area",
    diameter_aggregation="median",
    mm_per_pixel=0.769,
    speed="unknown",
    cultivar="unknown",
    display_mode="rgb"
):
    """
    初始化一次实验的输出文件夹和 config.json。
    """

    exp_dir = os.path.join(output_root, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        "experiment_id": experiment_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,

        "model_input_size": list(model_input_size),
        "original_image_size": list(original_image_size),

        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "tracker_iou_threshold": tracker_iou_threshold,
        "max_missing": max_missing,

        "start_line": start_line,
        "grading_line": grading_line,
        "lane_1_boundary": line_1,
        "lane_2_boundary": line_2,

        "size_method": size_method,
        "diameter_aggregation": diameter_aggregation,
        "mm_per_pixel": mm_per_pixel,

        "speed": speed,
        "cultivar": cultivar,
        "display_mode": display_mode
    }

    config_path = os.path.join(exp_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    paths = {
        "exp_dir": exp_dir,
        "config": config_path,
        "frame_level": os.path.join(exp_dir, "frame_level_records.csv"),
        "apple_level": os.path.join(exp_dir, "apple_level_features.csv"),
        "final_results": os.path.join(exp_dir, "final_sorting_results.csv")
    }

    return paths

def append_csv_row(csv_path, fieldnames, row):
    """
    追加一行到 CSV。
    如果文件不存在，则自动写入表头。
    """

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

FRAME_LEVEL_FIELDS = [
    "experiment_id",
    "apple_id",
    "obj_id",
    "block_id",
    "timestamp",
    "lane",

    "x_center",
    "y_center",

    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",

    "pred_class",
    "confidence",

    "equivalent_diameter_mm",
    "mask_area_px",
    "ellipse_long_axis_mm",
    "ellipse_short_axis_mm",

    "is_between_start_and_grading_line",
    "is_crossed_grading_line"
]


APPLE_LEVEL_FIELDS = [
    "experiment_id",
    "apple_id",
    "obj_id",
    "true_label",
    "lane",
    "n_frames",

    "median_diameter",
    "mean_diameter",
    "std_diameter",
    "min_diameter",
    "max_diameter",
    "diameter_range",

    "mask_area_median",
    "mask_area_mean",
    "mask_area_std",

    "count_0",
    "count_1",
    "count_2",
    "ratio_0",
    "ratio_1",
    "ratio_2",

    "mean_conf_0",
    "mean_conf_1",
    "mean_conf_2",

    "max_conf_0",
    "max_conf_1",
    "max_conf_2",

    "median_conf_0",
    "median_conf_1",
    "median_conf_2",

    "weighted_score_0",
    "weighted_score_1",
    "weighted_score_2",

    "top1_class",
    "top1_ratio",
    "top1_mean_conf",
    "top1_max_conf",

    "class_switch_count",

    "current_rule_result",
    "current_rule_reason"
]


FINAL_RESULT_FIELDS = [
    "experiment_id",
    "apple_id",
    "obj_id",
    "timestamp",
    "lane",

    "final_class",
    "sent_signal",
    "diameter_used_mm",
    "grading_reason",

    "true_label",
    "is_correct"
]        



def build_apple_level_features(
    experiment_id,
    apple_id,
    obj_id,
    lane,
    data,
    final_class,
    grading_reason,
    true_label=None
):
    """
    将一个 obj_id 的多帧记录汇总成 apple-level 特征。
    """

    classes = np.array(data.get("class", []), dtype=int)
    confidences = np.array(data.get("confidence", []), dtype=float)
    diameters = np.array(data.get("Diameter", []), dtype=float)
    mask_areas = np.array(data.get("mask_area", []), dtype=float)

    n_frames = len(classes)

    row = {
        "experiment_id": experiment_id,
        "apple_id": apple_id,
        "obj_id": int(obj_id),
        "true_label": "" if true_label is None else int(true_label),
        "lane": lane,
        "n_frames": int(n_frames),

        "median_diameter": 0.0,
        "mean_diameter": 0.0,
        "std_diameter": 0.0,
        "min_diameter": 0.0,
        "max_diameter": 0.0,
        "diameter_range": 0.0,

        "mask_area_median": 0.0,
        "mask_area_mean": 0.0,
        "mask_area_std": 0.0,

        "count_0": 0,
        "count_1": 0,
        "count_2": 0,

        "ratio_0": 0.0,
        "ratio_1": 0.0,
        "ratio_2": 0.0,

        "mean_conf_0": 0.0,
        "mean_conf_1": 0.0,
        "mean_conf_2": 0.0,

        "max_conf_0": 0.0,
        "max_conf_1": 0.0,
        "max_conf_2": 0.0,

        "median_conf_0": 0.0,
        "median_conf_1": 0.0,
        "median_conf_2": 0.0,

        "weighted_score_0": 0.0,
        "weighted_score_1": 0.0,
        "weighted_score_2": 0.0,

        "top1_class": -1,
        "top1_ratio": 0.0,
        "top1_mean_conf": 0.0,
        "top1_max_conf": 0.0,

        "class_switch_count": 0,

        "current_rule_result": int(final_class),
        "current_rule_reason": grading_reason
    }

    if n_frames == 0:
        return row

    # ==============================
    # 尺寸统计
    # ==============================
    if len(diameters) > 0:
        row["median_diameter"] = float(np.median(diameters))
        row["mean_diameter"] = float(np.mean(diameters))
        row["std_diameter"] = float(np.std(diameters))
        row["min_diameter"] = float(np.min(diameters))
        row["max_diameter"] = float(np.max(diameters))
        row["diameter_range"] = float(np.max(diameters) - np.min(diameters))

    if len(mask_areas) > 0:
        row["mask_area_median"] = float(np.median(mask_areas))
        row["mask_area_mean"] = float(np.mean(mask_areas))
        row["mask_area_std"] = float(np.std(mask_areas))

    # ==============================
    # 类别统计
    # ==============================
    for cls in [0, 1, 2]:
        cls_mask = classes == cls
        cls_count = int(np.sum(cls_mask))

        row[f"count_{cls}"] = cls_count
        row[f"ratio_{cls}"] = float(cls_count / n_frames) if n_frames > 0 else 0.0

        cls_confs = confidences[cls_mask]

        if len(cls_confs) > 0:
            row[f"mean_conf_{cls}"] = float(np.mean(cls_confs))
            row[f"max_conf_{cls}"] = float(np.max(cls_confs))
            row[f"median_conf_{cls}"] = float(np.median(cls_confs))

            # 一个简单的 weighted score：类别比例 × 该类别平均置信度
            row[f"weighted_score_{cls}"] = float(row[f"ratio_{cls}"] * row[f"mean_conf_{cls}"])
        else:
            row[f"mean_conf_{cls}"] = 0.0
            row[f"max_conf_{cls}"] = 0.0
            row[f"median_conf_{cls}"] = 0.0
            row[f"weighted_score_{cls}"] = 0.0

    # ==============================
    # top1 class
    # ==============================
    ratios = {
        0: row["ratio_0"],
        1: row["ratio_1"],
        2: row["ratio_2"]
    }

    top1_class = max(ratios, key=ratios.get)

    row["top1_class"] = int(top1_class)
    row["top1_ratio"] = float(row[f"ratio_{top1_class}"])
    row["top1_mean_conf"] = float(row[f"mean_conf_{top1_class}"])
    row["top1_max_conf"] = float(row[f"max_conf_{top1_class}"])

    # ==============================
    # class switch count
    # ==============================
    if len(classes) > 1:
        row["class_switch_count"] = int(np.sum(classes[1:] != classes[:-1]))

    return row


# ========== 检测线程 ==========
def detection_loop(packet_queue, lane_queues, stop_event,
                   model_grade, file_path, class_names, colors,
                   output_root="experiment_outputs",
                   experiment_id=None,
                   model_path="best.pt",
                   speed="unknown",
                   cultivar="unknown",
                   true_label_map=None,
                   save_queue=None,
                   fps_stats=None):

    Grading_line = 1024 - 100
    Start_line = 100
    Line_1 = 256
    Line_2 = 512
    conf_thres = 0.25
    img_size = [768, 1024]  # [H, W]
    max_missing = 15
    iou_thres = 0.3
    if experiment_id is None:
        experiment_id = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")

    output_paths = init_experiment_output(
        output_root=output_root,
        experiment_id=experiment_id,
        model_path=model_path,
        model_input_size=(1024, 768),
        original_image_size=(2048, 1536),
        conf_threshold=conf_thres,
        iou_threshold=0.5,
        tracker_iou_threshold=iou_thres,
        max_missing=max_missing,
        start_line=Start_line,
        grading_line=Grading_line,
        line_1=Line_1,
        line_2=Line_2,
        size_method="equivalent_area",
        diameter_aggregation="median",
        mm_per_pixel=0.769,
        speed=speed,
        cultivar=cultivar,
        display_mode="rgb"
    )

    print(f"[EXPERIMENT] Output folder: {output_paths['exp_dir']}")

    tracker = TrackerManager(
        img_width=img_size[1],
        max_missing=max_missing,
        iou_threshold=iou_thres
    )
    object_records: Dict[int, Dict[str, List]] = {}
    crossed_ids = set()

    # obj_id 是 tracker 内部 ID；apple_id 是实验记录 ID。
    # 这里按最终输出 lane 分别计数，格式为：Lane x_y
    # 例如：Lane 1_1, Lane 1_2, Lane 2_1
    obj_to_apple_id = {}
    lane_output_counter = {0: 0, 1: 0, 2: 0}

    static_overlay = create_static_overlay(1024, 768, Start_line, Grading_line, Line_1, Line_2)
    processing_fps = 0.0
    fps_window_start = time.perf_counter()
    fps_window_count = 0

    while not stop_event.is_set():
        try:
            packet = packet_queue.get(timeout=1)
        except queue.Empty:
            continue

        if packet is None:
            continue

        # =====================================================
        # 1. 从同一个 packet 中取出检测帧和显示帧
        # =====================================================
        if not isinstance(packet, dict):
            print(f"⚠️ Invalid packet type: {type(packet)}")
            continue

        block_id = packet.get("block_id", None)
        detection_frame = packet.get("detection_frame", None)
        display_frame = packet.get("display_frame", None)
        display_mode = packet.get("display_mode", "unknown")

        if detection_frame is None or display_frame is None:
            print("⚠️ packet missing detection_frame or display_frame")
            continue

        # =====================================================
        # 2. 模型输入：5-channel multispectral image
        # =====================================================
        detect_img = cv2.resize(detection_frame, (1024, 768))
        detect_img = np.ascontiguousarray(detect_img)

        if detect_img.ndim != 3 or detect_img.shape[2] != 5:
            print(f"⚠️ detection_frame shape error: {detect_img.shape}")
            continue

        # =====================================================
        # 3. 显示图像：3-channel BGR image
        # =====================================================
        display_img = cv2.resize(display_frame, (1024, 768))

        if display_img.ndim == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        elif display_img.ndim == 3 and display_img.shape[2] == 4:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGRA2BGR)
        elif display_img.ndim == 3 and display_img.shape[2] == 5:
            # 保险处理：如果误传了 5-channel，这里只取前三个通道显示
            rgb_vis = display_img[:, :, :3]
            display_img = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

        display_img = np.ascontiguousarray(display_img)

        if display_img.ndim != 3 or display_img.shape[2] != 3:
            print(f"⚠️ display_frame shape error: {display_img.shape}")
            continue

        # =====================================================
        # 4. 模型检测：只使用 detect_img，不使用 display_img
        # =====================================================
        results_grade_list = model_grade(
            detect_img,
            show=False,
            save=False,
            show_labels=False,
            show_conf=False,
            conf=0.25,
            iou=0.5,
            save_txt=False,
            verbose=False,
            device=0
        )

        if len(results_grade_list) == 0:
            print("⚠️ 模型未返回结果，跳过该帧")
            enqueue_save_task(save_queue, block_id, detection_frame)
            continue

        results_grade = results_grade_list[0]

        # =====================================================
        # 5. 可视化绘图：只画到 display_img 上
        # =====================================================
        display_img = cv2.addWeighted(display_img, 1.0, static_overlay, 1.0, 0)

        # 可选：显示当前显示模式和 block_id
        cv2.putText(
            display_img,
            f"ID: {block_id} | Display: {display_mode}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        # =====================================================
        # 6. 主要判定逻辑
        # =====================================================
        if getattr(results_grade, "masks", None) is not None:
            try:
                masks = results_grade.masks.data.cpu().numpy()
            except Exception:
                masks = np.array([])

            try:
                boxes_np = results_grade.boxes.data.cpu().numpy()
            except Exception:
                boxes_np = np.array([])

            detections = []
            masks_bin = []

            for seg, box in zip(masks[:len(boxes_np)], boxes_np[:len(masks)]):
                if len(box) < 6:
                    continue

                conf = box[4]
                # print(f"There is a detection with confidence {conf:.2f}")
                cls = int(box[5])

                if conf < conf_thres:
                    continue

                xmin, ymin, xmax, ymax = map(int, box[:4])
                detections.append([xmin, ymin, xmax, ymax, conf, cls])

                mask_bin = (seg > 0.5).astype(np.uint8)

                # YOLO segmentation mask 的尺寸有时可能和显示/检测尺寸不一致，统一 resize 到 1024x768
                if mask_bin.shape[:2] != (img_size[0], img_size[1]):
                    mask_bin = cv2.resize(
                        mask_bin,
                        (img_size[1], img_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                masks_bin.append(mask_bin)

            # 关键修改：tracker.update 返回 tracker-object 与 detection-index 的匹配关系。
            # 后面根据 det_idx 去取 detections[det_idx] 和 masks_bin[det_idx]，避免错位。
            matched_objects = tracker.update([d[:4] for d in detections])

            draw_tasks = []

            for obj, det_idx in matched_objects:
                if det_idx >= len(detections) or det_idx >= len(masks_bin):
                    print(f"⚠️ {obj.id}: det_idx={det_idx} 超出 detections/masks_bin 范围，跳过")
                    continue

                det = detections[det_idx]
                mask = masks_bin[det_idx]

                xmin, ymin, xmax, ymax = map(int, obj.bbox)

                X_central = (xmin + xmax) * 0.5
                Y_central = (ymin + ymax) * 0.5
                obj_id = obj.id

                contour = mask.astype(np.uint8)
                contours, _ = cv2.findContours(
                    contour,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE  # 保留完整轮廓点，避免 SIMPLE 把矩形压缩成 4 个点
                )

                if len(contours) == 0:
                    print(f"⚠️ {obj.id}: 无有效轮廓，跳过该目标")
                    continue

                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)

                if area < 100:
                    print(f"⚠️ {obj.id}: 轮廓面积过小 ({area:.1f})，跳过该目标")
                    continue

                if len(cnt) < 5:
                    print(f"⚠️ {obj.id}: 轮廓点数不足 ({len(cnt)} < 5)，跳过该目标")
                    continue

                try:
                    # determine_size 内部会画长短轴，所以必须传 display_img，不能传 5-channel detect_img
                    Diameter, Size_1, Size_2, center, equivalent_radius_px = determine_size(display_img, cnt)
                except cv2.error as e:
                    print(f"⚠️ {obj.id}: 椭圆拟合失败: {e}")
                    continue

                if Start_line <= xmax <= Grading_line:
                    conf = float(det[4])
                    cls = int(det[5])

                    # 当前帧所属 lane 只用于 frame-level 记录。
                    # 最终 apple_id 仍然在过 Grading_line 时，按照最终输出 lane 分配。
                    if Y_central < Line_1:
                        current_lane_idx = 0
                    elif Y_central < Line_2:
                        current_lane_idx = 1
                    else:
                        current_lane_idx = 2
                    current_lane_name = f"Lane {current_lane_idx + 1}"

                    if obj_id not in object_records:
                        object_records[obj_id] = {
                            "apple_id": "",
                            "lane": current_lane_name,

                            "block_id": [],
                            "timestamp": [],

                            "x_center": [],
                            "y_center": [],
                            "bbox": [],

                            "class": [],
                            "confidence": [],

                            "Diameter": [],
                            "mask_area": [],

                            "Size_1": [],
                            "Size_2": []
                        }

                    rec = object_records[obj_id]
                    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    rec["lane"] = current_lane_name
                    rec["block_id"].append(block_id)
                    rec["timestamp"].append(timestamp_now)
                    rec["x_center"].append(float(X_central))
                    rec["y_center"].append(float(Y_central))
                    rec["bbox"].append([int(xmin), int(ymin), int(xmax), int(ymax)])

                    rec["class"].append(cls)
                    rec["confidence"].append(conf)
                    rec["Diameter"].append(float(Diameter))
                    rec["mask_area"].append(float(area))
                    rec["Size_1"].append(float(Size_1))
                    rec["Size_2"].append(float(Size_2))

                    # 由于 apple_id 按最终输出 lane 分配，过线前可能为空。
                    # 后续可通过 obj_id 与 final_sorting_results.csv 关联回填。
                    frame_apple_id = obj_to_apple_id.get(obj_id, rec.get("apple_id", ""))

                    frame_row = {
                        "experiment_id": experiment_id,
                        "apple_id": frame_apple_id,
                        "obj_id": int(obj_id),
                        "block_id": block_id,
                        "timestamp": timestamp_now,
                        "lane": current_lane_name,

                        "x_center": float(X_central),
                        "y_center": float(Y_central),

                        "bbox_xmin": int(xmin),
                        "bbox_ymin": int(ymin),
                        "bbox_xmax": int(xmax),
                        "bbox_ymax": int(ymax),

                        "pred_class": int(cls),
                        "confidence": float(conf),

                        "equivalent_diameter_mm": float(Diameter),
                        "mask_area_px": float(area),
                        "ellipse_long_axis_mm": float(Size_1),
                        "ellipse_short_axis_mm": float(Size_2),

                        "is_between_start_and_grading_line": 1,
                        "is_crossed_grading_line": 0
                    }

                    append_csv_row(
                        output_paths["frame_level"],
                        FRAME_LEVEL_FIELDS,
                        frame_row
                    )

                    draw_tasks.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "cls": cls,
                        "conf": conf,
                        "obj_id": int(obj_id),
                        "seg": contour,
                        "text": f"EqD: {Diameter:.2f} mm",
                        "center": center,
                        "long_axis": equivalent_radius_px
                    })

                if Grading_line <= X_central and obj_id not in crossed_ids:
                    if obj_id not in object_records:
                        object_records[obj_id] = {
                            "apple_id": "",
                            "lane": None,

                            "block_id": [],
                            "timestamp": [],

                            "x_center": [],
                            "y_center": [],
                            "bbox": [],

                            "class": [],
                            "confidence": [],

                            "Diameter": [],
                            "mask_area": [],

                            "Size_1": [],
                            "Size_2": []
                        }

                    data = object_records[obj_id]
                    Diameters = data.get("Diameter", [])
                    grades = data.get("class", [])
                    t_confidences = data.get("confidence", [])

                    if Diameters:
                        Diameter = float(np.median(Diameters))
                    else:
                        Diameter = 0.0

                    final_class, grading_reason = final_grading(
                        grades,
                        np.array(t_confidences).tolist(),
                        Diameter
                    )
                    final_class = int(final_class)

                    # =====================================================
                    # 按最终过线位置判断输出 lane，并按 lane 生成 apple_id
                    # apple_id 格式：Lane x_y
                    # x = lane 编号；y = 这条 lane 上输出的第几个苹果
                    # =====================================================
                    if Y_central < Line_1:
                        lane_idx = 0
                    elif Y_central < Line_2:
                        lane_idx = 1
                    else:
                        lane_idx = 2

                    lane_name = f"Lane {lane_idx + 1}"

                    if obj_id not in obj_to_apple_id:
                        lane_output_counter[lane_idx] += 1
                        obj_to_apple_id[obj_id] = f"Lane {lane_idx + 1}_{lane_output_counter[lane_idx]}"

                    apple_id = obj_to_apple_id[obj_id]
                    data["apple_id"] = apple_id
                    data["lane"] = lane_name

                    crossed_ids.add(obj_id)

                    sent_signal = final_class + 1

                    try:
                        lane_queues[lane_idx].put(sent_signal, block=False)
                    except Exception:
                        lane_queues[lane_idx].put(sent_signal)

                    true_label = None
                    if true_label_map is not None:
                        true_label = true_label_map.get(apple_id, None)

                    is_correct = ""
                    if true_label is not None:
                        is_correct = int(int(true_label) == int(final_class))

                    final_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    # 写入 final_sorting_results.csv
                    final_row = {
                        "experiment_id": experiment_id,
                        "apple_id": apple_id,
                        "obj_id": int(obj_id),
                        "timestamp": final_timestamp,
                        "lane": lane_name,

                        "final_class": int(final_class),
                        "sent_signal": int(sent_signal),
                        "diameter_used_mm": float(Diameter),
                        "grading_reason": grading_reason,

                        "true_label": "" if true_label is None else int(true_label),
                        "is_correct": is_correct
                    }

                    append_csv_row(
                        output_paths["final_results"],
                        FINAL_RESULT_FIELDS,
                        final_row
                    )

                    # 写入 apple_level_features.csv，用于后续 Random Forest 训练
                    apple_feature_row = build_apple_level_features(
                        experiment_id=experiment_id,
                        apple_id=apple_id,
                        obj_id=obj_id,
                        lane=lane_name,
                        data=data,
                        final_class=final_class,
                        grading_reason=grading_reason,
                        true_label=true_label
                    )

                    append_csv_row(
                        output_paths["apple_level"],
                        APPLE_LEVEL_FIELDS,
                        apple_feature_row
                    )

                    # 额外写入一行 crossed frame，方便 frame-level 表中直接看到 apple_id 与过线时刻。
                    crossed_frame_row = {
                        "experiment_id": experiment_id,
                        "apple_id": apple_id,
                        "obj_id": int(obj_id),
                        "block_id": block_id,
                        "timestamp": final_timestamp,
                        "lane": lane_name,

                        "x_center": float(X_central),
                        "y_center": float(Y_central),

                        "bbox_xmin": int(xmin),
                        "bbox_ymin": int(ymin),
                        "bbox_xmax": int(xmax),
                        "bbox_ymax": int(ymax),

                        "pred_class": int(det[5]),
                        "confidence": float(det[4]),

                        "equivalent_diameter_mm": float(Diameter),
                        "mask_area_px": float(area),
                        "ellipse_long_axis_mm": float(Size_1),
                        "ellipse_short_axis_mm": float(Size_2),

                        "is_between_start_and_grading_line": 0,
                        "is_crossed_grading_line": 1
                    }

                    append_csv_row(
                        output_paths["frame_level"],
                        FRAME_LEVEL_FIELDS,
                        crossed_frame_row
                    )

                    print(
                        f"🍎 {apple_id} | Object ID {obj_id} graded as Class {final_class + 1} "
                        f"Diameter: {Diameter:.2f} mm, Reason: {grading_reason}, "
                        f"sent to {lane_name}"
                    )

                    # 保留原来的 JSON 输出，方便兼容旧逻辑。
                    # 新实验数据建议主要看 final_sorting_results.csv。
                    new_record = {
                        "timestamp": final_timestamp,
                        "apple_id": apple_id,
                        "ID": int(obj_id),
                        "grades": int(final_class),
                        "sent_signal": int(sent_signal),
                        "Diameter": float(Diameter),
                        "Reason": grading_reason,
                        "Lane": lane_name
                    }

                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                output = json.load(f)
                                if not isinstance(output, list):
                                    output = [output]
                        except Exception:
                            output = []
                    else:
                        output = []

                    output.append(new_record)

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(output, f, indent=4, ensure_ascii=False)

            display_img = draw_results(display_img, draw_tasks, colors, class_names)

        fps_window_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 1.0:
            processing_fps = fps_window_count / elapsed
            fps_window_start = now
            fps_window_count = 0

        saving_fps = 0.0
        if fps_stats is not None:
            with fps_stats["lock"]:
                saving_fps = fps_stats.get("saving_fps", 0.0)

        cv2.putText(
            display_img,
            f"Processing FPS: {processing_fps:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            display_img,
            f"Saving FPS: {saving_fps:.2f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        enqueue_save_task(save_queue, block_id, detection_frame)

        # =====================================================
        # 7. 显示：只显示 display_img
        # =====================================================
        cv2.imshow("frame", display_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    for q in lane_queues:
        q.put(None)

    cv2.destroyAllWindows()
    print("检测线程结束")


