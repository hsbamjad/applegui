#!/usr/bin/env python3
import eBUS as eb
import sys
import numpy as np
import cv2
import threading
import queue

sys.path.append("C:\\Yuyuan\\apple_sorting_project\\apple_integration_code\\apple_integration\\lib")
import PvSampleUtils as psu

BUFFER_COUNT = 16
kb = psu.PvKb()   # 键盘监听

# 🔹 全局队列，用于传给 online_test
# 一个队列同时传递 detection_frame 和 display_frame，避免两路队列不同步
packet_queue = queue.Queue(maxsize=40)

# 为了兼容旧代码命名，frame_queue 仍然保留为 packet_queue 的别名
frame_queue = packet_queue


latest_frames = {
    "Source0": None,
    "Source1": None,
    "Source2": None
}
frames_lock = threading.Lock()  # 避免多线程同时写入冲突

class SourceStream:
    def __init__(self, device, connection_id, source_name, detection_channel, show_window=False, display_mode="rgb"):
        self.device = device
        self.connection_id = connection_id
        self.source_name = source_name
        self.detection_channel = detection_channel
        self.show_window = show_window
        self.display_mode = display_mode  # 可选: "rgb", "nir1", "nir2"
        self.stream = None
        self.pipeline = None
        self.running = False
        self.capture_thread = None
        self.display_queue = queue.Queue(maxsize=5)
        self.frame_count = 0
        self.display_interval = 1  # 每帧都显示

    def open(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)

        result, channel = self.device.GetParameters().GetIntegerValue("SourceIDValue")
        if result.IsFailure():
            result, channel = self.device.GetParameters().GetIntegerValue("SourceStreamChannel")
            if result.IsFailure():
                print(f"[{self.source_name}] Cannot determine stream channel.")
                return False

        self.stream = eb.PvStreamGEV()
        if self.stream.Open(self.connection_id, 0, channel).IsFailure():
            print(f"[{self.source_name}] Failed to open stream.")
            return False

        ip = self.stream.GetLocalIPAddress()
        port = self.stream.GetLocalPort()
        self.device.SetStreamDestination(ip, port, channel)

        payload_size = self.device.GetPayloadSize()
        self.pipeline = eb.PvPipeline(self.stream)
        self.pipeline.SetBufferSize(payload_size)
        self.pipeline.SetBufferCount(BUFFER_COUNT)
        self.pipeline.Start()

        return True

    def start_acquisition(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)
        self.device.StreamEnable()
        self.device.GetParameters().Get("AcquisitionStart").Execute()

    def stop_acquisition(self):
        stack = eb.PvGenStateStack(self.device.GetParameters())
        stack.SetEnumValue("SourceSelector", self.source_name)
        self.device.GetParameters().Get("AcquisitionStop").Execute()
        self.device.StreamDisable()

    def close(self):
        self.pipeline.Stop()
        self.stream.Close()


    def run(self):
        self.running = True

        while self.running and not kb.is_stopping():
            result, buffer, op_result = self.pipeline.RetrieveNextBuffer(1000)

            if not result.IsOK():
                continue

            try:
                if not op_result.IsOK():
                    continue

                image = buffer.GetImage()
                if not image:
                    continue

                width, height = image.GetWidth(), image.GetHeight()
                pixel_type = image.GetPixelType()
                ptr = image.GetDataPointer()
                block_id = buffer.GetBlockID()

                # ==========================================================
                # ① 转换为 numpy 图像
                # ==========================================================
                # Source0: RGB/Bayer -> BGR, H x W x 3
                # Source1/Source2: Mono8 -> gray, H x W
                # ==========================================================

                if pixel_type == eb.PvPixelMono8:
                    np_image = np.ctypeslib.as_array(
                        ptr, shape=(height, width)
                    ).copy()

                elif pixel_type == 0x01080009:  # BayerRG8
                    bayer = np.ctypeslib.as_array(
                        ptr, shape=(height, width)
                    ).copy()
                    # np_image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
                    np_image = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

                elif pixel_type == eb.PvPixelRGB8:
                    rgb = np.ctypeslib.as_array(
                        ptr, shape=(height, width, 3)
                    ).copy()
                    np_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                else:
                    print(f"⚠️ Unsupported pixel type: {pixel_type}")
                    continue

                # ==========================================================
                # ② 缓存三个 Source，并按 block_id 配对
                # ==========================================================
                rgb_frame = None
                nir1_frame = None
                nir2_frame = None
                packet_block_id = None

                with frames_lock:
                    latest_frames[self.source_name] = {
                        "image": np_image.copy(),
                        "block_id": block_id
                    }

                    if (
                        latest_frames.get("Source0") is not None
                        and latest_frames.get("Source1") is not None
                        and latest_frames.get("Source2") is not None
                    ):
                        rgb_item = latest_frames["Source0"]
                        nir1_item = latest_frames["Source1"]
                        nir2_item = latest_frames["Source2"]

                        rgb_block_id = rgb_item["block_id"]
                        nir1_block_id = nir1_item["block_id"]
                        nir2_block_id = nir2_item["block_id"]

                        if rgb_block_id == nir1_block_id == nir2_block_id:
                            rgb_frame = rgb_item["image"].copy()
                            nir1_frame = nir1_item["image"].copy()
                            nir2_frame = nir2_item["image"].copy()
                            packet_block_id = rgb_block_id

                            latest_frames["Source0"] = None
                            latest_frames["Source1"] = None
                            latest_frames["Source2"] = None

                        else:
                            print(
                                f"⚠️ Source block_id mismatch: "
                                f"Source0={rgb_block_id}, "
                                f"Source1={nir1_block_id}, "
                                f"Source2={nir2_block_id}"
                            )

                            min_block_id = min(rgb_block_id, nir1_block_id, nir2_block_id)

                            if rgb_block_id == min_block_id:
                                latest_frames["Source0"] = None
                            if nir1_block_id == min_block_id:
                                latest_frames["Source1"] = None
                            if nir2_block_id == min_block_id:
                                latest_frames["Source2"] = None

                # ==========================================================
                # ③ 构建 packet:
                # detection_frame: H x W x 5, [R, G, B, NIR1, NIR2]
                # display_frame: H x W x 3, BGR，用于显示和绘图
                # ==========================================================
                if (
                    rgb_frame is not None
                    and nir1_frame is not None
                    and nir2_frame is not None
                ):
                    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
                        print(f"⚠️ Source0 RGB shape error: {rgb_frame.shape}")
                        continue

                    h, w = rgb_frame.shape[:2]

                    nir1_gray = self.ensure_gray(nir1_frame)
                    nir2_gray = self.ensure_gray(nir2_frame)

                    if nir1_gray.shape[:2] != (h, w):
                        nir1_gray = cv2.resize(nir1_gray, (w, h))

                    if nir2_gray.shape[:2] != (h, w):
                        nir2_gray = cv2.resize(nir2_gray, (w, h))

                    # Source0 是 OpenCV BGR；训练时顺序是 [R, G, B, NIR1, NIR2]
                    B = rgb_frame[:, :, 0]
                    G = rgb_frame[:, :, 1]
                    R = rgb_frame[:, :, 2]

                    detection_frame = np.dstack([
                        R,
                        G,
                        B,
                        nir1_gray,
                        nir2_gray
                    ]).astype(np.uint8)

                    display_frame = self.build_display_frame(
                        rgb_bgr=rgb_frame,
                        nir1_gray=nir1_gray,
                        nir2_gray=nir2_gray,
                        mode=self.display_mode
                    )

                    packet = {
                        "block_id": packet_block_id,
                        "detection_frame": detection_frame,
                        "display_frame": display_frame,
                        "display_mode": self.display_mode
                    }

                    try:
                        packet_queue.put_nowait(packet)
                    except queue.Full:
                        pass

                # ==========================================================
                # ④ 单个 source 的独立预览窗口，可选
                # ==========================================================
                self.frame_count += 1

                if self.show_window and self.frame_count % self.display_interval == 0:
                    source_preview = self.to_display_bgr(np_image)

                    try:
                        self.display_queue.put_nowait((block_id, source_preview))
                    except queue.Full:
                        pass

            finally:
                self.pipeline.ReleaseBuffer(buffer)

    def ensure_gray(self, img):
        """
        确保 NIR 图像是 H x W 单通道。
        """
        if img.ndim == 2:
            return img

        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0]
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        raise ValueError(f"Unsupported NIR image shape: {img.shape}")

    def normalize_gray_for_display(self, gray):
        """
        将 NIR 灰度图归一化成 8-bit BGR，便于显示。
        """
        gray = self.ensure_gray(gray)

        if gray.dtype != np.uint8:
            gray_8u = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray_8u = gray_8u.astype(np.uint8)
        else:
            gray_8u = gray

        return cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)

    def build_display_frame(self, rgb_bgr, nir1_gray, nir2_gray, mode="rgb"):
        """
        根据 mode 选择显示图像。
        mode 可选：
            "rgb"  -> 显示 Source0 的 RGB/BGR 图像
            "nir1" -> 显示 Source1 的 NIR1 灰度图
            "nir2" -> 显示 Source2 的 NIR2 灰度图

        返回值永远是 H x W x 3 的 BGR 图像，便于 cv2 画框和 imshow。
        """
        mode = str(mode).lower()

        if mode == "rgb":
            return rgb_bgr.copy()

        if mode == "nir1":
            return self.normalize_gray_for_display(nir1_gray)

        if mode == "nir2":
            return self.normalize_gray_for_display(nir2_gray)

        print(f"⚠️ Unknown display_mode={mode}, fallback to rgb")
        return rgb_bgr.copy()

    def to_display_bgr(self, img):
        """
        将任意图像转成可以 cv2.imshow 显示的 BGR 图像。
        注意：这里只用于单 source 预览，不参与模型输入。
        """
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            if img.shape[2] == 3:
                return img
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.shape[2] == 5:
                rgb = img[:, :, :3]
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raise ValueError(f"Unsupported display image shape: {img.shape}")

    def display_loop(self):
        while self.running:
            try:
                block_id, img = self.display_queue.get(timeout=0.5)
                cv2.putText(img, f"{self.source_name} - ID: {block_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(f"{self.source_name}", img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                    self.running = False
            except queue.Empty:
                continue

    def start_thread(self, show_window=False):
        self.show_window = show_window
        self.capture_thread = threading.Thread(target=self.run)
        self.capture_thread.start()
        if self.show_window:
            threading.Thread(target=self.display_loop, daemon=True).start()

    def stop_thread(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
