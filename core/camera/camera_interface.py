"""
core/camera/camera_interface.py
================================
Camera interface — mock and real JAI eBUS backends.

Supports two backends controlled by config["camera"]["mode"]:
  "mock" → MockCamera  (synthetic frames, works on any machine)
  "jai"  → JAICamera   (real hardware, requires JAI eBUS SDK + 10 GigE NIC)

Frame triplet format:
  Each acquisition returns a FrameTriplet of 3 NumPy arrays:
    ch1: np.ndarray shape (1536, 2048, 3) dtype uint8  — Color BGR (BayerBG2BGR)
    ch2: np.ndarray shape (1536, 2048)    dtype uint8  — NIR1 ~800nm (Mono8, normalized)
    ch3: np.ndarray shape (1536, 2048)    dtype uint8  — NIR2 ~900nm (Mono8, normalized)

Threading:
  CameraInterface is NOT thread-safe on its own.
  Use CameraWorker (gui/workers/camera_worker.py) to wrap it in a QThread.
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

log = logging.getLogger(__name__)


@dataclass
class FrameTriplet:
    """One synchronized capture from all 3 JAI sensors."""
    ch1:        np.ndarray    # Color BGR  shape (H, W, 3) uint8
    ch2:        np.ndarray    # NIR1 ~800nm shape (H, W)   uint8
    ch3:        np.ndarray    # NIR2 ~900nm shape (H, W)   uint8
    timestamp:  float         # time.time() at acquisition
    frame_idx:  int           # monotonically increasing frame counter
    block_id:   int = -1      # GEV block ID (all 3 channels must match for sync)


# ── JAI eBUS backend ──────────────────────────────────────────────────────────

class _JAISource:
    """
    One physical sensor on the FS-3200T.
    Mirrors the Source class proven in scripts/camera_probe_jai.py.
    """

    BUFFER_COUNT = 16

    def __init__(self, device, connection_id: str, source_name: str, ch_index: int):
        self._device         = device
        self._connection_id  = connection_id
        self._source_name    = source_name
        self._ch_index       = ch_index
        self.stream          = None
        self.pipeline        = None
        self.source_channel  = 0
        self.pixel_format    = "Mono8"

    @staticmethod
    def _get_p(nm, name: str) -> str:
        try:
            param = nm.Get(name)
            if param is None:
                return "N/A"
            try:
                r, v = param.GetValue()
                if r.IsOK():
                    return str(v)
            except Exception:
                pass
            try:
                r, v = param.GetValueString()
                if r.IsOK():
                    return v
            except Exception:
                pass
        except Exception:
            pass
        return "N/A"

    def open(self) -> bool:
        try:
            import eBUS as eb
        except ImportError:
            log.error("eBUS SDK not found — cannot open JAI source")
            return False

        nm    = self._device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self._source_name)

        # Read integer source channel ID
        result, self.source_channel = nm.GetIntegerValue("SourceIDValue")
        if result.IsFailure():
            result, self.source_channel = nm.GetIntegerValue("SourceStreamChannel")
        if result.IsFailure():
            self.source_channel = self._ch_index

        self.pixel_format = self._get_p(nm, "PixelFormat")
        log.info(
            "  %s  ch_id=%d  fmt=%s  %s×%s",
            self._source_name, self.source_channel,
            self.pixel_format,
            self._get_p(nm, "Width"), self._get_p(nm, "Height"),
        )

        # Open dedicated GEV stream for this source channel
        self.stream = eb.PvStreamGEV()
        r = self.stream.Open(self._connection_id, 0, self.source_channel)
        if r.IsFailure():
            log.error("  %s: stream.Open failed: %s",
                      self._source_name, r.GetCodeString())
            return False

        lip = self.stream.GetLocalIPAddress()
        lp  = self.stream.GetLocalPort()
        self._device.SetStreamDestination(lip, lp, self.source_channel)
        log.info("       → %s:%d", lip, lp)

        # Pipeline
        payload_size = self._device.GetPayloadSize()
        self.pipeline = eb.PvPipeline(self.stream)
        self.pipeline.SetBufferSize(payload_size)
        self.pipeline.SetBufferCount(self.BUFFER_COUNT)
        self.pipeline.Start()
        return True

    def start_acquisition(self) -> None:
        import eBUS as eb
        nm    = self._device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self._source_name)
        self._device.StreamEnable()
        nm.Get("AcquisitionStart").Execute()

    def stop_acquisition(self) -> None:
        import eBUS as eb
        nm    = self._device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self._source_name)
        nm.Get("AcquisitionStop").Execute()
        self._device.StreamDisable()

    def grab(self, timeout_ms: int = 500) -> tuple[Optional[np.ndarray], int]:
        """
        Retrieve one frame from the pipeline.
        Returns (raw_array, block_id) or (None, -1) on timeout/error.
        """
        result, buffer, op_result = self.pipeline.RetrieveNextBuffer(timeout_ms)
        if result.IsFailure() or not op_result.IsOK():
            return None, -1

        image    = buffer.GetImage()
        raw      = image.GetDataPointer().copy()
        block_id = buffer.GetBlockID()
        self.pipeline.ReleaseBuffer(buffer)
        return raw, block_id

    def close(self) -> None:
        if self.pipeline:
            self.pipeline.Stop()
        if self.stream:
            self.stream.Close()


class JAICamera:
    """
    Real JAI FS-3200T camera backend using eBUS Python SDK.
    Simultaneous 3-source MultiSource acquisition with hardware sync.
    """

    WARMUP_S     = 2.0
    DRAIN_FRAMES = 30
    DISPLAY_W    = 640    # resize to this BEFORE emitting — keeps Qt rendering fast
    DISPLAY_H    = 480

    def __init__(self, config: dict) -> None:
        self._cfg          = config
        self._device       = None
        self._sources: list[_JAISource] = []
        self._running      = False
        self._frame_idx    = 0
        self._latest: Optional[FrameTriplet] = None
        self._latest_lock  = threading.Lock()
        self._grab_thread: Optional[threading.Thread] = None
        # EMA-stabilized gain for NIR channels — prevents per-frame flicker
        self._nir_ema_min  = [0.0, 0.0]     # one per NIR source
        self._nir_ema_max  = [64.0, 64.0]   # one per NIR source
        self._EMA_ALPHA    = 0.05            # low = slow/stable, high = fast/reactive

    def connect(self) -> bool:
        try:
            import eBUS as eb
        except ImportError:
            log.error("eBUS SDK not installed — cannot connect to JAI camera")
            return False

        jai_cfg    = self._cfg.get("jai", {})
        target_mac = jai_cfg.get("mac", None)   # MAC is stable; IP can change
        target_ip  = jai_cfg.get("ip",  None)   # IP as fallback hint

        # ── Discover ──────────────────────────────────────────────
        log.info("JAICamera: scanning network for FS-3200T …")
        try:
            sys_obj = eb.PvSystem()
            sys_obj.Find()
        except Exception as e:
            log.error("JAICamera: PvSystem.Find() failed: %s", e)
            return False

        connection_id = None
        for i in range(sys_obj.GetInterfaceCount()):
            iface = sys_obj.GetInterface(i)
            for j in range(iface.GetDeviceCount()):
                dev = iface.GetDeviceInfo(j)
                mac = str(dev.GetMACAddress())
                ip  = str(dev.GetIPAddress())
                log.info("  Found: %s  IP=%s  MAC=%s", dev.GetDisplayID(), ip, mac)

                # Prefer MAC match (stable), fall back to IP match, else first device
                if connection_id is None:
                    if target_mac and mac.lower() == target_mac.lower():
                        connection_id = dev.GetConnectionID()
                        log.info("  → Selected by MAC match")
                    elif target_ip and ip == target_ip:
                        connection_id = dev.GetConnectionID()
                        log.info("  → Selected by IP match")

        # Last resort: use first found device if no match
        if connection_id is None:
            for i in range(sys_obj.GetInterfaceCount()):
                iface = sys_obj.GetInterface(i)
                for j in range(iface.GetDeviceCount()):
                    connection_id = iface.GetDeviceInfo(j).GetConnectionID()
                    log.info("  → No MAC/IP match — using first found device")
                    break
                if connection_id:
                    break

        if connection_id is None:
            log.error("JAICamera: no camera found on network")
            return False

        # ── Connect ───────────────────────────────────────────────
        try:
            result, self._device = eb.PvDevice.CreateAndConnect(connection_id)
        except Exception as e:
            log.error("JAICamera: CreateAndConnect raised: %s", e)
            return False

        if self._device is None:
            log.error("JAICamera: connect failed: %s — is eBUS Player open?",
                      result.GetCodeString())
            return False
        log.info("JAICamera: connected (GEV: %s)", isinstance(self._device, eb.PvDeviceGEV))

        # Negotiate packet size BEFORE opening any streams (prevents dark NIR)
        if isinstance(self._device, eb.PvDeviceGEV):
            r = self._device.NegotiatePacketSize()
            log.info("NegotiatePacketSize: %s", r.GetCodeString())

        # ── Enumerate sources ─────────────────────────────────────
        nm              = self._device.GetParameters()
        source_selector = nm.GetEnum("SourceSelector")
        source_names    = []
        if source_selector:
            result, count = source_selector.GetEntriesCount()
            for i in range(count):
                result, entry = source_selector.GetEntryByIndex(i)
                if entry:
                    result, name = entry.GetName()
                    source_names.append(name)
        log.info("Sources: %s", source_names)

        # ── Open all streams simultaneously ───────────────────────
        for ch_idx, src_name in enumerate(source_names):
            src = _JAISource(self._device, connection_id, src_name, ch_idx)
            if src.open():
                self._sources.append(src)

        if not self._sources:
            log.error("JAICamera: no sources opened")
            self._device.Disconnect()
            eb.PvDevice.Free(self._device)
            return False

        log.info("JAICamera: %d streams open simultaneously", len(self._sources))

        # ── Start acquisition + warmup + drain ────────────────────
        for src in self._sources:
            src.start_acquisition()

        log.info("JAICamera: warming up %.1fs …", self.WARMUP_S)
        time.sleep(self.WARMUP_S)

        counts = {src._source_name: 0 for src in self._sources}
        for _ in range(self.DRAIN_FRAMES):
            for src in self._sources:
                r, buf, op = src.pipeline.RetrieveNextBuffer(200)
                if r.IsOK():
                    src.pipeline.ReleaseBuffer(buf)
                    counts[src._source_name] += 1
        log.info("JAICamera: drained %s — ready", counts)

        self._running = True
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="JAI-grab"
        )
        self._grab_thread.start()
        log.info("JAICamera: background grab thread started — ready")
        return True

    # ── Background grab loop ──────────────────────────────────────────────────

    def _grab_loop(self) -> None:
        """
        Runs in a daemon thread at full camera speed (30fps).
        Always stores the latest processed triplet in self._latest.
        The worker thread reads this cache non-blocking via grab().
        """
        try:
            while self._running:
                raws     = []
                block_id = -1
                ok       = True

                for src in self._sources:
                    raw, bid = src.grab(timeout_ms=100)
                    if raw is None:
                        ok = False
                        break
                    raws.append((raw, src.pixel_format, bid))
                    if block_id == -1:
                        block_id = bid

                if not ok or len(raws) < len(self._sources):
                    continue

                triplet = self._process_raws(raws, block_id)
                with self._latest_lock:
                    self._latest = triplet
        except Exception as e:
            log.error("JAI grab thread crashed: %s", e, exc_info=True)

    def _process_raws(
        self,
        raws: list,
        block_id: int,
    ) -> FrameTriplet:
        """Convert raw buffers to display-ready FrameTriplet."""
        raw0, pf0, _ = raws[0]
        raw1, _,   _ = raws[1]
        raw2, _,   _ = raws[2]

        # 1. CH1 Bayer demosaic at full res → Resize with fast bilinear interpolation
        if pf0 == "BayerRG8":
            ch1 = cv2.cvtColor(raw0, cv2.COLOR_BayerBG2BGR)
        else:
            ch1 = raw0
        ch1 = cv2.resize(ch1, (self.DISPLAY_W, self.DISPLAY_H), interpolation=cv2.INTER_LINEAR)

        # 2. Extract raw peak maximums and minimums at full resolution (<0.3ms) to preserve hot pixels and highlight levels
        cur_min1, cur_max1 = float(raw1.min()), float(raw1.max())
        cur_min2, cur_max2 = float(raw2.min()), float(raw2.max())

        # 3. Downsample raw Mono8 NIR frames FIRST to 640x480 using fast bilinear interpolation
        raw1_small = cv2.resize(raw1, (self.DISPLAY_W, self.DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        raw2_small = cv2.resize(raw2, (self.DISPLAY_W, self.DISPLAY_H), interpolation=cv2.INTER_LINEAR)

        mn1, mx1 = int(raw1_small.min()), int(raw1_small.max())
        mn2, mx2 = int(raw2_small.min()), int(raw2_small.max())

        if self._frame_idx % 90 == 0:
            log.info("NIR stats — CH2: min=%d max=%d  |  CH3: min=%d max=%d",
                     mn1, mx1, mn2, mx2)

        # 4. Perform EMA-stabilized Min-Max normalization on 640x480 using cv2.convertScaleAbs.
        # This completely subtracts the dynamic black offset (pedestal) and stretches the contrast,
        # perfectly matching cv2.normalize(..., cv2.NORM_MINMAX) from the simple video to eliminate gray glare!
        def _ema_normalize(
            raw_small: np.ndarray,
            cur_min: float,
            cur_max: float,
            ch_idx: int,
        ) -> np.ndarray:
            if self._frame_idx == 0:
                self._nir_ema_min[ch_idx] = cur_min
                self._nir_ema_max[ch_idx] = cur_max
            else:
                self._nir_ema_min[ch_idx] = (
                    (1 - self._EMA_ALPHA) * self._nir_ema_min[ch_idx]
                    + self._EMA_ALPHA * cur_min
                )
                self._nir_ema_max[ch_idx] = (
                    (1 - self._EMA_ALPHA) * self._nir_ema_max[ch_idx]
                    + self._EMA_ALPHA * cur_max
                )

            diff = self._nir_ema_max[ch_idx] - self._nir_ema_min[ch_idx]
            diff = max(diff, 1.0)
            scale = 255.0 / diff
            offset = -self._nir_ema_min[ch_idx] * scale
            return cv2.convertScaleAbs(raw_small, alpha=scale, beta=offset)

        ch2 = _ema_normalize(raw1_small, cur_min1, cur_max1, 0)
        ch3 = _ema_normalize(raw2_small, cur_min2, cur_max2, 1)

        self._frame_idx += 1
        return FrameTriplet(
            ch1       = ch1,
            ch2       = ch2,
            ch3       = ch3,
            timestamp = time.time(),
            frame_idx = self._frame_idx,
            block_id  = block_id,
        )


    def grab(self) -> Optional[FrameTriplet]:
        """
        Non-blocking: return the latest frame from the background grab thread.
        Never blocks on the camera — always returns immediately.
        """
        if not self._running:
            return None
        with self._latest_lock:
            return self._latest

    # ── Live camera controls ──────────────────────────────────────────────────

    def set_exposure(self, exposure_us: int) -> bool:
        """
        Set sensor exposure time in microseconds while streaming.

        The exposure time controls how long each sensor pixel integrates light:
          - Short exposure  (1000–5000 µs)  → darker image, freezes fast motion
          - Medium exposure (5000–15000 µs) → balanced for 1 apple/s conveyor
          - Long exposure   (>15000 µs)     → brighter image, risk of motion blur

        Hardware constraint — FPS caps the maximum exposure:
          30 FPS  → max 33,333 µs   (1,000,000 / 30)
          60 FPS  → max 16,666 µs
          107 FPS → max  9,345 µs
        The camera firmware enforces this automatically; we log if value is clamped.

        Args:
            exposure_us: Desired exposure in microseconds. Range: 100–100,000.

        Returns:
            True on success, False on failure.
        """
        if self._device is None:
            log.warning("set_exposure: no device connected")
            return False
        try:
            print(f"[CAM] set_exposure: writing {exposure_us} µs to device")
            nm = self._device.GetParameters()
            # Use GetEnum (not Get) for enum parameters in eBUS Python API
            ae = nm.GetEnum("ExposureAuto")
            if ae:
                ae.SetValue("Off")
            param = nm.GetFloat("ExposureTime")
            if param is None:
                print("[CAM] set_exposure: ExposureTime param is None!")
                log.error("set_exposure: ExposureTime parameter not found on device")
                return False
            r = param.SetValue(float(exposure_us))
            if r.IsOK():
                # Read back actual value — camera may have clamped it
                _, actual = param.GetValue()
                print(f"[CAM] set_exposure: OK, actual={actual:.0f} µs")
                if abs(actual - exposure_us) > 50:
                    log.warning("set_exposure: requested %d µs, camera clamped to %.0f µs "
                                "(FPS limit — reduce frame rate to allow longer exposure)",
                                exposure_us, actual)
                else:
                    log.info("Camera: ExposureTime = %d µs", exposure_us)
                return True
            print(f"[CAM] set_exposure: REJECTED by camera: {r.GetCodeString()}")
            log.error("set_exposure: camera rejected value %d µs: %s",
                      exposure_us, r.GetCodeString())
            return False
        except Exception as e:
            print(f"[CAM] set_exposure: EXCEPTION: {e}")
            log.error("set_exposure exception: %s", e)
            return False

    def set_fps(self, fps: float) -> bool:
        """
        Set acquisition frame rate while streaming.

        Changing FPS has two important side-effects:
          1. Max allowed exposure shrinks: max_exp_us = 1,000,000 / fps
             (firmware enforces this — any existing exposure above the new limit
              will be silently clamped by the camera)
          2. NIR EMA gain will temporarily flicker for ~20 frames as the
             background illumination/sensor output adjusts — this is normal.

        The FS-3200T supports 1–107 FPS at full 2048×1536 resolution.
        Lower FPS → more light per frame → brighter NIR. Useful in dark conditions.
        Higher FPS → faster throughput → shorter max exposure.

        Args:
            fps: Desired frame rate in frames per second. Range: 1–107.

        Returns:
            True on success, False on failure.
        """
        if self._device is None:
            log.warning("set_fps: no device connected")
            return False
        try:
            print(f"[CAM] set_fps: writing {fps} FPS to device")
            nm = self._device.GetParameters()
            # Must enable frame rate control before setting value
            enable = nm.GetBoolean("AcquisitionFrameRateEnable")
            if enable:
                enable.SetValue(True)
            param = nm.GetFloat("AcquisitionFrameRate")
            if param is None:
                print("[CAM] set_fps: AcquisitionFrameRate param is None!")
                log.error("set_fps: AcquisitionFrameRate parameter not found on device")
                return False
            r = param.SetValue(float(fps))
            if r.IsOK():
                _, actual = param.GetValue()
                print(f"[CAM] set_fps: OK, actual={actual:.1f} FPS")
                log.info("Camera: AcquisitionFrameRate = %.1f FPS (max exposure now %.0f µs)",
                         actual, 1_000_000 / max(actual, 1))
                return True
            print(f"[CAM] set_fps: REJECTED by camera: {r.GetCodeString()}")
            log.error("set_fps: camera rejected %.1f FPS: %s", fps, r.GetCodeString())
            return False
        except Exception as e:
            print(f"[CAM] set_fps: EXCEPTION: {e}")
            log.error("set_fps exception: %s", e)
            return False

    def get_exposure(self) -> int:
        """
        Read the current ExposureTime from the device firmware.
        Returns the ACTUAL value the camera is using (may differ from
        what was requested if the FPS limit clamped it).
        Returns -1 on failure.
        """
        if self._device is None:
            return -1
        try:
            nm    = self._device.GetParameters()
            param = nm.GetFloat("ExposureTime")
            if param is None:
                return -1
            _, val = param.GetValue()
            return int(val)
        except Exception as e:
            log.error("get_exposure exception: %s", e)
            return -1

    def disconnect(self) -> None:
        import eBUS as eb
        self._running = False
        for src in self._sources:
            src.stop_acquisition()
        for src in self._sources:
            src.close()
        self._sources.clear()
        if self._device:
            self._device.Disconnect()
            eb.PvDevice.Free(self._device)
            self._device = None
        log.info("JAICamera: disconnected")


# ── Unified CameraInterface ───────────────────────────────────────────────────

class CameraInterface:
    """
    Unified camera interface — wraps mock or real JAI camera.
    Backend selected by config["mode"]: "mock" or "jai".
    """

    def __init__(self, config: dict) -> None:
        self._cfg      = config
        self._mode     = config.get("mode", "mock")
        self._backend  = None
        self._frame_idx = 0

    def connect(self) -> bool:
        """Connect to camera. Returns True on success."""
        if self._mode == "jai":
            self._backend = JAICamera(self._cfg)
            try:
                ok = self._backend.connect()
            except Exception as e:
                log.error("JAICamera.connect() raised unexpected error: %s", e)
                ok = False
            if not ok:
                log.warning("JAICamera failed — falling back to mock mode")
                self._mode    = "mock"
                self._backend = None
            else:
                return True


        # Mock backend
        log.info("MockCamera: connected (synthetic frames)")
        self._backend = None   # mock uses inline generation
        return True

    def disconnect(self) -> None:
        if self._backend and self._mode == "jai":
            self._backend.disconnect()
        self._backend = None
        log.info("Camera disconnected")

    def grab(self) -> Optional[FrameTriplet]:
        """Grab next synchronized frame triplet."""
        if self._mode == "jai" and self._backend:
            return self._backend.grab()
        return self._mock_frame()

    def _mock_frame(self) -> FrameTriplet:
        """Animated apple blob moving across frame — informative mock for development."""
        H, W = 480, 640    # display-sized mock (no need to generate 2048×1536 of noise)
        fps  = self._cfg.get("mock", {}).get("fps", 30)

        t      = self._frame_idx / fps
        period = 3.0
        phase  = (t % period) / period
        cx     = int(phase * (W + 140)) - 70
        cy     = H // 2 + int(18 * np.sin(t * 1.1))

        Y, X = np.mgrid[0:H, 0:W]
        ax, ay   = 52, 44
        dist_sq  = (X - cx) ** 2 / ax ** 2 + (Y - cy) ** 2 / ay ** 2
        inside   = (dist_sq < 1.0).astype(np.float32)
        glow     = np.clip(1.8 - dist_sq, 0.0, 1.0) * 0.35

        rng = np.random.default_rng(seed=self._frame_idx % 200)
        bg1 = rng.integers(18, 52, (H, W), dtype=np.uint8)
        bg2 = rng.integers(28, 62, (H, W), dtype=np.uint8)
        bg3 = rng.integers(10, 38, (H, W), dtype=np.uint8)

        # CH1: color apple (red tones) → BGR
        r_ch = np.clip(bg1.astype(np.float32) + inside * 180 + glow * 220, 0, 255).astype(np.uint8)
        g_ch = np.clip(bg1.astype(np.float32) + inside * 60  + glow * 80,  0, 255).astype(np.uint8)
        b_ch = np.clip(bg1.astype(np.float32) + inside * 40  + glow * 50,  0, 255).astype(np.uint8)
        ch1  = np.stack([b_ch, g_ch, r_ch], axis=2)   # BGR

        # CH2: NIR1 — diffuse glow (internal structure)
        ch2 = np.clip(bg2.astype(np.float32) + inside * 110 + glow * 160, 0, 255).astype(np.uint8)

        # CH3: NIR2 — dimmer, subtle contrast (water content)
        ch3 = np.clip(bg3.astype(np.float32) + inside * 80  + glow * 120, 0, 255).astype(np.uint8)

        self._frame_idx += 1
        time.sleep(1.0 / fps)

        return FrameTriplet(
            ch1       = ch1,
            ch2       = ch2,
            ch3       = ch3,
            timestamp = time.time(),
            frame_idx = self._frame_idx,
            block_id  = self._frame_idx,
        )


    def set_exposure(self, exposure_us: int) -> bool:
        """Delegate to JAICamera.set_exposure(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_exposure(exposure_us)
        log.debug("set_exposure: mock mode — ignored")
        return True

    def set_fps(self, fps: float) -> bool:
        """Delegate to JAICamera.set_fps(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_fps(fps)
        log.debug("set_fps: mock mode — ignored")
        return True

    def get_exposure(self) -> int:
        """Read actual ExposureTime from firmware. Returns -1 on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_exposure()
        return -1

    @property
    def mode(self) -> str:
        return self._mode
