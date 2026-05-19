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
    ch1_bid:    int = -1      # Channel 1 Block ID
    ch2_bid:    int = -1      # Channel 2 Block ID
    ch3_bid:    int = -1      # Channel 3 Block ID



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

    def __init__(self, config: dict) -> None:
        self._cfg          = config
        self._device       = None
        self._sources: list[_JAISource] = []
        self._running      = False
        self._frame_idx    = 0
        self._latest: Optional[FrameTriplet] = None
        self._latest_lock  = threading.Lock()
        self._grab_thread: Optional[threading.Thread] = None
        # Camera-side FPS tracking (grab thread measures actual acquisition rate)
        self._grab_fps      = 0.0
        self._grab_count    = 0
        self._grab_fps_t    = 0.0

    @property
    def grab_fps(self) -> float:
        """Actual camera acquisition FPS measured in the background grab thread."""
        return self._grab_fps

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
        Runs in a daemon thread at camera acquisition speed.
        Measures actual grab FPS independently of display FPS.
        Always stores the latest processed triplet in self._latest.
        """
        self._grab_fps_t = time.time()
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

                # Track actual camera FPS
                self._grab_count += 1
                elapsed = time.time() - self._grab_fps_t
                if elapsed >= 1.0:
                    self._grab_fps   = self._grab_count / elapsed
                    self._grab_count = 0
                    self._grab_fps_t = time.time()

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
        """
        Convert raw sensor buffers into a FrameTriplet.

        NO downsampling — frames are returned at full sensor resolution (2048×1536).
        NO normalization — NIR pixel values are exactly what the sensor captured.
        The display widget (image_display.py) handles scaling via Qt SmoothTransformation.

        CH1: Bayer demosaic only (BayerRG8 → BGR). Full res.
        CH2: Raw Mono8 pass-through. Full res.
        CH3: Raw Mono8 pass-through. Full res.
        """
        raw0, pf0, bid0 = raws[0]
        raw1, _,   bid1 = raws[1]
        raw2, _,   bid2 = raws[2]

        # CH1 — Bayer demosaic at full resolution, no resize
        if pf0 == "BayerRG8":
            ch1 = cv2.cvtColor(raw0, cv2.COLOR_BayerBG2BGR)
        else:
            ch1 = raw0

        # CH2, CH3 — raw sensor values, untouched
        ch2 = raw1
        ch3 = raw2

        if self._frame_idx % 90 == 0:
            log.info(
                "NIR raw stats — CH2: min=%d max=%d  |  CH3: min=%d max=%d",
                int(ch2.min()), int(ch2.max()),
                int(ch3.min()), int(ch3.max()),
            )

        self._frame_idx += 1
        return FrameTriplet(
            ch1       = ch1,
            ch2       = ch2,
            ch3       = ch3,
            timestamp = time.time(),
            frame_idx = self._frame_idx,
            block_id  = block_id,
            ch1_bid   = bid0,
            ch2_bid   = bid1,
            ch3_bid   = bid2,
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

    def _write_exposure_direct_to_source(self, src, value_us: int) -> int:
        """Helper to write exposure time directly to one specific source."""
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", src._source_name)

            ae = nm.GetEnum("ExposureAuto")
            if ae:
                ae.SetValue("Off")

            param = nm.GetFloat("ExposureTime")
            if param is None:
                return -1

            r = param.SetValue(float(value_us))
            if r.IsOK():
                _, actual = param.GetValue()
                return int(actual)
            return -1
        except Exception:
            return -1

    def _apply_exposure_loop(self, target_exposures: list[int]) -> list[int]:
        """
        Ramps target exposures simultaneously and incrementally on all open sources.
        Pads shorter ramping paths to complete in unison, preventing sync loss.
        """
        currents = self.get_exposures_per_source()
        if not currents or len(currents) < len(self._sources):
            currents = [5000] * len(self._sources)

        max_step = 4000  # µs step size ceiling
        delay_s  = 0.03  # ~1 frame period at 30 FPS

        steps_list = []
        for i, src in enumerate(self._sources):
            tgt = target_exposures[i] if i < len(target_exposures) else target_exposures[0]
            curr = currents[i]

            ch_steps = []
            diff = tgt - curr
            if abs(diff) <= max_step:
                ch_steps = [tgt]
            else:
                c = curr
                if diff > 0:
                    while c < tgt:
                        c = min(c + max_step, tgt)
                        ch_steps.append(c)
                else:
                    while c > tgt:
                        c = max(c - max_step, tgt)
                        ch_steps.append(c)
            steps_list.append(ch_steps)

        max_steps_len = max(len(s) for s in steps_list)

        # Pad shorter sequences at the start so all ramps conclude at the same time
        for s in steps_list:
            while len(s) < max_steps_len:
                s.insert(0, s[0])

        # Execute synchronized steps
        actuals = [-1] * len(self._sources)
        for step_idx in range(max_steps_len):
            for i, src in enumerate(self._sources):
                val = steps_list[i][step_idx]
                act = self._write_exposure_direct_to_source(src, val)
                if step_idx == max_steps_len - 1:
                    actuals[i] = act
            if step_idx < max_steps_len - 1:
                time.sleep(delay_s)

        return actuals

    def set_exposure(self, exposure_us: int) -> int:
        """
        Set same exposure on ALL 3 sources. Returns actual readback from Source0.
        Used for global exposure controls and Reset.
        """
        if self._device is None:
            log.warning("set_exposure: no device connected")
            return exposure_us
        try:
            actuals = self._apply_exposure_loop([exposure_us] * len(self._sources))
            readback = actuals[0] if actuals else exposure_us
            log.info("Camera: ExposureTime = %d µs (all), readback = %d µs", exposure_us, readback)
            return readback
        except Exception as e:
            log.error("set_exposure exception: %s", e)
            return exposure_us

    def set_exposures_per_source(self, exposures: list[int]) -> list[int]:
        """
        Set independent exposures per source while streaming.
        exposures[0] → Source0 (Color / CH1)
        exposures[1] → Source1 (NIR1  / CH2)
        exposures[2] → Source2 (NIR2  / CH3)
        Returns list of actual readback values from firmware.
        """
        if self._device is None:
            log.warning("set_exposures_per_source: no device connected")
            return exposures
        try:
            actuals = self._apply_exposure_loop(exposures)
            log.info("Camera: Per-source exposure req=%s actual=%s", exposures, actuals)
            return actuals
        except Exception as e:
            log.error("set_exposures_per_source exception: %s", e)
            return exposures

    def get_exposure(self) -> int:
        """Read actual ExposureTime from Source0. Returns -1 on failure."""
        if self._device is None:
            return -1
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)
            param = nm.GetFloat("ExposureTime")
            if param:
                _, val = param.GetValue()
                return int(val)
            return -1
        except Exception as e:
            log.error("get_exposure exception: %s", e)
            return -1

    def get_exposures_per_source(self) -> list[int]:
        """
        Read current ExposureTime (µs) from ALL sources independently.
        Returns list of microsecond values (one per source), or [] on failure.
        """
        if self._device is None:
            return []
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            exposures = []
            for src in self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)
                param = nm.GetFloat("ExposureTime")
                if param:
                    _, val = param.GetValue()
                    exposures.append(int(val))
                else:
                    exposures.append(-1)
            return exposures
        except Exception as e:
            log.error("get_exposures_per_source exception: %s", e)
            return []

    def set_fps(self, fps: float) -> bool:
        """
        Set acquisition frame rate while streaming.

        Changing FPS has one important side-effect:
          Max allowed exposure shrinks: max_exp_us = 1,000,000 / fps
          (firmware enforces this — any existing exposure above the new limit
           will be silently clamped by the camera).

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

    def auto_white_balance(self) -> bool:
        """
        Executes a One-Push Auto White Balance on the visible color channel (Source0).
        Temporarily sets SourceSelector to the color channel, activates 'Once' auto WB,
        and waits up to 2 seconds for the camera firmware to finish calibrating.
        """
        if self._device is None:
            log.warning("auto_white_balance: no device connected")
            return False
        try:
            import eBUS as eb
            nm = self._device.GetParameters()

            # 1. Point SourceSelector to Source0 (the color channel)
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            # 2. Get the BalanceWhiteAuto parameter
            param = nm.GetEnum("BalanceWhiteAuto")
            if param is None:
                log.error("auto_white_balance: BalanceWhiteAuto parameter not found on device")
                return False

            # 3. Trigger One-Push Auto White Balance
            log.info("Triggering One-Push Auto White Balance on Source0...")
            r = param.SetValue("Once")
            if not r.IsOK():
                log.error("auto_white_balance failed to write 'Once': %s", r.GetCodeString())
                return False

            # 4. Wait for JAI firmware calibration to stabilize (value returns to 'Off')
            import time
            start_time = time.time()
            success = False
            while time.time() - start_time < 3.0:
                time.sleep(0.1)
                # Correct eBUS API usage: GetValueString returns (PvResult, str)
                r_status, val = param.GetValueString()
                if r_status.IsOK() and val == "Off":
                    success = True
                    break

            if success:
                log.info("Auto White Balance calibration successful!")
                r_ratio, b_ratio = self.get_white_balance_ratios()
                log.info("Readback ratios after AWB — Red: %.2f | Blue: %.2f", r_ratio, b_ratio)
                return True
            else:
                log.warning("Auto White Balance timed out (calibration still in progress)")
                return False
        except Exception as e:
            log.error("auto_white_balance exception: %s", e)
            return False

    def get_white_balance_ratios(self) -> tuple[float, float]:
        """
        Read current Red and Blue BalanceRatio values from Source0.
        Falls back to GainSelector -> Red/Blue under Source0 if BalanceRatio is absent.
        """
        if self._device is None:
            return 1.0, 1.0
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            r_ratio, b_ratio = 1.0, 1.0
            ratio_sel = nm.GetEnum("BalanceRatioSelector")
            ratio_val = nm.GetFloat("BalanceRatio")
            if ratio_sel and ratio_val:
                ratio_sel.SetValue("Red")
                _, r_ratio = ratio_val.GetValue()
                ratio_sel.SetValue("Blue")
                _, b_ratio = ratio_val.GetValue()
            else:
                # JAI FS-3200T fallback: uses Red/Blue GainSelector sub-channels
                gs = nm.GetEnum("GainSelector")
                g_val = nm.GetFloat("Gain")
                if gs and g_val:
                    gs.SetValue("Red")
                    _, r_ratio = g_val.GetValue()
                    gs.SetValue("Blue")
                    _, b_ratio = g_val.GetValue()
                    log.info("[CAM AWB] Read white balance using GainSelector Red/Blue: %.2f / %.2f", r_ratio, b_ratio)
            return r_ratio, b_ratio
        except Exception as e:
            log.error("get_white_balance_ratios exception: %s", e)
            return 1.0, 1.0

    def set_white_balance_ratios(self, r_ratio: float, b_ratio: float) -> bool:
        """
        Manually write specific Red and Blue BalanceRatio values to Source0.
        Falls back to GainSelector -> Red/Blue under Source0 if BalanceRatio is absent.
        """
        if self._device is None:
            return False
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            # Ensure Auto WB is turned OFF before setting manual ratios
            param = nm.GetEnum("BalanceWhiteAuto")
            if param:
                res = param.SetValue("Off")
                log.info("[CAM AWB] Set BalanceWhiteAuto='Off' result: %s", res.GetCodeString())

            ratio_sel = nm.GetEnum("BalanceRatioSelector")
            ratio_val = nm.GetFloat("BalanceRatio")
            
            if ratio_sel and ratio_val:
                # Standard path
                r1 = ratio_sel.SetValue("Red")
                r2 = ratio_val.SetValue(float(r_ratio))
                log.info("[CAM AWB] Set Red Selector result: %s | Set Red Ratio %.3f result: %s", 
                         r1.GetCodeString(), r_ratio, r2.GetCodeString())

                r3 = ratio_sel.SetValue("Blue")
                r4 = ratio_val.SetValue(float(b_ratio))
                log.info("[CAM AWB] Set Blue Selector result: %s | Set Blue Ratio %.3f result: %s", 
                         r3.GetCodeString(), b_ratio, r4.GetCodeString())

                # Verification Readback
                ratio_sel.SetValue("Red")
                _, verified_r = ratio_val.GetValue()
                ratio_sel.SetValue("Blue")
                _, verified_b = ratio_val.GetValue()
                log.info("[CAM AWB] Verify Readback — Red: %.3f (target: %.3f) | Blue: %.3f (target: %.3f)",
                         verified_r, r_ratio, verified_b, b_ratio)

                return r2.IsOK() and r4.IsOK()
            else:
                # JAI FS-3200T fallback: uses Red/Blue GainSelector sub-channels
                gs = nm.GetEnum("GainSelector")
                g_val = nm.GetFloat("Gain")
                if gs and g_val:
                    r1 = gs.SetValue("Red")
                    r2 = g_val.SetValue(float(r_ratio))
                    log.info("[CAM AWB] Set GainSelector='Red' result: %s | Set Red Gain %.2f result: %s",
                             r1.GetCodeString(), r_ratio, r2.GetCodeString())
                    
                    r3 = gs.SetValue("Blue")
                    r4 = g_val.SetValue(float(b_ratio))
                    log.info("[CAM AWB] Set GainSelector='Blue' result: %s | Set Blue Gain %.2f result: %s",
                             r3.GetCodeString(), b_ratio, r4.GetCodeString())
                    
                    # Verification Readback
                    gs.SetValue("Red")
                    _, verified_r = g_val.GetValue()
                    gs.SetValue("Blue")
                    _, verified_b = g_val.GetValue()
                    log.info("[CAM AWB] Fallback Verify Readback — Red: %.2f (target: %.2f) | Blue: %.2f (target: %.2f)",
                             verified_r, r_ratio, verified_b, b_ratio)
                    return r2.IsOK() and r4.IsOK()
            log.error("[CAM AWB] No WB controls found (neither BalanceRatio nor GainSelector Red/Blue)")
            return False
        except Exception as e:
            log.error("set_white_balance_ratios exception: %s", e)
            return False


    # ── Gain helpers ──────────────────────────────────────────────────────────

    def _apply_gain_loop(self, gains_per_source: list[float]) -> list[float]:
        """
        Core gain-writing loop shared by set_gain() and set_gain_per_source().

        gains_per_source[i] is written to self._sources[i].
        Only writes to '*All' GainSelector entries — never Red/Green/Blue which
        would destroy white balance on the color channel.
        Returns list of actual readback values (same length as self._sources).
        """
        import eBUS as eb
        nm      = self._device.GetParameters()
        actuals = []

        for i, src in enumerate(self._sources):
            gain_db = gains_per_source[i] if i < len(gains_per_source) else gains_per_source[0]

            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", src._source_name)

            ag = nm.GetEnum("GainAuto")
            if ag:
                ag.SetValue("Off")

            gs = nm.GetEnum("GainSelector")
            selectors_to_try: list[str | None] = []
            if gs:
                try:
                    _, count = gs.GetEntriesCount()
                    for j in range(count):
                        _, entry = gs.GetEntryByIndex(j)
                        if entry:
                            _, name = entry.GetName()
                            if name.lower().endswith("all"):
                                selectors_to_try.append(name)
                            # else: skip Red/Green/Blue/NIR sub-channel selectors
                except Exception:
                    selectors_to_try = ["DigitalAll", "AnalogAll", "All"]
                if not selectors_to_try:
                    selectors_to_try = [None]
            else:
                selectors_to_try = [None]

            actual   = gain_db
            wrote    = False
            for sel in selectors_to_try:
                if sel is not None and gs:
                    r_sel = gs.SetValue(sel)
                    if not r_sel.IsOK():
                        continue

                param = nm.GetFloat("Gain")
                if param is None:
                    continue

                try:
                    _, g_min = param.GetMin()
                    _, g_max = param.GetMax()
                    clamped  = float(max(g_min, min(g_max, gain_db)))
                except Exception:
                    clamped = float(gain_db)

                r = param.SetValue(clamped)
                if r.IsOK():
                    _, v = param.GetValue()
                    actual = float(v)
                    print(f"[CAM] Gain {src._source_name} [{sel}]: "
                          f"req={gain_db:.1f} actual={actual:.1f} dB")
                    wrote = True
                else:
                    print(f"[CAM] Gain {src._source_name} [{sel}]: "
                          f"REJECTED {gain_db:.1f} dB — {r.GetCodeString()}")

            if not wrote:
                log.error("set_gain: could not write to source %s", src._source_name)

            actuals.append(actual)
            # stack destroyed here → SourceSelector reverts for next iteration

        return actuals

    def set_gain(self, gain_db: float) -> float:
        """
        Set same gain on ALL 3 sources. Returns actual readback from Source0.
        Used for global gain and Reset.
        """
        if self._device is None:
            log.warning("set_gain: no device connected")
            return gain_db
        try:
            actuals  = self._apply_gain_loop([gain_db] * len(self._sources))
            readback = actuals[0] if actuals else gain_db
            log.info("Camera: Gain=%.1f dB (all), readback=%.1f dB", gain_db, readback)
            return readback
        except Exception as e:
            log.error("set_gain exception: %s", e)
            return gain_db

    def set_gain_per_source(self, gains: list[float]) -> list[float]:
        """
        Set independent gain per source while streaming.
        gains[0] → Source0 (Color / CH1)
        gains[1] → Source1 (NIR1  / CH2)
        gains[2] → Source2 (NIR2  / CH3)
        Returns list of actual readback values from firmware.
        """
        if self._device is None:
            log.warning("set_gain_per_source: no device connected")
            return gains
        try:
            actuals = self._apply_gain_loop(gains)
            log.info("Camera: Per-source gain req=%s actual=%s",
                     [f"{g:.1f}" for g in gains],
                     [f"{a:.1f}" for a in actuals])
            return actuals
        except Exception as e:
            log.error("set_gain_per_source exception: %s", e)
            return gains

    def get_gains_per_source(self) -> list[float]:
        """
        Read current Gain (dB) from ALL sources independently.
        Returns list of dB values (one per source), or [] on failure.
        """
        if self._device is None:
            return []
        try:
            import eBUS as eb
            nm    = self._device.GetParameters()
            gains = []
            for src in self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)
                gs = nm.GetEnum("GainSelector")
                if gs:
                    r, _ = gs.SetValue("DigitalAll")
                    if not r.IsOK():
                        try:
                            _, entry = gs.GetEntryByIndex(0)
                            if entry:
                                _, name = entry.GetName()
                                gs.SetValue(name)
                        except Exception:
                            pass
                param = nm.GetFloat("Gain")
                if param:
                    _, val = param.GetValue()
                    gains.append(float(val))
                else:
                    gains.append(-1.0)
            return gains
        except Exception as e:
            log.error("get_gains_per_source exception: %s", e)
            return []



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
            ch1_bid   = self._frame_idx,
            ch2_bid   = self._frame_idx,
            ch3_bid   = self._frame_idx,
        )


    def set_exposure(self, exposure_us: int) -> int:
        """Delegate to JAICamera.set_exposure(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_exposure(exposure_us)
        log.debug("set_exposure: mock mode — ignored")
        return exposure_us

    def set_exposures_per_source(self, exposures: list[int]) -> list[int]:
        """Delegate to JAICamera.set_exposures_per_source(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_exposures_per_source(exposures)
        log.debug("set_exposures_per_source: mock mode — ignored")
        return exposures

    def set_fps(self, fps: float) -> bool:
        """Delegate to JAICamera.set_fps(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_fps(fps)
        log.debug("set_fps: mock mode — ignored")
        return True

    def set_gain(self, gain_db: float) -> float:
        """Delegate to JAICamera.set_gain(). Returns actual readback. No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_gain(gain_db)
        log.debug("set_gain: mock mode — ignored")
        return gain_db

    def set_gain_per_source(self, gains: list[float]) -> list[float]:
        """Delegate to JAICamera.set_gain_per_source(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_gain_per_source(gains)
        log.debug("set_gain_per_source: mock mode — ignored")
        return gains

    def get_exposure(self) -> int:
        """Read actual ExposureTime from firmware. Returns -1 on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_exposure()
        return -1

    def get_exposures_per_source(self) -> list[int]:
        """Read ExposureTime from all 3 sources independently. Returns [] on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_exposures_per_source()
        return []

    def get_gains_per_source(self) -> list[float]:
        """Read Gain (dB) from all 3 sources. Returns [] on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_gains_per_source()
        return []

    def auto_white_balance(self) -> bool:
        """Trigger One-Push Auto White Balance on the color channel (Source0). No-op in mock."""
        if self._mode == "jai" and self._backend:
            return self._backend.auto_white_balance()
        log.debug("auto_white_balance: mock mode — simulated success")
        return True

    def get_white_balance_ratios(self) -> tuple[float, float]:
        """Read current Red and Blue BalanceRatio values. Returns 1.0, 1.0 in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_white_balance_ratios()
        return 1.0, 1.0

    def set_white_balance_ratios(self, r_ratio: float, b_ratio: float) -> bool:
        """Manually write specific Red and Blue BalanceRatio values. No-op in mock."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_white_balance_ratios(r_ratio, b_ratio)
        log.debug("set_white_balance_ratios: mock mode — ignored")
        return True

    def grab_fps(self) -> float:
        """Actual camera acquisition FPS from grab thread. 0.0 in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.grab_fps
        return 0.0

    @property
    def mode(self) -> str:
        return self._mode

