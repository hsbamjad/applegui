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
        raw0, pf0, _ = raws[0]
        raw1, _,   _ = raws[1]
        raw2, _,   _ = raws[2]

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
        Set sensor exposure time in microseconds on ALL 3 sources while streaming.

        The JAI FS-3200T has INDEPENDENT ExposureTime per source (Color / NIR1 / NIR2).
        This method iterates every open source and writes the value via SourceSelector
        so that NIR channels are actually affected (previously only Source0 was written).

        Hardware constraint — FPS caps the maximum exposure:
          30 FPS  → max 33,333 µs   (1,000,000 / 30)
          60 FPS  → max 16,666 µs
          107 FPS → max  9,345 µs
        The camera firmware enforces this automatically; we log if value is clamped.

        Args:
            exposure_us: Desired exposure in microseconds. Range: 100–100,000.

        Returns:
            True if ALL sources accepted the value, False on any failure.
        """
        if self._device is None:
            log.warning("set_exposure: no device connected")
            return False
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            all_ok = True
            for src in self._sources:
                # Select source so ExposureAuto and ExposureTime target this sensor
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)

                ae = nm.GetEnum("ExposureAuto")
                if ae:
                    ae.SetValue("Off")

                param = nm.GetFloat("ExposureTime")
                if param is None:
                    log.error("set_exposure: ExposureTime not found for source %s",
                              src._source_name)
                    all_ok = False
                    continue

                r = param.SetValue(float(exposure_us))
                if r.IsOK():
                    _, actual = param.GetValue()
                    print(f"[CAM] set_exposure {src._source_name}: actual={actual:.0f} µs")
                    if abs(actual - exposure_us) > 50:
                        log.warning(
                            "set_exposure %s: requested %d µs, clamped to %.0f µs",
                            src._source_name, exposure_us, actual,
                        )
                else:
                    log.error("set_exposure %s: rejected %d µs — %s",
                              src._source_name, exposure_us, r.GetCodeString())
                    all_ok = False

            log.info("Camera: ExposureTime = %d µs applied to %d source(s)",
                     exposure_us, len(self._sources))
            return all_ok
        except Exception as e:
            print(f"[CAM] set_exposure: EXCEPTION: {e}")
            log.error("set_exposure exception: %s", e)
            return False

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

    def get_exposure(self) -> int:
        """
        Read the actual ExposureTime from Source0 (color channel).
        Uses SourceSelector to ensure a consistent read regardless of what the
        last set_exposure() loop left the selector pointing to.
        Returns -1 on failure.
        """
        if self._device is None:
            return -1
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)
            param = nm.GetFloat("ExposureTime")
            if param is None:
                return -1
            _, val = param.GetValue()
            return int(val)
        except Exception as e:
            log.error("get_exposure exception: %s", e)
            return -1


    def set_gain(self, gain_db: float) -> float:
        """
        Set gain on ALL 3 sources while streaming. Returns actual gain readback from
        Source0 (so the GUI spinbox can be synced to truth).

        Strategy — tries ALL known GainSelector values on each source:
          'DigitalAll' → digital amplifier stage (most cameras)
          'AnalogAll'  → analog stage before ADC (if present)
          No selector  → fallback if GainSelector enum doesn't exist

        This exhaustive approach ensures no hidden gain stage stays elevated when
        the user reduces gain (the root cause of the "can't reduce gain" bug).

        Range:   camera minimum (often 0.0 dB) – 24.0 dB
        Returns: actual dB value read back from firmware, or gain_db on failure.
        """
        if self._device is None:
            log.warning("set_gain: no device connected")
            return gain_db
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            actual_readback = gain_db

            for src in self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)

                # Always disable auto-gain first
                ag = nm.GetEnum("GainAuto")
                if ag:
                    ag.SetValue("Off")

                gs = nm.GetEnum("GainSelector")

                # IMPORTANT: only write to master '*All' selectors (DigitalAll,
                # AnalogAll, All). NEVER write to individual channel selectors
                # (Red, Green, Blue, NIR1, NIR2) — doing so destroys white balance
                # on the color channel and causes pink/green tint artifacts.
                selectors_to_try: list[str | None] = []
                if gs:
                    # Probe which entries this camera actually supports.
                    # ONLY keep master '*All' selectors — never Red/Green/Blue/NIR
                    # sub-channel selectors, which would destroy white balance.
                    try:
                        _, count = gs.GetEntriesCount()
                        for i in range(count):
                            _, entry = gs.GetEntryByIndex(i)
                            if entry:
                                _, name = entry.GetName()
                                if name.lower().endswith("all"):
                                    selectors_to_try.append(name)
                                    print(f"[CAM] GainSelector [{src._source_name}]: using '{name}'")
                                else:
                                    print(f"[CAM] GainSelector [{src._source_name}]: skipping '{name}' (per-channel)")
                    except Exception:
                        selectors_to_try = ["DigitalAll", "AnalogAll", "All"]
                    # If no '*All' selector found, write Gain without selector
                    if not selectors_to_try:
                        selectors_to_try = [None]
                else:
                    selectors_to_try = [None]  # no GainSelector — write Gain directly


                wrote_any = False
                for sel in selectors_to_try:
                    if sel is not None and gs:
                        r_sel = gs.SetValue(sel)
                        if not r_sel.IsOK():
                            continue   # this selector not valid on this source

                    param = nm.GetFloat("Gain")
                    if param is None:
                        continue

                    # Clamp to camera's own min/max to avoid rejection
                    try:
                        _, g_min = param.GetMin()
                        _, g_max = param.GetMax()
                        clamped = float(max(g_min, min(g_max, gain_db)))
                    except Exception:
                        clamped = float(gain_db)

                    r = param.SetValue(clamped)
                    if r.IsOK():
                        _, actual = param.GetValue()
                        print(f"[CAM] set_gain {src._source_name} [{sel}]: "
                              f"requested={gain_db:.1f} actual={actual:.1f} dB")
                        wrote_any = True
                        if src is self._sources[0] and sel == selectors_to_try[0]:
                            actual_readback = float(actual)
                    else:
                        print(f"[CAM] set_gain {src._source_name} [{sel}]: "
                              f"REJECTED {gain_db:.1f} dB — {r.GetCodeString()}")

                if not wrote_any:
                    log.error("set_gain: could not write gain on source %s",
                              src._source_name)

            log.info("Camera: Gain target=%.1f dB, readback=%.1f dB, %d source(s)",
                     gain_db, actual_readback, len(self._sources))
            return actual_readback
        except Exception as e:
            print(f"[CAM] set_gain: EXCEPTION: {e}")
            log.error("set_gain exception: %s", e)
            return gain_db

    def get_gain(self) -> float:
        """
        Read current Gain (dB) from Source0, trying all known GainSelector values.
        Returns the first successful readback, or -1.0 on failure.
        """
        if self._device is None:
            return -1.0
        try:
            import eBUS as eb
            nm = self._device.GetParameters()
            if self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)
            gs = nm.GetEnum("GainSelector")
            if gs:
                # Try DigitalAll first, fall back to first available entry
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
            if param is None:
                return -1.0
            _, val = param.GetValue()
            return float(val)
        except Exception as e:
            log.error("get_gain exception: %s", e)
            return -1.0

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

    def set_gain(self, gain_db: float) -> bool:
        """Delegate to JAICamera.set_gain(). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_gain(gain_db)
        log.debug("set_gain: mock mode — ignored")
        return True

    def get_exposure(self) -> int:
        """Read actual ExposureTime from firmware. Returns -1 on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_exposure()
        return -1

    def get_gain(self) -> float:
        """Read actual Gain (dB) from firmware. Returns -1.0 on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_gain()
        return -1.0

    def grab_fps(self) -> float:
        """Actual camera acquisition FPS from grab thread. 0.0 in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.grab_fps
        return 0.0

    @property
    def mode(self) -> str:
        return self._mode

