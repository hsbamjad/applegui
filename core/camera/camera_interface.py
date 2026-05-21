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
        # Saved WB ratios for Revert (set just before One-Push AWB is triggered)
        self._saved_wb: Optional[tuple[float, float, float]] = None

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

        Block ID validation: all 3 sources must report the same GEV block ID
        for a triplet to be considered synchronized. If they diverge (e.g.
        after an ROI stop/start), the loop self-corrects by advancing the
        lagging source(s) until they catch up to the most advanced block ID.
        This recovers sync within 1-3 frames without requiring a full restart.
        """
        self._grab_fps_t = time.time()
        try:
            while self._running:
                raws = []
                bids = []
                ok   = True

                for src in self._sources:
                    raw, bid = src.grab(timeout_ms=150)
                    if raw is None:
                        ok = False
                        break
                    raws.append((raw, src.pixel_format, bid))
                    bids.append(bid)

                if not ok or len(raws) < len(self._sources):
                    continue

                # ── Block ID validation ────────────────────────────────────
                # All 3 bids must match. After ROI restart, sources may be
                # 1-2 frames out of phase — advance lagging ones to recover.
                if not (bids[0] == bids[1] == bids[2]):
                    max_bid  = max(bids)
                    log.debug("Sync mismatch bids=%s — re-syncing to bid=%d",
                              bids, max_bid)
                    for i, src in enumerate(self._sources):
                        attempts = 0
                        while bids[i] < max_bid and self._running and attempts < 16:
                            raw, bid = src.grab(timeout_ms=150)
                            if raw is None:
                                ok = False
                                break
                            raws[i] = (raw, src.pixel_format, bid)
                            bids[i] = bid
                            attempts += 1
                        if not ok:
                            break

                    if not ok or not (bids[0] == bids[1] == bids[2]):
                        # Could not recover — skip and try next iteration
                        log.debug("Sync recovery incomplete bids=%s — skipping", bids)
                        continue

                    log.debug("Sync recovered — bids=%s", bids)

                block_id = bids[0]

                # ── Track actual camera FPS ────────────────────────────────
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



    # ── White Balance helpers (Source0 / Color CH1 only) ──────────────────────

    def _get_wb_selector_names(self, nm) -> dict:
        """
        Enumerate GainSelector entries on the currently-scoped source and
        return a dict mapping role → entry_name for the WB channels.

        The JAI FS-3200T may name these entries:
          "Red" / "Green" / "Blue"   or
          "DigitalRed" / "DigitalGreen" / "DigitalBlue"  etc.

        We match by checking whether the entry name contains 'red', 'green',
        or 'blue' (case-insensitive). First match per role wins.

        Returns e.g. {'red': 'Red', 'green': 'Green', 'blue': 'Blue'}
        Returns empty dict if GainSelector is not available.
        """
        result: dict = {}
        gs = nm.GetEnum("GainSelector")
        if gs is None:
            return result
        try:
            _, count = gs.GetEntriesCount()
            for i in range(count):
                _, entry = gs.GetEntryByIndex(i)
                if entry is None:
                    continue
                _, name = entry.GetName()
                lower = name.lower()
                if "red" in lower and "red" not in result:
                    result["red"] = name
                elif "green" in lower and "green" not in result:
                    result["green"] = name
                elif "blue" in lower and "blue" not in result:
                    result["blue"] = name
        except Exception as e:
            log.warning("_get_wb_selector_names: %s", e)
        return result

    def get_white_balance_ratios(self) -> tuple:
        """
        Read current R/G/B WB ratios from Source0 (Color CH1) firmware registers.
        Enumerates GainSelector on Source0 and reads the Gain float for each
        of the Red / Green / Blue selectors.
        Returns (r, g, b) as floats. Returns (1.0, 1.0, 1.0) on any failure.
        """
        if self._device is None or not self._sources:
            return (1.0, 1.0, 1.0)
        try:
            import eBUS as eb
            nm    = self._device.GetParameters()
            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            wb_names = self._get_wb_selector_names(nm)
            if not wb_names:
                # Log all available GainSelector entries to help diagnose firmware naming
                gs_probe = nm.GetEnum("GainSelector")
                if gs_probe:
                    try:
                        _, cnt = gs_probe.GetEntriesCount()
                        all_names = []
                        for i in range(cnt):
                            _, e = gs_probe.GetEntryByIndex(i)
                            if e:
                                _, n = e.GetName()
                                all_names.append(n)
                        log.warning(
                            "get_white_balance_ratios: no R/G/B entries found. "
                            "All GainSelector entries on %s: %s",
                            self._sources[0]._source_name, all_names
                        )
                    except Exception as probe_e:
                        log.warning("get_white_balance_ratios: GainSelector probe failed: %s", probe_e)
                else:
                    log.warning("get_white_balance_ratios: GainSelector parameter not found on %s",
                                self._sources[0]._source_name)
                return (1.0, 1.0, 1.0)

            gs = nm.GetEnum("GainSelector")
            ratios: dict = {"red": 1.0, "green": 1.0, "blue": 1.0}
            for role, sel_name in wb_names.items():
                r_sel = gs.SetValue(sel_name)
                if not r_sel.IsOK():
                    log.warning("get_wb: GainSelector.SetValue(%s) failed", sel_name)
                    continue
                param = nm.GetFloat("Gain")
                if param is None:
                    continue
                _, val = param.GetValue()
                ratios[role] = float(val)

            log.info("WB readback Source0: R=%.4f G=%.4f B=%.4f",
                     ratios["red"], ratios["green"], ratios["blue"])
            return (ratios["red"], ratios["green"], ratios["blue"])
        except Exception as e:
            log.error("get_white_balance_ratios exception: %s", e)
            return (1.0, 1.0, 1.0)

    def set_white_balance_ratios(self, r: float, g: float, b: float) -> tuple:
        """
        Write explicit R/G/B WB ratios to Source0 GenICam registers.
        - Scopes to Source0 only via PvGenStateStack
        - Disables BalanceWhiteAuto first (sets to Off)
        - Enumerates GainSelector, writes Gain for Red/Green/Blue entries
        - Returns actual readback (r, g, b)
        """
        if self._device is None or not self._sources:
            return (r, g, b)
        try:
            import eBUS as eb
            nm    = self._device.GetParameters()
            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            # Disable auto WB before manual write
            bwa = nm.GetEnum("BalanceWhiteAuto")
            if bwa:
                bwa.SetValue("Off")

            wb_names = self._get_wb_selector_names(nm)
            if not wb_names:
                log.warning("set_white_balance_ratios: no R/G/B GainSelector entries found")
                return (r, g, b)

            gs      = nm.GetEnum("GainSelector")
            targets = {"red": r, "green": g, "blue": b}
            actuals: dict = {"red": r, "green": g, "blue": b}

            for role, sel_name in wb_names.items():
                r_sel = gs.SetValue(sel_name)
                if not r_sel.IsOK():
                    log.warning("set_wb: GainSelector.SetValue(%s) failed", sel_name)
                    continue
                param = nm.GetFloat("Gain")
                if param is None:
                    continue
                try:
                    _, g_min = param.GetMin()
                    _, g_max = param.GetMax()
                    clamped  = float(max(g_min, min(g_max, targets[role])))
                except Exception:
                    clamped = float(targets[role])
                r_write = param.SetValue(clamped)
                if r_write.IsOK():
                    _, v = param.GetValue()
                    actuals[role] = float(v)
                    print(f"[CAM] WB Source0 [{sel_name}]: "
                          f"req={targets[role]:.4f} actual={actuals[role]:.4f}")
                else:
                    log.warning("set_wb: Gain.SetValue(%s, %.4f) failed: %s",
                                sel_name, clamped, r_write.GetCodeString())

            log.info("WB written Source0: R=%.4f G=%.4f B=%.4f",
                     actuals["red"], actuals["green"], actuals["blue"])
            return (actuals["red"], actuals["green"], actuals["blue"])
        except Exception as e:
            log.error("set_white_balance_ratios exception: %s", e)
            return (r, g, b)

    def trigger_auto_white_balance(self) -> tuple:
        """
        Trigger One-Push Auto White Balance on Source0 (Color CH1 only).

        Steps:
          1. Save current R/G/B ratios as revert target (self._saved_wb)
          2. Set BalanceWhiteAuto = Once  (hardware calibration starts)
          3. Poll BalanceWhiteAuto every 50 ms until it returns 'Off'
             (firmware auto-reverts when calibration is complete)
          4. Read back resulting R/G/B ratios from firmware

        Returns:
          (success: bool, r: float, g: float, b: float)
        """
        if self._device is None or not self._sources:
            log.warning("trigger_auto_white_balance: no device / no sources")
            return (False, 1.0, 1.0, 1.0)
        try:
            import eBUS as eb

            # 1. Save current ratios as revert target — only on the FIRST AWB since
            #    the last Revert (or since startup).  If _saved_wb is already set it
            #    means the user ran AWB before without reverting; we must NOT overwrite
            #    the original snapshot or Revert would only undo back to the previous
            #    AWB result instead of the true pre-calibration baseline.
            if self._saved_wb is None:
                self._saved_wb = self.get_white_balance_ratios()
                log.info("AWB: saved pre-calibration WB = R=%.4f G=%.4f B=%.4f",
                         *self._saved_wb)
            else:
                log.info("AWB: _saved_wb already set (R=%.4f G=%.4f B=%.4f) — "
                         "keeping original snapshot for Revert", *self._saved_wb)

            nm    = self._device.GetParameters()
            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            bwa = nm.GetEnum("BalanceWhiteAuto")
            if bwa is None:
                log.error("AWB: BalanceWhiteAuto parameter not found on Source0")
                return (False, 1.0, 1.0, 1.0)

            # 2. Trigger One-Push
            r_set = bwa.SetValue("Once")
            if not r_set.IsOK():
                log.error("AWB: BalanceWhiteAuto.SetValue('Once') failed: %s",
                          r_set.GetCodeString())
                return (False, 1.0, 1.0, 1.0)
            log.info("AWB: BalanceWhiteAuto = Once — calibrating…")

            # 3. Poll until firmware reverts flag to 'Off' (max 3 s)
            deadline = time.time() + 3.0
            poll_interval = 0.05
            done = False
            while time.time() < deadline:
                time.sleep(poll_interval)
                try:
                    # PvGenEnum uses GetValueString() in the eBUS Python SDK
                    _, cur_str = bwa.GetValueString()
                except AttributeError:
                    try:
                        _, cur_str = bwa.GetValue()
                        cur_str = str(cur_str)
                    except Exception:
                        cur_str = ""
                cur_str = cur_str.lower()
                if "off" in cur_str or cur_str == "0":
                    done = True
                    break
                log.debug("AWB: BalanceWhiteAuto still = %s …", cur_str)

            if not done:
                log.warning("AWB: timed out waiting for BalanceWhiteAuto to return Off")

            # 4. Read back resulting ratios
            ratios = self.get_white_balance_ratios()
            log.info("AWB complete: R=%.4f G=%.4f B=%.4f", *ratios)
            return (True, ratios[0], ratios[1], ratios[2])

        except Exception as e:
            log.error("trigger_auto_white_balance exception: %s", e)
            return (False, 1.0, 1.0, 1.0)

    def revert_white_balance(self) -> tuple:
        """
        Restore the WB ratios saved before the last One-Push AWB calibration.
        Returns (success, r, g, b). Fails gracefully if no save point exists.
        Clears _saved_wb on success so the next AWB snapshots from the restored baseline.
        """
        if self._saved_wb is None:
            log.warning("revert_white_balance: no saved WB target — AWB not yet triggered")
            return (False, 1.0, 1.0, 1.0)
        r, g, b = self._saved_wb
        log.info("AWB revert: restoring R=%.4f G=%.4f B=%.4f", r, g, b)
        actual = self.set_white_balance_ratios(r, g, b)
        self._saved_wb = None   # reset snapshot so next AWB captures from the reverted baseline
        return (True, actual[0], actual[1], actual[2])

    # ── Black Level helpers (per-source, hardware pedestal) ────────────────────

    def _apply_black_level_loop(self, levels: list[float]) -> list[float]:
        """
        Write BlackLevel to each source independently.

        GenICam pattern:
          1. Scope to source via PvGenStateStack + SourceSelector
          2. Set BlackLevelSelector = 'All' (or first available selector)
          3. Write BlackLevel float value (firmware clamps to valid range)
          4. Read back actual value

        Returns list of actual readback values (one per source).
        """
        import eBUS as eb
        nm      = self._device.GetParameters()
        actuals = []

        for i, src in enumerate(self._sources):
            level = levels[i] if i < len(levels) else levels[0]

            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", src._source_name)

            # Set BlackLevelSelector — try 'All' first, then enumerate to find best entry
            bls = nm.GetEnum("BlackLevelSelector")
            if bls:
                # Try 'All' directly
                r_sel = bls.SetValue("All")
                if not r_sel.IsOK():
                    # Enumerate and use first available entry
                    try:
                        _, count = bls.GetEntriesCount()
                        for j in range(count):
                            _, entry = bls.GetEntryByIndex(j)
                            if entry:
                                _, name = entry.GetName()
                                r_try = bls.SetValue(name)
                                if r_try.IsOK():
                                    log.debug("BlackLevelSelector: using '%s' on %s",
                                              name, src._source_name)
                                    break
                    except Exception as e:
                        log.warning("BlackLevelSelector enumeration failed on %s: %s",
                                    src._source_name, e)

            param = nm.GetFloat("BlackLevel")
            if param is None:
                log.warning("BlackLevel parameter not found on %s", src._source_name)
                actuals.append(level)
                continue

            try:
                _, bl_min = param.GetMin()
                _, bl_max = param.GetMax()
                clamped   = float(max(bl_min, min(bl_max, level)))
            except Exception:
                clamped = float(level)

            r_write = param.SetValue(clamped)
            if r_write.IsOK():
                _, v = param.GetValue()
                actual = float(v)
                print(f"[CAM] BlackLevel {src._source_name}: "
                      f"req={level:.1f} actual={actual:.1f} DN")
            else:
                actual = level
                log.warning("BlackLevel.SetValue(%.1f) failed on %s: %s",
                            clamped, src._source_name, r_write.GetCodeString())

            actuals.append(actual)
            # stack destroyed → SourceSelector reverts for next iteration

        return actuals

    def set_black_levels_per_source(self, levels: list[float]) -> list[float]:
        """
        Set independent BlackLevel (DN) per source while streaming.
        levels[0] → Source0 (Color / CH1)
        levels[1] → Source1 (NIR1  / CH2)
        levels[2] → Source2 (NIR2  / CH3)
        Returns list of actual firmware-accepted values.
        """
        if self._device is None:
            log.warning("set_black_levels_per_source: no device connected")
            return levels
        try:
            actuals = self._apply_black_level_loop(levels)
            log.info("Camera: BlackLevel per-source req=%s actual=%s", levels, actuals)
            return actuals
        except Exception as e:
            log.error("set_black_levels_per_source exception: %s", e)
            return levels

    def get_black_levels_per_source(self) -> list[float]:
        """
        Read current BlackLevel (DN) from ALL sources independently.
        Returns list of float values (one per source), or [] on failure.
        """
        if self._device is None:
            return []
        try:
            import eBUS as eb
            nm     = self._device.GetParameters()
            levels = []
            for src in self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)

                bls = nm.GetEnum("BlackLevelSelector")
                if bls:
                    r_sel, _ = bls.SetValue("All")
                    if not r_sel.IsOK():
                        try:
                            _, entry = bls.GetEntryByIndex(0)
                            if entry:
                                _, name = entry.GetName()
                                bls.SetValue(name)
                        except Exception:
                            pass

                param = nm.GetFloat("BlackLevel")
                if param:
                    _, val = param.GetValue()
                    levels.append(float(val))
                else:
                    levels.append(0.0)
            return levels
        except Exception as e:
            log.error("get_black_levels_per_source exception: %s", e)
            return []

    # ── ROI (Region of Interest) ───────────────────────────────────────────────

    def _align(self, value: int, step: int) -> int:
        """Round value DOWN to the nearest multiple of step (firmware alignment)."""
        return max(0, (value // max(step, 1)) * max(step, 1))

    def get_roi_limits(self) -> dict:
        """
        Read the absolute sensor limits from firmware.
        Returns dict with keys: max_width, max_height, offset_x_step,
        offset_y_step, width_step, height_step.
        Falls back to JAI FS-3200T defaults if parameters are unavailable.
        """
        defaults = {
            "max_width":      2048,
            "max_height":     1536,
            "offset_x_step":   16,   # confirmed: 200 → 192
            "offset_y_step":    8,   # confirmed:  10 →   8
            "width_step":      16,   # same physical constraint as OffsetX
            "height_step":      8,   # same physical constraint as OffsetY
        }
        if self._device is None:
            return defaults
        try:
            nm = self._device.GetParameters()
            result = dict(defaults)
            for key, param_name in [("max_width",  "WidthMax"),
                                     ("max_height", "HeightMax")]:
                p = nm.GetInteger(param_name)
                if p:
                    _, v = p.GetValue()
                    result[key] = int(v)
            # Step / increment from Width parameter
            for key, param_name in [("width_step",    "Width"),
                                     ("height_step",   "Height"),
                                     ("offset_x_step", "OffsetX"),
                                     ("offset_y_step", "OffsetY")]:
                p = nm.GetInteger(param_name)
                if p:
                    try:
                        _, inc = p.GetIncrement()
                        result[key] = max(1, int(inc))
                    except Exception:
                        pass
            return result
        except Exception as e:
            log.warning("get_roi_limits exception: %s — using defaults", e)
            return defaults

    def set_roi(
        self, offset_x: int, offset_y: int, width: int, height: int
    ) -> tuple:
        """
        Apply ROI to ALL sources (they share the same physical FOV).

        IMPORTANT: Width/Height/OffsetX/OffsetY are locked while streaming.
        We must stop acquisition on all sources, write the registers, then
        restart acquisition. This causes a ~0.5s frame gap — acceptable for
        a deliberate calibration action.

        Safe write order per source:
          1. Stop acquisition
          2. Reset offsets to 0 (so Width/Height can expand to full sensor)
          3. Set Width/Height to max (clear any previous crop)
          4. Set new OffsetX, OffsetY
          5. Set new Width, Height
          6. Restart acquisition

        Returns (actual_x, actual_y, actual_w, actual_h) read back from Source0.
        """
        if self._device is None:
            log.warning("set_roi: no device connected")
            return (offset_x, offset_y, width, height)
        try:
            import eBUS as eb
            limits = self.get_roi_limits()
            mw = limits["max_width"]
            mh = limits["max_height"]
            xs = limits["offset_x_step"]
            ys = limits["offset_y_step"]
            ws = limits["width_step"]
            hs = limits["height_step"]

            # Align all values to firmware step requirements
            ox = self._align(max(0, offset_x), xs)
            oy = self._align(max(0, offset_y), ys)
            w  = self._align(max(ws, min(width,  mw - ox)), ws)
            h  = self._align(max(hs, min(height, mh - oy)), hs)

            log.info("ROI: stopping acquisition to apply OffsetX=%d OffsetY=%d "
                     "Width=%d Height=%d …", ox, oy, w, h)

            # ── Step 1: Stop all acquisitions ──────────────────────────────
            for src in self._sources:
                try:
                    src.stop_acquisition()
                except Exception as e:
                    log.warning("ROI: stop_acquisition failed on %s: %s",
                                src._source_name, e)

            # ── Step 1b: Drain stale pipeline buffers ───────────────────────
            # After AcquisitionStop, up to BUFFER_COUNT (16) pre-ROI frames
            # remain queued in each source's PvPipeline. If not flushed they
            # mix with post-restart frames → block ID mismatch.
            log.info("ROI: draining pipeline buffers …")
            for src in self._sources:
                drained = 0
                while True:
                    r, buf, op = src.pipeline.RetrieveNextBuffer(20)  # 20 ms
                    if r.IsFailure():
                        break
                    src.pipeline.ReleaseBuffer(buf)
                    drained += 1
                log.debug("ROI: drained %d stale frames from %s",
                          drained, src._source_name)

            # ── Step 2: Write ROI to each source ───────────────────────────
            nm = self._device.GetParameters()
            for src in self._sources:
                stack = eb.PvGenStateStack(nm)
                stack.SetEnumValue("SourceSelector", src._source_name)

                p_ox = nm.GetInteger("OffsetX")
                p_oy = nm.GetInteger("OffsetY")
                p_w  = nm.GetInteger("Width")
                p_h  = nm.GetInteger("Height")

                if p_ox and p_oy and p_w and p_h:
                    # GenICam SFNC write order — CRITICAL:
                    # Constraint: OffsetX + Width  <= MaxWidth  (2048)
                    #             OffsetY + Height <= MaxHeight (1536)
                    #
                    # Step 1: Reset both offsets to 0 so size can expand freely
                    p_ox.SetValue(0)
                    p_oy.SetValue(0)
                    # Step 2: Reset size to full sensor (clear any previous crop)
                    p_w.SetValue(mw)
                    p_h.SetValue(mh)
                    # Step 3: Set target WIDTH and HEIGHT *first*
                    #   e.g. Width=1648 → 0+1648=1648 ≤ 2048 ✓
                    r_w = p_w.SetValue(w)
                    r_h = p_h.SetValue(h)
                    # Step 4: Now set OffsetX/OffsetY (size already reduced to fit)
                    #   e.g. OffsetX=400 → 400+1648=2048 ≤ 2048 ✓
                    r_ox = p_ox.SetValue(ox)
                    r_oy = p_oy.SetValue(oy)
                    if not r_w.IsOK():
                        log.warning("ROI: Width.SetValue(%d) failed on %s: %s",
                                    w, src._source_name, r_w.GetCodeString())
                    if not r_h.IsOK():
                        log.warning("ROI: Height.SetValue(%d) failed on %s: %s",
                                    h, src._source_name, r_h.GetCodeString())
                    if not r_ox.IsOK():
                        log.warning("ROI: OffsetX.SetValue(%d) failed on %s: %s",
                                    ox, src._source_name, r_ox.GetCodeString())
                    if not r_oy.IsOK():
                        log.warning("ROI: OffsetY.SetValue(%d) failed on %s: %s",
                                    oy, src._source_name, r_oy.GetCodeString())
                else:
                    log.warning("ROI: integer params not found on %s", src._source_name)
                # stack destroyed → SourceSelector reverts

            # ── Step 3: Restart all acquisitions ───────────────────────────
            for src in self._sources:
                try:
                    src.start_acquisition()
                except Exception as e:
                    log.warning("ROI: start_acquisition failed on %s: %s",
                                src._source_name, e)

            # ── Step 3b: Post-restart mini-drain ───────────────────────────
            # Sources restart sequentially so Source0 produces 1-2 frames
            # before Source2 even starts. Discard the first 5 frames from
            # each source so the grab loop begins with all sources at the
            # same frame number, avoiding block ID divergence.
            log.info("ROI: post-restart stabilisation drain …")
            time.sleep(0.05)   # 50 ms — let all sources begin transmitting
            for _ in range(5):
                for src in self._sources:
                    r, buf, op = src.pipeline.RetrieveNextBuffer(100)
                    if r.IsOK():
                        src.pipeline.ReleaseBuffer(buf)
            log.info("ROI: pipeline stabilised — resuming grab")

            # ── Step 4: Read back actual values from Source0 ────────────────
            actual = self.get_roi()
            log.info("ROI confirmed: OffsetX=%d OffsetY=%d Width=%d Height=%d",
                     *actual)
            print(f"[CAM] ROI applied: ({actual[0]}, {actual[1]}) "
                  f"{actual[2]}×{actual[3]} px")
            return actual

        except Exception as e:
            log.error("set_roi exception: %s", e)
            # Best-effort: try to restart acquisition if we stopped it
            try:
                for src in self._sources:
                    src.start_acquisition()
            except Exception:
                pass
            return (offset_x, offset_y, width, height)

    def get_roi(self) -> tuple:
        """
        Read current ROI from Source0.
        Returns (offset_x, offset_y, width, height) as ints.
        """
        if self._device is None:
            return (0, 0, 2048, 1536)
        try:
            import eBUS as eb
            nm    = self._device.GetParameters()
            stack = eb.PvGenStateStack(nm)
            stack.SetEnumValue("SourceSelector", self._sources[0]._source_name)

            vals = {}
            for name in ("OffsetX", "OffsetY", "Width", "Height"):
                p = nm.GetInteger(name)
                if p:
                    _, v = p.GetValue()
                    vals[name] = int(v)
                else:
                    vals[name] = {"OffsetX": 0, "OffsetY": 0,
                                  "Width": 2048, "Height": 1536}[name]
            return (vals["OffsetX"], vals["OffsetY"],
                    vals["Width"],   vals["Height"])
        except Exception as e:
            log.error("get_roi exception: %s", e)
            return (0, 0, 2048, 1536)

    def reset_roi(self) -> tuple:
        """
        Reset ROI to full sensor frame on all sources.
        Returns (0, 0, max_width, max_height).
        """
        limits = self.get_roi_limits()
        return self.set_roi(0, 0, limits["max_width"], limits["max_height"])

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

    def get_white_balance_ratios(self) -> tuple:
        """Read R/G/B WB ratios from Source0 firmware registers. Returns (1.0, 1.0, 1.0) in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_white_balance_ratios()
        return (1.0, 1.0, 1.0)

    def set_white_balance_ratios(self, r: float, g: float, b: float) -> tuple:
        """Write R/G/B WB ratios to Source0 GenICam registers. No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_white_balance_ratios(r, g, b)
        log.debug("set_white_balance_ratios: mock mode — ignored")
        return (r, g, b)

    def trigger_auto_white_balance(self) -> tuple:
        """Trigger One-Push AWB on Source0. Returns (success, r, g, b). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.trigger_auto_white_balance()
        log.debug("trigger_auto_white_balance: mock mode — no-op")
        return (False, 1.0, 1.0, 1.0)

    def revert_white_balance(self) -> tuple:
        """Restore pre-AWB WB ratios on Source0. Returns (success, r, g, b). No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.revert_white_balance()
        log.debug("revert_white_balance: mock mode — no-op")
        return (False, 1.0, 1.0, 1.0)

    def set_black_levels_per_source(self, levels: list[float]) -> list[float]:
        """Write BlackLevel (DN) to all 3 sources independently. No-op in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_black_levels_per_source(levels)
        log.debug("set_black_levels_per_source: mock mode — ignored")
        return levels

    def get_black_levels_per_source(self) -> list[float]:
        """Read BlackLevel (DN) from all 3 sources. Returns [] on failure or in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_black_levels_per_source()
        return []

    def get_roi_limits(self) -> dict:
        """Read sensor ROI limits from firmware. Returns defaults if in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_roi_limits()
        return {"max_width": 2048, "max_height": 1536,
                "offset_x_step": 4, "offset_y_step": 2,
                "width_step": 4, "height_step": 2}

    def set_roi(
        self, offset_x: int, offset_y: int, width: int, height: int
    ) -> tuple:
        """Apply ROI to all sources. Returns (actual_x, actual_y, actual_w, actual_h)."""
        if self._mode == "jai" and self._backend:
            return self._backend.set_roi(offset_x, offset_y, width, height)
        log.debug("set_roi: mock mode — no-op")
        return (offset_x, offset_y, width, height)

    def get_roi(self) -> tuple:
        """Read current ROI from Source0. Returns (x, y, w, h)."""
        if self._mode == "jai" and self._backend:
            return self._backend.get_roi()
        return (0, 0, 2048, 1536)

    def reset_roi(self) -> tuple:
        """Reset ROI to full frame. Returns (0, 0, max_w, max_h)."""
        if self._mode == "jai" and self._backend:
            return self._backend.reset_roi()
        log.debug("reset_roi: mock mode — no-op")
        return (0, 0, 2048, 1536)

    def grab_fps(self) -> float:
        """Actual camera acquisition FPS from grab thread. 0.0 in mock mode."""
        if self._mode == "jai" and self._backend:
            return self._backend.grab_fps
        return 0.0

    @property
    def mode(self) -> str:
        return self._mode

