"""
scripts/camera_live_view.py
============================
JAI FS-3200T — Live 3-channel synchronized viewer
Uses the same MultiSource pipeline pattern as camera_probe_jai.py.

Displays CH1 (Color), CH2 (NIR1), CH3 (NIR2) side-by-side in real time.
NIR channels are display-normalized (NORM_MINMAX) for visibility.

Controls:
  Q       → quit
  S       → save snapshot to scripts/probe_output/snapshot_<n>.png
  N       → toggle NIR normalization on/off (see raw vs normalized)

Usage:
  conda activate applegui
  python scripts/camera_live_view.py
"""

import time
import sys
from pathlib import Path

import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
BUFFER_COUNT  = 16
TIMEOUT_MS    = 500        # short timeout — we're in a live loop
DISPLAY_W     = 640        # width per channel panel
DISPLAY_H     = 480        # height per channel panel
OUT_DIR       = Path("scripts/probe_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── NIR display normalization ────────────────────────────────────────────────
# Root cause of darkness:  NORM_MINMAX uses pixel min and max.  Even a single
# hot pixel (value=255) while the scene is at DN 5–30 compresses everything
# to near-black on display.  Fix: use the 1st/99th percentile as black/white
# points — this is robust to outlier pixels.
#
# Root cause of flicker:   percentile values still fluctuate slightly frame to
# frame due to sensor noise.  Fix: apply a fast EMA (α=0.30) on the range so
# the normalisation window settles in ~3 frames but tracks real changes.
#
# Tunable:
NIR_EMA   = 0.30   # EMA speed: 0=frozen, 1=per-frame  (0.30 → settles in ~3 frames)
NIR_LO_P  = 1      # percentile used as black point
NIR_HI_P  = 99     # percentile used as white point
NIR_MIN_SPAN = 6   # minimum DN range; below this falls back to full min/max
# Per-source running range state (initialised on first frame)
_nir_lo = [None, None, None]
_nir_hi = [None, None, None]

# ── Import eBUS ───────────────────────────────────────────────────────────────
print("\n── eBUS Import ───────────────────────────────────────────────────────")
try:
    import eBUS as eb
    print("  ✅  import eBUS OK")
except ImportError as e:
    print(f"  ❌  {e}")
    sys.exit(1)

# ── Helper ────────────────────────────────────────────────────────────────────
def get_p(nm, name):
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

# ── Source class (same as probe) ──────────────────────────────────────────────
class Source:
    BUFFER_COUNT = 16

    def __init__(self, device, connection_id, source_name, ch_index):
        self.device         = device
        self.connection_id  = connection_id
        self.source_name    = source_name
        self.ch_index       = ch_index
        self.stream         = None
        self.pipeline       = None
        self.source_channel = 0
        self.pixel_format   = "Mono8"

    def open(self):
        nm = self.device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self.source_name)

        result, self.source_channel = nm.GetIntegerValue("SourceIDValue")
        if result.IsFailure():
            result, self.source_channel = nm.GetIntegerValue("SourceStreamChannel")
        if result.IsFailure():
            self.source_channel = self.ch_index

        self.pixel_format = get_p(nm, "PixelFormat")
        w = get_p(nm, "Width")
        h = get_p(nm, "Height")
        print(f"  {self.source_name}  ch_id={self.source_channel}  "
              f"fmt={self.pixel_format}  {w}×{h}")

        self.stream = eb.PvStreamGEV()
        r = self.stream.Open(self.connection_id, 0, self.source_channel)
        if r.IsFailure():
            print(f"  ❌  {self.source_name}: stream open failed: {r.GetCodeString()}")
            return False

        lip = self.stream.GetLocalIPAddress()
        lp  = self.stream.GetLocalPort()
        self.device.SetStreamDestination(lip, lp, self.source_channel)
        print(f"       → {lip}:{lp}")

        payload_size = self.device.GetPayloadSize()
        self.pipeline = eb.PvPipeline(self.stream)
        self.pipeline.SetBufferSize(payload_size)
        self.pipeline.SetBufferCount(self.BUFFER_COUNT)
        self.pipeline.Start()
        return True

    def start_acquisition(self):
        nm    = self.device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self.source_name)
        self.device.StreamEnable()
        nm.Get("AcquisitionStart").Execute()

    def stop_acquisition(self):
        nm    = self.device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self.source_name)
        nm.Get("AcquisitionStop").Execute()
        self.device.StreamDisable()

    def grab(self):
        """Grab latest frame. Returns (img_bgr, block_id) or (None, None)."""
        result, buffer, op_result = self.pipeline.RetrieveNextBuffer(TIMEOUT_MS)
        if result.IsFailure() or not op_result.IsOK():
            return None, None

        image    = buffer.GetImage()
        raw      = image.GetDataPointer().copy()
        block_id = buffer.GetBlockID()
        self.pipeline.ReleaseBuffer(buffer)
        return raw, block_id

    def close(self):
        if self.pipeline:
            self.pipeline.Stop()
        if self.stream:
            self.stream.Close()


# ── 1. Discover & connect ─────────────────────────────────────────────────────
print("\n── Device Discovery ──────────────────────────────────────────────────")
sys_obj = eb.PvSystem()
sys_obj.Find()

connection_id = None
for i in range(sys_obj.GetInterfaceCount()):
    iface = sys_obj.GetInterface(i)
    for j in range(iface.GetDeviceCount()):
        dev = iface.GetDeviceInfo(j)
        print(f"  Found: {dev.GetDisplayID()}")
        if connection_id is None:
            connection_id = dev.GetConnectionID()

if connection_id is None:
    print("  ❌  No camera found.")
    sys.exit(1)

print("\n── Connect ───────────────────────────────────────────────────────────")
result, device = eb.PvDevice.CreateAndConnect(connection_id)
if device is None:
    print(f"  ❌  {result.GetCodeString()} — Close eBUS Player first")
    sys.exit(1)
print(f"  ✅  Connected  (GEV: {isinstance(device, eb.PvDeviceGEV)})")

if isinstance(device, eb.PvDeviceGEV):
    r = device.NegotiatePacketSize()
    print(f"  NegotiatePacketSize: {r.GetCodeString()}")

# ── Read current AcquisitionFrameRate (info only — do NOT set Enable flag) ───
# NOTE: Setting AcquisitionFrameRateEnable=True disables the JAI hardware
# multi-source sync mechanism → SYNC!! and unequal blockIDs.
# Frame rate must be configured in eBUS Player or JAI SDK before running.
print("\n── Camera Frame Rate (read-only) ─────────────────────────────────────")
try:
    _nm = device.GetParameters()
    _fr = _nm.Get("AcquisitionFrameRate")
    if _fr is not None:
        _r, _cur = _fr.GetValue()
        print(f"  AcquisitionFrameRate = {_cur:.2f} fps")
        if _cur < 25:
            print(f"  ⚠️  Frame rate is {_cur:.1f} fps — expected 30.")
            print("      → Open eBUS Player → connect → set AcquisitionFrameRate=30 → close.")
            print("      → The register value persists for the power session.")
    else:
        print("  AcquisitionFrameRate parameter not found")
except Exception as _e:
    print(f"  Could not read frame rate: {_e}")

# ── 2. Enumerate & open streams ───────────────────────────────────────────────
print("\n── Open Streams ──────────────────────────────────────────────────────")
nm = device.GetParameters()
source_selector = nm.GetEnum("SourceSelector")
source_names = []
if source_selector:
    result, count = source_selector.GetEntriesCount()
    for i in range(count):
        result, entry = source_selector.GetEntryByIndex(i)
        if entry:
            result, name = entry.GetName()
            source_names.append(name)

sources = []
for ch_idx, src_name in enumerate(source_names):
    src = Source(device, connection_id, src_name, ch_idx)
    if src.open():
        sources.append(src)

if not sources:
    print("  ❌  No sources opened.")
    sys.exit(1)
print(f"  ✅  {len(sources)} streams open simultaneously")

# ── 3. Start acquisition ──────────────────────────────────────────────────────
print("\n── Start Acquisition ─────────────────────────────────────────────────")
for src in sources:
    src.start_acquisition()
print(f"  ✅  All sources streaming — warming up 2s …")
time.sleep(2.0)

# Drain each source's pipeline to empty individually.
# Round-robin draining (the old approach) does not guarantee each buffer is
# fully cleared — stale extra frames in one source cause blockID mismatch
# (SYNC !!) from the very first displayed frame.
print("  Draining pipeline buffers …")
for src in sources:
    n = 0
    while True:
        r, buf, op = src.pipeline.RetrieveNextBuffer(50)  # 50 ms timeout
        if r.IsFailure():
            break
        src.pipeline.ReleaseBuffer(buf)
        n += 1
    print(f"    {src.source_name}: drained {n} frames")

print("  ✅  Ready — opening live view window")
print("       Q = quit  |  S = save snapshot  |  N = toggle NIR normalize\n")

# ── 4. Live display loop ──────────────────────────────────────────────────────
LABELS    = ["CH1  Color (BayerRG8)", "CH2  NIR1 (Mono8)", "CH3  NIR2 (Mono8)"]
FONT      = cv2.FONT_HERSHEY_SIMPLEX
normalize_nir = True     # toggle with N
snapshot_n    = 0
fps_times     = []
last_frames   = [None, None, None]   # keep last good frame per source

try:
    while True:
        t0 = time.perf_counter()

        panels = []
        block_ids = []

        for i, src in enumerate(sources):
            raw, bid = src.grab()

            if raw is not None:
                last_frames[i] = (raw, bid)

            if last_frames[i] is None:
                # No frame yet — show black panel
                panels.append(np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8))
                block_ids.append(None)
                continue

            raw, bid = last_frames[i]
            block_ids.append(bid)

            # Convert to BGR display image
            if src.pixel_format == "BayerRG8":
                img_bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
            else:
                # ── NIR: percentile-clip + fast EMA on range ──────────────
                # NORM_MINMAX (min/max) is destroyed by even a single hot
                # pixel: if max=255 but scene is at DN 5-30, everything maps
                # to near-black.  Using 1st/99th percentile excludes outliers.
                # Flicker comes from the range itself jumping frame-to-frame;
                # we smooth it with a fast EMA (α=0.30, settles in ~3 frames).
                if normalize_nir:
                    p_lo = float(np.percentile(raw, NIR_LO_P))
                    p_hi = float(np.percentile(raw, NIR_HI_P))
                    if p_hi - p_lo < NIR_MIN_SPAN:   # very uniform scene
                        p_lo = float(raw.min())       # fall back to full range
                        p_hi = float(raw.max())
                    if p_hi <= p_lo:
                        p_hi = p_lo + 1.0
                    if _nir_lo[i] is None:            # first frame: initialise
                        _nir_lo[i] = p_lo
                        _nir_hi[i] = p_hi
                    else:                             # subsequent: EMA smooth
                        _nir_lo[i] += NIR_EMA * (p_lo - _nir_lo[i])
                        _nir_hi[i] += NIR_EMA * (p_hi - _nir_hi[i])
                    span = max(_nir_hi[i] - _nir_lo[i], 1.0)
                    normed = np.clip(
                        (raw.astype(np.float32) - _nir_lo[i]) / span * 255,
                        0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

            # Resize to display panel
            panel = cv2.resize(img_bgr, (DISPLAY_W, DISPLAY_H))

            # Overlay label
            cv2.rectangle(panel, (0, 0), (DISPLAY_W, 28), (30, 30, 30), -1)
            cv2.putText(panel, LABELS[i], (6, 20), FONT, 0.55, (200, 200, 200), 1)

            # BlockID badge
            if bid is not None:
                bid_txt = f"blockID={bid}"
                cv2.putText(panel, bid_txt, (DISPLAY_W - 130, 20),
                            FONT, 0.45, (100, 255, 100), 1)

            panels.append(panel)

        # Sync indicator — show if all blockIDs match
        synced = (len(set(b for b in block_ids if b is not None)) == 1
                  and None not in block_ids)

        # Compose side-by-side display
        display = np.hstack(panels)

        # FPS counter
        fps_times.append(time.perf_counter())
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0

        # Status bar at bottom
        bar_y = DISPLAY_H - 28
        cv2.rectangle(display, (0, bar_y), (DISPLAY_W * len(panels), DISPLAY_H),
                      (20, 20, 20), -1)
        sync_txt  = "SYNC OK" if synced else "SYNC !!"
        sync_col  = (0, 220, 0) if synced else (0, 60, 220)
        norm_txt  = "NIR: normalized" if normalize_nir else "NIR: raw"
        status    = f"FPS: {fps:.1f}  |  {sync_txt}  |  {norm_txt}  |  [Q] quit  [S] snap  [N] toggle NIR"
        cv2.putText(display, sync_txt, (10, DISPLAY_H - 8), FONT, 0.55, sync_col, 1)
        cv2.putText(display, f"FPS: {fps:.1f}", (90, DISPLAY_H - 8), FONT, 0.55, (180, 180, 180), 1)
        cv2.putText(display, norm_txt, (200, DISPLAY_H - 8), FONT, 0.5, (140, 200, 255), 1)
        cv2.putText(display, "[Q]quit  [S]snap  [N]NIR toggle",
                    (DISPLAY_W * len(panels) - 340, DISPLAY_H - 8),
                    FONT, 0.45, (120, 120, 120), 1)

        cv2.imshow("JAI FS-3200T — Live 3-Channel View", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC
            break
        elif key == ord('s'):
            snap_path = OUT_DIR / f"snapshot_{snapshot_n:03d}.png"
            cv2.imwrite(str(snap_path), display)
            snapshot_n += 1
            print(f"  📸  Snapshot saved: {snap_path}")
        elif key == ord('n'):
            normalize_nir = not normalize_nir
            print(f"  NIR normalization: {'ON' if normalize_nir else 'OFF'}")

finally:
    print("\n── Shutting down ─────────────────────────────────────────────────────")
    cv2.destroyAllWindows()
    for src in sources:
        src.stop_acquisition()
    for src in sources:
        src.close()
    device.Disconnect()
    eb.PvDevice.Free(device)
    print("  ✅  Disconnected cleanly")
