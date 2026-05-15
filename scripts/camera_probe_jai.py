"""
scripts/camera_probe_jai.py
============================
JAI Camera Probe — eBUS Python API v6.6.1
Simultaneous 3-source acquisition using the official Pleora MultiSource pattern.

Key differences from single-stream approach:
  - 3 × PvStreamGEV opened simultaneously (one per source channel)
  - SetStreamDestination called with source_channel index (3rd arg)
  - PvPipeline per source (SetBufferSize + SetBufferCount + Start)
  - ALL streams opened BEFORE any AcquisitionStart → hardware-synchronized triplets
  - PvGenStateStack for clean per-source parameter context

Prerequisites:
  - eBUS SDK installed (JAI 64-bit)
  - pip install ebus_python-6.6.1-...-py310-none-win_amd64.whl
  - pip install "numpy<2"
  - eBUS Player must be CLOSED (exclusive device access)

Usage:
  conda activate applegui
  python scripts/camera_probe_jai.py
"""

import time
import sys
from pathlib import Path

import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
BUFFER_COUNT = 16
TIMEOUT_MS   = 5000
OUT_DIR      = Path("scripts/probe_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import eBUS ───────────────────────────────────────────────────────────────
print("\n── eBUS Import ───────────────────────────────────────────────────────")
try:
    import eBUS as eb
    print("  ✅  import eBUS OK")
except ImportError as e:
    print(f"  ❌  {e}")
    sys.exit(1)

# ── Helper: read any GenICam parameter ───────────────────────────────────────
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

# ── Source class — mirrors official MultiSource.py pattern ───────────────────
class Source:
    """One physical sensor on the FS-3200T. Each has its own stream + pipeline."""

    BUFFER_COUNT = 16

    def __init__(self, device, connection_id, source_name, ch_index):
        self.device        = device
        self.connection_id = connection_id
        self.source_name   = source_name   # e.g. "Source0"
        self.ch_index      = ch_index      # 0-based display index
        self.stream        = None
        self.pipeline      = None
        self.source_channel = 0            # integer channel ID from SourceIDValue

    def open(self):
        nm = self.device.GetParameters()

        # Use PvGenStateStack to temporarily select this source (auto-restored on scope exit)
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self.source_name)

        # Read the integer source channel ID (used to route the stream)
        result, self.source_channel = nm.GetIntegerValue("SourceIDValue")
        if result.IsFailure():
            result, self.source_channel = nm.GetIntegerValue("SourceStreamChannel")
        if result.IsFailure():
            print(f"  ⚠️  {self.source_name}: could not read SourceIDValue — defaulting to {self.ch_index}")
            self.source_channel = self.ch_index

        pf  = get_p(nm, "PixelFormat")
        w   = get_p(nm, "Width")
        h   = get_p(nm, "Height")
        print(f"  {self.source_name}  ch_id={self.source_channel}  "
              f"fmt={pf}  {w}×{h}")

        # Open a dedicated GEV stream for this source channel
        self.stream = eb.PvStreamGEV()
        r = self.stream.Open(self.connection_id, 0, self.source_channel)
        if r.IsFailure():
            print(f"  ❌  {self.source_name}: stream.Open failed: {r.GetCodeString()}")
            return False

        lip  = self.stream.GetLocalIPAddress()
        lp   = self.stream.GetLocalPort()
        # Route this source on the device to our stream's local address
        self.device.SetStreamDestination(lip, lp, self.source_channel)
        print(f"       → stream → {lip}:{lp}")


        # Set up pipeline (manages buffer pool internally)
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

    def retrieve_one(self):
        """Retrieve one frame from the pipeline. Returns (numpy_array, pixel_format_str)."""
        nm    = self.device.GetParameters()
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", self.source_name)
        pf = get_p(nm, "PixelFormat")

        result, buffer, op_result = self.pipeline.RetrieveNextBuffer(TIMEOUT_MS)
        if result.IsFailure():
            print(f"  ❌  {self.source_name}: RetrieveNextBuffer: {result.GetCodeString()}")
            return None, pf
        if not op_result.IsOK():
            print(f"  ⚠️  {self.source_name}: op_result: {op_result.GetCodeString()}")
            self.pipeline.ReleaseBuffer(buffer)
            return None, pf

        image    = buffer.GetImage()
        img_data = image.GetDataPointer().copy()   # copy before releasing buffer
        block_id = buffer.GetBlockID()
        self.pipeline.ReleaseBuffer(buffer)
        print(f"  ✅  {self.source_name}: {image.GetWidth()}×{image.GetHeight()}  "
              f"blockID={block_id}  min={img_data.min()} max={img_data.max()} "
              f"mean={img_data.mean():.1f}")
        return img_data, pf

    def close(self):
        if self.pipeline:
            self.pipeline.Stop()
        if self.stream:
            self.stream.Close()


# ── 1. Discover device ────────────────────────────────────────────────────────
print("\n── Device Discovery ──────────────────────────────────────────────────")
sys_obj = eb.PvSystem()
sys_obj.Find()

connection_id   = None
dev_info_cached = None
for i in range(sys_obj.GetInterfaceCount()):
    iface = sys_obj.GetInterface(i)
    for j in range(iface.GetDeviceCount()):
        dev = iface.GetDeviceInfo(j)
        print(f"  Found: {dev.GetDisplayID()}")
        if connection_id is None:
            connection_id   = dev.GetConnectionID()
            dev_info_cached = dev

if connection_id is None:
    print("  ❌  No camera found.")
    sys.exit(1)
print(f"  → Connection ID: {connection_id}")

# ── 2. Connect ────────────────────────────────────────────────────────────────
print("\n── Connect ───────────────────────────────────────────────────────────")
result, device = eb.PvDevice.CreateAndConnect(connection_id)
if device is None:
    print(f"  ❌  {result.GetCodeString()}: {result.GetDescription()}")
    print("  → Close eBUS Player (it holds exclusive camera access)")
    sys.exit(1)
print(f"  ✅  Connected  (GEV: {isinstance(device, eb.PvDeviceGEV)})")

# Negotiate max packet size across the network path (CRITICAL — prevents packet loss)
# Must be called BEFORE opening any streams
if isinstance(device, eb.PvDeviceGEV):
    r = device.NegotiatePacketSize()
    print(f"  NegotiatePacketSize: {r.GetCodeString()}")

# ── 3. Print key parameters ───────────────────────────────────────────────────
print("\n── Parameters ────────────────────────────────────────────────────────")
nm = device.GetParameters()
for name in ["DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion",
             "DeviceTemperature", "AcquisitionFrameRate", "GevSCPSPacketSize",
             "TriggerMode", "TriggerSource", "PayloadSize"]:
    print(f"  {name:<32}: {get_p(nm, name)}")

# ── 4. Enumerate sources ──────────────────────────────────────────────────────
print("\n── Enumerate Sources ─────────────────────────────────────────────────")
source_selector = nm.GetEnum("SourceSelector")
source_names = []
if source_selector:
    result, count = source_selector.GetEntriesCount()
    for i in range(count):
        result, entry = source_selector.GetEntryByIndex(i)
        if entry:
            result, name = entry.GetName()
            source_names.append(name)
print(f"  Sources: {source_names}")

# ── 5. Open all streams simultaneously ───────────────────────────────────────
print("\n── Open Streams (simultaneous) ───────────────────────────────────────")
sources = []
for ch_idx, src_name in enumerate(source_names):
    src = Source(device, connection_id, src_name, ch_idx)
    if src.open():
        sources.append(src)

if not sources:
    print("  ❌  No sources opened.")
    device.Disconnect()
    eb.PvDevice.Free(device)
    sys.exit(1)
print(f"  ✅  {len(sources)} streams open simultaneously")

# ── 6. Start acquisition on ALL sources ──────────────────────────────────────
print("\n── Start Acquisition (all sources) ───────────────────────────────────")
for src in sources:
    src.start_acquisition()
print(f"  ✅  AcquisitionStart sent to all {len(sources)} sources")

# Allow sensors to reach steady-state exposure
WARMUP_S     = 2.0    # seconds
DRAIN_FRAMES = 30     # frames to drain per source

print(f"  Warming up ({WARMUP_S}s) …", end="", flush=True)
time.sleep(WARMUP_S)
print(" done")

# Drain frames in round-robin so all pipelines advance at the same rate
# → prevents blockID drift that happens with sequential per-source drain
print(f"  Draining {DRAIN_FRAMES} frames (interleaved) …")
counts = {src.source_name: 0 for src in sources}
for _ in range(DRAIN_FRAMES):
    for src in sources:
        r, buf, op = src.pipeline.RetrieveNextBuffer(200)
        if r.IsOK():
            src.pipeline.ReleaseBuffer(buf)
            counts[src.source_name] += 1
for src in sources:
    print(f"    {src.source_name}: discarded {counts[src.source_name]} frames")

# ── 7. Retrieve one synchronized frame per source ────────────────────────────
print("\n── Retrieve Synchronized Triplet ─────────────────────────────────────")
frames = {}   # source_name → (numpy_array, pixel_format)
block_ids = {}

for src in sources:
    img_data, pf = src.retrieve_one()
    frames[src.source_name]    = (img_data, pf)

# Synchronization check — blockIDs printed per-source above; verify they match
print("  ↑ Verify blockIDs above are identical across all 3 sources")


# Stream packet-loss diagnostics (dark images → check for errors here)
print("\n── Stream Statistics (packet loss check) ─────────────────────────────")
for src in sources:
    sm = src.stream.GetParameters()
    lost   = get_p(sm, "PacketResendRequestCount")
    errors = get_p(sm, "PacketErrorCount")
    missed = get_p(sm, "MissingPacketCount")
    fps    = get_p(sm, "AcquisitionRate")
    bw     = get_p(sm, "Bandwidth")
    print(f"  {src.source_name}: FPS={fps}  BW={bw}  "
          f"Resend={lost}  Errors={errors}  Missing={missed}")


# ── 8. Stop all acquisitions ──────────────────────────────────────────────────
print("\n── Stop Acquisition ──────────────────────────────────────────────────")
for src in sources:
    src.stop_acquisition()

# ── 9. Save PNGs ──────────────────────────────────────────────────────────────
print("\n── Save PNGs ─────────────────────────────────────────────────────────")
source_results = []
for ch_idx, src in enumerate(sources):
    img_data, pf = frames[src.source_name]
    if img_data is None:
        print(f"  ⚠️  CH{ch_idx+1} ({src.source_name}): no data — skipped")
        continue

    fname = OUT_DIR / f"ch{ch_idx+1}.png"

    if pf == "BayerRG8":
        img_save = cv2.cvtColor(img_data, cv2.COLOR_BayerBG2BGR)
    else:
        img_save = img_data   # Mono8 NIR — raw, unchanged

    # Raw save (correct pixel values for processing)
    cv2.imwrite(str(fname), img_save)

    # Normalized save (for visual inspection — stretches dark images to full range)
    norm_fname = OUT_DIR / f"ch{ch_idx+1}_norm.png"
    img_norm = cv2.normalize(img_save, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(norm_fname), img_norm)
    print(f"  ✅  {fname.name}  (raw)    min={img_data.min()} max={img_data.max()} mean={img_data.mean():.1f}")
    print(f"       {norm_fname.name}  (display-normalized)")

    source_results.append({
        "source":       src.source_name,
        "pixel_format": pf,
        "width":        img_data.shape[1],
        "height":       img_data.shape[0],
        "exposure_us":  get_p(nm, "ExposureTime"),
        "gain":         get_p(nm, "Gain"),
    })

# ── 10. Close all streams + disconnect ────────────────────────────────────────
print("\n── Cleanup ───────────────────────────────────────────────────────────")
for src in sources:
    src.close()
device.Disconnect()
eb.PvDevice.Free(device)
print("  ✅  Disconnected cleanly")

# ── 11. Config summary ────────────────────────────────────────────────────────
print(f"""
{'═'*62}
  PASTE INTO config.yaml → camera.jai
{'═'*62}
  ip:     "{dev_info_cached.GetIPAddress()}"
  mac:    "{dev_info_cached.GetMACAddress()}"
  serial: "{dev_info_cached.GetSerialNumber()}"
  fps:    {get_p(nm, 'AcquisitionFrameRate')}
  sources:""")
for sr in source_results:
    print(f"    {sr['source']}:")
    print(f"      pixel_format: \"{sr['pixel_format']}\"")
    print(f"      resolution:   [{sr['width']}, {sr['height']}]")
    print(f"      exposure_us:  {sr['exposure_us']}")
    print(f"      gain:         {sr['gain']}")
print(f"""{'═'*62}
  PNGs → {OUT_DIR.resolve()}
{'═'*62}
""")
