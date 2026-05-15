"""
scripts/camera_probe_jai.py
============================
JAI Camera Probe — eBUS Python API v6.6.1
Based on the official Pleora PvStreamSample documentation pattern.
No GenTL / .cti file required.

Prerequisites:
  - eBUS SDK installed (JAI 64-bit)
  - pip install ebus_python-6.6.1-...-py310-none-win_amd64.whl
  - pip install "numpy<2"
  - eBUS Player must be CLOSED (exclusive device access)

Usage:
  conda activate applegui
  python scripts/camera_probe_jai.py
"""

import sys
import importlib
from pathlib import Path

import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
BUFFER_COUNT = 16
TIMEOUT_MS   = 5000   # ms to wait for first buffer
OUT_DIR      = Path("scripts/probe_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import eBUS ───────────────────────────────────────────────────────────────
print("\n── eBUS Import ───────────────────────────────────────────────────────")
try:
    import eBUS as eb
    print("  ✅  import eBUS OK")
except ImportError as e:
    print(f"  ❌  {e}")
    print("  Install: pip install ebus_python-...-py310-none-win_amd64.whl")
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

# ── 1. Discover devices ───────────────────────────────────────────────────────
print("\n── Device Discovery ──────────────────────────────────────────────────")
sys_obj = eb.PvSystem()
sys_obj.Find()

connection_ID = None
for i in range(sys_obj.GetInterfaceCount()):
    iface = sys_obj.GetInterface(i)
    for j in range(iface.GetDeviceCount()):
        dev = iface.GetDeviceInfo(j)
        print(f"  Found: {dev.GetDisplayID()}")
        print(f"    IP : {dev.GetIPAddress()}")
        print(f"    MAC: {dev.GetMACAddress()}")
        print(f"    S/N: {dev.GetSerialNumber()}")
        if connection_ID is None:
            connection_ID = dev.GetConnectionID()
            dev_info_cached = dev      # save for config output

if connection_ID is None:
    print("  ❌  No camera found.")
    sys.exit(1)

print(f"\n  → Using connection ID: {connection_ID}")

# ── 2. Connect to device ──────────────────────────────────────────────────────
print("\n── connect_to_device ─────────────────────────────────────────────────")
result, device = eb.PvDevice.CreateAndConnect(connection_ID)
if device is None:
    print(f"  ❌  {result.GetCodeString()}: {result.GetDescription()}")
    print("  → Close eBUS Player (it holds exclusive camera access)")
    sys.exit(1)
print("  ✅  Connected")
print(f"     GEV device: {isinstance(device, eb.PvDeviceGEV)}")

# ── 3. Read parameters ────────────────────────────────────────────────────────
print("\n── Parameters ────────────────────────────────────────────────────────")
nm = device.GetParameters()
PARAM_NAMES = [
    "DeviceModelName", "DeviceVendorName", "DeviceSerialNumber",
    "DeviceFirmwareVersion", "DeviceTemperature",
    "Width", "Height", "WidthMax", "HeightMax",
    "PixelFormat", "PixelSize",
    "AcquisitionFrameRate", "AcquisitionFrameRateEnable",
    "ResultingFrameRate", "AcquisitionMode",
    "ExposureTime", "ExposureMode", "ExposureAuto",
    "Gain", "GainAuto", "BlackLevel",
    "TriggerMode", "TriggerSource",
    "PayloadSize", "GevSCPSPacketSize",
    "SourceSelector", "ComponentSelector",
]
params = {}
for name in PARAM_NAMES:
    v = get_p(nm, name)
    params[name] = v
    mark = "✅" if v != "N/A" else "·"
    print(f"  {mark}  {name:<32}: {v}")

# GenICam AcquisitionStart / Stop commands
start_cmd = nm.Get("AcquisitionStart")
stop_cmd  = nm.Get("AcquisitionStop")

# ── 4. Open stream ────────────────────────────────────────────────────────────
print("\n── open_stream ───────────────────────────────────────────────────────")
result, stream = eb.PvStream.CreateAndOpen(connection_ID)
if stream is None:
    print(f"  ❌  {result.GetCodeString()}: {result.GetDescription()}")
    device.Disconnect()
    eb.PvDevice.Free(device)
    sys.exit(1)
print(f"  ✅  Stream open")

# ── 5. Configure stream (GEV-specific) ───────────────────────────────────────
print("\n── configure_stream ──────────────────────────────────────────────────")
if isinstance(device, eb.PvDeviceGEV):
    r = device.NegotiatePacketSize()
    print(f"  NegotiatePacketSize     : {r.GetCodeString()}")
    lip  = stream.GetLocalIPAddress()
    lp   = stream.GetLocalPort()
    r = device.SetStreamDestination(lip, lp)
    print(f"  SetStreamDestination    : {r.GetCodeString()}  ({lip}:{lp})")
else:
    print("  ⚠️  Not a PvDeviceGEV — skipping GEV config")

# ── 6. Allocate buffers ───────────────────────────────────────────────────────
print("\n── configure_stream_buffers ──────────────────────────────────────────")
payload_size  = device.GetPayloadSize()
buffer_count  = min(stream.GetQueuedBufferMaximum(), BUFFER_COUNT)
print(f"  Payload size: {payload_size} bytes")
print(f"  Buffer count: {buffer_count}")

buffer_list = []
for _ in range(buffer_count):
    pvbuf = eb.PvBuffer()
    pvbuf.Alloc(payload_size)
    buffer_list.append(pvbuf)
for pvbuf in buffer_list:
    stream.QueueBuffer(pvbuf)
print(f"  Allocated and queued {buffer_count} buffers")

# ── 7. Enumerate sources & acquire one buffer per source ─────────────────────
print("\n── acquire_images (all sources) ──────────────────────────────────────")

# Find which SourceSelector values exist
source_param = nm.Get("SourceSelector")
source_names = []
if source_param is not None:
    try:
        entries = source_param.GetEntries()
        source_names = [e.GetName() for e in entries]
    except Exception:
        pass

if not source_names:
    source_names = ["Source0", "Source1", "Source2"]   # fallback

print(f"  Sources found: {source_names}")

source_results = []

for ch_idx, src_name in enumerate(source_names):
    print(f"\n  ── {src_name} (CH{ch_idx+1}) ──────────────────────────────────────")

    # Switch source selector
    try:
        r, _ = source_param.SetValue(src_name)
        if not r.IsOK():
            print(f"  ⚠️  SetValue({src_name}) failed: {r.GetCodeString()}")
    except Exception as e:
        print(f"  ⚠️  SetValue error: {e}")

    pf   = get_p(nm, "PixelFormat")
    w    = get_p(nm, "Width")
    h    = get_p(nm, "Height")
    exp  = get_p(nm, "ExposureTime")
    gain = get_p(nm, "Gain")
    print(f"  PixelFormat={pf}  {w}×{h}  Exp={exp}µs  Gain={gain}")

    # Allocate a fresh buffer for this source
    p_size = device.GetPayloadSize()
    pvbuf  = eb.PvBuffer()
    pvbuf.Alloc(p_size)
    stream.QueueBuffer(pvbuf)

    # Start / stop acquisition for this source
    device.StreamEnable()
    start_cmd.Execute()
    time.sleep(0.3)

    result, pvbuffer, op_result = stream.RetrieveBuffer(3000)

    w_src, h_src = 0, 0

    if result.IsOK() and op_result.IsOK():
        pt = pvbuffer.GetPayloadType()
        if pt == eb.PvPayloadTypeImage:
            image     = pvbuffer.GetImage()
            w_src     = image.GetWidth()
            h_src     = image.GetHeight()
            img_data  = image.GetDataPointer()   # numpy array
            fname     = OUT_DIR / f"ch{ch_idx+1}.png"
            cv2.imwrite(str(fname), img_data)
            print(f"  ✅  {fname.name}  {w_src}×{h_src}  "
                  f"min={img_data.min()} max={img_data.max()} mean={img_data.mean():.1f}")
        else:
            print(f"  PayloadType={pt} (unexpected for single source)")
        stream.QueueBuffer(pvbuffer)
    else:
        if result.IsOK():
            print(f"  ⚠️  op_result: {op_result.GetCodeString()}")
            stream.QueueBuffer(pvbuffer)
        else:
            print(f"  ❌  {result.GetCodeString()}")

    stop_cmd.Execute()
    device.StreamDisable()

    source_results.append({
        "source":       src_name,
        "pixel_format": pf,
        "width":        w_src or int(w),
        "height":       h_src or int(h),
        "exposure_us":  exp,
        "gain":         gain,
    })

# Re-queue any remaining buffers and drain
stream.AbortQueuedBuffers()
while stream.GetQueuedBufferCount() > 0:
    r, pvb, _ = stream.RetrieveBuffer()

# ── 8. Stop and clean up ──────────────────────────────────────────────────────
print("\n── Cleanup ───────────────────────────────────────────────────────────")
stop_cmd.Execute()
device.StreamDisable()

stream.AbortQueuedBuffers()
while stream.GetQueuedBufferCount() > 0:
    r, pvb, _ = stream.RetrieveBuffer()

stream.Close()
eb.PvStream.Free(stream)
device.Disconnect()
eb.PvDevice.Free(device)
print("  ✅  Disconnected cleanly")

# ── 9. Config summary ─────────────────────────────────────────────────────────
print(f"""
{'═'*62}
  PASTE INTO config.yaml → camera.jai
{'═'*62}
  ip:     "{dev_info_cached.GetIPAddress()}"
  mac:    "{dev_info_cached.GetMACAddress()}"
  serial: "{dev_info_cached.GetSerialNumber()}"
  fps:    {params.get('AcquisitionFrameRate','?')}
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
