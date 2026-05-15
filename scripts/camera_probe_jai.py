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

# ── 7. Acquire ────────────────────────────────────────────────────────────────
print("\n── acquire_images ────────────────────────────────────────────────────")
device.StreamEnable()
start_cmd.Execute()
print(f"  AcquisitionStart sent — waiting up to {TIMEOUT_MS}ms for buffer…")

import time
time.sleep(0.5)

result, pvbuffer, op_result = stream.RetrieveBuffer(TIMEOUT_MS)

w_buf, h_buf = 0, 0

if result.IsOK():
    if op_result.IsOK():
        pt = pvbuffer.GetPayloadType()
        print(f"  ✅  Buffer received — PayloadType = {pt}")

        if pt == eb.PvPayloadTypeImage:
            image     = pvbuffer.GetImage()
            w_buf     = image.GetWidth()
            h_buf     = image.GetHeight()
            pix_type  = image.GetPixelType()
            img_data  = image.GetDataPointer()   # numpy array (uint8)
            print(f"  Single image: {w_buf}×{h_buf}  PixelType={pix_type}")
            cv2.imwrite(str(OUT_DIR / "ch1.png"), img_data)
            print(f"  ch1.png saved  shape={img_data.shape}  "
                  f"min={img_data.min()} max={img_data.max()} mean={img_data.mean():.1f}")

        elif pt == eb.PvPayloadTypeMultiPart:
            container = pvbuffer.GetMultiPartContainer()
            n_parts   = container.GetPartCount()
            print(f"  MultiPart: {n_parts} parts (3-sensor JAI)")
            for i in range(n_parts):
                part    = container.GetPart(i)
                img2    = part.GetImage()
                w2, h2  = img2.GetWidth(), img2.GetHeight()
                pix2    = img2.GetPixelType()
                data2   = img2.GetDataPointer()
                if i == 0:
                    w_buf, h_buf = w2, h2
                fname = OUT_DIR / f"ch{i+1}.png"
                cv2.imwrite(str(fname), data2)
                print(f"  CH{i+1}: {w2}×{h2}  fmt={pix2}  "
                      f"min={data2.min()} max={data2.max()} mean={data2.mean():.1f}")

        else:
            print(f"  Payload type {pt} not handled by this probe")

    else:
        print(f"  ⚠️  Buffer retrieved but op_result: {op_result.GetCodeString()}")

    stream.QueueBuffer(pvbuffer)   # re-queue
else:
    print(f"  ❌  RetrieveBuffer failed: {result.GetCodeString()}")
    print("  Possible causes:")
    print("    - NegotiatePacketSize / SetStreamDestination not applied")
    print("    - Firewall blocking UDP (disable Windows Firewall temporarily)")
    print("    - Wrong NIC / subnet  (camera IP: 169.254.133.151)")

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
if w_buf:
    params["Width"]  = str(w_buf)
    params["Height"] = str(h_buf)

print(f"""
{'═'*62}
  PASTE INTO config.yaml → camera.jai
{'═'*62}
  ip:           "{dev_info_cached.GetIPAddress()}"
  mac:          "{dev_info_cached.GetMACAddress()}"
  serial:       "{dev_info_cached.GetSerialNumber()}"
  resolution:   [{params.get('Width','?')}, {params.get('Height','?')}]
  pixel_format: "{params.get('PixelFormat','?')}"
  fps:          {params.get('AcquisitionFrameRate', params.get('ResultingFrameRate','?'))}
  exposure_us:  {params.get('ExposureTime','?')}
  gain:         {params.get('Gain','?')}
{'═'*62}
  PNGs → {OUT_DIR.resolve()}
{'═'*62}
""")
