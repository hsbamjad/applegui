"""
scripts/camera_probe_jai.py
============================
JAI Camera Probe — eBUS Python API (no GenTL / .cti needed).
Uses: ebus_python-6.6.1 (jai variant, installed from .whl)

Usage (applegui env, project root, eBUS Player CLOSED):
    python scripts/camera_probe_jai.py

Output:
  - All readable GenICam parameters
  - scripts/probe_output/ch1.png  (ch2, ch3 if MultiPart)
  - config.yaml-ready block
"""

import sys
import importlib
from pathlib import Path

OUT_DIR = Path("scripts/probe_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import eBUS module ────────────────────────────────────────────────────────
CANDIDATES = ["eBUS", "ebus_python", "eBUSPython", "PyeBUSSDK"]
eb = None
print("\n── Detecting eBUS Python module ──────────────────────────────────────")
for name in CANDIDATES:
    try:
        eb = importlib.import_module(name)
        print(f"  ✅  Imported: {name}")
        break
    except ImportError:
        print(f"  ❌  {name}")

if eb is None:
    print("❌  No eBUS module found. Install the JAI Python Wrapper .whl first.")
    sys.exit(1)

# ── Camera discovery ──────────────────────────────────────────────────────────
print("\n── Camera Discovery ──────────────────────────────────────────────────")
sys_obj = eb.PvSystem()
sys_obj.Find()
n_iface = sys_obj.GetInterfaceCount()
print(f"  Interfaces: {n_iface}")

dev_info = None
for i in range(n_iface):
    iface = sys_obj.GetInterface(i)
    if iface.GetDeviceCount() > 0:
        dev_info = iface.GetDeviceInfo(0)
        print(f"  Camera on interface [{i}]: {dev_info.GetDisplayID()}")
        print(f"    IP : {dev_info.GetIPAddress()}")
        print(f"    MAC: {dev_info.GetMACAddress()}")
        print(f"    S/N: {dev_info.GetSerialNumber()}")
        break

if dev_info is None:
    print("❌  No camera found. Check connection and power.")
    sys.exit(1)

conn_id = dev_info.GetConnectionID()
print(f"  Connection ID: {conn_id}")

# ── Connect device ────────────────────────────────────────────────────────────
print("\n── Connecting device ────────────────────────────────────────────────")
result, device = eb.PvDevice.CreateAndConnect(conn_id)
if not result.IsOK():
    print(f"  ❌  {result.GetCodeString()}")
    print("  → Close eBUS Player first (it holds exclusive camera access)")
    sys.exit(1)
print("  ✅  Connected")

nm = device.GetParameters()

# ── Read parameters ───────────────────────────────────────────────────────────
def get_p(name):
    try:
        param = nm.Get(name)
        if param is None:
            return "N/A"
        # Try GetValue() first (int / float / bool)
        try:
            r, v = param.GetValue()
            if r.IsOK():
                return str(v)
        except Exception:
            pass
        # Try GetValueString() (enum / string)
        try:
            r, v = param.GetValueString()
            if r.IsOK():
                return v
        except Exception:
            pass
        return "N/A"
    except Exception:
        return "N/A"

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

print("\n── Parameters ───────────────────────────────────────────────────────")
params = {}
for name in PARAM_NAMES:
    v = get_p(name)
    params[name] = v
    mark = "✅" if v != "N/A" else "·"
    print(f"  {mark}  {name:<32}: {v}")

# ── Open stream with GEV setup ────────────────────────────────────────────────
print("\n── Opening stream (GEV setup) ───────────────────────────────────────")
result, stream = eb.PvStream.CreateAndOpen(conn_id)
if not result.IsOK():
    print(f"  ❌  Stream open failed: {result.GetCodeString()}")
    eb.PvDevice.Free(device)
    sys.exit(1)
print(f"  ✅  Stream open  port={stream.GetLocalPort()}")

# Critical GEV setup: negotiate packet size + tell camera where to stream
device_gev = eb.PvDeviceGEV.Cast(device) if hasattr(eb, "PvDeviceGEV") else None
stream_gev  = eb.PvStreamGEV.Cast(stream)  if hasattr(eb, "PvStreamGEV")  else None

if device_gev and stream_gev:
    r = device_gev.NegotiatePacketSize()
    print(f"  NegotiatePacketSize     : {r.GetCodeString()}")
    lip  = stream_gev.GetLocalIPAddress()
    lp   = stream_gev.GetLocalPort()
    r = device_gev.SetStreamDestination(lip, lp)
    print(f"  SetStreamDestination    : {r.GetCodeString()} ({lip}:{lp})")
else:
    print("  ⚠️  GEV cast unavailable — frames may not arrive")

# ── Acquire one buffer ────────────────────────────────────────────────────────
import time, numpy as np, cv2

pipeline = eb.PvPipeline(stream)
pipeline.SetBufferCount(16)
pipeline.Start()
device.StreamEnable()
nm.Get("AcquisitionStart").Execute()

print("\n── Grabbing buffer (5 s timeout) ────────────────────────────────────")
time.sleep(1.0)
result, buf, op_result = pipeline.RetrieveNextBuffer(5000)

w_buf, h_buf = 0, 0

if result.IsOK():
    pt = buf.GetPayloadType()
    print(f"  ✅  Buffer OK  PayloadType={pt}")

    if pt == eb.PvPayloadTypeImage:
        img       = buf.GetImage()
        w_buf     = img.GetWidth()
        h_buf     = img.GetHeight()
        fmt       = img.GetPixelType()
        print(f"  Single image: {w_buf}×{h_buf}  fmt={fmt}")
        raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8)
        arr = raw.reshape(h_buf, w_buf) if raw.size == h_buf * w_buf else raw
        cv2.imwrite(str(OUT_DIR / "ch1.png"), arr)
        print(f"  ch1.png saved  min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")

    elif pt == eb.PvPayloadTypeMultiPart:
        mp  = buf.GetMultiPartContainer()
        cnt = mp.GetSectionCount()
        print(f"  MultiPart: {cnt} sections")
        for i in range(cnt):
            sec = mp.GetSection(i)
            if hasattr(sec, "GetImage"):
                img2 = sec.GetImage()
                w2, h2 = img2.GetWidth(), img2.GetHeight()
                if i == 0:
                    w_buf, h_buf = w2, h2
                raw2 = np.frombuffer(img2.GetDataPointer(), dtype=np.uint8)
                arr2 = raw2.reshape(h2, w2)
                cv2.imwrite(str(OUT_DIR / f"ch{i+1}.png"), arr2)
                print(f"  ch{i+1}.png  {w2}×{h2}  "
                      f"min={arr2.min()} max={arr2.max()} mean={arr2.mean():.1f}")
    else:
        print(f"  Unknown payload type: {pt}")

    pipeline.ReleaseBuffer(buf)
else:
    print(f"  ❌  {result.GetCodeString()}")
    print("  Possible causes:")
    print("    - NegotiatePacketSize / SetStreamDestination failed")
    print("    - Camera needs 'AcquisitionFrameRateEnable' set to True first")

# ── Stop cleanly ──────────────────────────────────────────────────────────────
nm.Get("AcquisitionStop").Execute()
device.StreamDisable()
pipeline.Stop()
eb.PvStream.Free(stream)
eb.PvDevice.Free(device)

# Update resolution from buffer if not read from params
if w_buf:
    params["Width"]  = str(w_buf)
    params["Height"] = str(h_buf)

# ── Config summary ────────────────────────────────────────────────────────────
print(f"""
{'═'*62}
  PASTE INTO config.yaml
{'═'*62}
  camera:
    jai:
      ip: "{dev_info.GetIPAddress()}"
      mac: "{dev_info.GetMACAddress()}"
      serial: "{dev_info.GetSerialNumber()}"
      resolution: [{params.get('Width','?')}, {params.get('Height','?')}]
      pixel_format: "{params.get('PixelFormat','BayerRG8')}"
      fps: {params.get('AcquisitionFrameRate','?')}
      exposure_us: {params.get('ExposureTime','?')}
{'═'*62}
  PNGs → {OUT_DIR.resolve()}
{'═'*62}
""")
