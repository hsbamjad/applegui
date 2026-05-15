"""
scripts/camera_probe_jai.py
============================
JAI Camera Probe — using JAI eBUS Python API directly.
No GenTL / Harvesters / .cti file required.
Uses: ebus_python-6.6.1 (jai variant, installed from .whl)

Usage (from project root, applegui env active):
    python scripts/camera_probe_jai.py

Output:
  - Full camera parameter report (paste into config.yaml)
  - scripts/probe_output/ch1.png, ch2.png, ch3.png
"""

import sys
import importlib
from pathlib import Path

OUT_DIR = Path("scripts/probe_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Import the eBUS Python module ─────────────────────────────────────
# Wheel name: ebus_python-6.6.1-7475_jai-py310-none-win_amd64.whl
# Import name derived from wheel package name (underscored)

CANDIDATES = [
    "ebus_python",      # ← most likely: matches wheel package name exactly
    "eBUSPython",       # alternate capitalisation
    "PyeBUSSDK",
    "jai",
    "jai_sdk",
]

eb = None
eb_name = None
print("\n── Detecting eBUS Python module ──────────────────────────────────────")
for name in CANDIDATES:
    try:
        eb = importlib.import_module(name)
        print(f"  ✅  Imported: {name}")
        eb_name = name
        break
    except ImportError:
        print(f"  ❌  {name}")

if eb is None:
    print("""
❌  No eBUS Python module found.
    Make sure you ran:
      pip install ebus_python-6.6.1-7475_jai-py310-none-win_amd64.whl
    with the (applegui) environment active.
""")
    sys.exit(1)

# ── Step 2: Print all public members ─────────────────────────────────────────
print(f"\n── Public API of '{eb_name}' ──────────────────────────────────────────")
members = sorted(m for m in dir(eb) if not m.startswith("_"))
for m in members:
    obj = getattr(eb, m)
    kind = "class" if isinstance(obj, type) else type(obj).__name__
    print(f"  {m:<40}  ({kind})")
print(f"\n  Total: {len(members)} public names\n")

# ── Step 3: Try to discover cameras ──────────────────────────────────────────
print("── Camera Discovery ──────────────────────────────────────────────────")

def try_pvsystem():
    """Pleora PvSystem pattern — most common for eBUS SDK Python."""
    sys_obj = eb.PvSystem()
    result  = sys_obj.Find()
    print(f"  PvSystem.Find() → {result}")
    n_iface = sys_obj.GetInterfaceCount()
    print(f"  Interfaces found: {n_iface}")
    for i in range(n_iface):
        iface = sys_obj.GetInterface(i)
        n_dev = iface.GetDeviceCount()
        for j in range(n_dev):
            dev = iface.GetDeviceInfo(j)
            print(f"\n  ── Device [{j}] ────────────────────────────")
            for attr in ["GetDisplayID", "GetModelName", "GetVendorName",
                         "GetSerialNumber", "GetIPAddress", "GetMACAddress"]:
                if hasattr(dev, attr):
                    print(f"    {attr:<20}: {getattr(dev, attr)()}")
    return sys_obj, n_iface

def try_connect_and_probe(sys_obj):
    """Connect to first camera and read all parameters."""
    try:
        iface = sys_obj.GetInterface(0)
        dev_info = iface.GetDeviceInfo(0)

        print("\n── Connecting ────────────────────────────────────────────────────")
        device = eb.PvDevice.CreateAndConnect(dev_info)
        if device is None:
            print("  ❌  PvDevice.CreateAndConnect returned None")
            return

        print("  ✅  Connected")
        nm = device.GetParameters()

        # Read key parameters safely
        def get_p(name):
            try:
                p = nm.Get(name)
                v = p.GetValue() if hasattr(p, "GetValue") else str(p)
                return v
            except Exception:
                return "N/A"

        params = {
            "DeviceModelName":        get_p("DeviceModelName"),
            "DeviceVendorName":       get_p("DeviceVendorName"),
            "DeviceSerialNumber":     get_p("DeviceSerialNumber"),
            "DeviceFirmwareVersion":  get_p("DeviceFirmwareVersion"),
            "Width":                  get_p("Width"),
            "Height":                 get_p("Height"),
            "PixelFormat":            get_p("PixelFormat"),
            "AcquisitionFrameRate":   get_p("AcquisitionFrameRate"),
            "ExposureTime":           get_p("ExposureTime"),
            "Gain":                   get_p("Gain"),
            "TriggerMode":            get_p("TriggerMode"),
        }

        print("\n── Parameters ────────────────────────────────────────────────────")
        for k, v in params.items():
            print(f"  {k:<30}: {v}")

        # Stream one buffer
        print("\n── Streaming one buffer ──────────────────────────────────────────")
        stream = eb.PvStream.CreateAndOpen(dev_info)
        pipeline = eb.PvPipeline(stream)
        pipeline.SetBufferCount(4)
        pipeline.Start()
        device.StreamEnable()
        cmd = nm.Get("AcquisitionStart")
        cmd.Execute()

        import time; time.sleep(0.5)
        result, buf, op_result = pipeline.RetrieveNextBuffer(1000)
        if result.IsOK() and buf:
            payload = buf.GetPayload()
            n_comp  = payload.GetImageCount() if hasattr(payload, "GetImageCount") else 1
            print(f"  Buffer OK — {n_comp} image(s) in payload")

            try:
                import numpy as np, cv2
                for ci in range(n_comp):
                    img = payload.GetImage(ci) if hasattr(payload, "GetImage") else payload
                    w   = img.GetWidth()
                    h   = img.GetHeight()
                    fmt = img.GetPixelType() if hasattr(img, "GetPixelType") else "?"
                    arr = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).reshape(h, w)
                    cv2.imwrite(str(OUT_DIR / f"ch{ci+1}.png"), arr)
                    print(f"  CH{ci+1}: {w}×{h}  fmt={fmt}  "
                          f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
            except Exception as e:
                print(f"  Image extraction error: {e}")

            pipeline.ReleaseBuffer(buf)

        cmd_stop = nm.Get("AcquisitionStop")
        cmd_stop.Execute()
        device.StreamDisable()
        pipeline.Stop()
        stream.Close()
        eb.PvStream.Free(stream)
        eb.PvDevice.Free(device)

        # Print config.yaml block
        print(f"""
{'═'*62}
  PASTE INTO config.yaml
{'═'*62}
  camera:
    jai:
      resolution: [{params['Width']}, {params['Height']}]
      pixel_format: "{params['PixelFormat']}"
      fps: {params['AcquisitionFrameRate']}
      exposure_us: {params['ExposureTime']}
{'═'*62}
  Saved PNGs → {OUT_DIR.resolve()}
{'═'*62}
""")
    except Exception as e:
        print(f"  Connect/probe error: {e}")
        import traceback; traceback.print_exc()

# ── Run ───────────────────────────────────────────────────────────────────────
try:
    sys_obj, n_iface = try_pvsystem()
    if n_iface > 0:
        try_connect_and_probe(sys_obj)
    else:
        print("  No cameras found. Check camera is powered and connected.")
except AttributeError as e:
    print(f"\n  PvSystem not available in this module: {e}")
    print("  The module API may differ — share the member list above and we'll adapt.")
except Exception as e:
    print(f"\n  Unexpected error: {e}")
    import traceback; traceback.print_exc()
