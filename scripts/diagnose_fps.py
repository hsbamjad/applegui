"""
scripts/diagnose_fps.py
=======================
Connects to the JAI camera and lists all FPS / frame-rate related
parameters with their names, types, access mode, and current values.

Run:
    conda activate applegui
    python scripts/diagnose_fps.py
"""

import os
import sys

# ── Register eBUS DLL paths ────────────────────────────────────────────────
_DLL_PATHS = [
    r"C:\Program Files\Common Files\Pleora\eBUS SDK",
    r"C:\Program Files\JAI\eBUS SDK",
    r"C:\Program Files\JAI\eBUS SDK\lib",
    r"C:\Program Files (x86)\Common Files\Pleora\eBUS SDK",
]
for p in _DLL_PATHS:
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

try:
    import eBUS as eb
except ImportError as e:
    print(f"ERROR: Cannot import eBUS — {e}")
    sys.exit(1)

print("eBUS imported OK")
print()

# ── Discover camera ───────────────────────────────────────────────────────
print("Scanning network for JAI camera...")
sys_obj = eb.PvSystem()
sys_obj.Find()

connection_id = None
for i in range(sys_obj.GetInterfaceCount()):
    iface = sys_obj.GetInterface(i)
    for j in range(iface.GetDeviceCount()):
        dev_info = iface.GetDeviceInfo(j)
        print(f"  Found: {dev_info.GetDisplayID()}  IP={dev_info.GetIPAddress()}  MAC={dev_info.GetMACAddress()}")
        if connection_id is None:
            connection_id = dev_info.GetConnectionID()

if connection_id is None:
    print("ERROR: No camera found on network.")
    sys.exit(1)

# ── Connect ───────────────────────────────────────────────────────────────
print(f"\nConnecting to camera...")
result, device = eb.PvDevice.CreateAndConnect(connection_id)
if device is None:
    print(f"ERROR: Could not connect — {result.GetCodeString()}")
    sys.exit(1)
print("Connected OK")
print()

nm = device.GetParameters()

# ── Check specific FPS-related parameter names ────────────────────────────
fps_param_names = [
    "AcquisitionFrameRate",
    "AcquisitionFrameRateEnable",
    "AcquisitionFrameRateEnabled",
    "AcquisitionFrameRateMode",
    "AcquisitionFrameRateAuto",
    "ResultingFrameRate",
    "ResultingFramePeriod",
]

print("=" * 60)
print("FPS PARAMETER PROBE")
print("=" * 60)
for name in fps_param_names:
    param = nm.Get(name)
    if param is None:
        print(f"  {name:45s}  NOT FOUND")
    else:
        try:
            access = param.GetAccessMode()
            type_str = type(param).__name__
            val_str = "?"
            try:
                r, v = param.GetValue()
                val_str = str(v) if r.IsOK() else f"read error: {r.GetCodeString()}"
            except Exception:
                try:
                    r, v = param.GetValueString()
                    val_str = v if r.IsOK() else "?"
                except Exception:
                    pass
            print(f"  {name:45s}  FOUND  type={type_str}  val={val_str}")
        except Exception as ex:
            print(f"  {name:45s}  FOUND  (error reading: {ex})")

# ── Dump ALL parameters containing "frame" or "rate" ─────────────────────
print()
print("=" * 60)
print("ALL PARAMETERS CONTAINING 'frame' OR 'rate' (case-insensitive)")
print("=" * 60)
try:
    count_result, total = nm.GetCount()
    for idx in range(total):
        result, param = nm.Get(idx)
        if param is None:
            continue
        try:
            r, pname = param.GetName()
            if not r.IsOK():
                continue
            if "frame" in pname.lower() or "rate" in pname.lower():
                type_str = type(param).__name__
                val_str = "?"
                try:
                    rv, v = param.GetValue()
                    val_str = str(v) if rv.IsOK() else "n/a"
                except Exception:
                    try:
                        rv, v = param.GetValueString()
                        val_str = v if rv.IsOK() else "n/a"
                    except Exception:
                        pass
                print(f"  {pname:50s}  {type_str:25s}  val={val_str}")
        except Exception:
            continue
except Exception as ex:
    print(f"  (Could not enumerate all parameters: {ex})")

print()
device.Disconnect()
eb.PvDevice.Free(device)
print("Done. Paste the full output above.")
