"""
scripts/diagnose_fps.py  (v2)
=============================
Connects to the JAI camera and probes all FPS-related parameters.
Also tries writing AcquisitionFrameRate directly.

Run (close GUI first):
    conda activate applegui
    python scripts/diagnose_fps.py
"""

import os
import sys

# ── Register eBUS DLL paths ───────────────────────────────────────────────
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
    print(f"ERROR: Cannot import eBUS - {e}")
    sys.exit(1)

print("eBUS imported OK\n")

# ── Discover & connect ────────────────────────────────────────────────────
print("Scanning network for JAI camera...")
sys_obj = eb.PvSystem()
sys_obj.Find()

connection_id = None
for i in range(sys_obj.GetInterfaceCount()):
    iface = sys_obj.GetInterface(i)
    for j in range(iface.GetDeviceCount()):
        dev_info = iface.GetDeviceInfo(j)
        print(f"  Found: {dev_info.GetDisplayID()}  IP={dev_info.GetIPAddress()}")
        if connection_id is None:
            connection_id = dev_info.GetConnectionID()

if connection_id is None:
    print("ERROR: No camera found.")
    sys.exit(1)

print("\nConnecting...")
result, device = eb.PvDevice.CreateAndConnect(connection_id)
if device is None:
    print(f"ERROR: {result.GetCodeString()}")
    sys.exit(1)
print("Connected OK\n")

nm = device.GetParameters()


def safe_read(param):
    """Try every known way to read a parameter value string."""
    for method in ("GetValue", "GetValueString"):
        try:
            ret = getattr(param, method)()
            if isinstance(ret, tuple):
                _, v = ret
                return str(v)
            return str(ret)
        except Exception:
            continue
    return "?"


# ── Probe specific parameter names ────────────────────────────────────────
print("=" * 65)
print("FPS PARAMETER PROBE")
print("=" * 65)
candidates = [
    "AcquisitionFrameRate",
    "AcquisitionFrameRateEnable",
    "AcquisitionFrameRateEnabled",
    "AcquisitionFrameRateMode",
    "AcquisitionFrameRateControlMode",
    "AcquisitionFrameRateAuto",
    "ResultingFrameRate",
    "ResultingAcquisitionFrameRate",
    "ResultingFramePeriod",
    "TriggerMode",
    "TriggerSelector",
]
for name in candidates:
    param = nm.Get(name)
    status = f"FOUND   val={safe_read(param)}" if param is not None else "NOT FOUND"
    print(f"  {name:50s}  {status}")

# ── Write test ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("WRITE TEST - try SetValue(30.0) on AcquisitionFrameRate")
print("=" * 65)
param = nm.GetFloat("AcquisitionFrameRate")
if param is None:
    print("  AcquisitionFrameRate (float) not found")
else:
    try:
        mn_ret = param.GetMin()
        mx_ret = param.GetMax()
        mn = mn_ret[1] if isinstance(mn_ret, tuple) else mn_ret
        mx = mx_ret[1] if isinstance(mx_ret, tuple) else mx_ret
        print(f"  Min={mn}  Max={mx}")
    except Exception as e:
        print(f"  Could not read min/max: {e}")
    try:
        r = param.SetValue(30.0)
        try:
            ok = r.IsOK()
            code = r.GetCodeString()
        except Exception:
            ok = False
            code = str(r)
        print(f"  SetValue(30.0) → {'OK' if ok else 'FAILED: ' + code}")
    except Exception as ex:
        print(f"  Exception: {ex}")

# ── Enumerate all params with 'frame' or 'rate' ───────────────────────────
print()
print("=" * 65)
print("ALL PARAMS CONTAINING 'frame' OR 'rate'")
print("=" * 65)
try:
    total = nm.GetCount()
    if isinstance(total, tuple):
        total = total[1]
    found = 0
    for idx in range(int(total)):
        try:
            param = nm.Get(idx)
            if param is None:
                continue
            name_ret = param.GetName()
            pname = name_ret[1] if isinstance(name_ret, tuple) else name_ret
            if "frame" in str(pname).lower() or "rate" in str(pname).lower():
                val = safe_read(param)
                print(f"  {str(pname):55s}  val={val}")
                found += 1
        except Exception:
            continue
    print(f"\n  Total found: {found}")
except Exception as ex:
    print(f"  Enumeration failed: {ex}")

print()
device.Disconnect()
eb.PvDevice.Free(device)
print("Done.")
