"""
scripts/camera_probe.py
========================
JAI Camera Probe — B4 data collection script.

Connects to the JAI FSFE-3200T-10GE via Harvesters/GenTL and:
  1. Reads all critical GenICam parameters
  2. Grabs one synchronized frame buffer
  3. Reports every component (channel): shape, dtype, min/max pixel
  4. Saves each channel as a PNG  → scripts/probe_output/ch1.png, ch2.png, ch3.png
  5. Prints a summary report you can paste directly into config.yaml

Usage (from project root, with applegui env active):
    python scripts/camera_probe.py

Requirements:
    - JAI eBUS SDK installed  (provides the .cti GenTL producer file)
    - harvesters pip package  (already in environment.yml)
    - Camera connected and streaming (verified in eBUS Player first)
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np

# ── 1. Locate the GenTL producer (.cti file) ─────────────────────────────────

CTI_SEARCH_PATHS = [
    "C:/Program Files/JAI/eBUS SDK/bin/PvGenTL.cti",
    "C:/Program Files (x86)/JAI/eBUS SDK/bin/PvGenTL.cti",
    "C:/Program Files/JAI/eBUS_SDK/bin/PvGenTL.cti",
    "D:/Program Files/JAI/eBUS SDK/bin/PvGenTL.cti",
]


def find_cti() -> str:
    for path in CTI_SEARCH_PATHS:
        if os.path.exists(path):
            return path
    # Brute-force search
    for drive in ["C:/", "D:/", "E:/"]:
        hits = glob.glob(f"{drive}**/PvGenTL.cti", recursive=True)
        if hits:
            return hits[0]
    return ""


# ── 2. Safe parameter reader ──────────────────────────────────────────────────

def read_param(node_map, name: str, default="N/A"):
    """Safely read a GenICam node value."""
    try:
        node = getattr(node_map, name)
        return node.value
    except Exception:
        return default


# ── 3. Main probe ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── Find CTI ──────────────────────────────────────────────────────────────
    cti_path = find_cti()
    if not cti_path:
        print("\n❌  Could not find PvGenTL.cti — is JAI eBUS SDK installed?")
        print("    Searched:", "\n    ".join(CTI_SEARCH_PATHS))
        sys.exit(1)
    print(f"\n✅  GenTL producer found:\n    {cti_path}\n")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path("scripts/probe_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Connect via Harvesters ────────────────────────────────────────────────
    try:
        from harvesters.core import Harvester
    except ImportError:
        print("❌  harvesters not installed.  Run:  pip install harvesters")
        sys.exit(1)

    h = Harvester()
    h.add_file(cti_path)
    h.update()

    if not h.device_info_list:
        print("❌  No cameras found on network.")
        print("    Make sure the camera is connected and eBUS Player can see it.")
        h.reset()
        sys.exit(1)

    print(f"✅  Cameras found: {len(h.device_info_list)}")
    for i, dev in enumerate(h.device_info_list):
        print(f"    [{i}] {dev}")

    print("\n── Connecting to camera [0] ──────────────────────────────────────")
    ia = h.create(0)

    # ── Read GenICam parameters ───────────────────────────────────────────────
    nm = ia.remote_device.node_map

    model       = read_param(nm, "DeviceModelName")
    vendor      = read_param(nm, "DeviceVendorName")
    serial      = read_param(nm, "DeviceSerialNumber")
    firmware    = read_param(nm, "DeviceFirmwareVersion")
    width       = read_param(nm, "Width")
    height      = read_param(nm, "Height")
    pix_fmt     = read_param(nm, "PixelFormat")
    fps         = read_param(nm, "AcquisitionFrameRate")
    fps_max     = read_param(nm, "AcquisitionFrameRateMax")
    exposure    = read_param(nm, "ExposureTime")
    gain        = read_param(nm, "Gain")
    trig_mode   = read_param(nm, "TriggerMode")
    temp        = read_param(nm, "DeviceTemperature")

    # ── Try reading payload type / component count ─────────────────────────────
    payload_type = read_param(nm, "PayloadType")

    print("\n── Acquiring one frame buffer ────────────────────────────────────")
    ia.start()

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("⚠️  opencv-python not installed — PNGs will not be saved.")

    with ia.fetch() as buf:
        components = buf.payload.components
        n_comp     = len(components)

        print(f"    Buffer acquired: {n_comp} component(s)\n")

        comp_info = []
        for i, comp in enumerate(components):
            arr = comp.data

            # Try to reshape if it's flat
            try:
                h_px = comp.height
                w_px = comp.width
                arr  = arr.reshape(h_px, w_px)
            except Exception:
                h_px = 0
                w_px = arr.size

            info = {
                "index":  i,
                "shape":  (h_px, w_px) if h_px else arr.shape,
                "dtype":  str(arr.dtype),
                "min":    int(arr.min()),
                "max":    int(arr.max()),
                "mean":   round(float(arr.mean()), 1),
            }
            comp_info.append(info)

            # Save PNG
            if has_cv2 and h_px > 0:
                png_path = out_dir / f"ch{i + 1}.png"
                cv2.imwrite(str(png_path), arr)
                print(f"    📸  CH{i + 1} saved → {png_path}")

    ia.stop()
    ia.destroy()
    h.reset()

    # ── Print summary report ───────────────────────────────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print("  JAI CAMERA PROBE REPORT")
    print(f"  Generated for: config/config.yaml")
    print(sep)

    print(f"""
DEVICE
  Vendor      : {vendor}
  Model       : {model}
  Serial      : {serial}
  Firmware    : {firmware}
  Temperature : {temp} °C

ACQUISITION
  Resolution  : {width} × {height}  px
  Pixel Fmt   : {pix_fmt}
  Frame Rate  : {fps}  FPS  (max reported: {fps_max})
  Exposure    : {exposure}  μs
  Gain        : {gain}  dB
  Trigger     : {trig_mode}
  Payload     : {payload_type}
""")

    print(f"COMPONENTS  ({n_comp} detected)")
    print(f"  {'CH':<4} {'Shape':<16} {'Dtype':<8} {'Min':>5} {'Max':>5} {'Mean':>7}")
    print(f"  {'─'*4} {'─'*16} {'─'*8} {'─'*5} {'─'*5} {'─'*7}")
    for c in comp_info:
        shape_str = f"{c['shape'][0]}×{c['shape'][1]}" if len(c['shape']) == 2 else str(c['shape'])
        print(f"  {c['index']+1:<4} {shape_str:<16} {c['dtype']:<8} {c['min']:>5} {c['max']:>5} {c['mean']:>7}")

    print(f"""
PASTE INTO config.yaml
  camera:
    jai:
      resolution: [{width}, {height}]
      pixel_format: "{pix_fmt}"
      fps: {fps}                  # change to desired fps (max ≈ {fps_max})
      exposure_us: {exposure}

  channels:   # {n_comp} component(s) confirmed
    ch1: {{ sensor_index: 0 }}    # buffer.payload.components[0]
    ch2: {{ sensor_index: 1 }}    # buffer.payload.components[1]
    ch3: {{ sensor_index: 2 }}    # buffer.payload.components[2]
""")

    print(sep)
    print(f"  Saved {n_comp} PNG(s) to: {out_dir.resolve()}")
    print(sep)


if __name__ == "__main__":
    main()
