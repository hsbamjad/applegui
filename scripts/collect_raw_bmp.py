"""
scripts/collect_raw_bmp.py
==========================
Standalone JAI data collector — NO GUI app, NO OpenCV windows.

Uses CameraInterface + config.yaml (same camera stack as the app).
Console keyboard controls (this terminal must stay focused).

Controls:
  S  /  SPACE   Toggle Save ON ↔ OFF
  Q             Quit
  Ctrl+C        Quit

Why not write BMP while live?
  Full-res BMP at 30 fps ≈ 450 MB/s + 90 file-creates/s.  That disk load
  eventually starves 10GigE packet processing and the grab loop dies
  (raw=None forever).  Instead we append bit-exact pixels to sequential
  ``stream.raw`` files while recording, then convert to BMP after the
  camera is disconnected (on Quit).

Live on-disk during Save::

    data/sessions/YYYYMMDD_HHMMSS/
      raw_frames/ch1/stream.raw
      raw_frames/ch2/stream.raw
      raw_frames/ch3/stream.raw
      raw_frames/index.jsonl

After Quit (camera down) → same folder gains::

      raw_frames/ch1/frame_000001.bmp   # full-res, uncompressed
      raw_frames/ch2/frame_000001.bmp
      raw_frames/ch3/frame_000001.bmp

Usage (from repo root, eBUS Player closed)::

    conda activate applegui
    python scripts/collect_raw_bmp.py
"""

from __future__ import annotations

import json
import msvcrt
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.camera.camera_interface import CameraInterface
from core.log import configure_root, get_logger
from utils.paths import APP_ROOT, CONFIG_PATH, SESSIONS_DIR

configure_root()
log = get_logger("collect_raw_bmp")

_WRITER_QUEUE = 24          # ~360 MB burst; sequential writes keep up far better than BMP
_CONNECT_RETRIES = 3
_FIRST_FRAME_TIMEOUT_S = 15.0
_STREAM_DEAD_S = 3.0        # no new frame → reconnect
_FILE_BUFFER = 16 * 1024 * 1024


# ── Sequential raw writer (live path) ─────────────────────────────────────────

class _RawWriter:
    """Append full-res frames to stream.raw / index.jsonl on a bg thread."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=_WRITER_QUEUE)
        self._stop = threading.Event()
        self.saved = 0
        self.dropped = 0
        self._files: dict[str, object] | None = None
        self._index_fh = None
        self._session: Path | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="raw-writer",
        )
        self._thread.start()

    def open_session(self, session_dir: Path) -> None:
        """Close previous session files (if any) and prepare for a new one."""
        self._q.put(("__open__", session_dir))

    def close_session(self) -> None:
        self._q.put(("__close__", None))

    def submit(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ch3: np.ndarray,
    ) -> None:
        try:
            self._q.put_nowait(("__frame__", (ch1, ch2, ch3)))
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 50, 200, 1000):
                log.warning(
                    "Writer behind — dropped %d (%d written)",
                    self.dropped, self.saved,
                )

    def drain_and_stop(self, timeout: float = 120.0) -> None:
        self.close_session()
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            if self._stop.is_set() and self._q.empty():
                self._close_files()
                return
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self._close_files()
                return
            kind, payload = item
            try:
                if kind == "__open__":
                    self._close_files()
                    self._open_files(payload)
                elif kind == "__close__":
                    self._close_files()
                elif kind == "__frame__":
                    if self._files is None:
                        continue
                    self._write_frame(*payload)
            except Exception:
                log.exception("raw-writer error (%s)", kind)

    def _open_files(self, session_dir: Path) -> None:
        root = session_dir / "raw_frames"
        files = {}
        for ch in ("ch1", "ch2", "ch3"):
            (root / ch).mkdir(parents=True, exist_ok=True)
            files[ch] = open(root / ch / "stream.raw", "wb", buffering=_FILE_BUFFER)
        self._index_fh = open(
            root / "index.jsonl", "w", encoding="utf-8", buffering=1024 * 1024,
        )
        self._files = files
        self._session = session_dir
        self.saved = 0
        log.info("Raw writer opened → %s", session_dir.name)

    def _close_files(self) -> None:
        if self._files is not None:
            for fh in self._files.values():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            self._files = None
        if self._index_fh is not None:
            try:
                self._index_fh.flush()
                self._index_fh.close()
            except Exception:
                pass
            self._index_fh = None
        if self._session is not None:
            log.info(
                "Raw writer closed → %s  frames=%d  dropped=%d",
                self._session.name, self.saved, self.dropped,
            )
            self._session = None

    def _write_frame(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ch3: np.ndarray,
    ) -> None:
        assert self._files is not None and self._index_fh is not None
        self.saved += 1
        n = self.saved
        entry: dict = {"i": n}
        for name, frame in (("ch1", ch1), ("ch2", ch2), ("ch3", ch3)):
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            self._files[name].write(memoryview(frame))
            entry[name] = {"shape": list(frame.shape), "dtype": str(frame.dtype)}
        self._index_fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        if n == 1 or n % 50 == 0:
            log.info("Wrote %d raw frames (%d dropped)", n, self.dropped)


# ── raw → BMP (camera must be down) ───────────────────────────────────────────

def convert_session_to_bmp(session_dir: Path) -> int:
    """Convert stream.raw + index.jsonl → per-frame BMP. Returns # images written."""
    root = session_dir / "raw_frames"
    index_path = root / "index.jsonl"
    if not index_path.exists():
        log.warning("No index.jsonl in %s — skip convert", session_dir.name)
        return 0

    streams: dict[str, object] = {}
    written = 0
    try:
        for ch in ("ch1", "ch2", "ch3"):
            p = root / ch / "stream.raw"
            if p.exists():
                streams[ch] = open(p, "rb", buffering=_FILE_BUFFER)
        if not streams:
            return 0

        log.info("Converting %s → BMP …", session_dir.name)
        t0 = time.perf_counter()
        with index_path.open("r", encoding="utf-8") as idx:
            for line in idx:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                n = int(entry["i"])
                fname = f"frame_{n:06d}.bmp"
                for ch in ("ch1", "ch2", "ch3"):
                    meta = entry.get(ch)
                    if meta is None or ch not in streams:
                        continue
                    shape = tuple(meta["shape"])
                    dtype = np.dtype(meta["dtype"])
                    nbytes = int(np.prod(shape) * dtype.itemsize)
                    buf = streams[ch].read(nbytes)
                    if len(buf) != nbytes:
                        log.warning("Short read %s frame %d", ch, n)
                        continue
                    arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
                    path = root / ch / fname
                    if not cv2.imwrite(str(path), arr):
                        log.error("imwrite failed: %s", path)
                    else:
                        written += 1
        log.info(
            "Convert done %s — %d BMPs in %.1fs",
            session_dir.name, written, time.perf_counter() - t0,
        )
    finally:
        for fh in streams.values():
            try:
                fh.close()
            except Exception:
                pass
        # Remove bulky intermediates
        for ch in ("ch1", "ch2", "ch3"):
            try:
                (root / ch / "stream.raw").unlink(missing_ok=True)
            except Exception:
                pass
        try:
            index_path.unlink(missing_ok=True)
        except Exception:
            pass
    return written


# ── Camera helpers ────────────────────────────────────────────────────────────

def _load_camera_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        full = yaml.safe_load(fh) or {}
    cfg = dict(full.get("camera", {}))
    cfg["mode"] = "jai"
    return cfg


def _new_session() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = SESSIONS_DIR / ts
    for ch in ("ch1", "ch2", "ch3"):
        (session / "raw_frames" / ch).mkdir(parents=True, exist_ok=True)
    return session


def _poll_key() -> str | None:
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        if msvcrt.kbhit():
            msvcrt.getch()
        return None
    try:
        return ch.decode("ascii", errors="ignore").lower()
    except Exception:
        return None


def _connect_with_retry(cam_cfg: dict) -> CameraInterface | None:
    for attempt in range(1, _CONNECT_RETRIES + 1):
        log.info("Connect attempt %d/%d …", attempt, _CONNECT_RETRIES)
        cam = CameraInterface(dict(cam_cfg))
        if not cam.connect():
            log.error("connect() failed")
            continue
        if cam._mode != "jai":  # noqa: SLF001
            log.error("JAI not available (mock fallback). Close eBUS Player?")
            cam.disconnect()
            continue

        log.info("Waiting for first live frame (%.0fs) …", _FIRST_FRAME_TIMEOUT_S)
        t0 = time.time()
        last_idx = -1
        while time.time() - t0 < _FIRST_FRAME_TIMEOUT_S:
            triplet = cam.grab()
            if triplet is not None and triplet.frame_idx != last_idx:
                log.info(
                    "Live frames OK  frame_idx=%d  ch1=%s",
                    triplet.frame_idx, getattr(triplet.ch1, "shape", None),
                )
                return cam
            time.sleep(0.01)

        log.error("No frames within %.0fs — retry", _FIRST_FRAME_TIMEOUT_S)
        cam.disconnect()
        time.sleep(1.0)
    return None


def main() -> int:
    log.info("Standalone collector  ·  root=%s", APP_ROOT)
    log.info("Sessions → %s", SESSIONS_DIR)
    log.info("Focus THIS terminal.  S/SPACE = toggle save   Q = quit")
    log.info(
        "Live writes stream.raw (bit-exact).  BMP convert runs on Quit "
        "after camera disconnect."
    )

    cam_cfg = _load_camera_cfg()
    cam = _connect_with_retry(cam_cfg)
    if cam is None:
        log.error(
            "Could not get live frames.\n"
            "  • Close eBUS Player / other camera apps\n"
            "  • Check 10GigE link\n"
            "  • Power-cycle camera if needed"
        )
        return 1

    writer = _RawWriter()
    sessions: list[Path] = []
    saving = False
    session_dir: Path | None = None
    frame_n = 0
    last_idx = -1
    last_frame_t = time.time()
    last_status_t = 0.0

    print()
    print("  Camera LIVE.  Press S to record, Q to quit.")
    print("  (BMP files are created on Quit — during Save we write .raw)")
    print()

    try:
        while True:
            key = _poll_key()
            if key == "q":
                break
            if key in ("s", " "):
                if not saving:
                    session_dir = _new_session()
                    sessions.append(session_dir)
                    frame_n = 0
                    saving = True
                    writer.open_session(session_dir)
                    log.info("SAVE ON  → %s", session_dir)
                    print(f"  >>> SAVE ON   {session_dir}")
                else:
                    saving = False
                    writer.close_session()
                    log.info(
                        "SAVE OFF → %s  (queued=%d written=%d dropped=%d)",
                        session_dir, frame_n, writer.saved, writer.dropped,
                    )
                    print(
                        f"  >>> SAVE OFF  queued={frame_n}  "
                        f"written={writer.saved}  dropped={writer.dropped}"
                    )
                    session_dir = None

            triplet = cam.grab()
            now = time.time()

            if triplet is None or triplet.frame_idx == last_idx:
                # Stream dead? reconnect.
                if now - last_frame_t >= _STREAM_DEAD_S:
                    was_saving = saving
                    if saving:
                        saving = False
                        writer.close_session()
                        print("  !!! Stream lost during SAVE — closing session")
                        session_dir = None
                    log.warning(
                        "No frames for %.1fs — reconnecting …", now - last_frame_t,
                    )
                    print("  !!! Reconnecting camera …")
                    try:
                        cam.disconnect()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    cam = _connect_with_retry(cam_cfg)
                    if cam is None:
                        log.error("Reconnect failed — quitting")
                        break
                    last_idx = -1
                    last_frame_t = time.time()
                    print("  Camera LIVE again.")
                    if was_saving:
                        print("  (Save was ON — press S to start a new session)")
                else:
                    time.sleep(0.001)
                continue

            last_idx = triplet.frame_idx
            last_frame_t = now

            if saving and session_dir is not None:
                frame_n += 1
                writer.submit(triplet.ch1, triplet.ch2, triplet.ch3)

            if now - last_status_t >= 2.0:
                last_status_t = now
                cam_fps = cam.grab_fps() if hasattr(cam, "grab_fps") else 0.0
                if saving:
                    print(
                        f"  REC  frames={frame_n}  written={writer.saved}  "
                        f"dropped={writer.dropped}  cam_fps={cam_fps:.1f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  IDLE  cam_fps={cam_fps:.1f}  (press S to record)",
                        flush=True,
                    )
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        saving = False
        log.info("Stopping writer …")
        writer.drain_and_stop()
        log.info("Disconnecting camera …")
        try:
            cam.disconnect()
        except Exception:
            pass

        # Convert all sessions to BMP now that GigE is down.
        total_bmp = 0
        for s in sessions:
            total_bmp += convert_session_to_bmp(s)
        log.info(
            "Done.  sessions=%d  BMP images=%d  raw-drops=%d",
            len(sessions), total_bmp, writer.dropped,
        )
        print(f"\n  Finished.  {len(sessions)} session(s), {total_bmp} BMP files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
