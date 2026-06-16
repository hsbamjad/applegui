"""
core/control/sorter_controller.py
==================================
Pneumatic sorting actuator controller.

Hardware: 3 sorting units (one per lane), each with:
  - Submodule A → Solenoid valve A → Air cylinder → Paddle → Outlet A (Fresh)
  - Submodule B → Solenoid valve B → Air cylinder → Paddle → Outlet B (Processing)
  - Default (no fire)                                       → Outlet C (Cull)
  - Arduino board receiving commands via USB/Serial (PySerial)

Command Protocol:
  Send ASCII string over serial: "<lane><submodule>\n"
  e.g. "1A\n" → Lane 1, Submodule A (Fresh)
       "2B\n" → Lane 2, Submodule B (Processing)
  No command → apple falls to Outlet C (Cull) — safe default

Timing:
  delay_ms = (camera_to_gate_m / conveyor_speed_m_s) * 1000
  Command is scheduled delay_ms after the grade decision is made.

Grades:
  Fresh      → Outlet A (Submodule A fires)
  Processing → Outlet B (Submodule B fires)
  Cull       → Outlet C (no command, default)
"""

from __future__ import annotations

from core.log import get_logger
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = get_logger(__name__)


class SortOutlet(Enum):
    """Physical outlet destinations on the sorting unit."""
    A = "A"   # Fresh — Submodule A fires
    B = "B"   # Processing — Submodule B fires
    C = "C"   # Cull — default, no command needed


# Grade → action digit mapping (confirmed with hardware person + Arduino sketch)
# Arduino doAction(): 1=Fresh, 2=Processing, 3=Cull
# Command format: "XYZ\n" — 3 chars, one per lane, 0 = no action for that lane
GRADE_TO_DIGIT: dict[str, int] = {
    "Fresh":      1,
    "Processing": 2,
    "Cull":       3,
}


@dataclass
class GradeCommand:
    """A pending sort command for one apple instance."""
    apple_id:       int          # Tracking ID from YOLO
    lane:           int          # 1, 2, or 3
    grade:          str          # "Fresh" | "Processing" | "Cull"
    confidence:     float        # 0.0–1.0 from model
    graded_at_ns:   int          # time.time_ns() when grade was decided
    fire_at_ns:     int          # time.time_ns() when command must be sent
    digit:          int          # Action digit for Arduino: 1/2/3


class SorterController:
    """
    Controls the 3-lane pneumatic sorting system via Arduino serial interface.

    Usage:
        controller = SorterController(config["sorter"], config["conveyor"])
        controller.start()
        controller.schedule(apple_id=42, lane=1, grade="Fresh", confidence=0.93)
        ...
        controller.stop()
    """

    def __init__(self, sorter_cfg: dict, conveyor_cfg: dict) -> None:
        self._mode             = sorter_cfg.get("mode", "simulation")
        self._port             = sorter_cfg.get("serial", {}).get("port", "COM3")
        self._baudrate         = sorter_cfg.get("serial", {}).get("baudrate", 9600)
        self._valve_pulse_ms   = sorter_cfg.get("valve_pulse_ms", 80)
        self._camera_to_gate_m = conveyor_cfg.get("camera_to_gate_m", 0.5)
        self._conveyor_speed   = conveyor_cfg.get("speed_apples_per_sec", 1)

        self._serial           = None      # serial.Serial instance when connected
        self._queue: list[GradeCommand] = []
        self._lock             = threading.Lock()
        self._running          = False
        self._thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {"total": 0, "fresh": 0, "processing": 0, "cull": 0, "missed": 0}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to Arduino (if serial mode) and start dispatch thread."""
        if self._mode == "serial":
            self._connect_serial()

        self._running = True
        self._thread  = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()
        log.info(f"SorterController started in '{self._mode}' mode.")

    def stop(self) -> None:
        """Stop dispatch thread and close serial connection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("SorterController stopped.")

    def schedule(
        self,
        apple_id: int,
        lane: int,
        grade: str,
        confidence: float,
        conveyor_speed_m_s: Optional[float] = None,
    ) -> None:
        """
        Schedule a sort command for an apple that has just been graded.

        Parameters
        ----------
        apple_id         : YOLO tracking ID
        lane             : Physical conveyor lane (1–3)
        grade            : "Fresh" | "Processing" | "Cull"
        confidence       : Model confidence (0.0–1.0)
        conveyor_speed_m_s : Live speed override (m/s). Uses config default if None.
        """
        speed    = conveyor_speed_m_s or self._conveyor_speed
        # camera_to_gate_m is 0.0 — Arduino handles timing via NIR sensor.
        # delay_ms kept for flexibility; with 0.0 it fires immediately.
        delay_ms = (self._camera_to_gate_m / speed) * 1000 if speed > 0 else 0.0

        now_ns   = time.time_ns()
        fire_ns  = now_ns + int(delay_ms * 1_000_000)
        digit    = GRADE_TO_DIGIT.get(grade, 3)   # unknown grade defaults to Cull

        cmd = GradeCommand(
            apple_id     = apple_id,
            lane         = lane,
            grade        = grade,
            confidence   = confidence,
            graded_at_ns = now_ns,
            fire_at_ns   = fire_ns,
            digit        = digit,
        )

        with self._lock:
            self._queue.append(cmd)
            self._queue.sort(key=lambda c: c.fire_at_ns)

        log.debug(
            f"Scheduled: apple={apple_id} lane={lane} grade={grade} "
            f"digit={digit} delay={delay_ms:.1f}ms"
        )

    def set_conveyor_speed(self, speed_m_s: float) -> None:
        """Update the live conveyor speed used for timing calculations."""
        self._conveyor_speed = speed_m_s

    def set_mode(self, mode: str) -> None:
        """
        Switch between 'simulation' and 'serial' at runtime.
        Called by the GUI Sorter toggle button.
        """
        if mode == self._mode:
            return
        if mode == "serial":
            self._connect_serial()
        elif mode == "simulation":
            if self._serial and self._serial.is_open:
                self._serial.close()
                log.info("Serial port closed — switching to simulation mode.")
        self._mode = mode
        log.info(f"SorterController mode → '{self._mode}'")

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def is_connected(self) -> bool:
        return self._mode == "simulation" or (
            self._serial is not None and self._serial.is_open
        )

    # ── Dispatch loop (background thread) ────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Background thread: fires commands at the scheduled time."""
        while self._running:
            now_ns = time.time_ns()
            with self._lock:
                due = [c for c in self._queue if c.fire_at_ns <= now_ns]
                for cmd in due:
                    self._queue.remove(cmd)

            for cmd in due:
                self._fire(cmd)

            time.sleep(0.001)   # 1ms polling resolution

    def _fire(self, cmd: GradeCommand) -> None:
        """
        Build and send the 3-char Arduino command for one graded apple.

        Format: "XYZ\n"
          X = Lane 1 action digit (0 if not this lane)
          Y = Lane 2 action digit (0 if not this lane)
          Z = Lane 3 action digit (0 if not this lane)

        Digits: 1=Fresh  2=Processing  3=Cull  0=no action (not queued by Arduino)
        """
        self._stats["total"] += 1
        grade_key = cmd.grade.lower()
        if grade_key in self._stats:
            self._stats[grade_key] += 1

        # Build 3-char command: digit in the correct lane slot, zeros elsewhere
        digits = ['0', '0', '0']
        digits[cmd.lane - 1] = str(cmd.digit)
        serial_cmd = "".join(digits) + "\n"

        if self._mode == "simulation":
            log.info(
                f"[SIM] FIRE: apple={cmd.apple_id} lane={cmd.lane} "
                f"grade={cmd.grade} digit={cmd.digit} cmd='{serial_cmd.strip()}' "
                f"conf={cmd.confidence:.2f}"
            )
        else:
            self._send_serial(serial_cmd)

    # ── Serial communication ──────────────────────────────────────────────────

    def _connect_serial(self) -> None:
        try:
            import serial
            self._serial = serial.Serial(
                port     = self._port,
                baudrate = self._baudrate,
                timeout  = 1.0,
            )
            log.info(f"Arduino connected on {self._port} @ {self._baudrate} baud.")
        except Exception as exc:
            log.error(f"Failed to connect to Arduino on {self._port}: {exc}")
            log.warning("Falling back to simulation mode.")
            self._mode = "simulation"

    def _send_serial(self, cmd: str) -> None:
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(cmd.encode("ascii"))
                log.debug(f"SERIAL TX: {cmd.strip()}")
            except Exception as exc:
                log.error(f"Serial write error: {exc}")
                self._stats["missed"] += 1
