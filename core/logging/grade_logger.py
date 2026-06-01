"""
core/logging/grade_logger.py
=============================
CSV grade logger for the apple sorting system.

Writes one row per committed apple grade to a timestamped CSV file.

Columns (from config.yaml logging.csv_columns):
  timestamp          ISO-8601 datetime of grade commit
  apple_id           sequential grading number (#1, #2, ...)
  lane               conveyor lane (1 / 2 / 3)
  grade              Fresh | Processing | Cull
  confidence         model confidence [0.0 - 1.0]
  size_px            peak bounding-box min-side in pixels (always present)
  size_mm            estimated equatorial diameter in mm (null until calibrated)
  size_cf            scale factor r(x) applied in mm/px (null until calibrated)
  conveyor_speed_aps apples-per-second setting at time of grade
  outlet_fired       A | B | C

Usage
-----
    logger = GradeLogger(cfg["logging"], output_dir="data/")
    logger.open()                         # creates file, writes header
    logger.write(rec, outlet, speed_aps)  # one call per GradeRecord
    logger.close()                        # flush + close
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class GradeLogger:
    """
    Thread-safe CSV writer for apple grade records.
    Call open() before write(), close() when done.
    """

    # Canonical column order - must match config.yaml logging.csv_columns
    COLUMNS = [
        "timestamp",
        "apple_id",
        "lane",
        "grade",
        "confidence",
        "size_px",
        "size_mm",
        "size_cf",
        "conveyor_speed_aps",
        "outlet_fired",
    ]

    def __init__(self, log_cfg: dict, output_dir: Optional[str] = None) -> None:
        self._enabled    = log_cfg.get("enabled", False)
        self._output_dir = Path(output_dir or log_cfg.get("output_dir", "data/"))
        self._file       = None
        self._writer     = None
        self._path: Optional[Path] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Create a new timestamped CSV file and write the header row."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._output_dir / f"grades_{ts}.csv"
        try:
            self._file = open(self._path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.COLUMNS,
                extrasaction="ignore",
            )
            self._writer.writeheader()
            self._file.flush()
            log.info("GradeLogger: opened %s", self._path)
        except OSError as e:
            log.error("GradeLogger: could not open file %s — %s", self._path, e)
            self._file   = None
            self._writer = None

    def write(self, rec, outlet: str, speed_aps: int) -> None:
        """
        Write one row for a committed GradeRecord.

        Parameters
        ----------
        rec      : GradeRecord  (from gui.workers.tracker)
        outlet   : "A" | "B" | "C"
        speed_aps: conveyor speed in apples-per-second (from left panel slider)
        """
        if self._writer is None:
            return

        # Compute scale factor from size_px and size_mm if both are available
        size_cf: Optional[float] = None
        if rec.size_px is not None and rec.size_mm is not None and rec.size_px > 0:
            size_cf = round(rec.size_mm / rec.size_px, 6)

        row = {
            "timestamp":          datetime.now().isoformat(timespec="milliseconds"),
            "apple_id":           rec.seq_id,
            "lane":               rec.lane,
            "grade":              rec.class_name,
            "confidence":         round(rec.confidence, 4),
            "size_px":            round(rec.size_px, 1) if rec.size_px is not None else "",
            "size_mm":            round(rec.size_mm, 1) if rec.size_mm is not None else "",
            "size_cf":            size_cf if size_cf is not None else "",
            "conveyor_speed_aps": speed_aps,
            "outlet_fired":       outlet,
        }
        try:
            self._writer.writerow(row)
            self._file.flush()   # ensure row is visible even if the process crashes
        except OSError as e:
            log.warning("GradeLogger: write failed — %s", e)

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
                log.info("GradeLogger: closed %s", self._path)
            except OSError:
                pass
            finally:
                self._file   = None
                self._writer = None

    @property
    def is_open(self) -> bool:
        return self._file is not None and not self._file.closed

    @property
    def path(self) -> Optional[Path]:
        return self._path
