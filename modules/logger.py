"""
Session logger module.

Writes one CSV row per analyzed frame with all signals and scores.
Buffers 30 rows before flushing to avoid constant disk I/O.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

FIELDNAMES = [
    "timestamp",
    "frame_index",
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "blink_rate",
    "eye_state_ear",
    "eye_state_cnn",
    "yawning",
    "yaw",
    "pitch",
    "roll",
    "head_direction",
    "attention_state",
    "raw_focus_score",
    "smoothed_focus_score",
    "focus_category",
    "alert_fired",
]

FLUSH_EVERY = 30


class SessionLogger:
    """Append engagement rows to CSV during a study session."""

    def __init__(
        self, output_path: str | Path = "outputs/session_log.csv"
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file        = open(self.path, "w", newline="")
        self._writer      = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._buffer: list = []
        self._session_start = time.time()

    # ------------------------------------------------------------------
    def log(self, data: Dict[str, Any]) -> None:
        """Buffer one row; auto-flush every FLUSH_EVERY rows."""
        data["timestamp"] = round(time.time() - self._session_start, 3)
        # Fill missing columns with empty string
        row = {k: data.get(k, "") for k in FIELDNAMES}
        self._buffer.append(row)
        if len(self._buffer) >= FLUSH_EVERY:
            self._flush()

    def _flush(self) -> None:
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()

    def close(self) -> Path:
        self._flush()
        self._file.close()
        print(f"[Logger] Session log → {self.path}")
        return self.path

    def session_duration(self) -> float:
        return time.time() - self._session_start

    # Context manager support
    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name

    logger = SessionLogger(output_path=tmp)
    for i in range(5):
        logger.log({
            "frame_index":         i * 3,
            "ear_left":            0.32,
            "ear_right":           0.30,
            "ear_avg":             0.31,
            "mar":                 0.15,
            "blink_rate":          14,
            "eye_state_ear":       "open",
            "eye_state_cnn":       "open",
            "yawning":             False,
            "yaw":                 2.1,
            "pitch":               -5.3,
            "roll":                0.8,
            "head_direction":      "forward",
            "attention_state":     "focused",
            "raw_focus_score":     88,
            "smoothed_focus_score":85,
            "focus_category":      "FOCUSED",
            "alert_fired":         "",
        })
    out = logger.close()
    print(f"\nWritten to {out}")
    with open(out) as f:
        print(f.read())
