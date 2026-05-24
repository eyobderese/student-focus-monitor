"""
Video ingestion and preprocessing module.

Handles webcam capture, horizontal flip, CLAHE contrast enhancement,
colour space conversion, and config-driven frame skipping.

CV concepts: image flipping, CLAHE contrast enhancement,
             color space transforms, frame sampling.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Union


def load_config(path: str | Path = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


class VideoIngestion:
    """Opens a webcam (or file), applies preprocessing, and yields frames."""

    WIDTH  = 640
    HEIGHT = 480

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        config: dict | None = None,
    ) -> None:
        self.source = int(source) if str(source).isdigit() else str(source)
        self.config = config or load_config()
        self._skip  = self.config.get("analysis_frame_skip", 3)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoIngestion":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    def stream(
        self,
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yield (frame_index, bgr_frame, rgb_frame, gray_eq_frame).

        Every frame is yielded (for smooth display).
        Full analysis should only run when frame_index % skip == 0.
        """
        if self._cap is None:
            self.open()

        idx = 0
        while True:
            ret, bgr = self._cap.read()
            if not ret:
                break

            bgr   = cv2.flip(bgr, 1)                               # mirror
            bgr   = cv2.resize(bgr, (self.WIDTH, self.HEIGHT))
            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray_eq = self._clahe.apply(gray)                      # CLAHE
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            yield idx, bgr, rgb, gray_eq
            idx += 1

    @property
    def should_analyze(self) -> callable:
        """Return a function that returns True on every Nth frame."""
        return lambda idx: idx % self._skip == 0


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, gray_eq in vid.stream():
            cv2.imshow("BGR",     bgr)
            cv2.imshow("CLAHE",   gray_eq)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 300:
                break
    cv2.destroyAllWindows()
    print("Ingestion test complete.")
