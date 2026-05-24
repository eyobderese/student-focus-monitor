"""
Single-face detection module using MediaPipe FaceDetection.

Uses the short-range model (model_selection=0) optimised for webcam distance
(<2 m). Only the first (most confident) detection is returned — this system
monitors a single student.

CV concepts: CNN-based face detection, bounding box extraction,
             absence event detection.
"""

from __future__ import annotations

import time
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class FaceDetector:
    """Detects a single face and tracks how long it has been absent."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        absence_timeout_seconds: float = 5.0,
    ) -> None:
        mp_fd = mp.solutions.face_detection  # type: ignore[attr-defined]
        self._detector = mp_fd.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self._timeout = absence_timeout_seconds
        self._absent_since: float | None = None   # timestamp when face last disappeared

    # ------------------------------------------------------------------
    def detect(
        self, rgb_frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Run detection on an RGB frame.

        Returns:
            (x1, y1, x2, y2) pixel bounding box of the first face, or None.
        """
        h, w  = rgb_frame.shape[:2]
        result = self._detector.process(rgb_frame)

        if not result.detections:
            if self._absent_since is None:
                self._absent_since = time.time()
            return None

        # Face found — reset absence timer
        self._absent_since = None
        det = result.detections[0]
        bb  = det.location_data.relative_bounding_box
        x1  = max(0, int(bb.xmin * w))
        y1  = max(0, int(bb.ymin * h))
        x2  = min(w, int((bb.xmin + bb.width)  * w))
        y2  = min(h, int((bb.ymin + bb.height) * h))
        return x1, y1, x2, y2

    @property
    def absence_duration(self) -> float:
        """Seconds since the face disappeared. 0 if the face is present."""
        if self._absent_since is None:
            return 0.0
        return time.time() - self._absent_since

    @property
    def is_absent_too_long(self) -> bool:
        return self.absence_duration >= self._timeout

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from ingestion import VideoIngestion

    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    detector = FaceDetector()

    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            bbox = detector.detect(rgb)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, "FACE", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                absent = detector.absence_duration
                cv2.putText(bgr, f"NO FACE  ({absent:.1f}s)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Detector test", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 300:
                break

    cv2.destroyAllWindows()
    detector.close()
