"""
Facial landmark extraction module using MediaPipe FaceMesh.

Extracts 468 landmarks (x, y, z) from a single face.
refine_landmarks=True enables iris landmarks (469–477).

CV concepts: facial landmark detection, coordinate normalisation,
             graph-based CNN inference (MediaPipe internals).
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Landmark index constants (used by all downstream modules)
# ---------------------------------------------------------------------------
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH      = [61, 291,  39, 269,   0,  17, 405, 181]
POSE_LM    = [1, 152, 33, 263, 61, 291]   # nose, chin, eye corners, mouth corners
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


class LandmarkExtractor:
    """Runs FaceMesh and returns pixel-space landmark tuples."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        mp_fm = mp.solutions.face_mesh  # type: ignore[attr-defined]
        self._mesh = mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # iris landmarks required
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    def extract(
        self, rgb_frame: np.ndarray
    ) -> Optional[List[Tuple[int, int, float]]]:
        """
        Extract landmarks from an RGB frame.

        Returns:
            List of 478 (x_px, y_px, z_norm) tuples, or None if not detected.
            Indices 0-467 are face mesh; 468-477 are iris (refine_landmarks).
        """
        h, w  = rgb_frame.shape[:2]
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        return [
            (int(lm.x * w), int(lm.y * h), float(lm.z))
            for lm in lms
        ]

    # ------------------------------------------------------------------
    def draw_landmarks(
        self,
        bgr_frame: np.ndarray,
        landmarks: List[Tuple[int, int, float]],
        draw_all: bool = False,
    ) -> np.ndarray:
        """Overlay selected or all landmarks onto a BGR frame."""
        if draw_all:
            for x, y, _ in landmarks:
                cv2.circle(bgr_frame, (x, y), 1, (0, 200, 200), -1)
        else:
            for indices, colour in [
                (LEFT_EYE,  (0, 255, 0)),
                (RIGHT_EYE, (0, 255, 0)),
                (MOUTH,     (0, 100, 255)),
                (POSE_LM,   (255, 100, 0)),
            ]:
                for i in indices:
                    x, y, _ = landmarks[i]
                    cv2.circle(bgr_frame, (x, y), 2, colour, -1)
        return bgr_frame

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "LandmarkExtractor":
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
    extractor = LandmarkExtractor()

    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            lms = extractor.extract(rgb)
            if lms:
                bgr = extractor.draw_landmarks(bgr, lms, draw_all=False)
                cv2.putText(bgr, f"{len(lms)} landmarks", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.imshow("Landmarks test", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 300:
                break

    cv2.destroyAllWindows()
    extractor.close()
