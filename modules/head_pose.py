"""
Head pose estimation using MediaPipe FaceMesh landmarks + OpenCV solvePnP.

Computes yaw, pitch, roll Euler angles and classifies head direction as
forward / down / side / up. Draws 3-D axis arrows on the nose tip.

CV concepts: camera projection model, PnP problem, Rodrigues rotation
             formula, RQ decomposition, 3D→2D axis projection overlay.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from modules.landmark_extractor import POSE_LM


# ---------------------------------------------------------------------------
# Generic 3-D face model (millimetres)
# ---------------------------------------------------------------------------
MODEL_POINTS_3D = np.array(
    [
        (0.0,    0.0,    0.0),      # nose tip         — lm 1
        (0.0,   -330.0, -65.0),     # chin             — lm 152
        (-225.0, 170.0, -135.0),    # left eye corner  — lm 33
        (225.0,  170.0, -135.0),    # right eye corner — lm 263
        (-150.0, -150.0, -125.0),   # left mouth       — lm 61
        (150.0,  -150.0, -125.0),   # right mouth      — lm 291
    ],
    dtype=np.float64,
)

_AXIS_3D = np.float32([[60, 0, 0], [0, 60, 0], [0, 0, 60], [0, 0, 0]])


class HeadPoseEstimator:
    """Estimates head orientation from 6 facial landmarks."""

    def __init__(self, config: dict) -> None:
        self._yaw_thresh       = config["head_yaw_distracted_threshold"]
        self._pitch_down_thresh = config["head_pitch_down_threshold"]
        self._pitch_up_thresh   = config["head_pitch_up_threshold"]

    # ------------------------------------------------------------------
    def estimate(
        self,
        landmarks: List[Tuple[int, int, float]],
        frame: np.ndarray,
    ) -> Optional[Dict[str, object]]:
        """
        Estimate head pose and return angles + direction.

        Args:
            landmarks: 468+ pixel-space (x, y, z) tuples from LandmarkExtractor.
            frame:     BGR frame (used for camera matrix dimensions and drawing).

        Returns:
            Dict {yaw, pitch, roll, direction, rvec, tvec} or None on failure.
        """
        h, w = frame.shape[:2]
        image_pts = np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in POSE_LM],
            dtype=np.float64,
        )

        focal_len     = w
        camera_matrix = np.array(
            [[focal_len, 0, w / 2], [0, focal_len, h / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw   = float(angles[1])
        roll  = float(angles[2])

        direction = self._classify(yaw, pitch)

        return {
            "yaw":       yaw,
            "pitch":     pitch,
            "roll":      roll,
            "direction": direction,
            "rvec":      rvec,
            "tvec":      tvec,
            "camera_matrix": camera_matrix,
            "dist_coeffs":   dist_coeffs,
        }

    # ------------------------------------------------------------------
    def draw_axes(
        self,
        frame: np.ndarray,
        pose_result: Dict,
        origin_lm: Tuple[int, int, float],
    ) -> np.ndarray:
        """
        Draw 3-D coordinate axes onto the frame originating from the nose tip.
        Red = X axis, Green = Y axis, Blue = Z axis.
        """
        origin = (int(origin_lm[0]), int(origin_lm[1]))
        imgpts, _ = cv2.projectPoints(
            _AXIS_3D,
            pose_result["rvec"],
            pose_result["tvec"],
            pose_result["camera_matrix"],
            pose_result["dist_coeffs"],
        )
        imgpts = imgpts.astype(int)
        nose   = tuple(imgpts[3].ravel())
        cv2.arrowedLine(frame, nose, tuple(imgpts[0].ravel()), (0,   0, 255), 2, tipLength=0.3)
        cv2.arrowedLine(frame, nose, tuple(imgpts[1].ravel()), (0, 255,   0), 2, tipLength=0.3)
        cv2.arrowedLine(frame, nose, tuple(imgpts[2].ravel()), (255, 0,   0), 2, tipLength=0.3)
        return frame

    # ------------------------------------------------------------------
    def _classify(self, yaw: float, pitch: float) -> str:
        if (abs(yaw) < self._yaw_thresh and
                self._pitch_down_thresh < pitch < self._pitch_up_thresh):
            return "forward"
        if pitch <= self._pitch_down_thresh:
            return "down"
        if abs(yaw) >= self._yaw_thresh:
            return "side"
        return "up"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json
    from modules.ingestion import VideoIngestion
    from modules.face_detector import FaceDetector
    from modules.landmark_extractor import LandmarkExtractor

    with open("config.json") as f:
        cfg = json.load(f)

    src       = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    detector  = FaceDetector()
    extractor = LandmarkExtractor()
    estimator = HeadPoseEstimator(cfg)

    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            if idx % cfg["analysis_frame_skip"] == 0:
                if detector.detect(rgb):
                    lms = extractor.extract(rgb)
                    if lms:
                        pose = estimator.estimate(lms, bgr)
                        if pose:
                            estimator.draw_axes(bgr, pose, lms[1])
                            cv2.putText(
                                bgr,
                                f"{pose['direction']}  Y:{pose['yaw']:.1f} "
                                f"P:{pose['pitch']:.1f} R:{pose['roll']:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 255), 2,
                            )
            cv2.imshow("Head Pose test", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 300:
                break

    cv2.destroyAllWindows()
    detector.close()
    extractor.close()
