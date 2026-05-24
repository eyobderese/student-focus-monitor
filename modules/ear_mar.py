"""
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink rate, and
yawn detection module.

This is the core classical CV module. Everything here is pure geometry —
no neural networks.

CV concepts: Euclidean distance geometry on landmarks, aspect ratio
             signal processing, temporal event detection, rolling window
             blink-rate statistics.
"""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance as dist

from modules.landmark_extractor import LEFT_EYE, RIGHT_EYE, MOUTH


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------
def compute_ear(eye_landmarks: List[Tuple[int, int, float]]) -> float:
    """
    Eye Aspect Ratio from 6 eye landmark points.

         |p2-p6| + |p3-p5|
    EAR = ──────────────────
              2 * |p1-p4|

    Points are ordered: left-corner, top-left, top-right,
                        right-corner, bottom-right, bottom-left.
    """
    p1, p2, p3, p4, p5, p6 = [(lm[0], lm[1]) for lm in eye_landmarks]
    v1 = dist.euclidean(p2, p6)
    v2 = dist.euclidean(p3, p5)
    h  = dist.euclidean(p1, p4)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_mar(mouth_landmarks: List[Tuple[int, int, float]]) -> float:
    """
    Mouth Aspect Ratio from 8 outer lip landmark points.

           |p2-p8| + |p3-p7| + |p4-p6|
    MAR = ─────────────────────────────
                   2 * |p1-p5|

    Points ordered: left-corner, top-left×3, right-corner, bottom-right×3.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = [(lm[0], lm[1]) for lm in mouth_landmarks]
    v  = dist.euclidean(p2, p8) + dist.euclidean(p3, p7) + dist.euclidean(p4, p6)
    h  = dist.euclidean(p1, p5)
    if h < 1e-6:
        return 0.0
    return v / (2.0 * h)


# ---------------------------------------------------------------------------
# Stateful analyser
# ---------------------------------------------------------------------------
class EARMARAnalyzer:
    """
    Maintains per-session counters for blink detection, yawn detection,
    and rolling blink-rate computation.
    """

    def __init__(self, config: dict) -> None:
        self._ear_thresh    = config["ear_drowsy_threshold"]
        self._ear_frames    = config["ear_closed_frames_threshold"]
        self._mar_thresh    = config["mar_yawn_threshold"]
        self._yawn_frames   = config["yawn_min_duration_frames"]

        self._ear_counter   = 0   # consecutive closed-eye frames
        self._yawn_counter  = 0   # consecutive open-mouth frames

        # Blink timestamps deque (rolling 60-second window)
        self._blink_times: collections.deque = collections.deque()
        self._total_blinks = 0
        self._total_yawns  = 0

        # Events list for logging: list of (timestamp, event_type)
        self.events: List[Tuple[float, str]] = []

    # ------------------------------------------------------------------
    def analyze(
        self,
        landmarks: List[Tuple[int, int, float]],
        frame_index: int,
    ) -> Dict[str, object]:
        """
        Compute EAR, MAR, blink rate and yawn flag for one frame.

        Returns dict with keys:
            ear, ear_left, ear_right, mar,
            blink_rate, eye_state, yawning,
            total_blinks, total_yawns
        """
        # ---- EAR per eye ----
        left_pts  = [landmarks[i] for i in LEFT_EYE]
        right_pts = [landmarks[i] for i in RIGHT_EYE]
        ear_l     = compute_ear(left_pts)
        ear_r     = compute_ear(right_pts)
        ear_avg   = (ear_l + ear_r) / 2.0

        # ---- MAR ----
        mouth_pts = [landmarks[i] for i in MOUTH]
        mar       = compute_mar(mouth_pts)

        # ---- Blink detection ----
        eye_closed = ear_avg < self._ear_thresh
        if eye_closed:
            self._ear_counter += 1
        else:
            if self._ear_counter >= self._ear_frames:
                # Eyes were closed long enough → count as blink
                now = time.time()
                self._blink_times.append(now)
                self._total_blinks += 1
                self.events.append((now, "BLINK"))
            self._ear_counter = 0

        # ---- Rolling blink rate (blinks in the last 60 s) ----
        cutoff = time.time() - 60.0
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        blink_rate = len(self._blink_times)   # blinks per minute

        # ---- Yawn detection ----
        mouth_open = mar > self._mar_thresh
        if mouth_open:
            self._yawn_counter += 1
        else:
            if self._yawn_counter >= self._yawn_frames:
                now = time.time()
                self._total_yawns += 1
                self.events.append((now, "YAWN"))
            self._yawn_counter = 0

        yawning   = self._yawn_counter >= self._yawn_frames
        eye_state = "closed" if self._ear_counter >= self._ear_frames else "open"
        drowsy    = self._ear_counter >= self._ear_frames

        return {
            "ear":          ear_avg,
            "ear_left":     ear_l,
            "ear_right":    ear_r,
            "mar":          mar,
            "blink_rate":   blink_rate,
            "eye_state":    eye_state,
            "yawning":      yawning,
            "drowsy":       drowsy,
            "total_blinks": self._total_blinks,
            "total_yawns":  self._total_yawns,
        }

    def reset(self) -> None:
        self._ear_counter  = 0
        self._yawn_counter = 0
        self._blink_times.clear()
        self._total_blinks = 0
        self._total_yawns  = 0
        self.events.clear()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import cv2
    import json
    from modules.ingestion import VideoIngestion
    from modules.face_detector import FaceDetector
    from modules.landmark_extractor import LandmarkExtractor

    with open("config.json") as f:
        cfg = json.load(f)

    src      = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    detector = FaceDetector()
    extractor = LandmarkExtractor()
    analyzer  = EARMARAnalyzer(cfg)

    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            if idx % cfg["analysis_frame_skip"] == 0:
                if detector.detect(rgb):
                    lms = extractor.extract(rgb)
                    if lms:
                        data = analyzer.analyze(lms, idx)
                        print(
                            f"EAR={data['ear']:.3f}  MAR={data['mar']:.3f}  "
                            f"blink_rate={data['blink_rate']:2d}  "
                            f"eye={data['eye_state']}  yawn={data['yawning']}"
                        )
                        color = (0, 0, 200) if data["drowsy"] else (0, 200, 0)
                        cv2.putText(bgr,
                            f"EAR:{data['ear']:.2f}  MAR:{data['mar']:.2f}  "
                            f"bpm:{data['blink_rate']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.imshow("EAR/MAR test", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 600:
                break

    cv2.destroyAllWindows()
    detector.close()
    extractor.close()
