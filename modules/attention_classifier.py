"""
Attention state CNN classifier (EyeStateNet).

Classifies each eye crop as open (1) or closed (0) using a lightweight
3-layer CNN trained on the MRL Eye Dataset.

Combines CNN prediction with EAR + head pose + yawn flag to produce a
single attention state label: focused / neutral / yawning / drowsy / distracted.

CV concepts: CNN architecture design, grayscale image classification,
             softmax confidence, multi-signal fusion.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.landmark_extractor import LEFT_EYE, RIGHT_EYE


# ---------------------------------------------------------------------------
# EyeStateNet architecture
# ---------------------------------------------------------------------------
class EyeStateNet(nn.Module):
    """
    Lightweight CNN for single-eye open/closed classification.
    Input: (1, 1, 32, 32) grayscale tensor, normalised to [0, 1].
    Output: logits for [closed=0, open=1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32,  kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1     = nn.Linear(128 * 4 * 4, 256)
        self.fc2     = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # → (32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # → (64,  8,  8)
        x = self.pool(F.relu(self.conv3(x)))   # → (128, 4,  4)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)              # → 2048
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Runtime classifier
# ---------------------------------------------------------------------------
class AttentionClassifier:
    """
    Extracts eye crops, runs EyeStateNet, and fuses with pose + EAR signals
    to produce a human-readable attention state.
    """

    EYE_SIZE       = 32
    DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "saved" / "eye_cnn.pt"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = torch.device("cpu")
        self.model  = EyeStateNet().to(self.device)

        weights = Path(model_path) if model_path else self.DEFAULT_WEIGHTS
        if weights.exists():
            state = torch.load(weights, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"[AttentionClassifier] Loaded {weights}")
        else:
            print(
                f"[AttentionClassifier] WARNING: weights not found at {weights}. "
                "Running with random weights — run models/train_eye_cnn.py first."
            )
        self.model.eval()

    # ------------------------------------------------------------------
    def predict(
        self,
        bgr_frame: np.ndarray,
        landmarks: List[Tuple[int, int, float]],
    ) -> Tuple[str, float]:
        """
        Classify eye state from the BGR frame using landmark-defined eye crops.

        Returns:
            (label, confidence) where label ∈ {'open', 'closed', 'uncertain'}.
        """
        left_state,  left_conf  = self._classify_eye(bgr_frame, landmarks, LEFT_EYE)
        right_state, right_conf = self._classify_eye(bgr_frame, landmarks, RIGHT_EYE)

        avg_conf = (left_conf + right_conf) / 2.0

        # Agree → confident result; disagree → uncertain
        if left_state == right_state:
            return left_state, avg_conf
        return "uncertain", avg_conf

    def get_attention_state(
        self,
        eye_state:      str,
        head_direction: str,
        ear:            float,
        yawning:        bool,
        config:         dict,
    ) -> str:
        """
        Fuse CNN eye state, head pose, EAR and yawn flag into one label.

        Returns one of: focused / neutral / yawning / drowsy / distracted.
        """
        if yawning:
            return "yawning"
        if eye_state == "closed" or ear < config["ear_drowsy_threshold"]:
            return "drowsy"
        if head_direction in ("side", "down"):
            return "distracted"
        if head_direction == "forward" and eye_state == "open":
            return "focused"
        return "neutral"

    # ------------------------------------------------------------------
    def _classify_eye(
        self,
        bgr_frame: np.ndarray,
        landmarks: List[Tuple[int, int, float]],
        eye_indices: List[int],
    ) -> Tuple[str, float]:
        """Crop one eye, run CNN, return (label, confidence)."""
        xs = [landmarks[i][0] for i in eye_indices]
        ys = [landmarks[i][1] for i in eye_indices]
        pad = 5
        h, w = bgr_frame.shape[:2]
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)

        if x2 <= x1 or y2 <= y1:
            return "uncertain", 0.5

        crop  = bgr_frame[y1:y2, x1:x2]
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray  = cv2.resize(gray, (self.EYE_SIZE, self.EYE_SIZE))
        tensor = (
            torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        ).to(self.device)   # (1, 1, 32, 32)

        with torch.no_grad():
            logits     = self.model(tensor)
            probs      = F.softmax(logits, dim=1)[0]
            pred_idx   = int(probs.argmax().item())
            confidence = float(probs.max().item())

        label = "open" if pred_idx == 1 else "closed"
        return label, confidence


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

    src        = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    detector   = FaceDetector()
    extractor  = LandmarkExtractor()
    classifier = AttentionClassifier()

    with VideoIngestion(source=src) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            if idx % cfg["analysis_frame_skip"] == 0:
                if detector.detect(rgb):
                    lms = extractor.extract(rgb)
                    if lms:
                        eye_label, conf = classifier.predict(bgr, lms)
                        cv2.putText(bgr, f"Eye: {eye_label} ({conf:.2f})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 200, 255), 2)
            cv2.imshow("Attention CNN test", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if idx > 300:
                break

    cv2.destroyAllWindows()
    detector.close()
    extractor.close()
