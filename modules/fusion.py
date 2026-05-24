"""
Focus score fusion module.

Combines EAR, head pose, CNN attention state, and blink rate into a
0–100 focus score with exponential moving average smoothing.
"""

from __future__ import annotations

from typing import Tuple


def compute_focus_score(
    ear: float,
    head_direction: str,
    attention_state: str,
    blink_rate: float,
    config: dict,
) -> Tuple[int, str]:
    """
    Compute weighted focus score and category.

    Weights (from config.json):
        ear            0.30
        head_pose      0.30
        attention_cnn  0.30
        blink_rate     0.10

    Returns:
        (raw_score 0-100, category string)
    """
    # ---- EAR sub-score ----
    # EAR ≥ 0.35 → fully open = 100; scales linearly down to 0
    ear_score = float(min(100, max(0, (ear / 0.35) * 100)))

    # ---- Head pose sub-score ----
    pose_score = float(
        {"forward": 100, "up": 60, "down": 20, "side": 0}.get(head_direction, 50)
    )

    # ---- Attention CNN sub-score ----
    cnn_score = float(
        {
            "focused":    100,
            "neutral":     60,
            "yawning":     30,
            "distracted":  10,
            "drowsy":       0,
            "uncertain":   50,
        }.get(attention_state, 50)
    )

    # ---- Blink rate sub-score ----
    b_min = config["blink_rate_normal_min"]
    b_max = config["blink_rate_normal_max"]
    if b_min <= blink_rate <= b_max:
        blink_score = 100.0
    elif blink_rate < b_min:
        blink_score = float(max(0, 100 - (b_min - blink_rate) * 10))
    else:
        blink_score = float(max(0, 100 - (blink_rate - b_max) * 5))

    # ---- Weighted sum ----
    w = config["focus_score_weights"]
    score = (
        w["ear"]           * ear_score   +
        w["head_pose"]     * pose_score  +
        w["attention_cnn"] * cnn_score   +
        w["blink_rate"]    * blink_score
    )
    raw = round(score)

    # ---- Category ----
    if raw >= 65:
        category = "FOCUSED"
    elif raw >= 40:
        category = "NEUTRAL"
    else:
        category = "DISTRACTED"

    return raw, category


class FusionState:
    """Holds per-session EMA state for score smoothing."""

    EMA_ALPHA = 0.25   # weight for new observation (0.75 for history)

    def __init__(self, initial_score: int = 50) -> None:
        self._prev = float(initial_score)

    def smooth(self, new_score: int) -> int:
        self._prev = self.EMA_ALPHA * new_score + (1 - self.EMA_ALPHA) * self._prev
        return round(self._prev)

    @property
    def current(self) -> int:
        return round(self._prev)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    with open("config.json") as f:
        cfg = json.load(f)

    state = FusionState()
    cases = [
        (0.30, "forward",  "focused",    15),
        (0.18, "forward",  "drowsy",      5),
        (0.28, "side",     "distracted", 12),
        (0.32, "forward",  "yawning",    20),
    ]
    for ear, direction, attn, bpm in cases:
        raw, cat = compute_focus_score(ear, direction, attn, bpm, cfg)
        smooth   = state.smooth(raw)
        print(
            f"EAR={ear:.2f} {direction:7s} {attn:11s} bpm={bpm:2d} → "
            f"raw={raw:3d} smooth={smooth:3d} [{cat}]"
        )
