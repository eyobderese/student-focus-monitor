"""
Alert management module.

Triggers visual banner overlays and audio beeps for DROWSY, DISTRACTED,
ABSENT, and YAWN events. Beep WAV files are auto-generated using numpy +
scipy if they do not exist (no external audio files required).

Respects per-type cooldown to avoid alert spam.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import scipy.io.wavfile as wav_io

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Alert appearance
_ALERT_COLORS: Dict[str, tuple] = {
    "DROWSY":     (0,   0, 220),    # red
    "DISTRACTED": (0, 140, 255),    # orange
    "ABSENT":     (180,  0, 180),   # magenta
    "YAWN":       (0, 200, 200),    # yellow
}

_ALERT_MESSAGES: Dict[str, str] = {
    "DROWSY":     "⚠ DROWSY — Please take a break!",
    "DISTRACTED": "⚠ DISTRACTED — Focus on the screen",
    "ABSENT":     "⚠ ABSENT — No face detected",
    "YAWN":       "⚠ YAWN detected",
}

# Beep frequencies per alert type (Hz)
_BEEP_FREQ: Dict[str, int] = {
    "DROWSY":     520,
    "DISTRACTED": 660,
    "ABSENT":     880,
    "YAWN":       440,
}


# ---------------------------------------------------------------------------
# WAV generator
# ---------------------------------------------------------------------------
def _generate_beep(freq: int, duration: float = 0.3, rate: int = 22050) -> np.ndarray:
    t    = np.linspace(0, duration, int(duration * rate), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    # Fade in/out to avoid clicks
    fade = int(rate * 0.02)
    wave[:fade]  = (wave[:fade]  * np.linspace(0, 1, fade)).astype(np.int16)
    wave[-fade:] = (wave[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    return wave


def ensure_alert_sounds() -> None:
    """Generate missing WAV files into assets/."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rate = 22050
    for alert_type, freq in _BEEP_FREQ.items():
        path = ASSETS_DIR / f"alert_{alert_type.lower()}.wav"
        if not path.exists():
            wav_io.write(str(path), rate, _generate_beep(freq))
            print(f"[Alerts] Generated {path.name}")


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------
class AlertManager:
    """Manages alert triggering with cooldowns and visual overlay drawing."""

    def __init__(self, config: dict) -> None:
        self._cooldown      = config["alert_cooldown_seconds"]
        self._last_alert:   Dict[str, float] = {}
        self._distracted_counter = 0
        self._distracted_limit   = 10  # frames before distraction alert fires
        self.last_alert: Optional[str] = None   # for logger

        ensure_alert_sounds()

        # Try to initialise pygame audio (soft failure if not available)
        self._pygame_ok = False
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._sounds: Dict[str, object] = {
                t: pygame.mixer.Sound(str(ASSETS_DIR / f"alert_{t.lower()}.wav"))
                for t in _BEEP_FREQ
            }
            self._pygame_ok = True
        except Exception as exc:
            print(f"[Alerts] pygame not available — audio disabled ({exc})")

    # ------------------------------------------------------------------
    def check(
        self,
        ear_data:        dict,
        pose_data:       dict,
        attention_state: str,
    ) -> None:
        """Evaluate all alert conditions for the current frame."""
        self.last_alert = None

        if ear_data.get("drowsy"):
            self._fire("DROWSY")
        elif attention_state == "yawning":
            self._fire("YAWN")

        # Distracted: must persist for several consecutive frames
        if pose_data.get("direction") in ("side", "down"):
            self._distracted_counter += 1
        else:
            self._distracted_counter = 0

        if self._distracted_counter >= self._distracted_limit:
            self._fire("DISTRACTED")

    def check_absence(self) -> None:
        self._fire("ABSENT")

    # ------------------------------------------------------------------
    def draw_alert_banner(
        self, frame: np.ndarray, alert_type: str
    ) -> np.ndarray:
        """Draw a semi-transparent coloured banner at the top of the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        color   = _ALERT_COLORS.get(alert_type, (0, 0, 200))
        cv2.rectangle(overlay, (0, 0), (w, 50), color, -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        cv2.putText(
            frame,
            _ALERT_MESSAGES.get(alert_type, alert_type),
            (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
        )
        return frame

    def get_active_alert(self) -> Optional[str]:
        """Return the most recent alert type if still within visual display window."""
        return self.last_alert

    # ------------------------------------------------------------------
    def _fire(self, alert_type: str) -> None:
        now = time.time()
        if now - self._last_alert.get(alert_type, 0.0) < self._cooldown:
            return
        self._last_alert[alert_type] = now
        self.last_alert = alert_type

        if self._pygame_ok:
            try:
                self._sounds[alert_type].play()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    with open("config.json") as f:
        cfg = json.load(f)

    mgr = AlertManager(cfg)
    print("Firing DROWSY alert …")
    mgr._fire("DROWSY")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = mgr.draw_alert_banner(frame, "DROWSY")
    cv2.imshow("Alert test", frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("Alert test done.")
