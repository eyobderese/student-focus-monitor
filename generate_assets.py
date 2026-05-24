"""
Generate alert WAV beep files into assets/.
Run once manually or it runs automatically on first AlertManager init.
"""

from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav_io

ASSETS = Path(__file__).parent / "assets"
RATE   = 22050

BEEPS = {
    "alert_drowsy.wav":     520,
    "alert_distracted.wav": 660,
    "alert_absent.wav":     880,
    "alert_yawn.wav":       440,
}


def _make_beep(freq: int, duration: float = 0.35) -> np.ndarray:
    t    = np.linspace(0, duration, int(duration * RATE), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    fade = int(RATE * 0.02)
    wave[:fade]  = (wave[:fade]  * np.linspace(0, 1, fade)).astype(np.int16)
    wave[-fade:] = (wave[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    return wave


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    for fname, freq in BEEPS.items():
        out = ASSETS / fname
        wav_io.write(str(out), RATE, _make_beep(freq))
        print(f"Generated {out}")
    print("Done.")
