"""
EAR threshold tuning + ROC curve on YawDD frames.

Usage:
    python eval/eval_ear_mar.py [yawdd_frames_dir]

Expected structure:
    data/yawdd_frames/
        open/    ← frames where eyes are open
        closed/  ← frames where eyes are closed

Sweeps EAR threshold 0.18–0.35, plots ROC, selects best F1 threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.landmark_extractor import LandmarkExtractor, LEFT_EYE, RIGHT_EYE
from modules.ear_mar import compute_ear

DATA_DIR = PROJECT_ROOT / "data" / "yawdd_frames"
CHARTS   = PROJECT_ROOT / "outputs" / "charts"


def evaluate(data_dir: Path = DATA_DIR) -> None:
    open_dir   = data_dir / "open"
    closed_dir = data_dir / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        print(
            f"Expected {data_dir}/open/ and {data_dir}/closed/ with extracted YawDD frames.\n"
            "Download YawDD Mirror subset and extract frames with:\n"
            "  python -c \"import cv2; ... \" or use the provided extract_frames.py helper."
        )
        return

    extractor = LandmarkExtractor()
    exts      = {".jpg", ".jpeg", ".png"}

    ears, labels = [], []

    for label, folder in [(1, open_dir), (0, closed_dir)]:
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        for p in paths[:500]:   # limit to 500 per class for speed
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            lms     = extractor.extract(img_rgb)
            if lms is None:
                continue
            left_pts  = [lms[i] for i in LEFT_EYE]
            right_pts = [lms[i] for i in RIGHT_EYE]
            ear = (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0
            ears.append(ear)
            labels.append(label)

    extractor.close()

    if not ears:
        print("No valid samples found. Check images have detectable faces.")
        return

    ears   = np.array(ears)
    labels = np.array(labels)
    print(f"Evaluated {len(ears)} images  (open={labels.sum()}, closed={len(labels)-labels.sum()})")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, ears)

    # Sweep thresholds for best F1
    best_f1, best_thresh = 0.0, 0.25
    for thresh in np.arange(0.18, 0.36, 0.01):
        preds = (ears >= thresh).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    preds_best = (ears >= best_thresh).astype(int)
    prec  = precision_score(labels, preds_best, zero_division=0)
    rec   = recall_score(labels,    preds_best, zero_division=0)

    print(f"\n=== EAR Threshold Evaluation ===")
    print(f"Best EAR threshold : {best_thresh:.2f}")
    print(f"F1 at best thresh  : {best_f1:.4f}")
    print(f"Precision          : {prec:.4f}")
    print(f"Recall             : {rec:.4f}")
    print(f"\nRecommended config: \"ear_drowsy_threshold\": {best_thresh:.2f}")

    # Plot ROC
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1a73e8", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"EAR ROC Curve  (best thresh={best_thresh:.2f}, F1={best_f1:.3f})")
    ax.grid(True, alpha=0.3)
    out = CHARTS / "ear_roc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved → {out}")


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    evaluate(d)
