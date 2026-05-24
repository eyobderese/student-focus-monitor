"""
SVM baseline trainer for eye state classification.

Features per eye crop (131-dim):
  • HOG on 32×32 grayscale crop (128-dim)
  • EAR value (1)
  • Mean pixel intensity (1)
  • Pixel intensity variance (1)

Trained on the same MRL dataset split as EyeStateNet.
Output: models/saved/svm_baseline.pkl

CV concepts: HOG feature extraction, SVM classification, feature scaling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "mrl_eyes"
SAVE_PATH    = PROJECT_ROOT / "models" / "saved" / "svm_baseline.pkl"
SEED         = 42
EYE_SIZE     = 32


# ---------------------------------------------------------------------------
def _extract_features(img_gray_32: np.ndarray, ear: float = 0.0) -> np.ndarray:
    """
    Build a 131-dimensional feature vector for one eye crop.

    img_gray_32: (32, 32) uint8 grayscale image.
    """
    hog_feat = hog(
        img_gray_32,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )  # → (128,)

    mean_pix = float(img_gray_32.mean())
    var_pix  = float(img_gray_32.var())
    return np.concatenate([hog_feat, [ear, mean_pix, var_pix]])


def _load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load MRL eye images and extract features."""
    open_dir   = DATA_DIR / "open"
    closed_dir = DATA_DIR / "closed"

    if not open_dir.exists() or not closed_dir.exists():
        print(
            f"ERROR: Expected {DATA_DIR}/open/ and {DATA_DIR}/closed/\n"
            "Run models/train_eye_cnn.py instructions to set up the dataset."
        )
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    X, y = [], []

    for label, folder in [(1, open_dir), (0, closed_dir)]:
        paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        for p in tqdm(paths, desc=f"{'open' if label else 'closed'}"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (EYE_SIZE, EYE_SIZE))
            X.append(_extract_features(img))
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading MRL eye dataset …")
    X, y = _load_dataset()
    print(f"Dataset: {len(X)} samples  (open={y.sum()}, closed={len(y)-y.sum()})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training SVM (rbf kernel) …")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              probability=True, random_state=SEED)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["closed", "open"]))

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": svm, "scaler": scaler}, SAVE_PATH)
    print(f"SVM saved → {SAVE_PATH}")
