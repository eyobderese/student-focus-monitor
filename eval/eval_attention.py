"""
CNN vs SVM attention classifier evaluation on MRL test split.

Usage:
    python eval/eval_attention.py

Requires trained weights:
    models/saved/eye_cnn.pt
    models/saved/svm_baseline.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.attention_classifier import EyeStateNet
from models.train_svm_baseline import _extract_features

DATA_DIR       = PROJECT_ROOT / "data" / "mrl_eyes"
CNN_WEIGHTS    = PROJECT_ROOT / "models" / "saved" / "eye_cnn.pt"
SVM_PATH       = PROJECT_ROOT / "models" / "saved" / "svm_baseline.pkl"
SEED           = 42
EYE_SIZE       = 32
TEST_LIMIT     = 5000   # max images per class for speed


def _load_test_images():
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    X_gray, y_labels = [], []
    for label, folder_name in [(1, "open"), (0, "closed")]:
        folder = DATA_DIR / folder_name
        if not folder.exists():
            print(f"Missing: {folder}")
            continue
        paths  = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        # Use the last 15% as test set (mirrors train_eye_cnn.py split)
        start  = int(len(paths) * 0.85)
        test_p = paths[start: start + TEST_LIMIT]
        for p in tqdm(test_p, desc=f"loading {folder_name}"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (EYE_SIZE, EYE_SIZE))
            X_gray.append(img)
            y_labels.append(label)
    return X_gray, np.array(y_labels)


def evaluate_cnn(X_gray: list, y: np.ndarray) -> None:
    if not CNN_WEIGHTS.exists():
        print(f"CNN weights not found: {CNN_WEIGHTS}")
        return

    device = torch.device("cpu")
    model  = EyeStateNet().to(device)
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device, weights_only=True))
    model.eval()

    preds = []
    with torch.no_grad():
        for img in X_gray:
            t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
            logits = model(t.to(device))
            preds.append(int(logits.argmax(1).item()))

    preds = np.array(preds)
    print("\n=== EyeStateNet (CNN) ===")
    print(f"Accuracy : {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds, target_names=["closed", "open"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))


def evaluate_svm(X_gray: list, y: np.ndarray) -> None:
    if not SVM_PATH.exists():
        print(f"SVM model not found: {SVM_PATH}")
        return

    saved  = joblib.load(SVM_PATH)
    svm    = saved["model"]
    scaler = saved["scaler"]

    X_feat = np.array([_extract_features(img) for img in tqdm(X_gray, desc="SVM features")])
    X_feat = scaler.transform(X_feat)
    preds  = svm.predict(X_feat)

    print("\n=== SVM Baseline ===")
    print(f"Accuracy : {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds, target_names=["closed", "open"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    print("Loading MRL test split …")
    X_gray, y = _load_test_images()
    if len(X_gray) == 0:
        print("No test images found.")
        sys.exit(1)

    print(f"Test set: {len(y)} images  (open={y.sum()}, closed={len(y)-y.sum()})")
    evaluate_cnn(X_gray, y)
    evaluate_svm(X_gray, y)
