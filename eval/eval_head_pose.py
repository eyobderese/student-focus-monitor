"""
Head pose accuracy evaluation on the 300W-LP sample.

Usage:
    python eval/eval_head_pose.py [pose_val_dir]

Expects images + angles.json in data/pose_val/.
angles.json format: {"img.jpg": {"yaw": 5.2, "pitch": -3.1, "roll": 1.0}, ...}

Targets: yaw MAE < 6°, pitch MAE < 7°, roll MAE < 5°.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.landmark_extractor import LandmarkExtractor
from modules.head_pose import HeadPoseEstimator

POSE_VAL_DIR = PROJECT_ROOT / "data" / "pose_val"


def _load_gt(val_dir: Path) -> dict:
    """Load ground truth angles from angles.json or per-image .mat files."""
    json_path = val_dir / "angles.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def _load_mat_gt(mat_path: Path) -> dict | None:
    try:
        import scipy.io as sio
        mat  = sio.loadmat(str(mat_path))
        pose = mat.get("Pose_Para")
        if pose is None:
            return None
        return {
            "pitch": float(np.degrees(pose[0, 0])),
            "yaw":   float(np.degrees(pose[0, 1])),
            "roll":  float(np.degrees(pose[0, 2])),
        }
    except Exception:
        return None


def evaluate(val_dir: Path = POSE_VAL_DIR) -> None:
    image_exts = {".jpg", ".jpeg", ".png"}
    images     = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in image_exts)

    if not images:
        print(f"No images in {val_dir}.")
        print("Download 300W-LP sample and place 200 images + angles.json there.")
        return

    gt_dict   = _load_gt(val_dir)
    extractor = LandmarkExtractor()

    # Dummy config with permissive thresholds (we only need the estimator)
    cfg = {
        "head_yaw_distracted_threshold": 90,
        "head_pitch_down_threshold": -90,
        "head_pitch_up_threshold": 90,
    }
    estimator = HeadPoseEstimator(cfg)

    yaw_errs, pitch_errs, roll_errs = [], [], []

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lms = extractor.extract(rgb)
        if lms is None:
            continue

        result = estimator.estimate(lms, bgr)
        if result is None:
            continue

        gt = gt_dict.get(img_path.name)
        if gt is None:
            mat_path = img_path.with_suffix(".mat")
            gt = _load_mat_gt(mat_path) if mat_path.exists() else None
        if gt is None:
            continue

        yaw_errs.append(abs(result["yaw"]   - gt["yaw"]))
        pitch_errs.append(abs(result["pitch"] - gt["pitch"]))
        roll_errs.append(abs(result["roll"]   - gt["roll"]))

    extractor.close()

    if not yaw_errs:
        print("No matched samples. Add angles.json or .mat sidecars alongside images.")
        return

    n = len(yaw_errs)
    print(f"\n=== Head Pose Evaluation — {n} images ===")
    print(f"Yaw   MAE : {np.mean(yaw_errs):.2f}°  (target < 6°)")
    print(f"Pitch MAE : {np.mean(pitch_errs):.2f}°  (target < 7°)")
    print(f"Roll  MAE : {np.mean(roll_errs):.2f}°  (target < 5°)")


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else POSE_VAL_DIR
    evaluate(d)
