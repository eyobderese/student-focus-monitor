"""
End-to-end latency profiler.

Runs 500 consecutive webcam frames through the full pipeline and reports
mean ± std latency per module and total.

Target: < 100ms per frame on CPU (≥10 fps analysis rate).

Usage:
    python eval/eval_latency.py [source]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.ingestion            import VideoIngestion
from modules.face_detector        import FaceDetector
from modules.landmark_extractor   import LandmarkExtractor
from modules.ear_mar              import EARMARAnalyzer
from modules.head_pose            import HeadPoseEstimator
from modules.attention_classifier import AttentionClassifier
from modules.fusion               import compute_focus_score

N_FRAMES = 500


def profile(source=0) -> None:
    with open(PROJECT_ROOT / "config.json") as f:
        cfg = json.load(f)

    detector   = FaceDetector()
    extractor  = LandmarkExtractor()
    ear_mar    = EARMARAnalyzer(cfg)
    head_pose  = HeadPoseEstimator(cfg)
    classifier = AttentionClassifier()

    times: dict[str, list[float]] = {
        "face_detect":  [],
        "landmarks":    [],
        "ear_mar":      [],
        "head_pose":    [],
        "cnn_classify": [],
        "fusion":       [],
        "total":        [],
    }

    analyzed = 0
    print(f"Profiling {N_FRAMES} analyzed frames from source {source!r} …")

    with VideoIngestion(source=source, config=cfg) as vid:
        for idx, bgr, rgb, _ in vid.stream():
            if idx % cfg["analysis_frame_skip"] != 0:
                continue

            t_total = time.perf_counter()

            t0 = time.perf_counter()
            bbox = detector.detect(rgb)
            times["face_detect"].append((time.perf_counter() - t0) * 1000)

            if bbox is None:
                continue

            t0 = time.perf_counter()
            lms = extractor.extract(rgb)
            times["landmarks"].append((time.perf_counter() - t0) * 1000)

            if lms is None:
                continue

            t0 = time.perf_counter()
            ear_data = ear_mar.analyze(lms, idx)
            times["ear_mar"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            pose = head_pose.estimate(lms, bgr)
            times["head_pose"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            eye_state, _ = classifier.predict(bgr, lms)
            times["cnn_classify"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            attn  = classifier.get_attention_state(
                eye_state, pose["direction"] if pose else "forward",
                ear_data["ear"], ear_data["yawning"], cfg
            )
            compute_focus_score(ear_data["ear"], pose["direction"] if pose else "forward",
                                attn, ear_data["blink_rate"], cfg)
            times["fusion"].append((time.perf_counter() - t0) * 1000)

            times["total"].append((time.perf_counter() - t_total) * 1000)

            analyzed += 1
            if analyzed % 50 == 0:
                print(f"  {analyzed}/{N_FRAMES} frames analyzed …")
            if analyzed >= N_FRAMES:
                break

    print(f"\n=== Latency Profile ({analyzed} frames) ===")
    header = f"{'Module':<18}  {'Mean (ms)':>10}  {'Std (ms)':>10}  {'Max (ms)':>10}"
    print(header)
    print("─" * len(header))
    for module, vals in times.items():
        if not vals:
            continue
        print(f"{module:<18}  {mean(vals):>10.2f}  {stdev(vals) if len(vals)>1 else 0:>10.2f}  {max(vals):>10.2f}")

    total_mean = mean(times["total"]) if times["total"] else 0
    fps        = 1000.0 / total_mean if total_mean > 0 else 0
    print(f"\nEffective analysis rate: {fps:.1f} fps  (target ≥ 10 fps / 100ms)")

    detector.close()
    extractor.close()


if __name__ == "__main__":
    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    profile(src)
