"""
Student Focus & Fatigue Monitor — main pipeline entry point.

Usage:
    python main.py [source]          # 0 = webcam (default), or path to video file
    python main.py --report-only     # regenerate PDF from existing CSV

Hotkeys:
    Q — quit
    P — pause / resume
    S — save report now
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from modules.ingestion            import VideoIngestion
from modules.face_detector        import FaceDetector
from modules.landmark_extractor   import LandmarkExtractor
from modules.ear_mar              import EARMARAnalyzer
from modules.head_pose            import HeadPoseEstimator
from modules.attention_classifier import AttentionClassifier
from modules.fusion               import compute_focus_score, FusionState
from modules.alerts               import AlertManager
from modules.logger               import SessionLogger
from modules.report               import generate_report

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH  = PROJECT_ROOT / "config.json"
LOG_PATH     = PROJECT_ROOT / "outputs" / "session_log.csv"
REPORT_PATH  = PROJECT_ROOT / "outputs" / "session_report.pdf"
CNN_WEIGHTS  = PROJECT_ROOT / "models" / "saved" / "eye_cnn.pt"


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student Focus Monitor")
    p.add_argument("source", nargs="?", default="0",
                   help="Webcam index (default 0) or video file path")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless — no cv2.imshow window")
    p.add_argument("--report-only", action="store_true",
                   help="Skip capture; regenerate PDF from existing CSV")
    return p.parse_args()


def _resolve_source(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


# ---------------------------------------------------------------------------
def _draw_hud(
    frame,
    smoothed: int,
    category: str,
    ear_data: dict,
    pose_data: dict | None,
    attention_state: str,
    alert_mgr: AlertManager,
) -> None:
    """Draw the heads-up display overlay onto the frame in-place."""
    cat_colors = {
        "FOCUSED":    (0, 200, 0),
        "NEUTRAL":    (0, 200, 200),
        "DISTRACTED": (0, 0, 220),
    }
    color = cat_colors.get(category, (200, 200, 200))

    # Score banner
    cv2.rectangle(frame, (0, 0), (640, 55), (20, 20, 20), -1)
    cv2.putText(frame, f"Focus: {smoothed}  [{category}]",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Sub-metrics row
    direction = pose_data["direction"] if pose_data else "—"
    cv2.putText(
        frame,
        f"EAR:{ear_data['ear']:.2f}  MAR:{ear_data['mar']:.2f}  "
        f"Blink:{ear_data['blink_rate']}/min  Pose:{direction}  {attention_state}",
        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
    )

    # Alert banner (if active)
    if alert_mgr.last_alert:
        frame[:] = alert_mgr.draw_alert_banner(frame, alert_mgr.last_alert)


# ---------------------------------------------------------------------------
def run(source, no_display: bool = False) -> None:
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    detector   = FaceDetector(
        absence_timeout_seconds=config["absence_timeout_seconds"]
    )
    extractor  = LandmarkExtractor()
    ear_mar    = EARMARAnalyzer(config)
    head_pose  = HeadPoseEstimator(config)
    classifier = AttentionClassifier(model_path=CNN_WEIGHTS)
    fusion_st  = FusionState(initial_score=50)
    alerts     = AlertManager(config)
    logger     = SessionLogger(output_path=LOG_PATH)

    paused = False
    print(f"[main] Source: {source!r}  |  Q=quit  P=pause  S=save report")

    with VideoIngestion(source=source, config=config) as vid:
        for idx, bgr, rgb, gray_eq in vid.stream():
            display = bgr.copy()

            if paused:
                cv2.putText(display, "PAUSED", (250, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("Focus Monitor", display)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            # Only run full analysis on every Nth frame
            if idx % config["analysis_frame_skip"] != 0:
                if not no_display:
                    cv2.imshow("Focus Monitor", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # ---- Face detection ----
            bbox = detector.detect(rgb)
            if bbox is None:
                if detector.is_absent_too_long:
                    alerts.check_absence()
                cv2.putText(display, "NO FACE DETECTED",
                            (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if not no_display:
                    cv2.imshow("Focus Monitor", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # ---- Landmarks ----
            lms = extractor.extract(rgb)
            if lms is None:
                continue

            # ---- EAR / MAR ----
            ear_data = ear_mar.analyze(lms, idx)

            # ---- Head pose ----
            pose_data = head_pose.estimate(lms, bgr)
            if pose_data:
                head_pose.draw_axes(display, pose_data, lms[1])

            # ---- CNN eye state + attention ----
            eye_state, _ = classifier.predict(bgr, lms)
            attn_state   = classifier.get_attention_state(
                eye_state,
                pose_data["direction"] if pose_data else "forward",
                ear_data["ear"],
                ear_data["yawning"],
                config,
            )

            # ---- Focus score fusion ----
            direction = pose_data["direction"] if pose_data else "forward"
            raw, category = compute_focus_score(
                ear_data["ear"], direction, attn_state,
                ear_data["blink_rate"], config,
            )
            smoothed = fusion_st.smooth(raw)

            # ---- Alerts ----
            alerts.check(ear_data, pose_data or {"direction": "forward"}, attn_state)

            # ---- Logging ----
            logger.log({
                "frame_index":          idx,
                "ear_left":             round(ear_data["ear_left"],  4),
                "ear_right":            round(ear_data["ear_right"], 4),
                "ear_avg":              round(ear_data["ear"],       4),
                "mar":                  round(ear_data["mar"],       4),
                "blink_rate":           ear_data["blink_rate"],
                "eye_state_ear":        ear_data["eye_state"],
                "eye_state_cnn":        eye_state,
                "yawning":              ear_data["yawning"],
                "yaw":                  round(pose_data["yaw"],   2) if pose_data else "",
                "pitch":                round(pose_data["pitch"], 2) if pose_data else "",
                "roll":                 round(pose_data["roll"],  2) if pose_data else "",
                "head_direction":       direction,
                "attention_state":      attn_state,
                "raw_focus_score":      raw,
                "smoothed_focus_score": smoothed,
                "focus_category":       category,
                "alert_fired":          alerts.last_alert or "",
            })

            # ---- HUD overlay ----
            _draw_hud(display, smoothed, category, ear_data, pose_data,
                      attn_state, alerts)

            # ---- Display ----
            if not no_display:
                cv2.imshow("Focus Monitor", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = True
                elif key == ord("s"):
                    logger._flush()
                    try:
                        generate_report(LOG_PATH, REPORT_PATH, config=config)
                        print(f"[main] Report → {REPORT_PATH}")
                    except Exception as exc:
                        print(f"[main] Report error: {exc}")

            if idx % 300 == 0:
                print(
                    f"[frame {idx:05d}]  focus={smoothed:3d} [{category}]  "
                    f"EAR={ear_data['ear']:.3f}  MAR={ear_data['mar']:.3f}  "
                    f"pose={direction}  attn={attn_state}"
                )

    # ---- Session end ----
    cv2.destroyAllWindows()
    saved = logger.close()
    print(f"\n[main] Session complete — log → {saved}")

    if saved.exists() and saved.stat().st_size > 100:
        try:
            generate_report(saved, REPORT_PATH, config=config)
        except Exception as exc:
            print(f"[main] Report error: {exc}")

    detector.close()
    extractor.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args   = _parse_args()
    source = _resolve_source(args.source)

    if args.report_only:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        generate_report(LOG_PATH, REPORT_PATH, config=cfg)
    else:
        run(source=source, no_display=args.no_display)
