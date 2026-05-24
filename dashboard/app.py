"""
Streamlit instructor dashboard for the Student Focus Monitor.

Tab 1 — Live session : real-time webcam monitoring with metric cards.
Tab 2 — Report viewer: upload CSV, explore charts, download PDF.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.face_detector        import FaceDetector
from modules.landmark_extractor   import LandmarkExtractor
from modules.ear_mar              import EARMARAnalyzer
from modules.head_pose            import HeadPoseEstimator
from modules.attention_classifier import AttentionClassifier
from modules.fusion               import compute_focus_score, FusionState
from modules.alerts               import AlertManager
from modules.logger               import SessionLogger
from modules.report               import generate_report_bytes

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Focus Monitor",
    page_icon="📚",
    layout="wide",
)

CONFIG_PATH = PROJECT_ROOT / "config.json"
LOG_PATH    = PROJECT_ROOT / "outputs" / "session_log.csv"
CNN_WEIGHTS = PROJECT_ROOT / "models" / "saved" / "eye_cnn.pt"


@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_pipeline(cfg: dict) -> dict:
    return {
        "detector":   FaceDetector(absence_timeout_seconds=cfg["absence_timeout_seconds"]),
        "extractor":  LandmarkExtractor(),
        "ear_mar":    EARMARAnalyzer(cfg),
        "head_pose":  HeadPoseEstimator(cfg),
        "classifier": AttentionClassifier(model_path=CNN_WEIGHTS),
        "fusion":     FusionState(),
        "alerts":     AlertManager(cfg),
    }


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
def plot_focus_timeline(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="timestamp", y="smoothed_focus_score",
                  title="Focus Score Over Session",
                  labels={"timestamp": "Time (s)", "smoothed_focus_score": "Focus Score"})
    fig.add_hline(y=65, line_dash="dash", line_color="green",  annotation_text="Focused")
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Distracted")
    fig.update_yaxes(range=[0, 105])
    return fig


def plot_ear_timeline(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = px.line(df, x="timestamp", y="ear_avg",
                  title="Eye Aspect Ratio (EAR) Over Time",
                  labels={"timestamp": "Time (s)", "ear_avg": "EAR"})
    fig.add_hline(y=threshold, line_dash="dot", line_color="red",
                  annotation_text=f"Drowsy ({threshold})")
    return fig


def plot_attention_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["attention_state"].value_counts().reset_index()
    counts.columns = ["State", "Count"]
    color_map = {
        "focused":    "#4caf50", "neutral":    "#ffeb3b",
        "distracted": "#ff9800", "drowsy":     "#f44336",
        "yawning":    "#9c27b0", "uncertain":  "#9e9e9e",
    }
    fig = px.pie(counts, names="State", values="Count",
                 title="Attention State Distribution",
                 color="State", color_discrete_map=color_map)
    return fig


def session_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    duration = df["timestamp"].max() if not df.empty else 0
    avg_focus = df["smoothed_focus_score"].mean()
    n_focused = int(df["focus_category"].eq("FOCUSED").sum())
    n_neutral = int(df["focus_category"].eq("NEUTRAL").sum())
    n_distracted = int(df["focus_category"].eq("DISTRACTED").sum())
    total_yawns = int(df["yawning"].astype(str).eq("True").sum())
    return pd.DataFrame({
        "Metric": ["Duration (s)", "Avg Focus Score", "Focused frames",
                   "Neutral frames", "Distracted frames", "Total yawns"],
        "Value": [f"{duration:.0f}", f"{avg_focus:.1f}", n_focused,
                  n_neutral, n_distracted, total_yawns],
    })


# ---------------------------------------------------------------------------
# Tab 1 — Live session
# ---------------------------------------------------------------------------
def tab_live(cfg: dict) -> None:
    st.header("Live Focus Session")

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        running  = st.toggle("Start monitoring", value=False)
        cam_idx  = st.number_input("Camera index", min_value=0, value=0, step=1)

    if not running:
        st.info("Toggle 'Start monitoring' to begin your study session.")
        return

    pipeline = load_pipeline(cfg)
    logger   = SessionLogger(output_path=LOG_PATH)

    video_ph = st.empty()
    c1, c2, c3, c4 = st.columns(4)
    m_score  = c1.empty()
    m_eye    = c2.empty()
    m_pose   = c3.empty()
    m_attn   = c4.empty()
    alert_ph = st.empty()
    chart_ph = st.empty()
    score_history: list[float] = []
    last_metric_ts = 0.0

    cap = cv2.VideoCapture(int(cam_idx))
    frame_idx = 0

    try:
        while running:
            ret, bgr = cap.read()
            if not ret:
                st.warning("Cannot read from camera.")
                break

            bgr       = cv2.flip(bgr, 1)
            bgr       = cv2.resize(bgr, (640, 480))
            rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_idx += 1

            if frame_idx % cfg["analysis_frame_skip"] != 0:
                video_ph.image(rgb, channels="RGB", use_container_width=True)
                continue

            display = bgr.copy()
            bbox = pipeline["detector"].detect(rgb)

            if bbox is None:
                cv2.putText(display, "NO FACE", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                video_ph.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                continue

            lms = pipeline["extractor"].extract(rgb)
            if lms is None:
                continue

            ear_data  = pipeline["ear_mar"].analyze(lms, frame_idx)
            pose_data = pipeline["head_pose"].estimate(lms, bgr)
            eye_state, _ = pipeline["classifier"].predict(bgr, lms)
            attn_state   = pipeline["classifier"].get_attention_state(
                eye_state,
                pose_data["direction"] if pose_data else "forward",
                ear_data["ear"], ear_data["yawning"], cfg,
            )
            direction = pose_data["direction"] if pose_data else "forward"
            raw, category = compute_focus_score(
                ear_data["ear"], direction, attn_state,
                ear_data["blink_rate"], cfg,
            )
            smoothed = pipeline["fusion"].smooth(raw)
            score_history.append(smoothed)

            pipeline["alerts"].check(ear_data, pose_data or {"direction": "forward"}, attn_state)

            logger.log({
                "frame_index": frame_idx,
                "ear_avg": ear_data["ear"], "mar": ear_data["mar"],
                "blink_rate": ear_data["blink_rate"],
                "eye_state_ear": ear_data["eye_state"],
                "eye_state_cnn": eye_state,
                "yawning": ear_data["yawning"],
                "yaw":   pose_data["yaw"]   if pose_data else "",
                "pitch": pose_data["pitch"] if pose_data else "",
                "roll":  pose_data["roll"]  if pose_data else "",
                "head_direction": direction,
                "attention_state": attn_state,
                "raw_focus_score": raw,
                "smoothed_focus_score": smoothed,
                "focus_category": category,
                "alert_fired": pipeline["alerts"].last_alert or "",
            })

            # HUD on display frame
            cat_colors = {"FOCUSED": (0,200,0), "NEUTRAL": (0,200,200), "DISTRACTED": (0,0,220)}
            color = cat_colors.get(category, (200, 200, 200))
            cv2.rectangle(display, (0, 0), (640, 50), (20, 20, 20), -1)
            cv2.putText(display, f"Focus: {smoothed}  [{category}]",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if pipeline["alerts"].last_alert:
                display = pipeline["alerts"].draw_alert_banner(display, pipeline["alerts"].last_alert)

            video_ph.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            # Update metric cards every 3 seconds
            now = time.time()
            if now - last_metric_ts > 3:
                m_score.metric("Focus Score", smoothed)
                m_eye.metric("Eye State",     eye_state)
                m_pose.metric("Head",         direction)
                m_attn.metric("Attention",    attn_state)
                last_metric_ts = now

                if pipeline["alerts"].last_alert:
                    alert_ph.warning(f"⚠ Alert: {pipeline['alerts'].last_alert}")
                else:
                    alert_ph.empty()

                if len(score_history) > 2:
                    chart_ph.line_chart(
                        pd.DataFrame({"Focus Score": score_history[-120:]})
                    )

    finally:
        cap.release()
        logger.close()


# ---------------------------------------------------------------------------
# Tab 2 — Report viewer
# ---------------------------------------------------------------------------
def tab_report(cfg: dict) -> None:
    st.header("Session Report Viewer")

    uploaded = st.file_uploader("Upload session CSV", type="csv")
    if not uploaded:
        if LOG_PATH.exists():
            st.info(f"Showing last session from `{LOG_PATH}`.")
            df = pd.read_csv(LOG_PATH)
        else:
            st.info("No session CSV yet. Complete a live session first.")
            return
    else:
        df = pd.read_csv(uploaded)

    df["smoothed_focus_score"] = pd.to_numeric(df["smoothed_focus_score"], errors="coerce")
    df["ear_avg"]              = pd.to_numeric(df["ear_avg"],              errors="coerce")

    st.subheader("Summary")
    st.dataframe(session_summary_table(df), use_container_width=True)

    st.subheader("Focus Score Timeline")
    st.plotly_chart(plot_focus_timeline(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("EAR Timeline")
        st.plotly_chart(
            plot_ear_timeline(df, cfg["ear_drowsy_threshold"]),
            use_container_width=True,
        )
    with col2:
        st.subheader("Attention Distribution")
        st.plotly_chart(plot_attention_pie(df), use_container_width=True)

    st.subheader("Export")
    if st.button("Generate PDF Report"):
        with st.spinner("Building PDF …"):
            try:
                pdf = generate_report_bytes(df, config=cfg)
                st.download_button("⬇ Download PDF", pdf,
                                   "session_report.pdf", "application/pdf")
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    st.title("📚 Student Focus & Fatigue Monitor")
    tab1, tab2 = st.tabs(["Live Session", "Session Report"])
    with tab1:
        tab_live(cfg)
    with tab2:
        tab_report(cfg)


if __name__ == "__main__":
    main()
