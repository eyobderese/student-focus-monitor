"""
Session report generation module.

Produces 4 Matplotlib charts and compiles a 4-page PDF using ReportLab.

Charts:
  1. Focus score timeline with alert markers
  2. EAR over time with blink event markers
  3. Attention state pie chart
  4. Blink rate per minute bar chart
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm as rl_cm
from reportlab.platypus import (
    Image as RLImage, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

CHARTS_DIR = Path("outputs/charts")


# ---------------------------------------------------------------------------
# Chart 1 — Focus score timeline
# ---------------------------------------------------------------------------
def _chart_focus_timeline(df: pd.DataFrame, alert_times: list) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = df["timestamp"] / 60.0   # → minutes

    ax.fill_between(t, 65, 100, alpha=0.08, color="green")
    ax.fill_between(t, 40,  65, alpha=0.08, color="orange")
    ax.fill_between(t,  0,  40, alpha=0.08, color="red")

    ax.plot(t, df["smoothed_focus_score"], color="#1a73e8", linewidth=1.8, label="Focus score")
    ax.axhline(65, color="green",  linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(40, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)

    for at, atype in alert_times:
        ax.axvline(at / 60.0, color="red", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.text(at / 60.0, 95, atype[:3], fontsize=6, color="red", rotation=90)

    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("Focus score")
    ax.set_title("Focus Score Timeline")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Chart 2 — EAR over time + blink event markers
# ---------------------------------------------------------------------------
def _chart_ear_timeline(df: pd.DataFrame, ear_threshold: float) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 3))
    t   = df["timestamp"] / 60.0
    ear = pd.to_numeric(df["ear_avg"], errors="coerce")

    ax.plot(t, ear, color="#e67e22", linewidth=1.5, label="EAR")
    ax.axhline(ear_threshold, color="red", linestyle="--",
               linewidth=1, label=f"Drowsy threshold ({ear_threshold})")

    # Mark blink events (EAR crossing back above threshold)
    closed = ear < ear_threshold
    crossings = closed.shift(1, fill_value=False) & ~closed
    blink_times = t[crossings]
    ax.plot(blink_times, [ear_threshold - 0.01] * len(blink_times),
            "v", color="blue", markersize=4, label="Blink", alpha=0.7)

    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("EAR")
    ax.set_title("Eye Aspect Ratio Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Chart 3 — Attention state pie chart
# ---------------------------------------------------------------------------
def _chart_attention_pie(df: pd.DataFrame) -> bytes:
    counts = df["attention_state"].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    cmap   = {
        "focused":    "#4caf50",
        "neutral":    "#ffeb3b",
        "distracted": "#ff9800",
        "drowsy":     "#f44336",
        "yawning":    "#9c27b0",
        "uncertain":  "#9e9e9e",
    }
    colors = [cmap.get(l, "#bdbdbd") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Attention State Distribution")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Chart 4 — Blink rate per minute bar chart
# ---------------------------------------------------------------------------
def _chart_blink_rate(df: pd.DataFrame, blink_min: int, blink_max: int) -> bytes:
    df = df.copy()
    df["minute"] = (df["timestamp"] / 60).astype(int)
    per_min = df.groupby("minute")["blink_rate"].mean()

    colors = [
        "#4caf50" if blink_min <= v <= blink_max else "#ff9800"
        for v in per_min.values
    ]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(per_min.index, per_min.values, color=colors, width=0.7)
    ax.axhline(blink_min, color="green",  linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(blink_max, color="green",  linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Blinks per minute")
    ax.set_title("Blink Rate per Minute")
    normal_patch = mpatches.Patch(color="#4caf50", label=f"Normal ({blink_min}–{blink_max} bpm)")
    abnormal_patch = mpatches.Patch(color="#ff9800", label="Abnormal")
    ax.legend(handles=[normal_patch, abnormal_patch], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _compute_summary(df: pd.DataFrame) -> dict:
    duration_s    = float(df["timestamp"].max()) if not df.empty else 0
    avg_focus     = round(pd.to_numeric(df["smoothed_focus_score"], errors="coerce").mean(), 1)
    total_blinks  = int(pd.to_numeric(df["blink_rate"], errors="coerce").max() or 0)
    total_yawns   = int(df["yawning"].astype(str).eq("True").sum())
    n_distracted  = int(df["focus_category"].eq("DISTRACTED").sum())
    alerts_fired  = df["alert_fired"].dropna().replace("", np.nan).dropna()

    # Longest focused streak (consecutive FOCUSED rows × frame skip × 30fps)
    streak, best = 0, 0
    for cat in df["focus_category"]:
        if cat == "FOCUSED":
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    return {
        "duration_min":     round(duration_s / 60, 1),
        "avg_focus":        avg_focus,
        "total_blinks":     total_blinks,
        "total_yawns":      total_yawns,
        "distracted_frames": n_distracted,
        "alert_count":      len(alerts_fired),
        "longest_streak_frames": best,
    }


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------
def generate_report(
    csv_path: str | Path,
    output_pdf: str | Path = "outputs/session_report.pdf",
    config: Optional[dict] = None,
) -> Path:
    """Build a 4-page PDF report and return its path."""
    import json

    csv_path   = Path(csv_path)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if config is None:
        try:
            with open("config.json") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {
                "ear_drowsy_threshold": 0.25,
                "blink_rate_normal_min": 10,
                "blink_rate_normal_max": 20,
            }

    df = pd.read_csv(csv_path)

    # Gather alert events from log
    alert_times = [
        (row["timestamp"], row["alert_fired"])
        for _, row in df.iterrows()
        if row.get("alert_fired") not in ("", None, float("nan"))
    ]

    summary = _compute_summary(df)

    # Generate charts
    png_focus   = _chart_focus_timeline(df, alert_times)
    png_ear     = _chart_ear_timeline(df, config["ear_drowsy_threshold"])
    png_pie     = _chart_attention_pie(df)
    png_blink   = _chart_blink_rate(
        df, config["blink_rate_normal_min"], config["blink_rate_normal_max"]
    )

    # Save chart PNGs to disk as well
    for name, data in [("focus_timeline.png", png_focus), ("ear_timeline.png", png_ear),
                       ("attention_pie.png", png_pie), ("blink_rate.png", png_blink)]:
        (CHARTS_DIR / name).write_bytes(data)

    # ---- Build PDF ----
    doc    = SimpleDocTemplate(str(output_pdf), pagesize=A4,
                               leftMargin=2 * rl_cm, rightMargin=2 * rl_cm,
                               topMargin=2 * rl_cm, bottomMargin=2 * rl_cm)
    styles = getSampleStyleSheet()
    H1, H2, BODY = styles["Heading1"], styles["Heading2"], styles["BodyText"]
    story  = []

    # ---- Page 1: Summary ----
    story.append(Paragraph("Student Focus & Fatigue Report", H1))
    story.append(Spacer(1, 0.4 * rl_cm))

    summary_rows = [
        ["Session duration",      f"{summary['duration_min']} min"],
        ["Overall avg focus",     f"{summary['avg_focus']} / 100"],
        ["Total blinks",          str(summary['total_blinks'])],
        ["Total yawns",           str(summary['total_yawns'])],
        ["Distracted frames",     str(summary['distracted_frames'])],
        ["Alerts fired",          str(summary['alert_count'])],
        ["Longest focused streak (frames)", str(summary['longest_streak_frames'])],
    ]
    tbl = Table(summary_rows, colWidths=[8 * rl_cm, 8 * rl_cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), rl_colors.lightgrey),
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [rl_colors.whitesmoke, rl_colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("PADDING",      (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ---- Page 2: Focus timeline ----
    story.append(Paragraph("Focus Score Timeline", H1))
    story.append(RLImage(io.BytesIO(png_focus), width=16 * rl_cm, height=7 * rl_cm))
    story.append(PageBreak())

    # ---- Page 3: EAR + Blink rate ----
    story.append(Paragraph("Eye Activity", H1))
    story.append(RLImage(io.BytesIO(png_ear),   width=16 * rl_cm, height=6 * rl_cm))
    story.append(Spacer(1, 0.4 * rl_cm))
    story.append(RLImage(io.BytesIO(png_blink), width=16 * rl_cm, height=5 * rl_cm))
    story.append(PageBreak())

    # ---- Page 4: Attention pie + stats ----
    story.append(Paragraph("Attention State Analysis", H1))
    story.append(RLImage(io.BytesIO(png_pie), width=10 * rl_cm, height=8 * rl_cm))
    story.append(Spacer(1, 0.3 * rl_cm))

    # Per-minute average table
    df["minute"] = (df["timestamp"] / 60).astype(int)
    per_min = (
        df.groupby("minute")
        .agg(
            avg_score=("smoothed_focus_score", "mean"),
            avg_ear=("ear_avg", "mean"),
            avg_blink=("blink_rate", "mean"),
        )
        .reset_index()
    )
    per_min = per_min.round(1)
    table_data = [["Minute", "Avg Focus", "Avg EAR", "Avg Blink/min"]]
    for _, row in per_min.iterrows():
        table_data.append([
            str(int(row["minute"])),
            str(row["avg_score"]),
            str(row["avg_ear"]),
            str(row["avg_blink"]),
        ])
    stat_tbl = Table(table_data,
                     colWidths=[3 * rl_cm, 4 * rl_cm, 4 * rl_cm, 5 * rl_cm])
    stat_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a73e8")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.whitesmoke, rl_colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ("PADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(stat_tbl)

    doc.build(story)
    print(f"[Report] PDF → {output_pdf}")
    return output_pdf


def generate_report_bytes(df: pd.DataFrame, config: Optional[dict] = None) -> bytes:
    """Generate PDF in-memory from a DataFrame (for Streamlit download button)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        out_path = tmp_pdf.name
    generate_report(tmp_csv.name, out_path, config=config)
    return Path(out_path).read_bytes()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    csv_in  = sys.argv[1] if len(sys.argv) > 1 else "outputs/session_log.csv"
    pdf_out = sys.argv[2] if len(sys.argv) > 2 else "outputs/session_report.pdf"
    generate_report(csv_in, pdf_out)
