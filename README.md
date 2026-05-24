# Student Focus & Fatigue Monitor

A real-time computer vision system that monitors a single student via webcam during self-study or online lectures. It detects drowsiness, distraction, yawning, and absence, logs all signals, and generates a personal productivity PDF report.

**Privacy first:** No facial recognition, no biometric storage, no cloud upload.

---

## What the System Measures

| Signal | Method | Type |
|--------|--------|------|
| Eye drowsiness (EAR) | Eye Aspect Ratio from 6 eye landmarks | Classical CV |
| Yawn detection (MAR) | Mouth Aspect Ratio from 8 lip landmarks | Classical CV |
| Head pose (yaw/pitch/roll) | MediaPipe FaceMesh + solvePnP | Classical CV + Geometry |
| Blink rate (bpm) | EAR threshold crossing counter, rolling 60s window | Signal processing |
| Eye state (open/closed) | EyeStateNet CNN on eye crop | Deep Learning |
| Attention state | Fusion of CNN + EAR + pose + yawn | Multi-signal |
| Absence detection | No face for > N seconds (config) | Rule-based |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Generate audio alert files

```bash
python generate_assets.py
```

### 3. Train EyeStateNet (one-time offline)

Download the [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) (`mrlEyes_2018_01` subset), then organise:

```
data/mrl_eyes/
    open/    ← images of open eyes
    closed/  ← images of closed eyes
```

Then train:

```bash
python models/train_eye_cnn.py       # → models/saved/eye_cnn.pt
python models/train_svm_baseline.py  # → models/saved/svm_baseline.pkl  (comparison baseline)
```

### 4. Run the live system

```bash
python main.py          # webcam (default)
python main.py 0        # same — explicit webcam index
python main.py path/to/video.mp4     # on a video file
python main.py --no-display          # headless (batch processing)
python main.py --report-only         # regenerate PDF from last session CSV
```

**Hotkeys during live session:** `Q` = quit · `P` = pause/resume · `S` = save report now

### 5. Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
project/
├── main.py                        ← full pipeline entry point
├── config.json                    ← all tunable thresholds (edit this)
├── generate_assets.py             ← generates alert beep WAV files
├── requirements.txt
├── modules/
│   ├── ingestion.py               ← webcam capture, CLAHE, flip, frame skip
│   ├── face_detector.py           ← MediaPipe FaceDetection, absence timer
│   ├── landmark_extractor.py      ← MediaPipe FaceMesh, 468 landmarks
│   ├── ear_mar.py                 ← EAR, MAR, blink rate, yawn detection
│   ├── head_pose.py               ← solvePnP → yaw/pitch/roll, 3D axis overlay
│   ├── attention_classifier.py    ← EyeStateNet CNN + attention state fusion
│   ├── fusion.py                  ← weighted focus score + EMA smoothing
│   ├── alerts.py                  ← audio beep + visual banner triggers
│   ├── logger.py                  ← CSV session logger (buffered)
│   └── report.py                  ← 4 Matplotlib charts + 4-page ReportLab PDF
├── models/
│   ├── train_eye_cnn.py           ← MRL dataset → EyeStateNet training
│   ├── train_svm_baseline.py      ← HOG + EAR → SVM baseline training
│   └── saved/
│       ├── eye_cnn.pt             ← trained EyeStateNet weights
│       └── svm_baseline.pkl       ← trained SVM + scaler
├── eval/
│   ├── eval_ear_mar.py            ← EAR ROC curve, best threshold on YawDD
│   ├── eval_head_pose.py          ← solvePnP MAE on 300W-LP sample
│   ├── eval_attention.py          ← CNN vs SVM accuracy/F1/confusion matrix
│   └── eval_latency.py            ← per-module timing (target <100ms/frame)
├── dashboard/
│   └── app.py                     ← Streamlit dashboard
├── data/
│   ├── mrl_eyes/open/ + closed/   ← MRL Eye Dataset (place here)
│   ├── yawdd_frames/open/+closed/ ← YawDD frames (place here)
│   ├── pose_val/                  ← 300W-LP sample + angles.json
│   └── self_recorded/             ← your own study session videos
├── assets/
│   ├── alert_drowsy.wav
│   ├── alert_distracted.wav
│   ├── alert_absent.wav
│   └── alert_yawn.wav
└── outputs/
    ├── session_log.csv
    ├── session_report.pdf
    └── charts/                    ← intermediate PNG charts
```

---

## config.json — All Tunable Thresholds

```json
{
  "ear_drowsy_threshold": 0.25,          ← EAR below this → eyes closing
  "ear_closed_frames_threshold": 15,     ← frames of closed eyes → drowsy alert
  "mar_yawn_threshold": 0.6,             ← MAR above this → mouth open (yawn)
  "yawn_min_duration_frames": 10,        ← frames of open mouth → yawn event
  "head_yaw_distracted_threshold": 25,   ← yaw degrees → looking away
  "head_pitch_down_threshold": -20,      ← pitch degrees → looking down
  "head_pitch_up_threshold": 15,         ← pitch degrees → head up
  "absence_timeout_seconds": 5,          ← seconds without face → absent alert
  "blink_rate_normal_min": 10,           ← blinks/min — lower bound of normal
  "blink_rate_normal_max": 20,           ← blinks/min — upper bound of normal
  "analysis_frame_skip": 3,             ← analyse every Nth frame
  "alert_cooldown_seconds": 30,          ← minimum gap between same alert
  "focus_score_weights": { ... }
}
```

Run `eval/eval_ear_mar.py` to find the optimal EAR threshold for your lighting conditions.

---

## Focus Score Formula

```
ear_score   = clamp(ear / 0.35 × 100, 0, 100)
pose_score  = {forward:100, up:60, down:20, side:0}
cnn_score   = {focused:100, neutral:60, yawning:30, distracted:10, drowsy:0}
blink_score = 100 if 10≤bpm≤20, penalised outside range

raw = 0.30×ear + 0.30×pose + 0.30×cnn + 0.10×blink
smoothed = 0.75×prev + 0.25×raw

FOCUSED    ≥ 65  (green)
NEUTRAL    40–64 (yellow)
DISTRACTED < 40  (red)
```

---

## Evaluation Scripts

```bash
python eval/eval_ear_mar.py          # ROC curve + optimal threshold (needs YawDD)
python eval/eval_head_pose.py        # MAE on 300W-LP (needs pose_val/ + angles.json)
python eval/eval_attention.py        # CNN vs SVM on MRL test split
python eval/eval_latency.py          # per-module timing on 500 live frames
```

---

## CV Concepts Demonstrated

- CLAHE contrast enhancement, image flipping, color space transforms
- CNN-based face detection (MediaPipe FaceDetection)
- 468-point facial landmark detection (MediaPipe FaceMesh, graph-based CNN)
- Eye Aspect Ratio + Mouth Aspect Ratio (Euclidean distance geometry)
- Blink rate with rolling 60-second window
- Camera projection model, PnP problem, Rodrigues rotation, RQ decomposition
- 3D→2D axis projection overlay
- Custom CNN (EyeStateNet) design + training
- HOG feature extraction + SVM classification (baseline)
- Multi-signal weighted fusion + EMA smoothing
- ROC curve, F1, MAE, confusion matrix evaluation metrics
