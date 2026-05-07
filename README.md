# Real-Time Illicit Activity Detection System

A research-grade surveillance AI prototype that detects illicit activity (fighting, robbery, anomalies) in real time using CCTV footage, with bounding box overlays, confidence scoring, and an alerting pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0_nightly-EE4C2C?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Model Stack](#model-stack)
- [Datasets](#datasets)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment Architecture](#deployment-architecture)
- [Quick Start — Deployment](#quick-start--deployment)
- [Running the Full System](#running-the-full-system)
- [Configuration](#configuration)
- [Frontend Features](#frontend-features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Evaluation Metrics](#evaluation-metrics)
- [Alert Logic](#alert-logic)

---

## Overview

This system processes live CCTV streams and classifies activity into four categories:

| Class       | Description                           |
| ----------- | ------------------------------------- |
| `Fighting`  | Physical altercations between persons |
| `Robbery`   | Theft or armed robbery events         |
| `Vandalism` | Property destruction or damage        |
| `Normal`    | No anomalous activity                 |

Key features:

- **Low-light enhancement** via Zero-DCE++ — two-stage trained (visual quality + detection-aware), triggered when frame brightness < 50
- **Person detection** via fine-tuned YOLOv11s (mAP50: 0.907)
- **Action recognition** via VideoMAE fine-tuned on UCF-Crime with 4-fold CV (Macro F1: 80.21% ± 5.64%)
- **Alert pipeline** with 30-second auto-clipped evidence videos uploaded to Supabase Storage
- **SMS/MMS alerts** via Twilio with video evidence attached
- **Live dashboard** via React + Supabase Realtime

---

## Pipeline

```
CCTV Stream (RTSP / mp4 / webcam)
    ↓
Frame Extraction @ 12 FPS (OpenCV)
    ↓
Zero-DCE++ Low-Light Enhancement  ← triggered if brightness < 50
    ↓                                 uses stage2_best.pth (detection-aware)
YOLOv11s Object Detection         ← person bounding boxes
    ↓
Annotated Frame → WebSocket → React Frontend (live view)
    ↓
Person detected ≥ 3 consecutive frames → fill 16-frame buffer
    ↓
Redis Stream (XADD)               ← 16-frame clip + 30s context + camera metadata
    ↓
VideoMAE Worker (XREAD)           ← polls Redis continuously
    ↓
VideoMAE Inference                ← Fighting / Robbery / Vandalism / Normal
    ↓
Confidence ≥ 0.40 + class ≠ Normal
    ↓
    ├── Supabase Storage           ← upload 30s evidence clip (.mp4)
    ├── Supabase PostgreSQL        ← insert incident row (camera, class, confidence, timestamp, clip_url)
    ├── Supabase Realtime          ← broadcast to frontend dashboard
    └── Twilio MMS                 ← SMS alert + video evidence to phone
```

---

### Zero-DCE++ Two-Stage Training

Zero-DCE++ is trained in two sequential stages before being used in the pipeline. This is an **offline training process** — the resulting weights (`stage2_best.pth`) are frozen and loaded at inference time.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — Visual Quality Training  (ExDARK, 200 epochs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ExDARK Dark Images (7,363 images, 12 classes)
    ↓
Zero-DCE++ (randomly initialized)
    ↓
L_total = 1.0·L_spa + 10.0·L_exp + 0.5·L_col + 200.0·L_tv
    ↓
Learns to produce visually bright, color-balanced,
noise-free enhanced images
    ↓
Saved → models/zerodce/stage1_best.pth  (val loss: 0.3348)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — Detection-Aware Fine-Tuning  (LLVIP, 100 epochs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLVIP Night Images (15,488 visible images)
    ↓
Zero-DCE++ (loaded from stage1_best.pth)
    ↓
Enhanced Image
    ↓              ↓
Enhancement    Frozen YOLOv11s (fine-tuned teacher)
Losses         ← domain-adapted to LLVIP/ExDARK
    ↓              ↓
L_enh          L_det (detection confidence signal)
    ↓              ↓
L_total = L_enh + λ · L_det

λ warmup schedule:
  Epochs  1–20  →  λ = 0.05  (stabilize first)
  Epochs 21–50  →  λ = 0.10  (gradually increase)
  Epochs 51–100 →  λ = 0.20  (full detection signal)

Zero-DCE++ learns to enhance images specifically to
maximize YOLOv11s person detection confidence.
YOLOv11s weights remain FROZEN throughout.
    ↓
Saved → models/zerodce/stage2_best.pth  (val loss: 0.1233)
                ↑
        ← USED IN PIPELINE
```

**Why two stages?**

- Stage 1 ensures Zero-DCE++ first learns what a good enhanced image looks like, establishing a stable base.
- Stage 2 then refines that enhancement objective toward detection utility — without Stage 1, the detection loss alone would destabilize training early on.
- The λ warmup in Stage 2 prevents the detection signal from overwhelming enhancement quality before the model has stabilized.

---

## Project Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── ucf_crime/                    # UCF-Crime videos + Action_Recognition_splits
│   │   ├── rwf2000/                      # RWF-2000 fight/nonfight videos
│   │   ├── exdark/                       # ExDARK low-light images
│   │   └── llvip/                        # LLVIP visible images + XML annotations
│   └── processed/
│       ├── clips/                        # (16, 224, 224, 3) float32 .npy clips
│       │   └── fold_N/train|test/ClassName/scene/*.npy
│       └── annotations_yolo/             # Converted YOLO format labels
│           ├── llvip/
│           └── exdark/
├── models/
│   ├── zerodce/
│   │   ├── stage1_best.pth               # ExDARK visual quality stage
│   │   └── stage2_best.pth               # LLVIP detection-aware stage ← pipeline
│   ├── yolov11/
│   │   ├── finetune_llvip_exdark/weights/best.pt
│   │   └── best.onnx                     # ONNX export ← pipeline
│   └── videomae/
│       ├── fold_1/videomae_best.pth
│       ├── fold_2/videomae_best.pth
│       ├── fold_3/videomae_best.pth
│       ├── fold_4/videomae_best.pth
│       └── pretrained/                   # MCG-NJU/videomae-base-finetuned-kinetics
├── src/
│   ├── preprocessing/
│   │   ├── convert_llvip_to_yolo.py
│   │   ├── convert_exdark_to_yolo.py
│   │   ├── build_dataset_yaml.py
│   │   └── verify_annotations.py
│   ├── training/
│   │   ├── zerodce_model.py
│   │   ├── zerodce_losses.py
│   │   ├── zerodce_dataset.py
│   │   ├── train_zerodce.py
│   │   ├── train_yolov11.py
│   │   ├── videomae_dataset.py           # ClipDataset + augmentation pipeline
│   │   ├── train_videomae.py             # 4-fold CV training + SWA
│   │   └── evaluate_tta.py               # Leak-free TTA evaluation
│   ├── inference/
│   │   ├── infer_zerodce.py
│   │   └── infer_yolov11.py
│   └── pipeline/
│       └── tracker.py                    # IoU-based person tracker
├── backend/
│   ├── app.py                            # FastAPI + WebSocket stream server
│   ├── worker_videomae.py                # Redis XREAD → VideoMAE inference worker
│   └── incident_handler.py              # Supabase upload + Twilio alert
├── frontend/                             # React dashboard
│   └── src/
│       └── App.jsx                       # Live feed + Supabase Realtime incident log
├── logs/
│   ├── zerodce/training_log.txt
│   └── videomae/training_log_fold{N}_v14.txt
└── outputs/
    ├── alerts/                           # Incident metadata logs
    └── clips/                            # 30s evidence .mp4 files
```

---

## Model Stack

### 1. Zero-DCE++ — Low-Light Enhancement

- **Architecture:** 8-iteration curve estimation (79,416 params)
- **Training:** Two-stage — ExDARK (visual quality) → LLVIP (detection-aware, frozen YOLOv11s as teacher)
- **Loss:** L_spa + L_exp + L_col + L_tv + λ·L_det
- **Trigger:** Only when mean frame brightness < 50

### 2. YOLOv11s — Person Detection

- **Base:** COCO pretrained, fine-tuned on LLVIP (12,025) + ExDARK People (609)
- **Input:** 640×640 | **Export:** ONNX for inference
- **Result:** mAP50 = 0.907, 2.0 ms/frame

### 3. VideoMAE — Action Recognition

- **Base:** MCG-NJU/videomae-base-finetuned-kinetics (86M params)
- **Training:** 4-fold CV on UCF-Crime official ActionRecognition splits — Fighting, Robbery, Vandalism, Normal
- **Clip format:** (16, 224, 224, 3) float32, stride-4 at 12 FPS
- **Key techniques:** progressive 5-stage unfreeze, MixUp (ep 3+), SWA (ep 14+), dropout schedule 0.5→0.3
- **Result: Macro F1 = 80.21% ± 5.64% across 4 folds**

---

## Datasets

| Dataset         | Purpose                                  | Size    | Location              |
| --------------- | ---------------------------------------- | ------- | --------------------- |
| UCF-Crime       | Primary crime classification (4-fold CV) | ~24.5GB | `data/raw/ucf_crime/` |
| RWF-2000        | Supplement Fighting / Normal class       | ~7.2GB  | `data/raw/rwf2000/`   |
| ExDARK          | Zero-DCE++ Stage 1 + YOLOv11 detection   | ~1.5GB  | `data/raw/exdark/`    |
| LLVIP (visible) | Zero-DCE++ Stage 2 + YOLOv11 fine-tuning | ~13GB   | `data/raw/llvip/`     |

**Total storage:** ~46GB

---

## Results

### Zero-DCE++ (Stage 2 — pipeline model)

| Metric           | Value                            |
| ---------------- | -------------------------------- |
| Enhancement Loss | 2.39 → 0.09 (converged)          |
| Val Loss         | 0.1233                           |
| Checkpoint       | `models/zerodce/stage2_best.pth` |

### YOLOv11s

| Metric          | Value                      |
| --------------- | -------------------------- |
| mAP50           | **0.907**                  |
| mAP50-95        | **0.534**                  |
| Precision       | 0.869                      |
| Recall          | 0.850                      |
| Inference Speed | 2.0 ms/frame               |
| Checkpoint      | `models/yolov11/best.onnx` |

### VideoMAE — 4-Fold Cross-Validation (v14)

| Class                | Macro F1           |
| -------------------- | ------------------ |
| Fighting             | 75.42% ± 10.48     |
| Robbery              | 78.30% ± 7.10      |
| Vandalism            | 80.15% ± 9.32      |
| Normal               | 86.55% ± 3.58      |
| **Overall Macro F1** | **80.21% ± 5.64%** |

---

## Setup

### Prerequisites

- Windows OS, NVIDIA GPU (8GB+ VRAM recommended)
- Python 3.11
- CUDA 12.8+
- Conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/illicit-detection.git
cd illicit-detection

# Create and activate conda environment
conda create -n illicit_detect python=3.11
conda activate illicit_detect

# Install PyTorch (nightly with CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Follow the download instructions in [`docs/dataset_setup.md`](docs/dataset_setup.md) for UCF-Crime, RWF-2000, ShanghaiTech, ExDARK, and LLVIP.

---

## Usage

### Annotation Conversion

```bash
# LLVIP XML → YOLO
python src/preprocessing/convert_llvip_to_yolo.py

# ExDARK bbGt → YOLO (People class only)
python src/preprocessing/convert_exdark_to_yolo.py

# Build unified dataset YAML
python src/preprocessing/build_dataset_yaml.py
```

### Training

```bash
# Stage 1: Zero-DCE++ on ExDARK
python src/training/train_zerodce.py --stage 1

# Stage 2: Detection-aware fine-tuning on LLVIP
python src/training/train_zerodce.py --stage 2

# YOLOv11s fine-tuning
python src/training/train_yolov11.py

# Extract clips for all 4 folds (run once, ~1.5h)
python src/training/extract_clips.py --all-folds

# VideoMAE 4-fold CV training
python src/training/train_videomae.py --all-folds --epochs 20

# Single fold
python src/training/train_videomae.py --fold 2 --epochs 20

# Print CV summary from saved checkpoints (no retraining)
python src/training/train_videomae.py --summarise-only
```

### Evaluation

```bash
# Leak-free TTA evaluation (all 4 folds, recommended)
python src/training/evaluate_tta.py

# Single fold TTA
python src/training/evaluate_tta.py --fold 1

# Reproduce training eval exactly (sanity check, no TTA)
python src/training/evaluate_tta.py --no-tta
```

### Inference

```bash
# Zero-DCE++ inference
python src/inference/infer_zerodce.py --input path/to/video

# YOLOv11s inference
python src/inference/infer_yolov11.py --input path/to/video
```

---

## Deployment Architecture

```
┌─────────────────┐
│  CCTV / Video   │
│     Stream      │
└────────┬────────┘
         │ RTSP / MP4 / Webcam
         ▼
┌────────────────────────────────────────────────┐
│           FastAPI Backend (Python)             │
├────────────────────────────────────────────────┤
│ • Frame extraction @ 12 FPS                    │
│ • Zero-DCE++ enhancement (brightness < 50)     │
│ • YOLOv11s person detection                    │
│ • WebSocket stream to frontend                 │
│ • Redis XADD clip enqueue                      │
└────┬───────────────────────────┬───────────────┘
     │ Annotated frames          │ 16-frame clips
     │ (live view)               │ (Redis queue)
     ▼                           ▼
┌──────────────────┐    ┌──────────────────────┐
│  React Frontend  │    │  VideoMAE Worker     │
├──────────────────┤    ├──────────────────────┤
│ • Live feed      │    │ • Redis XREAD        │
│ • Incident log   │    │ • Action recognition │
│ • Camera map     │    │ • Confidence scoring │
│ • Settings       │    │ • Alert trigger      │
└──────────────────┘    └────┬─────────────────┘
     ▲                        │
     └── Supabase Realtime ◄──┘
                               │
                    ┌──────────▼───────────┐
                    │    Alert Pipeline    │
                    ├──────────────────────┤
                    │ • Supabase Storage   │
                    │ • Supabase PostgreSQL│
                    │ • Twilio MMS         │
                    └──────────────────────┘
```

---

## Quick Start — Deployment

### Prerequisites

- Python 3.11, Node.js 18+
- Redis (local or remote)
- Supabase project (PostgreSQL + Storage + Realtime)
- Twilio account (for SMS/MMS alerts)

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Fill in:
# SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
# REDIS_URL
# TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO
```

### 3. Start Backend

```bash
python app.py
# Runs on http://localhost:8000
# WebSocket: ws://localhost:8000/ws
```

### 4. Start VideoMAE Worker (separate terminal)

```bash
python worker_videomae.py
# Polls Redis continuously for clip jobs
```

### 5. Frontend Setup

```bash
cd frontend
pnpm install
pnpm dev
# Runs on http://localhost:3000
```

---

## Running the Full System

```bash
# Terminal 1 — Backend
cd backend && python app.py

# Terminal 2 — VideoMAE Worker
cd backend && python worker_videomae.py

# Terminal 3 — Frontend
cd frontend && pnpm dev

# Terminal 4 — Send video stream (optional)
ffmpeg -rtsp_transport tcp -i <camera_url> -f mpegts udp://localhost:5005
```

---

## Configuration

### Key Environment Variables

| Variable                        | Description                  | Default                    |
| ------------------------------- | ---------------------------- | -------------------------- |
| `TARGET_FPS`                    | Frame extraction rate        | `12`                       |
| `YOLO_IMGSZ`                    | Detection input size         | `640`                      |
| `YOLO_EVERY_N_FRAMES`           | Run YOLO every N frames      | `3`                        |
| `ZERODCE_ENABLED`               | Enable low-light enhancement | `1`                        |
| `DARK_BRIGHTNESS_THRESHOLD`     | Trigger enhancement below    | `50`                       |
| `VIDEOMAE_MIN_CONF`             | Alert confidence threshold   | `0.40`                     |
| `INCIDENT_COOLDOWN_SECONDS`     | Alert rate-limit             | `15`                       |
| `REDIS_URL`                     | Redis connection string      | `redis://localhost:6379/0` |
| `NEXT_PUBLIC_SUPABASE_URL`      | Supabase project URL         | —                          |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon key            | —                          |

---

## Frontend Features

- **Live feed** — annotated MJPEG stream with YOLOv11 bounding boxes and FPS counter
- **Incident log** — filterable table (timestamp, camera, class, confidence) with 30s evidence video playback
- **Camera map** — live status indicators per camera
- **Realtime alerts** — Supabase Realtime broadcast, red pulse animation on alerting camera
- **Settings** — add/edit cameras (RTSP URL, name, GPS, zone), tune thresholds per class

---

## API Reference

| Method | Endpoint       | Description                                 |
| ------ | -------------- | ------------------------------------------- |
| `WS`   | `/ws`          | Annotated frame stream + detection metadata |
| `GET`  | `/health`      | Health check                                |
| `POST` | `/api/ingest`  | Upload video file for processing            |
| `GET`  | `/api/cameras` | List configured cameras                     |
| `POST` | `/api/cameras` | Register new camera                         |

---

## Troubleshooting

**No detections appearing**

- Lower `DARK_BRIGHTNESS_THRESHOLD` to trigger ZeroDCE earlier
- Verify YOLOv11 ONNX path in `.env`

**VideoMAE worker stuck**

- Check Redis: `redis-cli ping`
- Verify `VIDEOMAE_WEIGHTS_PATH` in `.env` points to a valid `videomae_best.pth`

**Supabase errors**

- Use `SUPABASE_SERVICE_ROLE_KEY` in backend (not anon key)
- Verify tables `incidents`, `cameras` exist and storage bucket `clips` is created

**Twilio alerts not sending**

- Check phone number format: `+<country><number>`
- Verify `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are correct

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| Backend         | FastAPI + Uvicorn (Python)             |
| Stream Handling | OpenCV                                 |
| ML Inference    | PyTorch / ONNX Runtime                 |
| Message Queue   | Redis Streams (XADD / XREAD)           |
| Database        | Supabase PostgreSQL                    |
| Video Storage   | Supabase Storage                       |
| Realtime Events | Supabase Realtime                      |
| Alerting        | Twilio SMS / MMS                       |
| Frontend        | React + WebSockets + Supabase Realtime |

---

## Evaluation Metrics

- **mAP50 / mAP50-95** — YOLOv11s object detection
- **Macro F1 + per-class F1** — VideoMAE action recognition (4-fold CV)
- **Confusion matrix** — per-fold and mean across folds
- **Inference latency** — ms/frame end-to-end
- **False positive rate** — alert pipeline

---

## Alert Logic

- Alert triggers when confidence ≥ **0.40** and predicted class ≠ Normal
- Auto-clips **30 seconds** of context frames → encoded as `.mp4` → uploaded to Supabase Storage
- Incident row inserted into Supabase PostgreSQL (`camera_id`, `class`, `confidence`, `timestamp`, `clip_url`)
- Supabase Realtime broadcasts event to all connected frontend clients instantly
- Twilio MMS sent to configured phone with class, confidence, camera name, and video attachment
- **Cooldown period** enforced between alerts per camera to prevent duplicates

---

## Hardware

| Component | Spec                       |
| --------- | -------------------------- |
| GPU       | NVIDIA RTX 5060 (8GB VRAM) |
| OS        | Windows                    |
| Python    | 3.11                       |
| PyTorch   | 2.12.0 nightly             |
| CUDA      | 12.8 (on 13.2 driver)      |
| Conda env | `illicit_detect`           |

---

## Acknowledgements

- [UCF-Crime Dataset](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) — University of Central Florida
- [RWF-2000 Dataset](https://huggingface.co/datasets/DanJoshua/RWF-2000)
- [ExDARK Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP)
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) — MCG-NJU
- [Zero-DCE++](https://li-chongyi.github.io/Proj_Zero-DCE++.html)
- [YOLOv11](https://github.com/ultralytics/ultralytics) — Ultralytics
