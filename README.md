# RT-DETR Sentinel Pro

A professional real-time object detection suite powered by **RT-DETR** (Real-Time Detection Transformer) and **Streamlit**.

---

## Features

| Page | Description |
|---|---|
| 🔴 **Live Detection** | Stream from webcam, IP camera, or video file with real-time RT-DETR inference, live FPS counter, per-class alert thresholds, and one-click frame capture |
| 📁 **File Detection** | Upload single/multiple images or videos, run batch folder inference, download annotated outputs and export as CSV/JSON |
| 📊 **Analytics** | Full detection history dashboard — class distribution, confidence histograms, detections-over-time charts, session history, all backed by SQLite |
| 🖍️ **Annotation Studio** | Draw bounding boxes over captured frames, assign COCO class labels, review dataset stats, export full dataset as ZIP |
| ⚙️ **Model Settings** | Switch model sizes (RT-DETR-L / X), load custom `.pt` weights, benchmark latency/FPS, launch fine-tuning on collected data |

---

## Project Structure

```
project/
├── app.py                        # Entry point (run this)
├── config.py                     # All constants — classes, paths, styles
├── requirements.txt
├── core/
│   ├── detector.py               # Model loading & inference wrapper
│   ├── video_stream.py           # Unified video source abstraction
│   └── logger.py                 # SQLite detection logger
├── pages/
│   ├── 1_live_detection.py
│   ├── 2_file_detection.py
│   ├── 3_analytics.py
│   ├── 4_annotation_studio.py
│   └── 5_model_settings.py
├── training_data/
│   ├── images/                   # Captured edge-case frames
│   └── labels/                   # YOLO-format annotation .txt files
├── detection_logs/               # SQLite DB
└── exports/                      # Annotated video outputs
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
py -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Model Weights

On first run, Ultralytics will automatically download the selected model weights (`rtdetr-l.pt` or `rtdetr-x.pt`).  
Custom weights can be loaded from **⚙️ Model Settings → Custom Model**.

---

## Data Pipeline

1. **Capture** edge-case frames during live detection (📸 button)
2. **Label** them in the Annotation Studio (YOLO format)
3. **Export** dataset ZIP from Annotation Studio
4. **Fine-tune** from Model Settings → Fine-Tune tab

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU inference works but is slower)
- Webcam or IP camera for live detection
