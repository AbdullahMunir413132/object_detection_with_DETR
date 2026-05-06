# =============================================================================
# stream_server.py — FastAPI MJPEG stream server for RT-DETR Sentinel Pro
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────┐
#   │  _capture_thread  ──► latest_raw_frame (always fresh)       │
#   │  _inference_thread ──► latest_annotated_frame + detections  │
#   │  MJPEG endpoint ──► reads latest_annotated at ~30 FPS       │
#   └─────────────────────────────────────────────────────────────┘
#
# Run standalone:  python stream_server.py
# Or via:          python launcher.py
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import uuid
import threading
from collections import Counter, deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BASE_DIR,
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMG_SIZE,
    AVAILABLE_MODELS, DEFAULT_MODEL_KEY,
    TRAINING_IMAGES_DIR, STREAM_SERVER_PORT,
    OPENVINO_MODELS, ONNX_MODELS,
)
from core.logger import log_detections, start_session, end_session, update_session_stats


# ---------------------------------------------------------------------------
# Shared mutable state  (all writes protected by state.lock)
# ---------------------------------------------------------------------------

class StreamState:
    def __init__(self):
        self.lock               = threading.Lock()
        self.running            = False
        self.latest_raw:        Optional[np.ndarray] = None
        self.latest_annotated:  Optional[np.ndarray] = None
        self.latest_detections: list[dict]           = []
        self.fps_inference:     float = 0.0
        self.fps_display:       float = 0.0
        self.frame_count:       int   = 0
        self.det_count:         int   = 0
        self.session_id:        str   = ""
        self.error_msg:         str   = ""

    def reset(self):
        with self.lock:
            self.running            = False
            self.latest_raw         = None
            self.latest_annotated   = None
            self.latest_detections  = []
            self.fps_inference      = 0.0
            self.fps_display        = 0.0
            self.frame_count        = 0
            self.det_count          = 0
            self.error_msg          = ""


_state = StreamState()


# ---------------------------------------------------------------------------
# Pydantic models for REST API
# ---------------------------------------------------------------------------

class StreamConfig(BaseModel):
    model_path:  str   = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]
    source_type: str   = "webcam"      # webcam | ipcam | file
    device_id:   int   = 0
    url:         str   = ""
    file_path:   str   = ""
    conf:        float = DEFAULT_CONF
    iou:         float = DEFAULT_IOU
    imgsz:       int   = DEFAULT_IMG_SIZE
    frame_skip:  int   = 2             # run inference every N frames (1 = every frame)


class LiveConfig(BaseModel):
    """Hot-update inference params without restarting the stream."""
    conf:       Optional[float] = None
    iou:        Optional[float] = None
    imgsz:      Optional[int]   = None
    frame_skip: Optional[int]   = None


# Mutable runtime config (inference thread always reads this)
_cfg = StreamConfig()


# ---------------------------------------------------------------------------
# Model loader — supports PyTorch, OpenVINO, ONNX
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """
    Load RT-DETR with the best available backend for the given weight path.

    Priority:
      1. If an OpenVINO-exported directory exists → use OpenVINO (fastest on CPU)
      2. If an ONNX-exported file exists → use ONNX Runtime (2–4× vs PyTorch)
      3. Fall back to PyTorch (.pt)
    """
    from ultralytics import RTDETR

    def _resolve_candidate_path(p: str) -> str:
        if os.path.isabs(p):
            return p
        local = os.path.join(BASE_DIR, p)
        if os.path.exists(local):
            return local
        return p

    requested = _resolve_candidate_path(model_path)
    fallback_candidates = [
        os.path.join(BASE_DIR, "rtdetr-l.pt"),
        os.path.join(BASE_DIR, "rtdetr-s.pt"),
        os.path.join(BASE_DIR, "rtdetr-x.pt"),
        "rtdetr-l.pt",
        "rtdetr-s.pt",
        "rtdetr-x.pt",
    ]

    chosen = requested
    if isinstance(requested, str) and requested.endswith(".pt") and not os.path.isfile(requested):
        for cand in fallback_candidates:
            if os.path.isfile(cand):
                chosen = cand
                print(f"[stream_server] Requested model missing ({requested}), using fallback: {cand}")
                break

    lookup_key = os.path.basename(chosen)
    ov_path = OPENVINO_MODELS.get(chosen, "") or OPENVINO_MODELS.get(lookup_key, "")
    onnx_path = ONNX_MODELS.get(chosen, "") or ONNX_MODELS.get(lookup_key, "")

    if ov_path and os.path.isdir(ov_path):
        print(f"[stream_server] Loading OpenVINO model: {ov_path}")
        return RTDETR(ov_path)

    if onnx_path and os.path.isfile(onnx_path):
        print(f"[stream_server] Loading ONNX model: {onnx_path}")
        return RTDETR(onnx_path)

    print(f"[stream_server] Loading PyTorch model: {chosen}")
    return RTDETR(chosen)


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _capture_thread(cfg: StreamConfig):
    """
    Thread 1 — reads frames as fast as possible from the source.
    Writes raw frames into _state.latest_raw.
    """
    global _state

    if cfg.source_type == "webcam":
        src = cfg.device_id
    elif cfg.source_type == "ipcam":
        src = cfg.url
    else:
        src = cfg.file_path

    cap = None
    if cfg.source_type == "webcam":
        # Windows webcam backend fallback chain (helps avoid blank/white feed)
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap_try = cv2.VideoCapture(src, backend)
            if cap_try.isOpened():
                cap = cap_try
                break
            cap_try.release()
    else:
        cap = cv2.VideoCapture(src)

    if cap is None:
        _state.error_msg = f"Cannot open source: {src!r}"
        _state.running   = False
        return

    if not cap.isOpened():
        _state.error_msg = f"Cannot open source: {src!r}"
        _state.running   = False
        return

    if cfg.source_type == "webcam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    fps_times: deque = deque(maxlen=30)

    while _state.running:
        ret, frame = cap.read()
        if not ret:
            if cfg.source_type == "file":
                # Loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            _state.running = False
            break

        with _state.lock:
            _state.latest_raw = frame

        fps_times.append(time.perf_counter())
        if len(fps_times) >= 2:
            _state.fps_display = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

        # Slight throttle to avoid spinning CPU needlessly
        time.sleep(0.005)

    cap.release()


def _inference_thread(model):
    """
    Thread 2 — runs RT-DETR on latest_raw, writes annotated frame + detections.
    Always reads latest _cfg so hot-config-updates take effect immediately.
    """
    global _state, _cfg

    fps_times: deque = deque(maxlen=20)
    frame_idx = 0

    while _state.running:
        # Always read latest config (supports hot-update via /config endpoint)
        cfg = _cfg

        with _state.lock:
            raw = _state.latest_raw

        if raw is None:
            time.sleep(0.01)
            continue

        frame_idx += 1

        # Frame-skip: on skipped frames, re-display last annotated result so
        # display thread stays smooth while inference is expensive on CPU
        if frame_idx % max(cfg.frame_skip, 1) != 0:
            time.sleep(0.008)
            continue

        # ---- Inference ----
        t0 = time.perf_counter()
        try:
            results = model.predict(
                source=raw,
                conf=cfg.conf,
                iou=cfg.iou,
                imgsz=cfg.imgsz,
                verbose=False,
            )
        except Exception as exc:
            print(f"[inference_thread] Prediction error: {exc}")
            time.sleep(0.1)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result     = results[0]
        annotated  = result.plot()
        boxes      = result.boxes

        detections: list[dict] = []
        if boxes is not None and len(boxes):
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs   = boxes.conf.cpu().numpy()
            xyxy    = boxes.xyxy.cpu().numpy()
            names   = result.names or {}
            for i, cid in enumerate(cls_ids):
                x1, y1, x2, y2 = xyxy[i]
                detections.append({
                    "class_id":   int(cid),
                    "class_name": names.get(int(cid), str(cid)),
                    "confidence": float(confs[i]),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                })

        with _state.lock:
            _state.latest_annotated  = annotated
            _state.latest_detections = detections
            _state.frame_count      += 1
            _state.det_count        += len(detections)

        fps_times.append(time.perf_counter())
        if len(fps_times) >= 2:
            _state.fps_inference = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

        # DB log every 10 inference frames (reduces I/O overhead)
        if _state.frame_count % 10 == 0 and detections:
            log_detections(
                _state.session_id, detections,
                cfg.source_type, cfg.model_path,
            )


# ---------------------------------------------------------------------------
# MJPEG frame generator
# ---------------------------------------------------------------------------

# Cached blank "waiting" frame so we don't allocate one every iteration
_BLANK_FRAME: Optional[bytes] = None

def _get_blank_jpeg() -> bytes:
    global _BLANK_FRAME
    if _BLANK_FRAME is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Stream not started", (140, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
        cv2.putText(blank, "Press  Start Stream  in the dashboard",
                    (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1)
        _, enc = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _BLANK_FRAME = enc.tobytes()
    return _BLANK_FRAME


def _mjpeg_generator():
    """
    Yields MJPEG boundary frames.  Runs in the FastAPI request context — reads
    from shared state, never blocks the inference thread.
    """
    while True:
        with _state.lock:
            annotated = _state.latest_annotated
            raw       = _state.latest_raw

        frame = annotated if annotated is not None else raw

        if frame is not None:
            ok, enc = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
            )
            jpeg_bytes = enc.tobytes() if ok else _get_blank_jpeg()
        else:
            jpeg_bytes = _get_blank_jpeg()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )

        # 30 FPS delivery cap — avoids saturating the network/browser
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RT-DETR Sentinel — Stream Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Endpoints ----

@app.get("/stream")
def video_stream():
    """MJPEG stream endpoint — embed with <img src='http://localhost:8502/stream'>"""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start")
def start_stream(cfg: StreamConfig):
    global _state, _cfg

    # Stop any existing stream first
    if _state.running:
        _state.running = False
        time.sleep(0.4)

    _state.reset()
    _cfg = cfg

    _state.running    = True
    _state.session_id = str(uuid.uuid4())[:8]

    # Load model (blocking — done before threads start so errors surface cleanly)
    try:
        model = _load_model(cfg.model_path)
    except Exception as exc:
        _state.running = False
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )

    start_session(_state.session_id, cfg.source_type, cfg.model_path)

    threading.Thread(target=_capture_thread,   args=(cfg,),        daemon=True, name="capture").start()
    threading.Thread(target=_inference_thread, args=(model,),      daemon=True, name="inference").start()

    return {"status": "ok", "session_id": _state.session_id}


@app.post("/stop")
def stop_stream():
    global _state
    if _state.running:
        sid = _state.session_id
        _state.running = False
        end_session(sid)
        update_session_stats(sid, _state.frame_count, _state.det_count)
        return {"status": "stopped", "frames": _state.frame_count, "detections": _state.det_count}
    return {"status": "already_stopped"}


@app.post("/config")
def update_config(cfg: LiveConfig):
    """Hot-update inference parameters without restarting the stream."""
    global _cfg
    if cfg.conf       is not None: _cfg.conf       = cfg.conf
    if cfg.iou        is not None: _cfg.iou        = cfg.iou
    if cfg.imgsz      is not None: _cfg.imgsz      = cfg.imgsz
    if cfg.frame_skip is not None: _cfg.frame_skip = cfg.frame_skip
    return {"status": "ok", "conf": _cfg.conf, "iou": _cfg.iou,
            "imgsz": _cfg.imgsz, "frame_skip": _cfg.frame_skip}


@app.get("/stats")
def get_stats():
    with _state.lock:
        dets   = list(_state.latest_detections)
        counts = dict(Counter(d["class_name"] for d in dets))
        return {
            "running":            _state.running,
            "fps_inference":      round(_state.fps_inference, 1),
            "fps_display":        round(_state.fps_display, 1),
            "frame_count":        _state.frame_count,
            "det_count":          _state.det_count,
            "current_detections": dets,
            "class_counts":       counts,
            "session_id":         _state.session_id,
            "error":              _state.error_msg,
        }


@app.post("/capture")
def capture_frame():
    """Save the current raw frame to training_data/images/."""
    with _state.lock:
        frame = _state.latest_raw

    if frame is None:
        return JSONResponse(
            {"status": "error", "message": "No frame available"},
            status_code=400,
        )

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    path = os.path.join(TRAINING_IMAGES_DIR, f"capture_{ts}.jpg")
    ok   = cv2.imwrite(path, frame)
    if not ok:
        return JSONResponse({"status": "error", "message": "cv2.imwrite failed"}, status_code=500)
    return {"status": "ok", "path": path}


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "running": _state.running,
        "error":   _state.error_msg,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[stream_server] Starting on http://0.0.0.0:{STREAM_SERVER_PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=STREAM_SERVER_PORT,
        log_level="warning",
        access_log=False,
    )
