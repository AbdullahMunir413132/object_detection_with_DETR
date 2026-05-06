# =============================================================================
# core/detector.py — Model loading & inference abstraction
# =============================================================================

from __future__ import annotations

import time
import numpy as np
import streamlit as st
from ultralytics import RTDETR

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR, DEFAULT_CONF, DEFAULT_IOU, MAX_DETECTIONS, DEFAULT_IMG_SIZE


# ---------------------------------------------------------------------------
# Cached loader — one model instance per Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_path: str) -> RTDETR:
    """Load (and cache) an RT-DETR model from *model_path*."""
    try:
        requested = model_path
        if not os.path.isabs(requested):
            local_requested = os.path.join(BASE_DIR, requested)
            if os.path.exists(local_requested):
                requested = local_requested

        chosen = requested
        if isinstance(requested, str) and requested.endswith(".pt") and not os.path.isfile(requested):
            for cand in (
                os.path.join(BASE_DIR, "rtdetr-l.pt"),
                os.path.join(BASE_DIR, "rtdetr-s.pt"),
                os.path.join(BASE_DIR, "rtdetr-x.pt"),
                "rtdetr-l.pt",
                "rtdetr-s.pt",
                "rtdetr-x.pt",
            ):
                if os.path.isfile(cand):
                    chosen = cand
                    break

        model = RTDETR(chosen)
        return model
    except Exception as exc:
        st.error(f"❌ Failed to load model `{model_path}`: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class DetectionResult:
    """Lightweight wrapper around a single Ultralytics result object."""

    def __init__(self, result, inference_ms: float):
        self.result       = result
        self.inference_ms = inference_ms
        self.boxes        = result.boxes          # Boxes object
        self.annotated    = result.plot()         # BGR numpy array

        # Derived fields (all CPU numpy for easy serialisation)
        if self.boxes is not None and len(self.boxes):
            self.cls_ids    = self.boxes.cls.cpu().numpy().astype(int)
            self.confs      = self.boxes.conf.cpu().numpy()
            self.xyxy       = self.boxes.xyxy.cpu().numpy()
        else:
            self.cls_ids = np.array([], dtype=int)
            self.confs   = np.array([], dtype=float)
            self.xyxy    = np.zeros((0, 4), dtype=float)

        self.class_names: list[str] = list(result.names.values()) if result.names else []

    def get_label(self, idx: int) -> str:
        try:
            return self.class_names[idx]
        except IndexError:
            return str(idx)

    def to_records(self) -> list[dict]:
        """Return list of dicts, one per detection."""
        records = []
        for i, cid in enumerate(self.cls_ids):
            x1, y1, x2, y2 = self.xyxy[i]
            records.append({
                "class_id":   int(cid),
                "class_name": self.get_label(int(cid)),
                "confidence": float(self.confs[i]),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
            })
        return records


def run_inference(
    model: RTDETR,
    source,
    conf:     float = DEFAULT_CONF,
    iou:      float = DEFAULT_IOU,
    classes:  list[int] | None = None,
    imgsz:    int   = DEFAULT_IMG_SIZE,
    max_det:  int   = MAX_DETECTIONS,
) -> DetectionResult:
    """Run RT-DETR inference and return a :class:`DetectionResult`."""
    t0 = time.perf_counter()
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        classes=classes,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return DetectionResult(results[0], elapsed_ms)
