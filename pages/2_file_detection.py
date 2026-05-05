# =============================================================================
# pages/2_file_detection.py — Image / Video file upload & batch processing
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import json
import csv
import time
import tempfile
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import (
    GLOBAL_CSS, AVAILABLE_MODELS, DEFAULT_MODEL_KEY,
    DEFAULT_CONF, DEFAULT_IOU, EXPORTS_DIR,
)
from core.detector import load_model, run_inference
from core.logger import log_detections

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="File Detection | RT-DETR Sentinel Pro",
                   page_icon="📁", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<p class="gradient-text">📁 File Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Upload images or videos — run inference, download results</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Inference Settings")
    model_key   = st.selectbox("Model", list(AVAILABLE_MODELS.keys()),
                               index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY))
    conf_thresh = st.slider("Confidence Threshold", 0.05, 1.0, DEFAULT_CONF, 0.05)
    iou_thresh  = st.slider("IoU Threshold",        0.05, 1.0, DEFAULT_IOU,  0.05)
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    export_csv  = st.checkbox("Export detections as CSV",  value=True)
    export_json = st.checkbox("Export detections as JSON", value=True)

model = load_model(AVAILABLE_MODELS[model_key])

# ---------------------------------------------------------------------------
# Mode tabs
# ---------------------------------------------------------------------------
tab_img, tab_vid, tab_batch = st.tabs(["🖼️ Image Upload", "🎬 Video Upload", "📂 Batch Folder"])

# ===========================================================================
# TAB 1 — Single / multi image upload
# ===========================================================================
with tab_img:
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_records: list[dict] = []

        for uf in uploaded_files:
            img_bytes = np.frombuffer(uf.read(), np.uint8)
            img_bgr   = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.warning(f"⚠️ Could not decode `{uf.name}` — skipping.")
                continue

            with st.spinner(f"Running inference on `{uf.name}`…"):
                det = run_inference(model, img_bgr, conf=conf_thresh, iou=iou_thresh)

            col_orig, col_ann = st.columns(2, gap="medium")
            with col_orig:
                st.markdown(f"**Original — `{uf.name}`**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
            with col_ann:
                st.markdown(f"**Annotated — `{uf.name}`**  "
                            f"<span style='color:#8899AA; font-size:0.85em;'>"
                            f"{len(det.cls_ids)} detection(s) | "
                            f"{det.inference_ms:.1f} ms</span>",
                            unsafe_allow_html=True)
                ann_rgb = cv2.cvtColor(det.annotated, cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)

                # Download annotated image
                _, buf = cv2.imencode(".jpg", det.annotated)
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=buf.tobytes(),
                    file_name=f"annotated_{uf.name}",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            recs = det.to_records()
            for r in recs:
                r["source_file"] = uf.name
            all_records.extend(recs)

            log_detections("file_upload", recs, "file_upload", AVAILABLE_MODELS[model_key])
            st.markdown("---")

        # ---- Bulk export ----
        if all_records:
            st.markdown("### 📊 All Detections Summary")
            with st.container(border=True):
                st.dataframe(all_records, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                if export_csv:
                    csv_buf = io.StringIO()
                    writer = csv.DictWriter(csv_buf, fieldnames=all_records[0].keys())
                    writer.writeheader()
                    writer.writerows(all_records)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv_buf.getvalue().encode(),
                        "detections.csv", "text/csv",
                        use_container_width=True,
                    )
            with dl2:
                if export_json:
                    st.download_button(
                        "⬇️ Download JSON",
                        json.dumps(all_records, indent=2).encode(),
                        "detections.json", "application/json",
                        use_container_width=True,
                    )

# ===========================================================================
# TAB 2 — Video file upload
# ===========================================================================
with tab_vid:
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_file.name)[1],
                                         delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap      = cv2.VideoCapture(tmp_path)
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(
            f"📹 **{video_file.name}** — {total_fr} frames, "
            f"{native_fps:.1f} FPS, {w}×{h}",
            icon="ℹ️",
        )

        sample_every = st.slider(
            "Process every N-th frame (1 = every frame, higher = faster preview)",
            1, 30, 5,
        )

        if st.button("🚀 Run Video Inference", type="primary", use_container_width=True):
            cap      = cv2.VideoCapture(tmp_path)
            out_path = os.path.join(EXPORTS_DIR, f"annotated_{video_file.name}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(out_path, fourcc, native_fps, (w, h))

            progress = st.progress(0, text="Processing…")
            preview  = st.empty()
            all_records_vid: list[dict] = []

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every == 0:
                    det = run_inference(model, frame, conf=conf_thresh, iou=iou_thresh)
                    out.write(det.annotated)
                    recs = det.to_records()
                    for r in recs:
                        r["frame"] = frame_idx
                    all_records_vid.extend(recs)
                    preview.image(
                        cv2.cvtColor(det.annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}/{total_fr}",
                        use_container_width=True,
                    )
                else:
                    out.write(frame)

                frame_idx += 1
                progress.progress(min(frame_idx / max(total_fr, 1), 1.0),
                                   text=f"Frame {frame_idx}/{total_fr}")

            cap.release()
            out.release()
            progress.empty()

            log_detections("video_upload", all_records_vid, "video_upload",
                           AVAILABLE_MODELS[model_key])

            st.success(f"✅ Done! {len(all_records_vid)} detections across {frame_idx} frames.")

            # Offer download
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Annotated Video",
                    f.read(),
                    file_name=f"annotated_{video_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                )

            if export_json and all_records_vid:
                st.download_button(
                    "⬇️ Download Detections JSON",
                    json.dumps(all_records_vid, indent=2).encode(),
                    "video_detections.json", "application/json",
                    use_container_width=True,
                )

        os.unlink(tmp_path) if os.path.exists(tmp_path) else None

# ===========================================================================
# TAB 3 — Batch folder processing
# ===========================================================================
with tab_batch:
    st.info(
        "Provide a folder path on disk containing images. "
        "The app will run inference on all images and produce a ZIP of annotated outputs.",
        icon="📂",
    )

    folder_path = st.text_input("Folder Path", placeholder="C:/images/my_dataset")
    col_ext, col_run = st.columns([1, 1])
    with col_ext:
        extensions = st.multiselect("Extensions", [".jpg", ".jpeg", ".png", ".bmp"],
                                    default=[".jpg", ".jpeg", ".png"])

    if st.button("🚀 Run Batch Inference", type="primary", use_container_width=True):
        if not folder_path or not os.path.isdir(folder_path):
            st.error("❌ Please enter a valid folder path.")
        else:
            image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in extensions
            ]
            if not image_paths:
                st.warning("No images found in that folder with the selected extensions.")
            else:
                st.info(f"Found **{len(image_paths)}** images. Running inference…")
                progress = st.progress(0)
                batch_records: list[dict] = []

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, img_path in enumerate(image_paths):
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is None:
                            continue
                        det  = run_inference(model, img_bgr, conf=conf_thresh, iou=iou_thresh)
                        _, buf = cv2.imencode(".jpg", det.annotated)
                        zf.writestr(f"annotated_{os.path.basename(img_path)}", buf.tobytes())

                        recs = det.to_records()
                        for r in recs:
                            r["source_file"] = os.path.basename(img_path)
                        batch_records.extend(recs)
                        progress.progress((i + 1) / len(image_paths))

                log_detections("batch", batch_records, "batch_folder", AVAILABLE_MODELS[model_key])
                st.success(f"✅ Done! {len(batch_records)} total detections.")

                st.download_button(
                    "⬇️ Download Annotated Images (ZIP)",
                    zip_buf.getvalue(),
                    "batch_annotated.zip", "application/zip",
                    use_container_width=True,
                )

                if export_csv and batch_records:
                    csv_buf = io.StringIO()
                    writer  = csv.DictWriter(csv_buf, fieldnames=batch_records[0].keys())
                    writer.writeheader()
                    writer.writerows(batch_records)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv_buf.getvalue().encode(),
                        "batch_detections.csv", "text/csv",
                        use_container_width=True,
                    )
