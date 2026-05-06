# =============================================================================
# app.py — RT-DETR Sentinel Pro  |  Entry point
# =============================================================================
# Run with:  py -m streamlit run app.py

import os
import sys
import subprocess
import time
import urllib.request

import streamlit as st
from config import APP_TITLE, APP_ICON, GLOBAL_CSS, STREAM_SERVER_PORT

# ---------------------------------------------------------------------------
# Auto-start stream server if it's not already running
# ---------------------------------------------------------------------------
def _stream_server_alive() -> bool:
    try:
        urllib.request.urlopen(
            f"http://localhost:{STREAM_SERVER_PORT}/health", timeout=1
        )
        return True
    except Exception:
        return False

if "stream_server_pid" not in st.session_state:
    st.session_state.stream_server_pid = None

if not _stream_server_alive():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "stream_server.py")],
        cwd=BASE_DIR,
    )
    st.session_state.stream_server_pid = proc.pid
    # Give it a moment to bind the port
    for _ in range(20):
        time.sleep(0.5)
        if _stream_server_alive():
            break

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar brand
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 12px 0 8px;">
            <span style="font-size:2.4em;">👁️</span><br>
            <span style="font-size:1.3em; font-weight:800;
                  background:linear-gradient(90deg,#00C9FF,#92FE9D);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                RT-DETR Sentinel Pro
            </span><br>
            <span style="color:#556070; font-size:0.78em;">
                Real-Time Object Detection Suite
            </span>
        </div>
        <hr style="border-color:#1E2535; margin:8px 0 16px;">
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.markdown('<p class="gradient-text">👁️ RT-DETR Sentinel Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Real-Time Object Detection Suite powered by RT-DETR</p>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 🔴 Live Detection")
        st.markdown(
            "Stream from webcam, IP camera, or video file with real-time "
            "RT-DETR inference, FPS counter, per-class alert thresholds, and "
            "one-click frame capture for retraining."
        )
        st.page_link("pages/1_live_detection.py", label="Open Live Detection →", icon="🔴")

with col2:
    with st.container(border=True):
        st.markdown("### 📁 File Detection")
        st.markdown(
            "Upload individual images or videos, run batch inference over "
            "entire folders, download annotated outputs and export results as "
            "CSV or JSON."
        )
        st.page_link("pages/2_file_detection.py", label="Open File Detection →", icon="📁")

with col3:
    with st.container(border=True):
        st.markdown("### 📊 Analytics")
        st.markdown(
            "Explore your full detection history — class distribution, "
            "confidence histograms, detections-over-time charts, and session "
            "summaries.  All backed by SQLite."
        )
        st.page_link("pages/3_analytics.py", label="Open Analytics →", icon="📊")

st.markdown("")

col4, col5, _ = st.columns(3, gap="large")

with col4:
    with st.container(border=True):
        st.markdown("### 🖍️ Annotation Studio")
        st.markdown(
            "Draw bounding boxes over captured edge-case frames, assign COCO "
            "class labels, review dataset stats per class, and export your "
            "dataset as a ready-to-train ZIP archive."
        )
        st.page_link("pages/4_annotation_studio.py", label="Open Annotation Studio →", icon="🖍️")

with col5:
    with st.container(border=True):
        st.markdown("### ⚙️ Model Settings")
        st.markdown(
            "Switch between RT-DETR model sizes, load a custom `.pt` weight "
            "file, view benchmark stats, and launch a fine-tuning run on your "
            "annotated training data."
        )
        st.page_link("pages/5_model_settings.py", label="Open Model Settings →", icon="⚙️")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#3D4F66; font-size:0.82em;'>"
    "RT-DETR Sentinel Pro &nbsp;|&nbsp; Built with Streamlit &amp; Ultralytics RT-DETR"
    "</div>",
    unsafe_allow_html=True,
)
