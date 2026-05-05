# =============================================================================
# pages/1_live_detection.py — Live detection via MJPEG + FastAPI stream server
#
# Architecture:
#   Browser <──MJPEG──> FastAPI :8502  (direct, no Streamlit overhead)
#   Streamlit controls ──REST──> FastAPI :8502
#   Stats panel auto-refreshes every 2s via st.fragment
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import streamlit as st

from config import (
    GLOBAL_CSS, AVAILABLE_MODELS, DEFAULT_MODEL_KEY,
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMG_SIZE,
    COCO_CLASSES, STREAM_SERVER_URL,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Live Detection | RT-DETR Sentinel Pro",
    page_icon="🔴",
    layout="wide",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<p class="gradient-text">🔴 Live Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-heading">MJPEG stream · zero Streamlit latency · FastAPI backend</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------
_TIMEOUT = 3


def _get(path: str) -> dict | None:
    try:
        r = requests.get(f"{STREAM_SERVER_URL}{path}", timeout=_TIMEOUT)
        return r.json()
    except Exception:
        return None


def _post(path: str, payload: dict | None = None) -> dict | None:
    try:
        r = requests.post(
            f"{STREAM_SERVER_URL}{path}",
            json=payload or {},
            timeout=_TIMEOUT,
        )
        return r.json()
    except Exception:
        return None


def _server_online() -> bool:
    return _get("/health") is not None


# ---------------------------------------------------------------------------
# Server-offline guard
# ---------------------------------------------------------------------------
if not _server_online():
    st.error(
        "**Stream Server is offline.**\n\n"
        "Start everything with the launcher:\n"
        "```\npython launcher.py\n```\n\n"
        "Or start only the stream server:\n"
        "```\npython stream_server.py\n```",
        icon="🔌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Video Source")
    source_type = st.radio("Input Source", ["Webcam", "IP Camera", "Video File"],
                           key="src_type")
    device_id  = 0
    ip_url     = ""
    video_path = ""

    if source_type == "Webcam":
        device_id = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
    elif source_type == "IP Camera":
        ip_url = st.text_input("RTSP / HTTP URL",
                               placeholder="rtsp://user:pass@192.168.1.1/stream")
    else:
        video_path = st.text_input("Video File Path", placeholder="C:/video.mp4")

    st.markdown("---")
    st.markdown("### 🤖 Model")
    model_key = st.selectbox(
        "Model",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
    )

    st.markdown("---")
    st.markdown("### ⚙️ Inference")

    conf_val = st.slider("Confidence",  0.05, 1.0, DEFAULT_CONF, 0.05, key="conf_sl")
    iou_val  = st.slider("IoU",         0.05, 1.0, 0.45,         0.05, key="iou_sl")
    imgsz_val = st.select_slider(
        "Inference size (px)",
        [320, 480, 640], value=480, key="imgsz_sl",
        help="Lower = faster on CPU · 320 recommended without GPU",
    )
    skip_val = st.slider(
        "Frame-skip (infer every N frames)",
        1, 10, 3, key="skip_sl",
        help="Higher = smoother display + less CPU load. 2–4 is ideal on CPU.",
    )

    st.markdown("---")
    st.markdown("### 🚨 Alert")
    alert_class_raw = st.selectbox(
        "Alert class",
        ["None"] + [f"{v}  ({k})" for k, v in COCO_CLASSES.items()],
        key="alert_cls",
    )
    alert_thresh = st.number_input("Count threshold", min_value=1, max_value=50, value=3,
                                   key="alert_thr")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
ctrl_col, feed_col, stats_col = st.columns([1, 2.8, 1.2], gap="medium")

# ── Action buttons ────────────────────────────────────────────────────────
with ctrl_col:
    with st.container(border=True):
        st.subheader("⚡ Control")
        start_btn   = st.button("▶️ Start Stream",  use_container_width=True, type="primary")
        stop_btn    = st.button("⏹️ Stop Stream",   use_container_width=True)
        capture_btn = st.button("📸 Capture Frame", use_container_width=True)
        update_btn  = st.button(
            "🔄 Apply Config",
            use_container_width=True,
            help="Push conf/iou/skip to a running stream without restarting",
        )

    with st.container(border=True):
        st.subheader("📶 Server")
        health = _get("/health") or {}
        if health.get("running"):
            st.markdown('<span class="badge-online">● LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-offline">● IDLE</span>', unsafe_allow_html=True)
        if health.get("error"):
            st.error(health["error"], icon="⚠️")

    with st.container(border=True):
        st.subheader("💡 CPU Tips")
        st.markdown(
            "- Use **320 px** inference size\n"
            "- Frame-skip **3–5** recommended\n"
            "- Export to **OpenVINO** in ⚙️ Settings\n"
            "  for 3–8× speedup on Intel CPUs\n"
            "- Use **RT-DETR-S** (small model)"
        )

# ── MJPEG feed ────────────────────────────────────────────────────────────
with feed_col:
    with st.container(border=True):
        stream_url = f"{STREAM_SERVER_URL}/stream"
        st.markdown(
            f"""
            <div style="text-align:center; background:#0B0F19;
                        border-radius:8px; padding:4px; overflow:hidden;">
                <img src="{stream_url}"
                     style="width:100%; border-radius:6px; display:block;"
                     onerror="this.style.opacity='0.2'">
            </div>
            <p style="text-align:center; color:#3D4F66; font-size:0.78em; margin-top:6px;">
                Live MJPEG · direct browser connection ·
                <a href="{stream_url}" target="_blank"
                   style="color:#00C9FF;">open standalone ↗</a>
            </p>
            """,
            unsafe_allow_html=True,
        )

# ── Auto-refresh stats fragment ───────────────────────────────────────────
with stats_col:
    stats_box  = st.container(border=True)
    chart_box  = st.container(border=True)


@st.fragment(run_every=2)
def _live_stats():
    stats = _get("/stats") or {}

    with stats_box:
        st.subheader("🔎 Stats")
        st.metric("Inference FPS", f"{stats.get('fps_inference', 0.0):.1f}",
                  help="RT-DETR throughput")
        st.metric("Capture FPS",   f"{stats.get('fps_display',   0.0):.1f}",
                  help="Camera read speed")
        st.metric("Frames done",   stats.get("frame_count", 0))
        st.metric("Total dets.",   stats.get("det_count",   0))

        counts: dict = stats.get("class_counts", {})
        if counts:
            st.markdown("**This frame:**")
            for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{cls}**: {cnt}")

            # Alert
            if alert_class_raw != "None":
                alert_name = alert_class_raw.split("  (")[0].strip()
                if counts.get(alert_name, 0) >= alert_thresh:
                    st.toast(
                        f"🚨 {counts[alert_name]}× {alert_name} detected!",
                        icon="🚨",
                    )
        elif stats.get("running"):
            st.caption("Nothing detected…")

    with chart_box:
        st.subheader("📉 Inf. FPS")
        fps = stats.get("fps_inference", 0.0)
        if "fps_hist" not in st.session_state:
            st.session_state.fps_hist = []
        st.session_state.fps_hist.append(fps)
        st.session_state.fps_hist = st.session_state.fps_hist[-60:]
        st.line_chart(st.session_state.fps_hist, height=130)


_live_stats()

# ---------------------------------------------------------------------------
# Button handlers
# ---------------------------------------------------------------------------
src_map = {"Webcam": "webcam", "IP Camera": "ipcam", "Video File": "file"}

if start_btn:
    payload = {
        "model_path":  AVAILABLE_MODELS[model_key],
        "source_type": src_map[source_type],
        "device_id":   device_id,
        "url":         ip_url,
        "file_path":   video_path,
        "conf":        conf_val,
        "iou":         iou_val,
        "imgsz":       imgsz_val,
        "frame_skip":  skip_val,
    }
    resp = _post("/start", payload)
    if resp and resp.get("status") == "ok":
        st.toast(f"✅ Stream started (session {resp.get('session_id')})", icon="▶️")
        st.session_state.fps_hist = []
    else:
        msg = (resp or {}).get("message", "Stream server unreachable")
        st.error(f"❌ {msg}")
    st.rerun()

if stop_btn:
    resp = _post("/stop")
    if resp:
        st.toast(
            f"⏹️ Stopped — {resp.get('frames',0)} frames, "
            f"{resp.get('detections',0)} detections logged",
            icon="⏹️",
        )
    st.rerun()

if capture_btn:
    resp = _post("/capture")
    if resp and resp.get("status") == "ok":
        st.toast("📸 Frame saved to training_data/images/", icon="📸")
    else:
        st.toast(f"⚠️ {(resp or {}).get('message','No active frame')}", icon="⚠️")

if update_btn:
    resp = _post("/config", {
        "conf":       conf_val,
        "iou":        iou_val,
        "imgsz":      imgsz_val,
        "frame_skip": skip_val,
    })
    if resp and resp.get("status") == "ok":
        st.toast(
            f"✅ conf={conf_val} · iou={iou_val} · "
            f"imgsz={imgsz_val} · skip={skip_val}",
            icon="✅",
        )
    else:
        st.error("❌ Config update failed. Is the stream running?")
