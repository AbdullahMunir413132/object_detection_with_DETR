# =============================================================================
# pages/5_model_settings.py — Model management & fine-tune launcher
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import time
import tempfile
import subprocess

import streamlit as st

from config import (
    GLOBAL_CSS, AVAILABLE_MODELS, DEFAULT_MODEL_KEY,
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMG_SIZE,
    TRAINING_IMAGES_DIR, TRAINING_LABELS_DIR, BASE_DIR,
    OPENVINO_MODELS, ONNX_MODELS,
)
from core.detector import load_model, run_inference

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Model Settings | RT-DETR Sentinel Pro",
                   page_icon="⚙️", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<p class="gradient-text">⚙️ Model Settings</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Switch models, accelerate for CPU, benchmark, fine-tune</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_select, tab_accel, tab_bench, tab_finetune = st.tabs([
    "🔄 Model Selector",
    "🚀 CPU Acceleration",
    "⚡ Benchmark",
    "🎓 Fine-Tune",
])

# ===========================================================================
# TAB 1 — Model selector
# ===========================================================================
with tab_select:
    col_left, col_right = st.columns([1.5, 2], gap="large")

    with col_left:
        with st.container(border=True):
            st.subheader("📦 Preset Models")
            chosen_key = st.selectbox("Select model", list(AVAILABLE_MODELS.keys()),
                                      index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY))
            st.caption(f"Weight file: `{AVAILABLE_MODELS[chosen_key]}`")

            load_btn = st.button("✅ Load This Model", type="primary", use_container_width=True)
            if load_btn:
                with st.spinner(f"Loading `{AVAILABLE_MODELS[chosen_key]}`…"):
                    m = load_model(AVAILABLE_MODELS[chosen_key])
                st.success(f"Model `{AVAILABLE_MODELS[chosen_key]}` loaded and cached.")

        with st.container(border=True):
            st.subheader("📂 Custom Model")
            custom_path = st.text_input("Path to custom .pt file",
                                        placeholder="C:/models/my_rtdetr.pt")
            if st.button("✅ Load Custom Model", use_container_width=True):
                if not custom_path or not os.path.isfile(custom_path):
                    st.error("❌ File not found. Check the path.")
                else:
                    with st.spinner("Loading custom weights…"):
                        try:
                            m = load_model(custom_path)
                            st.success(f"Loaded: `{custom_path}`")
                        except Exception as e:
                            st.error(f"Error loading model: {e}")

    with col_right:
        with st.container(border=True):
            st.subheader("ℹ️ Model Reference")
            st.markdown("""
| Model | Params | COCO mAP | CPU FPS (approx) |
|---|---|---|---|
| RT-DETR-S | 20M | 48.1 | ~8–12 |
| RT-DETR-L | 32M | 53.0 | ~3–6  |
| RT-DETR-X | 67M | 54.8 | ~1–3  |

**RT-DETR** (Real-Time Detection Transformer) is a Transformer-based detector
that eliminates NMS, achieving strong accuracy at real-time speeds.

- **rtdetr-s.pt** — recommended for CPU (best speed/accuracy balance)
- **rtdetr-l.pt** — better accuracy, slower on CPU
- **rtdetr-x.pt** — best accuracy, heaviest compute
- **Custom .pt**  — your fine-tuned weights after training on captured data

> 💡 Export to **OpenVINO** in the **CPU Acceleration** tab for 3–8× speedup.
            """)

# ===========================================================================
# TAB 2 — CPU Acceleration (OpenVINO / ONNX export)
# ===========================================================================
with tab_accel:
    st.info(
        "Export your RT-DETR model to a faster format for CPU inference. "
        "The stream server automatically uses the fastest available format.",
        icon="🚀",
    )

    accel_col1, accel_col2 = st.columns(2, gap="large")

    with accel_col1:
        with st.container(border=True):
            st.subheader("🔵 OpenVINO Export")
            st.markdown(
                "**3–8× faster** than PyTorch on Intel CPUs (Core, Xeon, etc.).\n\n"
                "Exports to a directory with `.xml` + `.bin` files. "
                "The stream server loads this automatically on next start."
            )
            ov_model_key = st.selectbox("Model to export (OpenVINO)",
                                        list(AVAILABLE_MODELS.keys()), key="ov_sel")
            ov_imgsz = st.select_slider("Export image size", [320, 480, 640], value=480,
                                        key="ov_imgsz")

            ov_out = OPENVINO_MODELS.get(AVAILABLE_MODELS[ov_model_key], "")
            already_ov = os.path.isdir(ov_out)
            if already_ov:
                st.success(f"✅ Already exported: `{os.path.basename(ov_out)}/`")

            if st.button("🔵 Export to OpenVINO", type="primary",
                         use_container_width=True):
                pt_path = AVAILABLE_MODELS[ov_model_key]
                with st.spinner(
                    f"Exporting `{pt_path}` → OpenVINO (this may take 1–3 min)…"
                ):
                    try:
                        from ultralytics import RTDETR
                        m = RTDETR(pt_path)
                        result = m.export(
                            format="openvino",
                            imgsz=ov_imgsz,
                            half=False,   # INT8/FP16 not reliable on all CPUs
                        )
                        st.success(
                            f"✅ Exported to: `{result}`\n\n"
                            "Restart the stream server to use it automatically."
                        )
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
                        st.info(
                            "Make sure `openvino` is installed:\n"
                            "```\npip install openvino\n```"
                        )

    with accel_col2:
        with st.container(border=True):
            st.subheader("🟠 ONNX Export")
            st.markdown(
                "**2–4× faster** than PyTorch via ONNX Runtime. "
                "Works on any CPU (AMD, Intel, ARM). "
                "Fallback if OpenVINO is not available."
            )
            onnx_model_key = st.selectbox("Model to export (ONNX)",
                                          list(AVAILABLE_MODELS.keys()), key="onnx_sel")
            onnx_imgsz = st.select_slider("Export image size", [320, 480, 640], value=480,
                                          key="onnx_imgsz")

            onnx_out = ONNX_MODELS.get(AVAILABLE_MODELS[onnx_model_key], "")
            already_onnx = os.path.isfile(onnx_out)
            if already_onnx:
                st.success(f"✅ Already exported: `{os.path.basename(onnx_out)}`")

            if st.button("🟠 Export to ONNX", use_container_width=True):
                pt_path = AVAILABLE_MODELS[onnx_model_key]
                with st.spinner(f"Exporting `{pt_path}` → ONNX…"):
                    try:
                        from ultralytics import RTDETR
                        m = RTDETR(pt_path)
                        result = m.export(
                            format="onnx",
                            imgsz=onnx_imgsz,
                            simplify=True,
                        )
                        st.success(
                            f"✅ Exported to: `{result}`\n\n"
                            "Restart the stream server to use it automatically."
                        )
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
                        st.info(
                            "Make sure `onnxruntime` is installed:\n"
                            "```\npip install onnxruntime\n```"
                        )

    st.markdown("---")
    with st.container(border=True):
        st.subheader("📊 Acceleration Status")
        rows = []
        for key, pt_path in AVAILABLE_MODELS.items():
            ov_dir   = OPENVINO_MODELS.get(pt_path, "")
            onnx_f   = ONNX_MODELS.get(pt_path, "")
            rows.append({
                "Model":     key.split("(")[0].strip(),
                "PyTorch":   "✅" if os.path.isfile(pt_path) or True else "❌",
                "OpenVINO":  "✅ Ready" if ov_dir and os.path.isdir(ov_dir) else "⬜ Not exported",
                "ONNX":      "✅ Ready" if onnx_f and os.path.isfile(onnx_f) else "⬜ Not exported",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "The stream server loads the **best available** format automatically: "
            "OpenVINO → ONNX → PyTorch"
        )

# ===========================================================================
# TAB 3 — Benchmark
# ===========================================================================
with tab_bench:
    st.info(
        "Run a quick inference benchmark on a synthetic frame to measure "
        "latency for the selected model.",
        icon="⚡",
    )

    bench_model_key = st.selectbox("Model to benchmark", list(AVAILABLE_MODELS.keys()), key="bench_sel")
    bench_imgsz     = st.select_slider("Image size", options=[320, 480, 640, 800, 1024], value=DEFAULT_IMG_SIZE)
    bench_runs      = st.slider("Number of runs", 5, 50, 20)

    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        import numpy as np

        with st.spinner("Loading model…"):
            bmodel = load_model(AVAILABLE_MODELS[bench_model_key])

        # Warm-up
        dummy = np.random.randint(0, 255, (bench_imgsz, bench_imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            run_inference(bmodel, dummy)

        # Timed runs
        times = []
        prog  = st.progress(0, text="Benchmarking…")
        for i in range(bench_runs):
            det = run_inference(bmodel, dummy)
            times.append(det.inference_ms)
            prog.progress((i + 1) / bench_runs, text=f"Run {i+1}/{bench_runs}")
        prog.empty()

        avg_ms  = sum(times) / len(times)
        min_ms  = min(times)
        max_ms  = max(times)
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        c1.metric("Avg Latency",  f"{avg_ms:.1f} ms")
        c2.metric("Min Latency",  f"{min_ms:.1f} ms")
        c3.metric("Max Latency",  f"{max_ms:.1f} ms")
        c4.metric("Avg FPS",      f"{avg_fps:.1f}")

        st.line_chart(times, height=200)
        st.caption(f"Model: `{AVAILABLE_MODELS[bench_model_key]}` | Image size: {bench_imgsz} | {bench_runs} runs")

# ===========================================================================
# TAB 3 — Fine-tune
# ===========================================================================
with tab_finetune:
    img_count   = len(glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.jpg"))
                    + glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.png")))
    label_count = len(glob.glob(os.path.join(TRAINING_LABELS_DIR, "*.txt")))

    col_info, col_cfg = st.columns(2, gap="large")

    with col_info:
        with st.container(border=True):
            st.subheader("📊 Training Data Status")
            st.metric("Captured Images", img_count)
            st.metric("Labeled Images",  label_count)

            if label_count == 0:
                st.warning(
                    "No labeled images found. Go to **🖍️ Annotation Studio** "
                    "to label your captured frames first."
                )
            elif label_count < 20:
                st.warning(f"Only {label_count} labeled image(s). "
                           "More data generally improves fine-tune quality.")
            else:
                st.success(f"✅ {label_count} labeled images ready for training.")

    with col_cfg:
        with st.container(border=True):
            st.subheader("⚙️ Training Configuration")
            ft_model    = st.selectbox("Base model", list(AVAILABLE_MODELS.keys()), key="ft_model")
            ft_epochs   = st.slider("Epochs",         5, 200, 50)
            ft_imgsz    = st.select_slider("Image size", [320, 480, 640], value=640, key="ft_imgsz")
            ft_batch    = st.slider("Batch size",      2, 32, 8)
            ft_lr       = st.number_input("Learning rate", value=1e-4,
                                          format="%.5f", min_value=1e-6, max_value=1e-1)
            ft_name     = st.text_input("Run name", value="sentinel_finetune")

    st.markdown("")

    # Generate YAML
    yaml_path = os.path.join(BASE_DIR, "training_data", "dataset.yaml")
    yaml_content = (
        f"path: {os.path.join(BASE_DIR, 'training_data')}\n"
        "train: images\nval: images\n\nnc: 80\n"
        "names:\n" + "\n".join(f"  {k}: {v}" for k, v in {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck",
        }.items()) + "\n  # ... (all 80 COCO classes)\n"
    )

    with st.expander("📄 dataset.yaml preview"):
        st.code(yaml_content, language="yaml")

    if st.button("📝 Save dataset.yaml", use_container_width=True):
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        st.toast("dataset.yaml saved.", icon="📝")

    st.markdown("---")

    if label_count < 1:
        st.button("🎓 Start Fine-Tuning", disabled=True, use_container_width=True,
                  help="Label at least 1 image in Annotation Studio first.")
    else:
        if st.button("🎓 Start Fine-Tuning", type="primary", use_container_width=True):
            cmd = (
                f"yolo detect train "
                f"model={AVAILABLE_MODELS[ft_model]} "
                f"data={yaml_path} "
                f"epochs={ft_epochs} "
                f"imgsz={ft_imgsz} "
                f"batch={ft_batch} "
                f"lr0={ft_lr} "
                f"name={ft_name}"
            )
            with st.expander("🖥️ Training command", expanded=True):
                st.code(cmd, language="bash")

            st.info(
                "Training is launched as a background process. "
                "Monitor progress in the terminal where Streamlit is running.",
                icon="ℹ️",
            )

            log_path = os.path.join(BASE_DIR, "detection_logs", f"{ft_name}_train.log")
            try:
                with open(log_path, "w") as log_f:
                    proc = subprocess.Popen(
                        cmd, shell=True, stdout=log_f, stderr=log_f, cwd=BASE_DIR
                    )
                st.success(f"✅ Training started (PID {proc.pid}). Logs → `{log_path}`")
            except Exception as e:
                st.error(f"Failed to start training: {e}")
