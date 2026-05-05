# =============================================================================
# pages/4_annotation_studio.py — Bounding-box labeling & dataset management
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import json
import io
import zipfile
from collections import Counter
from datetime import datetime

import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from config import (
    GLOBAL_CSS, COCO_CLASSES, TRAINING_IMAGES_DIR, TRAINING_LABELS_DIR,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Annotation Studio | RT-DETR Sentinel Pro",
                   page_icon="🖍️", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown('<p class="gradient-text">🖍️ Annotation Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Label edge-case frames captured during live detection</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_labels(img_path: str) -> list[dict]:
    """Load YOLO-format labels for an image if they exist."""
    base  = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(TRAINING_LABELS_DIR, base + ".txt")
    labels = []
    if os.path.isfile(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                    labels.append({"class_id": cls_id, "xc": xc, "yc": yc,
                                   "bw": bw, "bh": bh})
    return labels


def yolo_to_canvas_rect(label: dict, img_w: int, img_h: int) -> dict:
    """Convert YOLO normalised box → canvas object dict."""
    bx = label["xc"] * img_w
    by = label["yc"] * img_h
    bw = label["bw"] * img_w
    bh = label["bh"] * img_h
    return {
        "type": "rect",
        "left":   bx - bw / 2,
        "top":    by - bh / 2,
        "width":  bw,
        "height": bh,
        "scaleX": 1.0,
        "scaleY": 1.0,
    }


def dataset_stats() -> dict:
    """Count labeled images and per-class annotation stats."""
    label_files = glob.glob(os.path.join(TRAINING_LABELS_DIR, "*.txt"))
    class_counter: Counter = Counter()
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counter[int(parts[0])] += 1
    return {"labeled_images": len(label_files), "class_counts": dict(class_counter)}


# ---------------------------------------------------------------------------
# Left column — image selector + stats
# ---------------------------------------------------------------------------
saved_images = sorted(glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.jpg"))
                    + glob.glob(os.path.join(TRAINING_IMAGES_DIR, "*.png")))

if not saved_images:
    st.info(
        "No captured frames yet. Use the **🔴 Live Detection** page to capture edge-case frames first.",
        icon="📭",
    )
    st.stop()

left_col, canvas_col, right_col = st.columns([1.2, 3, 1.2], gap="medium")

with left_col:
    # ---- Image selector ----
    with st.container(border=True):
        st.subheader("🗂️ Images")
        img_names     = [os.path.basename(p) for p in saved_images]
        selected_name = st.selectbox("Select frame", img_names)
        selected_path = os.path.join(TRAINING_IMAGES_DIR, selected_name)

        st.caption(f"{len(saved_images)} image(s) in dataset")

        # Delete image button
        if st.button("🗑️ Delete This Image", use_container_width=True):
            os.remove(selected_path)
            base = os.path.splitext(selected_name)[0]
            lf   = os.path.join(TRAINING_LABELS_DIR, base + ".txt")
            if os.path.isfile(lf):
                os.remove(lf)
            st.toast(f"Deleted {selected_name}", icon="🗑️")
            st.rerun()

    # ---- Class selector ----
    with st.container(border=True):
        st.subheader("🏷️ Label")
        class_options = [f"{v}  (id:{k})" for k, v in COCO_CLASSES.items()]
        selected_class_str = st.selectbox("Class to annotate", class_options)
        label_class_id = int(selected_class_str.split("id:")[1].rstrip(")"))
        label_class_name = COCO_CLASSES[label_class_id]

        st.markdown("---")
        st.caption("**Drawing mode**: drag to draw a box. Canvas shows existing labels.")

        col_save, col_clear = st.columns(2)
        save_trigger  = col_save.button("💾 Save",  use_container_width=True, type="primary")
        clear_trigger = col_clear.button("🗑️ Clear", use_container_width=True)

    # ---- Existing labels for this image ----
    existing = load_existing_labels(selected_path)
    with st.container(border=True):
        st.subheader("📋 Current Labels")
        if existing:
            for lb in existing:
                cls_name = COCO_CLASSES.get(lb["class_id"], str(lb["class_id"]))
                st.markdown(f"- **{cls_name}** (id:{lb['class_id']})")
        else:
            st.caption("No labels saved yet.")

# ---------------------------------------------------------------------------
# Centre — canvas
# ---------------------------------------------------------------------------
with canvas_col:
    img = Image.open(selected_path)
    img_w, img_h = img.size

    # Scale canvas to max 900px wide for usability
    max_canvas_w = 900
    scale = min(max_canvas_w / img_w, 1.0)
    canvas_w = int(img_w * scale)
    canvas_h = int(img_h * scale)

    # Build initial objects from existing labels
    initial_drawing = {
        "version": "4.4.0",
        "objects": [yolo_to_canvas_rect(lb, canvas_w, canvas_h) for lb in existing],
    }

    canvas_result = st_canvas(
        fill_color="rgba(0, 201, 255, 0.15)",
        stroke_width=2,
        stroke_color="#00C9FF",
        background_image=img,
        initial_drawing=initial_drawing if existing else {"version": "4.4.0", "objects": []},
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key=f"canvas_{selected_name}",
    )

    # ---- Save action ----
    if save_trigger:
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if not objects:
                st.warning("⚠️ No boxes drawn.")
            else:
                base        = os.path.splitext(selected_name)[0]
                label_file  = os.path.join(TRAINING_LABELS_DIR, base + ".txt")
                written = 0
                with open(label_file, "w") as f:
                    for obj in objects:
                        if obj.get("type") != "rect":
                            continue
                        bw = obj["width"]  * obj.get("scaleX", 1.0)
                        bh = obj["height"] * obj.get("scaleY", 1.0)
                        xc = (obj["left"] + bw / 2) / canvas_w
                        yc = (obj["top"]  + bh / 2) / canvas_h
                        nw = bw / canvas_w
                        nh = bh / canvas_h
                        f.write(f"{label_class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                        written += 1
                st.toast(f"💾 Saved {written} label(s) for `{selected_name}`", icon="💾")
                st.rerun()

    if clear_trigger:
        base       = os.path.splitext(selected_name)[0]
        label_file = os.path.join(TRAINING_LABELS_DIR, base + ".txt")
        if os.path.isfile(label_file):
            os.remove(label_file)
        st.toast("Labels cleared.", icon="🗑️")
        st.rerun()

# ---------------------------------------------------------------------------
# Right — dataset stats + export
# ---------------------------------------------------------------------------
with right_col:
    stats = dataset_stats()

    with st.container(border=True):
        st.subheader("📊 Dataset Stats")
        st.metric("Total Images",  len(saved_images))
        st.metric("Labeled Images", stats["labeled_images"])
        unlabeled = len(saved_images) - stats["labeled_images"]
        st.metric("Unlabeled",      unlabeled)

    with st.container(border=True):
        st.subheader("🏷️ Per-Class Counts")
        if stats["class_counts"]:
            for cls_id, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
                cls_name = COCO_CLASSES.get(cls_id, str(cls_id))
                st.markdown(f"**{cls_name}**: {cnt}")
        else:
            st.caption("No annotations yet.")

    with st.container(border=True):
        st.subheader("📦 Export Dataset")
        st.caption("Pack all images + YOLO labels into a ZIP for training.")
        if st.button("⬇️ Export Dataset ZIP", use_container_width=True, type="primary"):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_path in saved_images:
                    zf.write(img_path, os.path.join("images", os.path.basename(img_path)))
                label_files = glob.glob(os.path.join(TRAINING_LABELS_DIR, "*.txt"))
                for lf in label_files:
                    zf.write(lf, os.path.join("labels", os.path.basename(lf)))

                # Write a minimal dataset.yaml
                yaml_content = (
                    "path: .\ntrain: images\nval: images\n\nnc: 80\n"
                    "names:\n" + "\n".join(f"  {k}: {v}" for k, v in COCO_CLASSES.items())
                )
                zf.writestr("dataset.yaml", yaml_content)

            st.download_button(
                "⬇️ Download ZIP",
                zip_buf.getvalue(),
                "training_dataset.zip",
                "application/zip",
                use_container_width=True,
            )
