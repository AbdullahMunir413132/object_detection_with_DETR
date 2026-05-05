import streamlit as st
import cv2
import torch
import os
import glob
from datetime import datetime
from ultralytics import RTDETR
from collections import Counter
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Setup Directories
os.makedirs(os.path.join("training_data", "images"), exist_ok=True)
os.makedirs(os.path.join("training_data", "labels"), exist_ok=True)

# 2. Page Configuration
st.set_page_config(page_title="RT-DETR Sentinel", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Aesthetics
st.markdown("""
<style>
    /* Dark, sleek background */
    .stApp { background-color: #0B0F19; }
    
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Sleek Buttons with Hover Glow */
    div.stButton > button:first-child {
        border-radius: 8px;
        border: 1px solid #333;
        background-color: #161B28;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #00C9FF;
        color: #00C9FF;
        box-shadow: 0px 0px 15px rgba(0, 201, 255, 0.3);
    }
    
    /* Hide top padding */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="gradient-text">👁️ RT-DETR: Vision Sentinel</p>', unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model
@st.cache_resource
def load_model():
    return RTDETR('rtdetr-l.pt')

model = load_model()

if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None

# 4. Create the Tabs
tab1, tab2 = st.tabs(["🔴 Live Sentinel", "🖍️ Labeling Studio"])

# ==========================================
# TAB 1: LIVE VIDEO STREAM
# ==========================================
with tab1:
    col1, col2, col3 = st.columns([1, 2.5, 1], gap="medium")

    with col1:
        with st.container(border=True):
            st.subheader("⚙️ Control Panel")
            conf_threshold = st.slider("Global Confidence Threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
            person_only = st.toggle("Filter: Person Only", value=False)
            run_stream = st.checkbox("🟢 Start Video Stream", value=False)
        
        with st.container(border=True):
            st.subheader("🛠️ Data Pipeline")
            st.caption("Click below to capture an edge case for retraining.")
            
            if st.button("🚨 Save Misclassification", use_container_width=True):
                if st.session_state.current_frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join("training_data", "images", f"edge_{timestamp}.jpg")
                    
                    success = cv2.imwrite(filename, st.session_state.current_frame)
                    if success:
                        # Using TOAST for sleek popup notifications!
                        st.toast("✅ Image Saved! Uncheck 'Start Video Stream' to label it.", icon="📸")
                    else:
                        st.error("❌ Error: Could not save the image.")
                else:
                    st.toast("⚠️ No video feed active to save.", icon="⚠️")

    with col2:
        with st.container(border=True):
            video_placeholder = st.empty()

    with col3:
        with st.container(border=True):
            st.subheader("📊 Live Analytics")
            metrics_placeholder = st.empty()

    if run_stream:
        cap = cv2.VideoCapture(0)
        while run_stream:
            ret, frame = cap.read()
            if not ret: break
                
            st.session_state.current_frame = frame.copy()
            target_classes = [0] if person_only else None
            
            results = model.predict(source=frame, conf=conf_threshold, classes=target_classes, verbose=False)
            
            annotated_frame = results[0].plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            detected_classes = results[0].boxes.cls.cpu().numpy()
            detected_names = [model.names[int(c)] for c in detected_classes]
            counts = Counter(detected_names)
            
            with metrics_placeholder.container():
                if len(counts) == 0:
                    st.info("Nothing detected.", icon="💤")
                for obj, count in counts.items():
                    st.metric(label=obj.capitalize(), value=count)
        cap.release()
    else:
        video_placeholder.info("Click 'Start Video Stream' in the Control Panel to begin.", icon="📺")

# ==========================================
# TAB 2: LABELING STUDIO
# ==========================================
with tab2:
    with st.container(border=True):
        st.subheader("🖍️ Data Annotation Tool")
        st.caption("Draw bounding boxes around the objects the AI missed. **Stop the video stream in Tab 1 first!**")
        
        saved_images = glob.glob(os.path.join("training_data", "images", "*.jpg"))
        
        if len(saved_images) == 0:
            st.info("No images saved yet. Go to the Live Sentinel tab and capture some edge cases!", icon="📭")
        else:
            col_a, col_b = st.columns([1, 3])
            
            with col_a:
                selected_img_path = st.selectbox("Select Image", saved_images)
                label_class = st.number_input("Class ID (e.g., 0 for Person)", min_value=0, max_value=79, value=0, step=1)
                
                if st.button("💾 Save Labels", use_container_width=True):
                    # Save Logic is triggered below based on session state
                    st.session_state.save_trigger = True
                else:
                    st.session_state.save_trigger = False

            with col_b:
                img = Image.open(selected_img_path)
                img_width, img_height = img.size
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 201, 255, 0.2)",  # Cyber blue fill
                    stroke_width=2,
                    stroke_color="#00C9FF",
                    background_image=img,
                    update_streamlit=True,
                    height=img_height,
                    width=img_width,
                    drawing_mode="rect",
                    key="canvas",
                )

                if st.session_state.save_trigger:
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if len(objects) > 0:
                            base_name = os.path.basename(selected_img_path)
                            label_filename = os.path.join("training_data", "labels", base_name.replace(".jpg", ".txt"))
                            
                            with open(label_filename, "w") as f:
                                for obj in objects:
                                    if obj["type"] == "rect":
                                        box_w = obj["width"] * obj["scaleX"]
                                        box_h = obj["height"] * obj["scaleY"]
                                        x_center = (obj["left"] + (box_w / 2)) / img_width
                                        y_center = (obj["top"] + (box_h / 2)) / img_height
                                        f.write(f"{label_class} {x_center:.6f} {y_center:.6f} {box_w/img_width:.6f} {box_h/img_height:.6f}\n")
                            
                            st.toast(f"Labels saved successfully!", icon="💾")
                        else:
                            st.warning("No bounding boxes drawn.")