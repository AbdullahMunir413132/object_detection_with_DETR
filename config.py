# =============================================================================
# config.py — Central configuration for RT-DETR Sentinel Pro
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGES_DIR = os.path.join(BASE_DIR, "training_data", "images")
TRAINING_LABELS_DIR = os.path.join(BASE_DIR, "training_data", "labels")
DETECTION_LOGS_DIR  = os.path.join(BASE_DIR, "detection_logs")
DB_PATH             = os.path.join(DETECTION_LOGS_DIR, "detections.db")
EXPORTS_DIR         = os.path.join(BASE_DIR, "exports")

for _d in [TRAINING_IMAGES_DIR, TRAINING_LABELS_DIR, DETECTION_LOGS_DIR, EXPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Stream server
# ---------------------------------------------------------------------------
STREAM_SERVER_PORT = 8502
STREAM_SERVER_URL  = f"http://localhost:{STREAM_SERVER_PORT}"

# ---------------------------------------------------------------------------
# Available Models
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "RT-DETR Large  (rtdetr-l.pt)":       os.path.join(BASE_DIR, "rtdetr-l.pt"),
    "RT-DETR Extra-Large (rtdetr-x.pt)":  os.path.join(BASE_DIR, "rtdetr-x.pt"),
}

# OpenVINO-exported model directories (populated after export in Model Settings)
OPENVINO_MODELS = {
    os.path.join(BASE_DIR, "rtdetr-s.pt"): os.path.join(BASE_DIR, "rtdetr-s_openvino_model"),
    os.path.join(BASE_DIR, "rtdetr-l.pt"): os.path.join(BASE_DIR, "rtdetr-l_openvino_model"),
    os.path.join(BASE_DIR, "rtdetr-x.pt"): os.path.join(BASE_DIR, "rtdetr-x_openvino_model"),
    "rtdetr-s.pt": os.path.join(BASE_DIR, "rtdetr-s_openvino_model"),
    "rtdetr-l.pt": os.path.join(BASE_DIR, "rtdetr-l_openvino_model"),
    "rtdetr-x.pt": os.path.join(BASE_DIR, "rtdetr-x_openvino_model"),
}

ONNX_MODELS = {
    os.path.join(BASE_DIR, "rtdetr-s.pt"): os.path.join(BASE_DIR, "rtdetr-s.onnx"),
    os.path.join(BASE_DIR, "rtdetr-l.pt"): os.path.join(BASE_DIR, "rtdetr-l.onnx"),
    os.path.join(BASE_DIR, "rtdetr-x.pt"): os.path.join(BASE_DIR, "rtdetr-x.onnx"),
    "rtdetr-s.pt": os.path.join(BASE_DIR, "rtdetr-s.onnx"),
    "rtdetr-l.pt": os.path.join(BASE_DIR, "rtdetr-l.onnx"),
    "rtdetr-x.pt": os.path.join(BASE_DIR, "rtdetr-x.onnx"),
}

DEFAULT_MODEL_KEY = "RT-DETR Large  (rtdetr-l.pt)"

# ---------------------------------------------------------------------------
# Inference Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONF       = 0.50
DEFAULT_IOU        = 0.45
MAX_DETECTIONS     = 300
DEFAULT_IMG_SIZE   = 640

# ---------------------------------------------------------------------------
# COCO 80-class names (index → name)
# ---------------------------------------------------------------------------
COCO_CLASSES = {
    0: "person",        1: "bicycle",       2: "car",           3: "motorcycle",
    4: "airplane",      5: "bus",           6: "train",         7: "truck",
    8: "boat",          9: "traffic light", 10: "fire hydrant", 11: "stop sign",
    12: "parking meter",13: "bench",        14: "bird",         15: "cat",
    16: "dog",          17: "horse",        18: "sheep",        19: "cow",
    20: "elephant",     21: "bear",         22: "zebra",        23: "giraffe",
    24: "backpack",     25: "umbrella",     26: "handbag",      27: "tie",
    28: "suitcase",     29: "frisbee",      30: "skis",         31: "snowboard",
    32: "sports ball",  33: "kite",         34: "baseball bat", 35: "baseball glove",
    36: "skateboard",   37: "surfboard",    38: "tennis racket",39: "bottle",
    40: "wine glass",   41: "cup",          42: "fork",         43: "knife",
    44: "spoon",        45: "bowl",         46: "banana",       47: "apple",
    48: "sandwich",     49: "orange",       50: "broccoli",     51: "carrot",
    52: "hot dog",      53: "pizza",        54: "donut",        55: "cake",
    56: "chair",        57: "couch",        58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet",       62: "tv",           63: "laptop",
    64: "mouse",        65: "remote",       66: "keyboard",     67: "cell phone",
    68: "microwave",    69: "oven",         70: "toaster",      71: "sink",
    72: "refrigerator", 73: "book",         74: "clock",        75: "vase",
    76: "scissors",     77: "teddy bear",   78: "hair drier",   79: "toothbrush",
}

# Reverse mapping: name → index
COCO_CLASS_IDS = {v: k for k, v in COCO_CLASSES.items()}

# ---------------------------------------------------------------------------
# UI / Styling
# ---------------------------------------------------------------------------
APP_TITLE       = "RT-DETR Sentinel Pro"
APP_ICON        = "👁️"
PRIMARY_COLOR   = "#00C9FF"
SECONDARY_COLOR = "#92FE9D"
BG_COLOR        = "#0B0F19"
CARD_BG         = "#161B28"

GLOBAL_CSS = f"""
<style>
    /* ---- Base ---- */
    .stApp {{ background-color: {BG_COLOR}; color: #E0E0E0; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {{ background-color: #10151F; border-right: 1px solid #1E2535; }}

    /* ---- Gradient heading ---- */
    .gradient-text {{
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6em;
        font-weight: 800;
        line-height: 1.1;
    }}
    .sub-heading {{
        color: #8899AA;
        font-size: 1em;
        margin-top: -6px;
        margin-bottom: 12px;
    }}

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #1E2535;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricValue"] {{ color: {PRIMARY_COLOR}; font-weight: 700; }}

    /* ---- Buttons ---- */
    div.stButton > button {{
        border-radius: 8px;
        border: 1px solid #2A3244;
        background-color: {CARD_BG};
        color: #C8D6E5;
        transition: all 0.25s ease;
    }}
    div.stButton > button:hover {{
        border-color: {PRIMARY_COLOR};
        color: {PRIMARY_COLOR};
        box-shadow: 0 0 14px rgba(0,201,255,0.25);
    }}
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #0077B6, #0096C7);
        border: none;
        color: white;
        font-weight: 600;
    }}

    /* ---- Expanders / containers ---- */
    [data-testid="stExpander"] {{
        border: 1px solid #1E2535 !important;
        border-radius: 10px !important;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 6px 6px 0 0;
        border: 1px solid #1E2535;
        color: #8899AA;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #0077B6, #0096C7) !important;
        color: white !important;
    }}

    /* ---- Divider ---- */
    hr {{ border-color: #1E2535; }}

    /* ---- Selectbox / slider ---- */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {{
        background-color: {CARD_BG};
        border: 1px solid #2A3244;
        color: #C8D6E5;
    }}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: #2A3244; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY_COLOR}; }}

    /* ---- Status badges ---- */
    .badge-online {{ color: #92FE9D; font-weight: 700; }}
    .badge-offline {{ color: #FF6B6B; font-weight: 700; }}
</style>
"""
