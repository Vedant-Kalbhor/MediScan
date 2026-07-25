import io
import os
import re
import requests
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Import models configuration
from config import MODELS_CONFIG

# --- Page Configuration ---
st.set_page_config(
    page_title="MediScan AI | Multi-Organ Medical Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling & Animations ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Header Gradient */
    .header-container {
        padding: 2.5rem 0rem 1.5rem 0rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(6, 182, 212, 0.05) 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(16, 185, 129, 0.1);
        text-align: center;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #10B981 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #9CA3AF;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* Use Case Cards Grid */
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2.5rem;
    }
    
    .use-case-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
    }
    
    .use-case-card:hover {
        transform: translateY(-4px);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 12px 20px -10px rgba(16, 185, 129, 0.3);
    }
    
    .use-case-card.active {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10B981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    
    .icon-badge {
        width: 48px;
        height: 48px;
        line-height: 48px;
        border-radius: 50%;
        background: rgba(16, 185, 129, 0.1);
        font-size: 1.5rem;
        display: inline-block;
        margin-bottom: 0.75rem;
        color: #10B981;
    }
    
    .active .icon-badge {
        background: #10B981;
        color: #FFFFFF;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #F3F4F6;
    }
    
    .card-meta {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    
    /* Analyzer Control Panel */
    .analyzer-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.5rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Result Box Styling */
    .prediction-box {
        padding: 1.75rem;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Scan Line Animation Overlay */
    .scan-container {
        position: relative;
        display: inline-block;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .scan-line {
        position: absolute;
        height: 4px;
        width: 100%;
        background: linear-gradient(to right, transparent, #10B981, transparent);
        animation: scan 3s infinite linear;
        z-index: 2;
        box-shadow: 0 0 12px #10B981;
    }
    
    @keyframes scan {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
BACKEND_REQUEST_TIMEOUT = (10, 240)

def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())

def pretty_label(value):
    cleaned = value.replace("_", " ").replace(".", " ").strip()
    return cleaned.title()

def is_normal_prediction(model_type, predicted_class):
    model_config = MODELS_CONFIG[model_type]
    predicted = normalize_label(predicted_class)
    normal_labels = {normalize_label(label) for label in model_config.get("normal_classes", [])}
    return predicted in normal_labels


def resolve_backend_url():
    backend_url = os.getenv("BACKEND_URL")
    if backend_url:
        return backend_url.rstrip("/")

    backend_hostport = os.getenv("BACKEND_HOSTPORT")
    if backend_hostport:
        return f"http://{backend_hostport}"

    return "http://127.0.0.1:8000"


def fetch_predictions(backend_url, limit=100):
    response = requests.get(
        f"{backend_url}/predictions",
        params={"limit": limit},
        timeout=BACKEND_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def draw_bone_detections(image, detections):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for detection in detections:
        box = detection.get("box", {})
        x1 = float(box.get("x1", 0))
        y1 = float(box.get("y1", 0))
        x2 = float(box.get("x2", 0))
        y2 = float(box.get("y2", 0))
        label = detection.get("class_name", "fracture")
        confidence = detection.get("confidence", 0.0)
        region = detection.get("image_region", "unknown")
        caption = f"{label} {confidence:.2f} | {region}"

        draw.rectangle([x1, y1, x2, y2], outline="#EF4444", width=4)

        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(0, x1)
        text_y = max(0, y1 - text_h - 8)
        draw.rectangle([text_x, text_y, text_x + text_w + 10, text_y + text_h + 6], fill="#111827")
        draw.text((text_x + 5, text_y + 3), caption, fill="#FFFFFF", font=font)

    return annotated


def render_dashboard(backend_url):
    st.markdown("## 📊 Prediction Analytics Dashboard")
    st.caption("View stored prediction history coming from PostgreSQL.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        limit = st.selectbox("Rows to load", options=[25, 50, 100, 250, 500], index=2)
    with col_b:
        refresh = st.button("🔄 Refresh Data", width='stretch')

    if refresh:
        st.rerun()

    try:
        rows = fetch_predictions(backend_url, limit=limit)
    except Exception as e:
        st.error(f"Could not load prediction history: {e}")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No prediction records found yet. Run a few scans first and they will appear here.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)

    metrics = st.columns(4)
    metrics[0].metric("Total Predictions", len(df))
    metrics[1].metric("Average Confidence", f"{df['confidence'].mean() * 100:.1f}%")
    metrics[2].metric("Unique Organs", df["organ"].nunique())
    metrics[3].metric("Latest Record", df.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M"))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("### Predictions by Organ")
        organ_counts = df["organ"].value_counts().sort_values(ascending=False)
        st.bar_chart(organ_counts)
    with chart_right:
        st.markdown("### Confidence Over Time")
        confidence_series = df.sort_values("timestamp").set_index("timestamp")["confidence"]
        st.line_chart(confidence_series)

    st.markdown("### Recent Prediction Records")
    st.dataframe(
        df[["id", "timestamp", "organ", "prediction", "confidence"]],
        width='stretch',
        hide_index=True,
    )

    try:
        export_response = requests.get(
            f"{backend_url}/predictions/export",
            params={"limit": limit},
            timeout=BACKEND_REQUEST_TIMEOUT,
        )
        export_response.raise_for_status()
        st.download_button(
            "Download CSV",
            data=export_response.content,
            file_name="mediscan_predictions.csv",
            mime="text/csv",
            width='stretch',
        )
    except Exception as e:
        st.warning(f"CSV export is temporarily unavailable: {e}")

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <div class="title-gradient">🧬 MediScan AI</div>
    <div class="subtitle">Multi-Organ Medical Diagnostic Assistant</div>
</div>
""", unsafe_allow_html=True)

# --- Use Cases Visual Grid (Informational Overview) ---
st.markdown("### 🔍 Diagnostics Categories")

cols = st.columns(5)
use_cases = [
    {"key": "brain", "emoji": "🧠", "name": "Brain MRI", "arch": "DenseNet121", "desc": "Glioma, Meningioma, Pituitary, Normal"},
    {"key": "chest", "emoji": "🫁", "name": "Chest CT", "arch": "EfficientNetV2", "desc": "Adenocarcinoma, Large Cell, Squamous, Normal"},
    {"key": "breast", "emoji": "🎀", "name": "Breast Ultrasound", "arch": "ResNet18", "desc": "Benign, Malignant, Normal"},
    {"key": "kidney", "emoji": "🧼", "name": "Kidney CT", "arch": "ResNet18", "desc": "Cyst, Stone, Tumor, Normal"},
    {"key": "bone", "emoji": "🦴", "name": "Bone X-ray", "arch": "YOLOv8", "desc": "Fracture Detection + Localization"}
]

# Set active usecase via Session State
if "selected_use_case" not in st.session_state:
    st.session_state.selected_use_case = "brain"

for i, uc in enumerate(use_cases):
    with cols[i]:
        is_active = st.session_state.selected_use_case == uc["key"]
        active_class = "active" if is_active else ""
        st.markdown(f"""
        <div class="use-case-card {active_class}">
            <div class="icon-badge">{uc['emoji']}</div>
            <div class="card-title">{uc['name']}</div>
            <div class="card-meta"><b>Network:</b> {uc['arch']}</div>
            <div class="card-meta" style="font-size: 0.75rem; margin-top: 5px; color: #9CA3AF;">{uc['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Selector & Control Panel ---
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Diagnostic Hub")
    app_mode = st.selectbox("Workspace:", ["Analyzer", "Admin Dashboard"])
    backend_url = resolve_backend_url()

    if app_mode == "Analyzer":
        selected_name = st.selectbox(
            "Select Active Scanner:",
            options=[uc["name"] for uc in use_cases],
            index=[uc["key"] for uc in use_cases].index(st.session_state.selected_use_case)
        )

        # Sync selected scanner back to session state
        for uc in use_cases:
            if uc["name"] == selected_name:
                st.session_state.selected_use_case = uc["key"]

        model_type = st.session_state.selected_use_case
        config = MODELS_CONFIG[model_type]

        st.markdown("---")
        st.markdown("#### 📁 Scan Specifications")
        st.info(f"**Input Type:** {config['input_type']}\n\n**Description:** {config['description']}")

        st.markdown("---")
        st.warning(
            "⚠️ **Clinical Disclaimer:** This tool is for training demonstration and initial benchmarking purposes only. "
            "All predictions must be validated by a board-certified professional radiologist before making clinical decisions."
        )
    else:
        st.markdown("#### Database Notes")
        st.info(
            "This view reads the stored prediction history from PostgreSQL and shows analytics "
            "for the latest inference runs."
        )

# --- Workspace Layout ---
if app_mode == "Admin Dashboard":
    render_dashboard(backend_url)
    st.stop()

col1, col2 = st.columns([6, 5], gap="large")

with col1:
    st.markdown(f"### 📤 Upload {config['name']} Scan")
    uploaded_file = st.file_uploader(
        f"Drop {config['input_type']} image here...",
        type=["jpg", "jpeg", "png"],
        help=f"Supported file formats: JPG, JPEG, PNG. Recommended size: 224x224."
    )
    
    if uploaded_file:
        st.markdown("#### 🖼️ Loaded Scan Preview")
        # Visual wrap for scan simulation
        st.markdown('<div class="scan-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Diagnostic Intelligence Report")
    
    if not uploaded_file:
        st.info("Please upload a medical image scan to start the automated evaluation.")
    else:
        st.write("Scan loaded successfully. Press the button below to initiate neural network classification.")
        
        analyze_btn = st.button("🧬 Analyze Scan & Run Inference")
        
        if analyze_btn:
            # Show animated scan overlay using HTML trick
            st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
            
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {"model_type": model_type}
            backend_url = resolve_backend_url()
            
            with st.spinner("Decoding scan and propagating through neural layers..."):
                try:
                    response = requests.post(
                        f"{backend_url}/predict",
                        files=files,
                        params=params,
                        timeout=BACKEND_REQUEST_TIMEOUT,
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_class = result["predicted_class"]
                        pretty_prediction = pretty_label(predicted_class)
                        confidence_pct = result["confidence"] * 100
                        details = result.get("details") or {}
                        
                        # Set prediction color based on abnormality
                        is_normal = is_normal_prediction(model_type, predicted_class)
                        badge_color = "#10B981" if is_normal else "#EF4444"
                        status_text = "NO ABNORMALITY DETECTED" if is_normal else "POTENTIAL PATHOLOGY DETECTED"
                        
                        st.markdown(f"""
<div class="prediction-box">
    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #9CA3AF; margin-bottom: 5px;">Scan Evaluation Result</div>
    <h2 style='color: {badge_color}; margin: 0 0 10px 0; font-size: 1.8rem;'>{pretty_prediction}</h2>
    <div style="display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: bold; background: rgba({(16 if is_normal else 239)}, {(185 if is_normal else 68)}, {(129 if is_normal else 68)}, 0.15); color: {badge_color}; margin-bottom: 15px;">
        {status_text}
    </div>
    
</div>
""", unsafe_allow_html=True)
                        
                        if model_type == "bone" and details:
                            detections = details.get("detections", [])
                            if detections:
                                best_detection = details.get("best_detection") or max(
                                    detections, key=lambda item: item.get("confidence", 0.0)
                                )
                                region = best_detection.get("image_region", "unknown")
                                box = best_detection.get("box", {})
                                st.markdown("#### Fracture Location")
                                st.success(
                                    f"Detected fracture in the **{region}** of the X-ray "
                                    f"at confidence **{best_detection.get('confidence', 0.0) * 100:.1f}%**."
                                )
                                st.caption(
                                    "Bounding box: "
                                    f"x1={box.get('x1', 0):.1f}, y1={box.get('y1', 0):.1f}, "
                                    f"x2={box.get('x2', 0):.1f}, y2={box.get('y2', 0):.1f}"
                                )

                                try:
                                    image_for_overlay = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                                    annotated = draw_bone_detections(image_for_overlay, detections)
                                    st.image(annotated, caption="Bone fracture localization", width="stretch")
                                except Exception as overlay_error:
                                    st.warning(f"Could not render fracture overlay: {overlay_error}")
                            else:
                                st.success("No fracture bounding boxes were returned by the YOLO model.")
                        
                        # Streamlit native progress bar for confidence
                        st.progress(result["confidence"])
                        
                        # Diagnostic action guides
                        if result["confidence"] < 0.70:
                            st.warning(
                                "⚠️ **Low Confidence Output:** The network confidence score is lower than the clinical safety threshold (70%). "
                                "Please check if the scan slice alignment is centered and clear of artifacts."
                            )
                        elif is_normal:
                            st.success(
                                "✅ **Triage Status:** The scan exhibits no visible pathological markers within the model's target range. "
                                "Always confirm negative results using clinical correlation."
                            )
                        else:
                            st.error(
                                "🚨 **Triage Status:** High-confidence pathological markers detected. "
                                "We recommend routing this case immediately to clinical review."
                            )
                            
                    elif response.status_code == 503:
                        st.error("❌ **Service Unavailable:** Model weights are not loaded on the backend server. Please verify model folders contain weights files.")
                    else:
                        st.error(f"❌ **API Error ({response.status_code}):** {response.text}")
                except Exception as e:
                    st.error(
                        f"❌ **Connection Error:** Could not communicate with the backend API. "
                        f"Ensure `main.py` is running on `127.0.0.1:8000` (Error: {str(e)})"
                    )

# --- Footer ---
st.markdown("<br><hr><center style='color: #6B7280; font-size: 0.8rem;'>MediScan v2.0 • Powered by PyTorch & YOLOv8 • Portfolio Demonstration</center>", unsafe_allow_html=True)
