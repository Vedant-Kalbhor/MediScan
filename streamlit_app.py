import os
import re

import requests
import streamlit as st
from PIL import Image

from config import MODELS_CONFIG


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


st.set_page_config(
    page_title="MediScan | AI Health Guide",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("MediScan AI")
    st.markdown("---")
    st.info("Tip: Use a clear, centered scan for the best results.")

    st.subheader("Model Selection")
    scan_options = {v["name"]: k for k, v in MODELS_CONFIG.items()}
    selected_name = st.selectbox("Choose Scan Type:", list(scan_options.keys()))
    model_type = scan_options[selected_name]

    st.markdown("---")
    st.warning(
        "Disclaimer: This tool is for educational and initial guidance only. "
        "It is not a replacement for professional medical advice, diagnosis, or treatment."
    )

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.title("Medical Scan Guide")
    st.write(MODELS_CONFIG[model_type]["description"])

    uploaded_file = st.file_uploader(
        f"Upload {MODELS_CONFIG[model_type]['input_type']}",
        type=["jpg", "jpeg", "png"],
        help=f"Accepts JPG, JPEG, and PNG images of {MODELS_CONFIG[model_type]['input_type']}",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Scan", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    st.write("Results will appear here after analysis.")

    if uploaded_file and st.button("Analyze Scan"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"model_type": model_type}
        backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

        with st.spinner("AI is analyzing the scan..."):
            try:
                response = requests.post(f"{backend_url}/predict", files=files, params=params, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    predicted_class = result["predicted_class"]
                    pretty_prediction = pretty_label(predicted_class)
                    confidence_pct = result["confidence"] * 100

                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h2 style='color: #4caf50; margin-bottom: 5px;'>Prediction: {pretty_prediction}</h2>
                            <p style='font-size: 1.2em;'>Confidence: <b>{confidence_pct:.2f}%</b></p>
                            <hr style='border: 0.1px solid rgba(255,255,255,0.1);'>
                            <p style='color: #888;'>Model: {result['scan_name']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if result["confidence"] < 0.7:
                        st.warning(
                            "Low confidence detected. Please ensure the scan is correct or consult a specialist."
                        )
                    elif is_normal_prediction(model_type, predicted_class):
                        st.success(
                            "The scan appears to be within the expected normal range. "
                            "Always verify with a qualified professional."
                        )
                    else:
                        st.error(
                            "Potential abnormality detected. Please schedule a follow-up with a clinician."
                        )

                elif response.status_code == 503:
                    st.error("Model weights are not available yet on the backend.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception:
                st.error(
                    "Connection error: could not reach the backend API. Make sure `main.py` is running."
                )

st.markdown("---")
st.caption("MediScan v2.0 - AI-assisted scan triage for educational use.")
