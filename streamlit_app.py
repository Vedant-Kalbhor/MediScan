import streamlit as st
import requests
from PIL import Image
from config import MODELS_CONFIG
import io

# Premium Page Config
st.set_page_config(
    page_title="MediScan | AI Health Guide",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
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
""", unsafe_allow_html=True)

# Sidebar - Settings & About
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864350.png", width=100)
    st.title("MediScan AI")
    st.markdown("---")
    st.info("💡 **Tip:** Ensure the image is clear and focused on the scan area for better accuracy.")
    
    st.subheader("Model Selection")
    scan_options = {v["name"]: k for k, v in MODELS_CONFIG.items()}
    selected_name = st.selectbox("Choose Scan Type:", list(scan_options.keys()))
    model_type = scan_options[selected_name]
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational/initial guide purposes ONLY. It is NOT a replacement for professional medical advice, diagnosis, or treatment. Always consult a qualified physician.")

# Main UI
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.title("🩺 Medical Scan Guide")
    st.write(MODELS_CONFIG[model_type]["description"])
    
    uploaded_file = st.file_uploader(
        f"Upload {MODELS_CONFIG[model_type]['input_type']}",
        type=["jpg", "jpeg", "png"],
        help=f"Accepts JPG, JPEG, PNG formats of {MODELS_CONFIG[model_type]['input_type']}"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Scan", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    st.write("Results will appear here after analysis.")
    
    if uploaded_file:
        if st.button("🚀 Analyze Scan"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {"model_type": model_type}
            with st.spinner("🤖 AI is analyzing the scan..."):
                try:
                    import os
                    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
                    response = requests.post(f"{BACKEND_URL}/predict", files=files, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style='color: #4caf50; margin-bottom: 5px;'>Prediction: {result['predicted_class'].title()}</h2>
                            <p style='font-size: 1.2em;'>Confidence: <b>{result['confidence'] * 100:.2f}%</b></p>
                            <hr style='border: 0.1px solid rgba(255,255,255,0.1);'>
                            <p style='color: #888;'>Model: {result['scan_name']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Dynamic recommendation based on confidence
                        if result['confidence'] < 0.7:
                            st.warning("⚠️ Low confidence detected. Please ensure the scan is correct or consult a specialist immediately.")
                        elif "normal" not in result['predicted_class'].lower() and "no tumor" not in result['predicted_class'].lower():
                            st.error("❗ Potential abnormality detected. We strongly recommend scheduling an appointment with your doctor for a detailed consultation.")
                        else:
                            st.success("✅ The scan appears to be within normal parameters. However, always verify with a professional.")
                            
                    elif response.status_code == 503:
                        st.error("🚀 Model is not yet available. We are currently integrating this specific scan model.")
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"📡 Connection error: Could not reach backend API. Make sure `main.py` is running.")

# Footer
st.markdown("---")
st.caption("MediScan v2.0 - Empowering patients with AI-driven insights.")
