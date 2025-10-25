import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="🧠🫁 MediScan Classifier", layout="centered")

st.title("🧠🫁 MediScan: Brain & Chest Disease Classifier")

# Step 1: Choose model
model_choice = st.radio("Select Model:", ["Brain Tumor (MRI)", "Chest CT Scan"])

# Step 2: Upload image
uploaded_file = st.file_uploader(
    f"Upload {'MRI' if 'Brain' in model_choice else 'CT Scan'} Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        model_type = "brain" if "Brain" in model_choice else "chest"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"model_type": model_type}

        with st.spinner("Running prediction... Please wait."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files, params=params)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Prediction: **{result['predicted_class']}**")
                    st.info(f"📊 Confidence: {result['confidence'] * 100:.2f}%")
                    st.caption(f"🧬 Model Used: {result['model_used'].capitalize()}")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
