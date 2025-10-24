import streamlit as st
from inference import load_model, predict_image

st.title("🧠 Brain Tumor Detector")
model = load_model()

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg","jpeg","png"])
if uploaded_file:
    pred = predict_image(model, uploaded_file.read())
    st.image(uploaded_file, caption=f"**Prediction: {pred}**", use_container_width=True)
