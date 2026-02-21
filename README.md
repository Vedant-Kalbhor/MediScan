# 🩺 MediScan: AI-Powered Medical Scan Guide

MediScan is a comprehensive medical imaging classification project designed to provide patients with an initial automated interpretation of their medical scans (MRI, CT, X-ray, Ultrasound) before consulting a doctor.

## 🚀 Features
- **Brain Tumor Detection (MRI)**: Classifies Glioma, Meningioma, Pituitary tumors, or Normal scans.
- **Chest Disease Classification (CT)**: Detects Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and Normal lungs.
- **Breast Cancer Analysis (Ultrasound)**: Classified as Benign, Malignant, or Normal.
- **Kidney Condition Detection (CT)**: Identifies Stones, Cysts, Tumors, or Normal kidneys.
- **Bone Fracture Detection (X-ray)**: Quick identification of fractures in radiographs.

## 📊 Datasets Used
| Scan Type | Source Link |
|-----------|-------------|
| Brain MRI | [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Chest CT | [Kaggle - Chest CT-Scan images](https://www.kaggle.com/datasets/mohamedhany2020/chest-ctscan-images) |
| Breast Ultrasound | [Kaggle - Breast Ultrasound Images](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| Kidney CT | [Kaggle - CT KIDNEY DATASET](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
<!-- | Bone X-ray | [Kaggle - Bone Fracture Multi-Region](https://www.kaggle.com/datasets/babban/bone-fracture-dataset-new) | -->

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Deep Learning**: PyTorch (ResNet18)
- **Inference**: Pillow, Torchvision

## 🏃 How to Run
1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn streamlit torch torchvision pillow requests
   ```
2. **Start Backend**:
   ```bash
   python main.py
   ```
3. **Start Frontend**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🧠 Training New Models (Kaggle)
To train models for the new categories without local storage constraints, use the respective templates provided:
- **Breast Cancer**: Use `breast_training_template.py`
- **Kidney Stone**: Use `kidney_training_template.py`

1. Create a New Notebook on Kaggle.
2. Add the respective dataset using the links above.
3. Paste the training code from the template.
4. Run the training and download the `best_model.pth` file.
5. Place the model file in the root directory and update `config.py`.

## ⚠️ Disclaimer
**MediScan is NOT a medical diagnostic tool.** The predictions are based on ML models and may contain errors. Always consult a licensed medical professional for clinical diagnosis.
