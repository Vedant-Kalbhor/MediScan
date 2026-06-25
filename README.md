# MediScan: AI-Powered Medical Scan Guide

MediScan is a medical imaging classification project that provides an initial automated interpretation of common scans before a clinician review.

## Features
- Brain tumor detection from MRI
- Chest disease classification from CT
- Breast cancer analysis from ultrasound
- Kidney condition detection from CT
- Bone fracture detection from X-ray

## Project Layout
Each model now lives in its own folder with a standard layout:

```text
brain/
  train.py
  inference.py
  best_brain_model.pth

breast/
  train.py
  inference.py
  best_breast_model.pth

chest/
  train.py
  inference.py
  best_chest_model.pth

kidney/
  train.py
  inference.py
  best_kidney_model.pth

bone/
  train.py
  inference.py
  best_bone_model.pth
```

## Datasets
| Scan Type | Source Link |
|-----------|-------------|
| Brain MRI | [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Chest CT | [Kaggle - Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| Breast Ultrasound | [Kaggle - Breast Ultrasound Images](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| Kidney CT | [Kaggle - CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| Bone X-ray | [Kaggle - Bone Fracture Detection Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) |

## Tech Stack
- FastAPI
- Streamlit
- PyTorch
- Torchvision
- Pillow
- Requests

## Run
1. Install dependencies:

   ```bash
   pip install fastapi uvicorn streamlit torch torchvision pillow requests python-multipart pydantic
   ```

2. Start the backend:

   ```bash
   python main.py
   ```

3. Start the frontend:

   ```bash
   streamlit run streamlit_app.py
   ```

## Training
Each folder has a `train.py` script and a Kaggle-ready `train.ipynb` notebook that benchmark multiple architectures and save:
- the best model weights as `best_<name>_model.pth`
- the discovered class order as `classes.json`
- benchmark results as `best_<name>_model_benchmark.csv`
- architecture metadata as `best_<name>_model_metadata.json`

The benchmark compares:
- CNN baseline
- ResNet50
- EfficientNetB0
- EfficientNetB3
- ViT
- optional ConvNeXt Tiny

You can override the dataset location with environment variables:
- `BRAIN_DATA_DIR`
- `BREAST_DATA_DIR`
- `CHEST_DATA_DIR`
- `KIDNEY_DATA_DIR`
- `BONE_DATA_DIR`

## Disclaimer
MediScan is not a medical diagnostic tool. Always consult a licensed medical professional for diagnosis and treatment.
