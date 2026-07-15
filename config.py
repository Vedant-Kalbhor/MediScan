"""Configuration for medical image models."""

import os


MODELS_CONFIG = {
    "brain": {
        "name": "Brain Tumor (MRI)",
        "model_path": "brain/best_brain_model.pth",
        "train_script": "brain/train.py",
        "inference_module": "brain/inference.py",
        "classes": ["glioma", "meningioma", "pituitary", "no tumor"],
        "normal_classes": ["no tumor"],
        "description": "Analyzes brain MRI scans to detect tumor types.",
        "input_type": "MRI Scan",
    },
    "chest": {
        "name": "Chest CT Scan",
        "model_path": "chest/best_chest_model.pth",
        "train_script": "chest/train.py",
        "inference_module": "chest/inference.py",
        "classes": ["adenocarcinoma", "large cell carcinoma", "normal", "squamous cell carcinoma"],
        "normal_classes": ["normal"],
        "description": "Detects lung cancer types and normal chest conditions from CT scans.",
        "input_type": "CT Scan",
    },
    "breast": {
        "name": "Breast Cancer (Ultrasound)",
        "model_path": "breast/best_breast_model.pth",
        "train_script": "breast/train.py",
        "inference_module": "breast/inference.py",
        "classes": ["benign", "malignant", "normal"],
        "normal_classes": ["benign", "normal"],
        "description": "Classifies breast ultrasound images as benign, malignant, or normal.",
        "input_type": "Ultrasound",
    },
    "kidney": {
        "name": "Kidney Stone (CT)",
        "model_path": "kidney/best_kidney_model.pth",
        "train_script": "kidney/train.py",
        "inference_module": "kidney/inference.py",
        "classes": ["cyst", "normal", "stone", "tumor"],
        "normal_classes": ["normal"],
        "description": "Identifies kidney stones, cysts, and tumors from axial CT scans.",
        "input_type": "CT Scan",
    },
    "bone": {
        "name": "Bone Fracture (X-ray)",
        "model_path": "bone/best_bone_model.pt",
        "train_script": "bone/train.py",
        "inference_module": "bone/inference.py",
        "classes": ["fracture", "not fractured"],
        "normal_classes": ["not fractured"],
        "description": "Detects fractures in bone X-ray images using YOLOv8 and reports bounding-box location.",
        "input_type": "X-ray",
    },
}

# Keep startup light on hosted platforms by avoiding a torch import at module load.
# The inference code resolves the actual device when it needs to run a model.
DEVICE = os.getenv("MEDISCAN_DEVICE", "cpu")

# Optional external URLs for downloading model weights when they are not baked into the repo.
# Example:
#   BRAIN_MODEL_URL=https://...
#   BREAST_MODEL_URL=https://...
#   CHEST_MODEL_URL=https://...
#   KIDNEY_MODEL_URL=https://...
#   BONE_MODEL_URL=https://...
MODEL_URL_ENV_MAP = {
    "brain": "BRAIN_MODEL_URL",
    "breast": "BREAST_MODEL_URL",
    "chest": "CHEST_MODEL_URL",
    "kidney": "KIDNEY_MODEL_URL",
    "bone": "BONE_MODEL_URL",
}

# Temporary hosted source for the kidney model while it lives on Hugging Face.
# Render will download this file on first use if the local weight file is missing.
KIDNEY_MODEL_URL = "https://huggingface.co/vedk08/saved_kidney_model/resolve/main/best_kidney_model.pth"
