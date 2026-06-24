"""Configuration for medical image models."""

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
        "model_path": "bone/best_bone_model.pth",
        "train_script": "bone/train.py",
        "inference_module": "bone/inference.py",
        "classes": ["fractured", "not fractured"],
        "normal_classes": ["not fractured"],
        "description": "Detects fractures in bone X-ray images.",
        "input_type": "X-ray",
    },
}

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
