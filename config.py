# config.py

# Configuration for Medical Models
MODELS_CONFIG = {
    "brain": {
        "name": "Brain Tumor (MRI)",
        "model_path": "best_brain_tumor_resnet18.pth",
        "classes": ["glioma", "meningioma", "pituitary", "no tumor"],
        "description": "Analyzes Brain MRI scans to detect different types of tumors.",
        "input_type": "MRI Scan"
    },
    "chest": {
        "name": "Chest CT Scan",
        "model_path": "best_chest_model.pth",
        "classes": ["adenocarcinoma", "large cell carcinoma", "normal", "squamous cell carcinoma"],
        "description": "Detects lung cancer types and normal chest conditions from CT scans.",
        "input_type": "CT Scan"
    },
    "breast": {
        "name": "Breast Cancer (Ultrasound)",
        "model_path": "best_breast_model.pth",
        "classes": ["benign", "malignant", "normal"],
        "description": "Classification of breast ultrasound images into Benign, Malignant, or Normal.",
        "input_type": "Ultrasound"
    },
    "kidney": {
        "name": "Kidney Stone (CT)",
        "model_path": "best_kidney_model.pth",
        "classes": ["cyst", "normal", "stone", "tumor"],
        "description": "Identifies kidney stones, cysts, and tumors from axial CT scans.",
        "input_type": "CT Scan"
    },
    "bone": {
        "name": "Bone Fracture (X-ray)",
        "model_path": "best_bone_model.pth",
        "classes": ["fractured", "not fractured"],
        "description": "Detects fractures in various bone X-rays.",
        "input_type": "X-ray"
    }
}

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
