import io
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from utils.hf_model_loader import get_model_path

CLASSES = ["fracture", "not fractured"]
MODEL_FILENAME = Path(__file__).with_name("best_bone_model.pt")


def load_model(model_path=None, device=None):
    model_path = model_path or get_model_path(
        MODEL_FILENAME,
        hf_repo=os.getenv("HF_MODEL_REPO", ""),
        filename=MODEL_FILENAME.name,
    )
    if model_path is None:
        raise FileNotFoundError("Bone model weights are missing and HF_MODEL_REPO is not configured.")

    return YOLO(str(model_path))


def predict_image(model, image_bytes, device=None):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return model.predict(source=image, verbose=False)
