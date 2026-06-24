import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from medical_inference_utils import load_resnet18_model, predict_with_resnet18

CLASSES = ["fractured", "not fractured"]
MODEL_FILENAME = Path(__file__).with_name("best_bone_model.pth")


def load_model(model_path=None, device=None):
    model_path = model_path or MODEL_FILENAME
    return load_resnet18_model(model_path, len(CLASSES), CLASSES, device=device)


def predict_image(model, image_bytes, device=None):
    return predict_with_resnet18(model, image_bytes, CLASSES, device=device)
