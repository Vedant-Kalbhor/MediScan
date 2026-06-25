import sys
import io
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

CLASSES = ["fractured", "not fractured"]
MODEL_FILENAME = Path(__file__).with_name("best_bone_model.pt")


def load_model(model_path=None, device=None):
    if YOLO is None:
        print("ultralytics is not installed. Please install it to use YOLOv8 models.")
        return None
    model_path = model_path or MODEL_FILENAME
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return None
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    return model


def predict_image(model, image_bytes, device=None):
    if model is None:
        return "error", 0.0
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Predict using YOLO (device will be handled automatically, or we pass it)
    # ultralytics uses "cpu" or "0" for cuda
    device_str = "cpu" if str(device) == "cpu" else "0" if "cuda" in str(device) else None
    results = model.predict(source=image, device=device_str, verbose=False)
    
    if not results or len(results[0].boxes) == 0:
        # No detections usually implies no fracture
        return "not fractured", 0.99
    
    # If boxes exist, let's find the one with highest confidence
    boxes = results[0].boxes
    best_conf = 0.0
    best_class_idx = 0
    
    for box in boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf > best_conf:
            best_conf = conf
            best_class_idx = cls
            
    # Map index to class name
    class_name = model.names[best_class_idx]
    
    return class_name, best_conf
