from pathlib import Path
import os
import shutil
import tempfile
from urllib.request import urlopen

from config import DEVICE, MODELS_CONFIG, MODEL_URL_ENV_MAP, KIDNEY_MODEL_URL
from utils.hf_model_loader import get_model_path

class ModelManager:
    def __init__(self):
        self.models = {}
        self.base_dir = Path(__file__).resolve().parent

    def _resolve_path(self, relative_path):
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _download_file(self, url: str, destination: Path) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=destination.suffix, dir=str(destination.parent))
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                with urlopen(url) as response:
                    shutil.copyfileobj(response, tmp_file)
            Path(tmp_name).replace(destination)
            return True
        except Exception as exc:
            print(f"Warning: failed to download model from {url}: {exc}")
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def ensure_model_file(self, model_type):
        config = MODELS_CONFIG[model_type]
        model_path = self._resolve_path(config["model_path"])
        if model_path.exists():
            return model_path

        hf_repo = os.getenv("HF_MODEL_REPO", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip() or None
        cache_dir = os.getenv("MODEL_CACHE_DIR", "").strip() or None

        downloaded_path = get_model_path(
            model_path,
            hf_repo=hf_repo,
            filename=Path(config["model_path"]).name,
            cache_dir=cache_dir,
            token=hf_token,
        )
        if downloaded_path is not None:
            return downloaded_path

        url_env = MODEL_URL_ENV_MAP.get(model_type)
        default_url = KIDNEY_MODEL_URL if model_type == "kidney" else ""
        model_url = os.getenv(url_env, default_url) if url_env else default_url
        if not model_url:
            print(f"Warning: model file {model_path} not found and {url_env} is not set.")
            return None

        print(f"Downloading model for {model_type} from {model_url} to {model_path}...")
        if self._download_file(model_url, model_path):
            return model_path
        return None

    def get_model(self, model_type):
        if model_type not in MODELS_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in self.models:
            return self.models[model_type]

        config = MODELS_CONFIG[model_type]
        model_path = self.ensure_model_file(model_type)
        if model_path is None:
            return None

        print(f"Loading model: {model_type} from {model_path}...")
        
        try:
            import io
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import timm

            from PIL import Image

            from torchvision import transforms

            try:
                from ultralytics import YOLO
            except ImportError:
                YOLO = None

            if model_type == "brain":
                # DenseNet121
                model = models.densenet121()
                model.classifier = nn.Linear(model.classifier.in_features, len(config["classes"]))
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                self.models[model_type] = model
                
            elif model_type == "breast":
                # ResNet18
                model = models.resnet18()
                model.fc = nn.Linear(model.fc.in_features, len(config["classes"]))
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                self.models[model_type] = model
                
            elif model_type == "chest":
                # tf_efficientnetv2_b0
                model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=len(config["classes"]))
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                self.models[model_type] = model
                
            elif model_type == "kidney":
                # ResNet18
                model = models.resnet18()
                model.fc = nn.Linear(model.fc.in_features, len(config["classes"]))
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                self.models[model_type] = model
                
            elif model_type == "bone":
                # YOLOv8
                if YOLO is None:
                    print("Warning: ultralytics is not installed.")
                    return None
                model = YOLO(str(model_path))
                self.models[model_type] = model
                
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            return None
            
        return self.models.get(model_type)

    def predict(self, model_type, image_bytes):
        model = self.get_model(model_type)
        if model is None:
            return "Model not found/loaded", 0.0, None

        config = MODELS_CONFIG[model_type]
        
        try:
            import io
            import torch
            from PIL import Image
            from torchvision import transforms

            device = torch.device(DEVICE if DEVICE == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))

            def _resolve_name(class_index):
                names = getattr(model, "names", {})
                if isinstance(names, dict):
                    return names.get(class_index, str(class_index))
                if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
                    return names[class_index]
                return str(class_index)

            def _image_region(center_x, center_y, image_width, image_height):
                if image_width <= 0 or image_height <= 0:
                    return "unknown"

                col = "left" if center_x < image_width / 3 else "center" if center_x < (2 * image_width / 3) else "right"
                row = "upper" if center_y < image_height / 3 else "middle" if center_y < (2 * image_height / 3) else "lower"

                if row == "middle" and col == "center":
                    return "center"
                return f"{row}-{col}"

            if model_type == "bone":
                # YOLO prediction logic
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_width, image_height = image.size
                device_str = "cpu" if str(device) == "cpu" else "0" if "cuda" in str(device) else None
                results = model.predict(source=image, device=device_str, verbose=False)
                
                if not results or len(results[0].boxes) == 0:
                    return "not fractured", 0.99, {
                        "fracture_detected": False,
                        "image_size": {"width": image_width, "height": image_height},
                        "detections": [],
                    }
                    
                boxes = results[0].boxes
                best_conf = 0.0
                best_class_idx = 0
                detections = []
                
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    class_name = _resolve_name(cls)
                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                        "center": {
                            "x": center_x,
                            "y": center_y,
                        },
                        "image_region": _image_region(center_x, center_y, image_width, image_height),
                    })
                    if conf > best_conf:
                        best_conf = conf
                        best_class_idx = cls
                        
                class_name = _resolve_name(best_class_idx)
                return class_name, best_conf, {
                    "fracture_detected": True,
                    "image_size": {"width": image_width, "height": image_height},
                    "detections": detections,
                    "best_detection": max(detections, key=lambda item: item["confidence"]) if detections else None,
                }
                
            else:
                # Classification prediction logic
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, preds = torch.max(probs, 1)
                    
                classes = config["classes"]
                predicted_class = classes[preds.item()]
                return predicted_class, confidence.item(), None
                
        except Exception as e:
            print(f"Error running prediction for {model_type}: {e}")
            return "Inference error", 0.0, None

# Singleton instance
manager = ModelManager()
