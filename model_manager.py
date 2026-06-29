import io
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import DEVICE, MODELS_CONFIG

class ModelManager:
    def __init__(self):
        self.models = {}
        self.base_dir = Path(__file__).resolve().parent

    def _resolve_path(self, relative_path):
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def get_model(self, model_type):
        if model_type not in MODELS_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in self.models:
            return self.models[model_type]

        config = MODELS_CONFIG[model_type]
        model_path = self._resolve_path(config["model_path"])
        if not model_path.exists():
            print(f"Warning: model file {model_path} not found.")
            return None

        print(f"Loading model: {model_type} from {model_path}...")
        
        try:
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
            return "Model not found/loaded", 0.0

        config = MODELS_CONFIG[model_type]
        
        try:
            if model_type == "bone":
                # YOLO prediction logic
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                device_str = "cpu" if str(DEVICE) == "cpu" else "0" if "cuda" in str(DEVICE) else None
                results = model.predict(source=image, device=device_str, verbose=False)
                
                if not results or len(results[0].boxes) == 0:
                    return "not fractured", 0.99
                    
                boxes = results[0].boxes
                best_conf = 0.0
                best_class_idx = 0
                
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_class_idx = cls
                        
                class_name = model.names[best_class_idx]
                return class_name, best_conf
                
            else:
                # Classification prediction logic
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, preds = torch.max(probs, 1)
                    
                classes = config["classes"]
                predicted_class = classes[preds.item()]
                return predicted_class, confidence.item()
                
        except Exception as e:
            print(f"Error running prediction for {model_type}: {e}")
            return "Inference error", 0.0

# Singleton instance
manager = ModelManager()
