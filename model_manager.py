import torch
from torchvision import models, transforms
from PIL import Image
import io
from config import MODELS_CONFIG, DEVICE
import os

class ModelManager:
    def __init__(self):
        self.models = {}
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_model(self, model_type):
        if model_type not in MODELS_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in self.models:
            return self.models[model_type]

        config = MODELS_CONFIG[model_type]
        model_path = config["model_path"]

        # If model file doesn't exist, we'll return None or raise warning
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Please train/download it.")
            return None

        # Load ResNet18 structure (common for all these for now)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(config["classes"]))
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        self.models[model_type] = model
        return model

    def predict(self, model_type, image_bytes):
        model = self.get_model(model_type)
        if model is None:
            return "Model not found/loaded", 0.0

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_t = self.transforms(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
            config = MODELS_CONFIG[model_type]
            predicted_class = config["classes"][preds.item()]
            
        return predicted_class, confidence.item()

# Singleton instance
manager = ModelManager()
