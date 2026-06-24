"""Shared inference helpers for image-classification models."""

from pathlib import Path
import io
import json

import torch
from PIL import Image
from torchvision import models, transforms


def load_class_names(model_path, fallback_classes):
    metadata_path = Path(model_path).with_name("classes.json")
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return list(fallback_classes)


def load_resnet18_model(model_path, num_classes, fallback_classes, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model.class_names = load_class_names(model_path, fallback_classes)
    return model


def predict_with_resnet18(model, image_bytes, fallback_classes, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)

    classes = getattr(model, "class_names", fallback_classes)
    predicted_class = classes[preds.item()]
    return predicted_class, confidence.item()
