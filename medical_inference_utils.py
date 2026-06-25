"""Shared inference helpers for benchmarked image-classification models."""

from __future__ import annotations

from pathlib import Path
import io
import json

import torch
from PIL import Image
from torchvision import transforms

from medical_training_utils import build_architecture_for_inference


def _metadata_path(model_path):
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_metadata.json")


def load_metadata(model_path):
    metadata_file = _metadata_path(model_path)
    if metadata_file.exists():
        try:
            return json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_class_names(model_path, fallback_classes):
    metadata = load_metadata(model_path)
    if metadata.get("class_names"):
        return list(metadata["class_names"])

    classes_file = Path(model_path).with_name("classes.json")
    if classes_file.exists():
        try:
            return json.loads(classes_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return list(fallback_classes)


def load_model(model_path, num_classes=None, fallback_classes=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    metadata = load_metadata(model_path)
    architecture = metadata.get("architecture", "resnet50")
    class_names = metadata.get("class_names") or list(fallback_classes or [])

    if num_classes is None:
        if class_names:
            num_classes = len(class_names)
        else:
            raise ValueError("num_classes or fallback_classes must be provided when metadata is missing.")

    model = build_architecture_for_inference(architecture, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.architecture = architecture
    model.class_names = class_names if class_names else list(fallback_classes or [])
    model.image_size = metadata.get("image_size", 224)
    return model


def load_resnet18_model(model_path, num_classes, fallback_classes, device=None):
    return load_model(model_path, num_classes=num_classes, fallback_classes=fallback_classes, device=device)


def transform_image(image_bytes, image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_with_model(model, image_bytes, fallback_classes, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = getattr(model, "image_size", 224)
    tensor = transform_image(image_bytes, image_size=image_size).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)

    classes = getattr(model, "class_names", fallback_classes)
    predicted_class = classes[preds.item()]
    return predicted_class, confidence.item()


def predict_with_resnet18(model, image_bytes, fallback_classes, device=None):
    return predict_with_model(model, image_bytes, fallback_classes, device=device)
