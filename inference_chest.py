import torch
from torchvision import transforms, models
from PIL import Image
import io

# Classes used during training
CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]

def load_model(model_path="best_chest_model.pth"):
    """Load chest CT model"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image(model, image_bytes):
    """Predict class for given image"""
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    return CLASSES[pred.item()], conf.item()
