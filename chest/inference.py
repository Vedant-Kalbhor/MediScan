import io
import torch
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path

CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
MODEL_FILENAME = Path(__file__).with_name("best_chest_model.pth")

def load_model(model_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or MODEL_FILENAME
    
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=len(CLASSES))
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_bytes, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    return CLASSES[preds.item()], confidence.item()
