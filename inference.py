import torch
from torchvision import models, transforms
from PIL import Image
import io

classes = ["glioma", "meningioma", "pituitary", "notumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path="best_brain_tumor_resnet18.pth"):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        predicted_class = classes[preds.item()]
    return predicted_class
