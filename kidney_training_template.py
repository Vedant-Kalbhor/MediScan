import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
import os
import copy
import time

# ==========================================
# CONFIGURATION (Kidney Dataset)
# ==========================================
# Path to the directory containing the class folders (Cyst, Normal, Stone, Tumor)
DATA_DIR = '/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone' 
MODEL_SAVE_NAME = 'best_kidney_model.pth'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2 

# Load full dataset once to detect classes
if not os.path.exists(DATA_DIR):
    print(f"ERROR: DATA_DIR {DATA_DIR} not found. Please verify the path on Kaggle.")
else:
    temp_ds = datasets.ImageFolder(DATA_DIR)
    NUM_CLASSES = len(temp_ds.classes)
    print(f"Detected Categories: {temp_ds.classes} ({NUM_CLASSES} classes)")

# ==========================================
# DATA LOADING & TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading Logic
if os.path.exists(DATA_DIR):
    # Load with one transform first to get indices
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Generate indices
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(VAL_SPLIT * dataset_size)
    
    torch.manual_seed(42)
    train_indices, val_indices = random_split(indices, [dataset_size - split, split])
    
    # Create two versions of the dataset with different transforms
    train_ds_full = datasets.ImageFolder(DATA_DIR, train_transform)
    val_ds_full = datasets.ImageFolder(DATA_DIR, val_transform)
    
    image_datasets = {
        'train': Subset(train_ds_full, train_indices),
        'val': Subset(val_ds_full, val_indices)
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
else:
    print("Dataset folder not found. Skipping dataloader initialization.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL DEFINITION (ResNet18 Transfer Learning)
# ==========================================
def train_model():
    if not os.path.exists(DATA_DIR):
        print("Training aborted: Dataset not found.")
        return None

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"Starting Training on {device}...")
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_NAME)
                print(f"Checkpoint saved: {MODEL_SAVE_NAME}")

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    train_model()
