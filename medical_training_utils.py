"""Shared training helpers for image-classification models."""

from pathlib import Path
import copy
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms


def train_image_folder_classifier(
    data_dir,
    model_save_name,
    class_names_output=None,
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    val_split=0.2,
    seed=42,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_path = data_dir / "train"
    val_path = data_dir / "val"
    has_explicit_splits = train_path.is_dir() and val_path.is_dir()

    if has_explicit_splits:
        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)
        class_names = train_dataset.classes
    else:
        full_dataset = datasets.ImageFolder(str(data_dir))
        class_names = full_dataset.classes

        dataset_size = len(full_dataset)
        if dataset_size < 2:
            raise ValueError("Dataset must contain at least two images to create a train/val split.")

        val_size = max(1, int(val_split * dataset_size))
        val_size = min(val_size, dataset_size - 1)
        train_size = dataset_size - val_size

        indices = list(range(dataset_size))
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(indices, [train_size, val_size], generator=generator)

        train_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=train_transform), train_indices.indices)
        val_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=val_transform), val_indices.indices)

    if class_names_output is not None:
        class_names_output = Path(class_names_output)
        class_names_output.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
    )

    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model_save_path = Path(model_save_name)

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f"Checkpoint saved: {model_save_path}")

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, class_names
