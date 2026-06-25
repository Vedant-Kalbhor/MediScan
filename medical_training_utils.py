"""Shared Kaggle-ready benchmark helpers for image classification models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import csv
import json
import os
import random
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    display_name: str
    image_size: int
    builder: Callable[[int], nn.Module]
    optional: bool = False


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _load_pretrained_model(factory, weight_enum_name: str | None = None):
    try:
        weights_enum = getattr(models, weight_enum_name) if weight_enum_name else None
        if weights_enum is not None:
            return factory(weights=weights_enum.DEFAULT)
    except Exception:
        pass
    try:
        return factory(weights=None)
    except TypeError:
        return factory(pretrained=False)


def build_architecture(key: str, num_classes: int) -> nn.Module:
    key = key.lower()
    if key == "cnn":
        return SimpleCNN(num_classes)
    if key == "resnet50":
        model = _load_pretrained_model(models.resnet50, "ResNet50_Weights")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if key == "efficientnet_b0":
        model = _load_pretrained_model(models.efficientnet_b0, "EfficientNet_B0_Weights")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if key == "efficientnet_b3":
        model = _load_pretrained_model(models.efficientnet_b3, "EfficientNet_B3_Weights")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if key == "vit_b_16":
        model = _load_pretrained_model(models.vit_b_16, "ViT_B_16_Weights")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    if key == "convnext_tiny":
        model = _load_pretrained_model(models.convnext_tiny, "ConvNeXt_Tiny_Weights")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {key}")


def build_architecture_for_inference(key: str, num_classes: int) -> nn.Module:
    key = key.lower()
    if key == "cnn":
        return SimpleCNN(num_classes)
    if key == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if key == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if key == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if key == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    if key == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {key}")


def get_default_architectures(include_optional: bool = True):
    specs = [
        ArchitectureSpec("cnn", "CNN Baseline", 224, lambda n: build_architecture("cnn", n)),
        ArchitectureSpec("resnet50", "ResNet50", 224, lambda n: build_architecture("resnet50", n)),
        ArchitectureSpec("efficientnet_b0", "EfficientNetB0", 224, lambda n: build_architecture("efficientnet_b0", n)),
        ArchitectureSpec("efficientnet_b3", "EfficientNetB3", 224, lambda n: build_architecture("efficientnet_b3", n)),
        ArchitectureSpec("vit_b_16", "Vision Transformer", 224, lambda n: build_architecture("vit_b_16", n)),
        ArchitectureSpec("convnext_tiny", "ConvNeXt Tiny", 224, lambda n: build_architecture("convnext_tiny", n), optional=True),
    ]
    if include_optional:
        return specs
    return [spec for spec in specs if not spec.optional]


def _resolve_dataset_splits(data_dir: Path, train_transform, val_transform, val_split: float, seed: int):
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    if train_path.is_dir() and val_path.is_dir():
        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)
        return train_dataset, val_dataset, train_dataset.classes

    full_dataset = datasets.ImageFolder(str(data_dir))
    class_names = full_dataset.classes
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two images to create a validation split.")

    val_size = max(1, int(val_split * dataset_size))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    indices = list(range(dataset_size))
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(indices, [train_size, val_size], generator=generator)

    train_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=train_transform), train_subset.indices)
    val_dataset = Subset(datasets.ImageFolder(str(data_dir), transform=val_transform), val_subset.indices)
    return train_dataset, val_dataset, class_names


def _confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, pred in zip(targets.view(-1), preds.view(-1)):
        matrix[target.long(), pred.long()] += 1
    return matrix


def _metrics_from_confusion(matrix: torch.Tensor):
    total = matrix.sum().item()
    correct = torch.trace(matrix).item()
    accuracy = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1_scores = []
    for idx in range(matrix.size(0)):
        tp = matrix[idx, idx].item()
        fp = matrix[:, idx].sum().item() - tp
        fn = matrix[idx, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def _evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(labels.detach().cpu())

    preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    matrix = _confusion_matrix(preds, targets, model.classifier[-1].out_features if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[-1], nn.Linear) else model.fc.out_features)
    return _metrics_from_confusion(matrix)


def _prepare_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def _train_single_architecture(
    arch_spec: ArchitectureSpec,
    train_dataset,
    val_dataset,
    class_names,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device,
    amp_enabled: bool,
):
    train_transform, val_transform = _prepare_transforms(arch_spec.image_size)

    def _rebuild_dataset(dataset, transform):
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, datasets.ImageFolder):
            rebuilt = datasets.ImageFolder(dataset.dataset.root, transform=transform)
            return Subset(rebuilt, dataset.indices)
        if isinstance(dataset, datasets.ImageFolder):
            return datasets.ImageFolder(dataset.root, transform=transform)
        return dataset

    train_dataset = _rebuild_dataset(train_dataset, train_transform)
    val_dataset = _rebuild_dataset(val_dataset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=amp_enabled,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=amp_enabled,
    )

    model = arch_spec.builder(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics = {"accuracy": 0.0, "f1": 0.0, "recall": 0.0, "precision": 0.0}
    best_score = -1.0

    for epoch in range(epochs):
        print(f"[{arch_spec.display_name}] Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_preds = []
            running_targets = []

            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=amp_enabled)
                labels = labels.to(device, non_blocking=amp_enabled)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_preds.append(preds.detach().cpu())
                running_targets.append(labels.detach().cpu())

            preds = torch.cat(running_preds) if running_preds else torch.empty(0, dtype=torch.long)
            targets = torch.cat(running_targets) if running_targets else torch.empty(0, dtype=torch.long)
            matrix = _confusion_matrix(preds, targets, len(class_names))
            metrics = _metrics_from_confusion(matrix)
            epoch_loss = running_loss / len(loader.dataset)
            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {metrics['accuracy']:.4f} "
                f"F1: {metrics['f1']:.4f} Recall: {metrics['recall']:.4f}"
            )

            if phase == "val" and (metrics["f1"] > best_score or (metrics["f1"] == best_score and metrics["accuracy"] > best_metrics["accuracy"])):
                best_score = metrics["f1"]
                best_metrics = metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_metrics


def train_image_folder_classifier(
    data_dir,
    model_save_name,
    class_names_output=None,
    benchmark_results_output=None,
    metadata_output=None,
    batch_size=16,
    epochs=5,
    learning_rate=1e-4,
    val_split=0.2,
    seed=42,
    include_optional=True,
    architecture_keys=None,
    preferred_metric="f1",
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    base_train_transform, base_val_transform = _prepare_transforms(224)
    train_dataset, val_dataset, class_names = _resolve_dataset_splits(
        data_dir=data_dir,
        train_transform=base_train_transform,
        val_transform=base_val_transform,
        val_split=val_split,
        seed=seed,
    )

    if class_names_output is not None:
        class_names_output = Path(class_names_output)
        class_names_output.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    if architecture_keys is None:
        architectures = get_default_architectures(include_optional=include_optional)
    else:
        architecture_map = {spec.key: spec for spec in get_default_architectures(include_optional=True)}
        architectures = [architecture_map[key] for key in architecture_keys if key in architecture_map]

    if not architectures:
        raise ValueError("No architectures selected for benchmarking.")

    results = []
    best_overall = None

    for arch_spec in architectures:
        print(f"\n=== Benchmarking {arch_spec.display_name} ===")
        try:
            model, metrics = _train_single_architecture(
                arch_spec=arch_spec,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                class_names=class_names,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
                amp_enabled=amp_enabled,
            )
            result_row = {
                "architecture": arch_spec.key,
                "display_name": arch_spec.display_name,
                "accuracy": round(metrics["accuracy"], 6),
                "f1": round(metrics["f1"], 6),
                "recall": round(metrics["recall"], 6),
                "precision": round(metrics["precision"], 6),
                "status": "ok",
            }
            results.append(result_row)

            score = metrics.get(preferred_metric, 0.0)
            best_score = best_overall["metrics"].get(preferred_metric, 0.0) if best_overall else -1.0
            if best_overall is None or score > best_score:
                best_overall = {
                    "architecture": arch_spec.key,
                    "display_name": arch_spec.display_name,
                    "metrics": metrics,
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "image_size": arch_spec.image_size,
                }
        except Exception as exc:
            print(f"Skipping {arch_spec.display_name}: {exc}")
            results.append(
                {
                    "architecture": arch_spec.key,
                    "display_name": arch_spec.display_name,
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "status": f"skipped: {exc}",
                }
            )

    if best_overall is None:
        raise RuntimeError("All benchmarked architectures failed.")

    model_save_name = Path(model_save_name)
    model_save_name.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall["state_dict"], model_save_name)

    metadata = {
        "architecture": best_overall["architecture"],
        "display_name": best_overall["display_name"],
        "class_names": class_names,
        "image_size": best_overall["image_size"],
        "preferred_metric": preferred_metric,
        "metrics": best_overall["metrics"],
        "dataset": str(data_dir),
    }

    if metadata_output is None:
        metadata_output = model_save_name.with_name(f"{model_save_name.stem}_metadata.json")
    else:
        metadata_output = Path(metadata_output)

    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if benchmark_results_output is None:
        benchmark_results_output = model_save_name.with_name(f"{model_save_name.stem}_benchmark.csv")
    else:
        benchmark_results_output = Path(benchmark_results_output)

    with benchmark_results_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["architecture", "display_name", "accuracy", "f1", "recall", "precision", "status"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nBenchmark summary:")
    for row in results:
        print(
            f"{row['display_name']}: acc={row['accuracy']:.4f}, "
            f"f1={row['f1']:.4f}, recall={row['recall']:.4f}"
        )
    print(f"Best model: {best_overall['display_name']} -> {model_save_name}")
    print(f"Metadata saved to: {metadata_output}")
    print(f"Benchmark table saved to: {benchmark_results_output}")

    return {
        "best_model_path": str(model_save_name),
        "metadata_path": str(metadata_output),
        "benchmark_csv": str(benchmark_results_output),
        "best_architecture": best_overall["architecture"],
        "class_names": class_names,
        "results": results,
    }
