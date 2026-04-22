"""
Child / Adult Classification – Training Script
===============================================
Supports three backbone families:

  • **YOLOv8-cls** – ultralytics YOLOv8 classification pipeline (fast, ONNX-exportable)
  • **EfficientNet-B0** – lightweight, SOTA accuracy / speed trade-off
  • **ResNet-18 / ResNet-50** – well-known, highly stable baseline

Expected dataset layout (ImageFolder-compatible)
-------------------------------------------------
    <data_dir>/
        train/
            adult/  …jpg / png images
            child/  …jpg / png images
        val/
            adult/
            child/

Usage examples
--------------
    # EfficientNet (default, recommended)
    python train.py --data_dir dataset --backbone efficientnet --epochs 50

    # ResNet-50
    python train.py --data_dir dataset --backbone resnet50 --epochs 50

    # YOLOv8 classification head
    python train.py --data_dir dataset --backbone yolov8 --epochs 50
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def get_transforms(img_size: int = 224):
    """Return (train_transform, val_transform) tuples.

    Training augmentations are designed for CCTV footage:
    low-resolution, variable lighting, occasional motion blur.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            # Simulate variable CCTV illumination
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.RandomRotation(10),
            # Simulate camera noise / compression artefacts
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    """Instantiate a pre-trained model and replace its head for *num_classes*."""
    backbone = backbone.lower()

    if backbone == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            "Choose from: efficientnet, resnet50, resnet18, yolov8"
        )

    return model


# ---------------------------------------------------------------------------
# PyTorch training loop (EfficientNet / ResNet)
# ---------------------------------------------------------------------------

def train_pytorch(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_tf, val_tf = get_transforms(args.img_size)

    train_ds = datasets.ImageFolder(str(Path(args.data_dir) / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(Path(args.data_dir) / "val"), transform=val_tf)

    num_workers = min(4, max(0, len(train_ds) // args.batch_size))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Classes   : {train_ds.classes}")
    print(f"Train size: {len(train_ds)}  |  Val size: {len(val_ds)}\n")

    model = build_model(args.backbone, num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_model.pth"
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- training ----
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            _, pred = out.max(1)
            tr_total += labels.size(0)
            tr_correct += pred.eq(labels).sum().item()

        scheduler.step()

        # ---- validation ----
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                va_loss += loss.item()
                _, pred = out.max(1)
                va_total += labels.size(0)
                va_correct += pred.eq(labels).sum().item()

        tr_acc = tr_correct / tr_total * 100
        va_acc = va_correct / va_total * 100

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train  loss={tr_loss/len(train_loader):.4f}  acc={tr_acc:.2f}% | "
            f"Val  loss={va_loss/len(val_loader):.4f}  acc={va_acc:.2f}%"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "classes": train_ds.classes,
                    "state_dict": model.state_dict(),
                    "val_acc": va_acc,
                },
                best_ckpt,
            )
            print(f"  → best model saved (val acc {va_acc:.2f}%)")

    print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------------
# YOLOv8 classification training
# ---------------------------------------------------------------------------

def train_yolov8(args: argparse.Namespace) -> None:
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.yolo_model)
    model.train(
        data=args.data_dir,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        project=args.output_dir,
        name="yolov8_cls",
        pretrained=True,
        patience=20,
        cos_lr=True,
        augment=True,
    )
    print(f"\nTraining complete.  Results saved to: {args.output_dir}/yolov8_cls")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a child/adult image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root dataset directory containing train/ and val/ sub-folders.",
    )
    parser.add_argument(
        "--backbone",
        default="efficientnet",
        choices=["efficientnet", "resnet50", "resnet18", "yolov8"],
        help="Model backbone.",
    )
    parser.add_argument(
        "--yolo_model",
        default="yolov8n-cls.pt",
        help="YOLOv8 classification weights (only used when --backbone yolov8).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--output_dir",
        default="runs/train",
        help="Directory to save checkpoints.",
    )

    args = parser.parse_args()

    if args.backbone == "yolov8":
        train_yolov8(args)
    else:
        train_pytorch(args)


if __name__ == "__main__":
    main()
