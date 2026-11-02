"""Train child/adult classifier from a flat folder of filename-labeled crops.

Dataset layout
--------------
A single directory of cropped person images whose filename contains the class:

    <data_dir>/
        frame_000123_adult.jpg
        frame_000456_child.jpg
        clipA_box2_adult.png
        ...

Files are split into train/val automatically (stratified per class).

Recommended pretrained model
----------------------------
**EfficientNetV2-S** (default) — pretrained on ImageNet-1k. Best
accuracy/throughput trade-off in this size class for ~22M params; finetunes
well on small CCTV datasets. Fallbacks:

  * ``efficientnet`` (EfficientNet-B0, ~5M params) — lightweight baseline
  * ``resnet50`` / ``resnet18`` — well-known, very stable
  * ``mobilenet_v3_l`` (in this script only) — fastest, for edge deploy

Usage
-----
    python train_flat.py --data_dir labeled --epochs 40
    python train_flat.py --data_dir labeled --backbone efficientnet --epochs 50
    python train_flat.py --data_dir labeled --backbone mobilenet_v3_l --img_size 224

Saves a checkpoint compatible with classify_realtime.py / eval_classifier.py:
    {epoch, backbone, classes, state_dict, val_acc} -> runs/train/best_model.pth
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image

from train import build_model, get_transforms

CLASSES = ["adult", "child"]  # index = label
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_from_name(path: Path):
    """Return 0 (adult), 1 (child), or None if neither token is in the stem.

    Token match is case-insensitive and prefers 'child' over 'adult' if both
    appear (which shouldn't happen, but is a safe choice for filenames like
    'parent_with_child_...').
    """
    name = path.stem.lower()
    has_child = "child" in name
    has_adult = "adult" in name
    if has_child:
        return 1
    if has_adult:
        return 0
    return None


class FilenameLabeledDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items  # list of (Path, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def gather_items(data_dir: Path, recursive: bool):
    iterator = data_dir.rglob("*") if recursive else data_dir.iterdir()
    items, skipped = [], 0
    for p in iterator:
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = label_from_name(p)
        if lbl is None:
            skipped += 1
            continue
        items.append((p, lbl))
    return items, skipped


def stratified_split(items, val_frac, seed):
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for it in items:
        by_class[it[1]].append(it)
    train, val = [], []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_frac)) if len(lst) > 1 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_model_extended(backbone, num_classes):
    """Wrap train.build_model with one extra option: mobilenet_v3_l."""
    if backbone == "mobilenet_v3_l":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)
        return model
    return build_model(backbone, num_classes)


def make_transforms(img_size: int):
    """Stronger augmentations than train.get_transforms; uses TrivialAugmentWide."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),  # learned augmentation policy
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    _, val_tf = get_transforms(img_size)
    return train_tf, val_tf


def class_balanced_sampler(train_items):
    counts = [0, 0]
    for _, lbl in train_items:
        counts[lbl] += 1
    weights = [1.0 / counts[lbl] for _, lbl in train_items]
    return WeightedRandomSampler(weights, num_samples=len(train_items),
                                 replacement=True), counts


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1) * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
        else:
            out = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1) * 100


def main():
    parser = argparse.ArgumentParser(
        description="Train child/adult classifier from filename-labeled crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Folder of crops with 'adult'/'child' in filenames.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories.")
    parser.add_argument("--backbone", default="efficientnet_v2_s",
                        choices=["efficientnet_v2_s", "efficientnet",
                                 "resnet50", "resnet18", "mobilenet_v3_l"],
                        help="Pretrained backbone.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/train")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision.")
    parser.add_argument("--no-balanced-sampler", action="store_true",
                        help="Disable class-balanced sampling.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stop after N epochs without val-acc improvement.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device : {device}  AMP={use_amp}")

    # ---- data ----
    items, skipped = gather_items(Path(args.data_dir), args.recursive)
    if not items:
        raise SystemExit(f"No labeled images found in {args.data_dir} "
                         "(expected 'adult' or 'child' in filename).")
    train_items, val_items = stratified_split(items, args.val_frac, args.seed)
    print(f"Found  : {len(items)} labeled  ({skipped} skipped, no class token)")
    print(f"Train  : {len(train_items)}  Val: {len(val_items)}")

    train_tf, val_tf = make_transforms(args.img_size)
    train_ds = FilenameLabeledDataset(train_items, train_tf)
    val_ds = FilenameLabeledDataset(val_items, val_tf)

    sampler = None
    if not args.no_balanced_sampler:
        sampler, counts = class_balanced_sampler(train_items)
        print(f"Train class counts: adult={counts[0]} child={counts[1]} "
              "(balanced sampler enabled)")

    nw = min(4, max(0, len(train_ds) // args.batch_size))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=sampler is None,
        num_workers=nw, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=device.type == "cuda",
    )

    # ---- model ----
    model = build_model_extended(args.backbone, num_classes=len(CLASSES)).to(device)
    print(f"Backbone: {args.backbone}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- train ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_model.pth"
    best_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"val loss={va_loss:.4f} acc={va_acc:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_acc > best_acc:
            best_acc = va_acc
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "backbone": args.backbone,
                "classes": CLASSES,
                "state_dict": model.state_dict(),
                "val_acc": va_acc,
            }, best_ckpt)
            print(f"  → best saved (val acc {va_acc:.2f}%)")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop: no improvement for {args.patience} epochs.")
                break

    print(f"\nBest val acc: {best_acc:.2f}%  →  {best_ckpt}")


if __name__ == "__main__":
    main()
