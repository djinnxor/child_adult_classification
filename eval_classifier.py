"""Offline evaluation of a trained child/adult classifier on the val split."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train import build_model, get_transforms


def confusion_matrix(labels, preds, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def per_class_report(labels, preds, classes):
    n = len(classes)
    cm = confusion_matrix(labels, preds, n)
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((c, precision, recall, f1, support))
    accuracy = cm.trace() / cm.sum() if cm.sum() else 0.0
    return rows, accuracy, cm


def roc_auc_binary(labels, scores):
    """AUC via the rank-sum (Mann-Whitney U) identity. labels in {0,1}."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    avg_ranks = rank_sums / counts
    ranks = avg_ranks[inverse]
    sum_pos_ranks = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def load_classifier(ckpt_path, device, img_size=224):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["backbone"], num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _, val_tf = get_transforms(img_size)
    return model, classes, val_tf


@torch.no_grad()
def evaluate(classifier, data_dir, batch_size, img_size, device):
    model, classes, val_tf = load_classifier(classifier, device, img_size)

    ds = datasets.ImageFolder(str(Path(data_dir)), transform=val_tf)
    if ds.classes != classes:
        print(f"WARNING: dataset classes {ds.classes} differ from "
              f"checkpoint classes {classes}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )

    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
        else:
            logits = model(imgs)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    print(f"\nDataset : {data_dir}  ({len(ds)} images)")
    print(f"Classes : {classes}\n")

    rows, accuracy, cm = per_class_report(labels, preds, classes)
    print(f"{'class':>10}  {'precision':>10}  {'recall':>10}  {'f1':>8}  {'support':>8}")
    for c, p, r, f1, s in rows:
        print(f"{c:>10}  {p:>10.4f}  {r:>10.4f}  {f1:>8.4f}  {s:>8d}")
    print(f"\nOverall accuracy: {accuracy:.4f}\n")

    print("Confusion matrix (rows=true, cols=pred):")
    print("          " + "  ".join(f"{c:>8}" for c in classes))
    for cls, row in zip(classes, cm):
        print(f"{cls:>8}  " + "  ".join(f"{v:>8d}" for v in row))

    if len(classes) == 2:
        auc = roc_auc_binary(labels, probs[:, 1])
        print(f"\nROC AUC ({classes[1]} as positive): {auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate child/adult classifier on a val split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--classifier", required=True,
                        help="Checkpoint from train.py (best_model.pth).")
    parser.add_argument("--data_dir", required=True,
                        help="ImageFolder root, e.g. dataset/val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    evaluate(args.classifier, args.data_dir, args.batch_size,
             args.img_size, device)


if __name__ == "__main__":
    main()
