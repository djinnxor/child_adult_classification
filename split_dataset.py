"""
Build YOLO train/valid/test splits.

Two modes
---------
- flat:  split a single flat folder of images + YOLO .txt labels.
- merge: walk a parent folder containing multiple YOLO-template datasets
         (each with some of train/valid/test, each having images/ + labels/),
         combine every pair, shuffle, and re-split into one dataset.

Output structure
----------------
    <output_dir>/
        train/  images/ *.jpg ...   labels/ *.txt ...
        valid/  images/             labels/
        test/   images/             labels/
        data.yaml

Usage
-----
    python split_dataset.py flat  --input_dir output_frames/video_name
    python split_dataset.py flat  --input_dir data --output_dir dataset \\
        --train 0.8 --val 0.1 --test 0.1
    python split_dataset.py merge --input_dir all_datasets --output_dir dataset
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "valid", "test")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "ds"


def _write_splits(
    pairs: list[tuple[Path, Path, str]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """pairs: list of (image_path, label_path, output_stem)."""
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train: n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, out_stem in split_pairs:
            shutil.copy2(img_path, img_dir / f"{out_stem}{img_path.suffix}")
            shutil.copy2(lbl_path, lbl_dir / f"{out_stem}.txt")

        print(f"  {split_name:5s}: {len(split_pairs)} pairs -> {img_dir.parent}")

    yaml_data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 1,
        "names": ["person"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\ndata.yaml saved to: {yaml_path}")


def _pairs_from_flat_folder(folder: Path) -> tuple[list[tuple[Path, Path]], int]:
    """Return (pairs, skipped) for a flat folder of images + .txt labels."""
    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = folder / f"{img_path.stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            skipped += 1
    return pairs, skipped


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    pairs, skipped = _pairs_from_flat_folder(input_dir)
    if skipped:
        print(f"Skipped {skipped} image(s) without matching .txt")
    if not pairs:
        raise SystemExit(f"No image/label pairs found in: {input_dir}")

    print(f"Found {len(pairs)} image/label pairs")
    _write_splits(
        [(img, lbl, img.stem) for img, lbl in pairs],
        output_dir, train_ratio, val_ratio, seed,
    )


def merge_datasets(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Each immediate subfolder is treated as a YOLO-template dataset.
    dataset_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not dataset_dirs:
        raise SystemExit(f"No dataset subfolders found in: {input_dir}")

    pairs: list[tuple[Path, Path, str]] = []
    total_skipped = 0
    for ds_dir in dataset_dirs:
        ds_prefix = _sanitize(ds_dir.name)
        ds_count = 0
        for split in SPLIT_NAMES:
            images_dir = ds_dir / split / "images"
            labels_dir = ds_dir / split / "labels"
            if not (images_dir.is_dir() and labels_dir.is_dir()):
                continue
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    total_skipped += 1
                    continue
                # Prefix keeps filenames unique across datasets/splits.
                out_stem = f"{ds_prefix}__{split}__{img_path.stem}"
                pairs.append((img_path, lbl_path, out_stem))
                ds_count += 1
        print(f"  {ds_dir.name}: {ds_count} pair(s)")

    if total_skipped:
        print(f"Skipped {total_skipped} image(s) without matching .txt")
    if not pairs:
        raise SystemExit(f"No image/label pairs found under: {input_dir}")

    print(f"Combined {len(pairs)} image/label pair(s) from "
          f"{len(dataset_dirs)} dataset folder(s)")
    _write_splits(pairs, output_dir, train_ratio, val_ratio, seed)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output_dir", default="dataset", help="Root output folder")
    p.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    p.add_argument("--val", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YOLO train/valid/test splits from flat or merged datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    flat_p = subparsers.add_parser(
        "flat", help="Split a single flat image+label folder.")
    flat_p.add_argument(
        "--input_dir", required=True, help="Folder with images and .txt labels")
    _add_common_args(flat_p)

    merge_p = subparsers.add_parser(
        "merge",
        help="Merge multiple YOLO-template datasets, shuffle, and re-split.")
    merge_p.add_argument(
        "--input_dir", required=True,
        help="Parent folder containing multiple YOLO-template dataset folders")
    _add_common_args(merge_p)

    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("--train + --val + --test must equal 1.0")

    fn = split_dataset if args.command == "flat" else merge_datasets
    fn(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
