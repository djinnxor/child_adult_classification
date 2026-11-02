"""
Build YOLO train/valid/test splits.

Two modes
---------
- flat:  split a single flat folder of images + YOLO .txt labels.
- merge: recursively walk a parent folder, collecting every image that has a
         sibling .txt label in the same directory, combine them all, shuffle,
         and split into one dataset.

         Designed for a tree like:
             <parent>/<source>/<clip>/  img.jpg + img.txt  (+ cropped/ ...)
         where labels sit next to their images.  Any directory named "cropped"
         is ignored, and frames are de-duplicated by filename stem so the same
         frame extracted into several folders never lands in two splits.

Output structure
----------------
    <output_dir>/
        train/  images/ *.jpg ...   labels/ *.txt ...
        valid/  images/             labels/
        test/   images/             labels/
        data.yaml

Usage
-----
    python person_detection/split_dataset.py flat  --input_dir output_frames/video_name
    python person_detection/split_dataset.py flat  --input_dir data --output_dir dataset \\
        --train 0.8 --val 0.1 --test 0.1
    python person_detection/split_dataset.py merge --input_dir "/home/rookie/Desktop/person dataset/labeled"
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "valid", "test")
# Substrings that mark a directory to skip while walking in merge mode (e.g.
# "cropped", "cropped after labeling" — crops that carry no labels and would
# otherwise be mistaken for data). Matched case-insensitively as substrings.
IGNORE_DIR_SUBSTRINGS = ("cropped",)


def _is_ignored_dir(name: str) -> bool:
    low = name.lower()
    return any(sub in low for sub in IGNORE_DIR_SUBSTRINGS)


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


def _iter_label_folders(root: Path):
    """Yield every directory under root (inclusive) that is not ignored.

    Directories named in IGNORE_DIRS (and everything below them) are pruned.
    """
    if _is_ignored_dir(root.name):
        return
    yield root
    for child in sorted(root.iterdir()):
        if child.is_dir() and not _is_ignored_dir(child.name):
            yield from _iter_label_folders(child)


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

    if not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    # Recursively gather (image, sibling .txt) pairs from every folder, ignoring
    # `cropped/` dirs. Frames are de-duplicated by stem so the same frame copied
    # into multiple folders never ends up split across train/valid/test.
    pairs: list[tuple[Path, Path, str]] = []
    seen_stems: dict[str, Path] = {}
    total_skipped = 0
    total_dupes = 0
    folders_with_data = 0

    for folder in _iter_label_folders(input_dir):
        folder_pairs, skipped = _pairs_from_flat_folder(folder)
        total_skipped += skipped
        if not folder_pairs:
            continue
        folders_with_data += 1
        kept = 0
        for img_path, lbl_path in folder_pairs:
            stem = img_path.stem
            if stem in seen_stems:
                total_dupes += 1
                continue
            seen_stems[stem] = img_path
            pairs.append((img_path, lbl_path, stem))
            kept += 1
        rel = folder.relative_to(input_dir)
        print(f"  {rel}: {kept} pair(s)" +
              (f" ({len(folder_pairs) - kept} dup)" if kept != len(folder_pairs)
               else ""))

    if total_skipped:
        print(f"Skipped {total_skipped} image(s) without matching .txt")
    if total_dupes:
        print(f"Skipped {total_dupes} duplicate frame(s) (same filename stem)")
    if not pairs:
        raise SystemExit(f"No image/label pairs found under: {input_dir}")

    print(f"Combined {len(pairs)} unique image/label pair(s) from "
          f"{folders_with_data} folder(s)")
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
