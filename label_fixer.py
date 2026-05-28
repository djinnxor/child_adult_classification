"""
Label Fixer
===========
For a given folder, reads all YOLO .txt annotation files and sets the class ID
(first field on each line) to 0, leaving all other fields unchanged.

Also supports cropping bboxes by class and cleaning a full YOLO dataset
(removing/renaming classes across train/valid/test, dropping images that
end up with no annotations).

Usage
-----
    python label_fixer.py fix --labels_dir path/to/labels --old_id 1 --new_id 0
    python label_fixer.py crop --base_dir path/to/split
    python label_fixer.py clean --dataset_dir path/to/dataset \\
        --remove 2 3 --rename 1:0 4:2
"""

import argparse
from pathlib import Path


def change_label_id(labels_dir: Path, old_id: str, new_id: str) -> None:
    txt_files = sorted(labels_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in: {labels_dir}")
        return

    changed_files = 0
    for txt_path in txt_files:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        file_changed = False
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                new_lines.append(line)
                continue
            if parts[0] == old_id:
                parts[0] = new_id
                file_changed = True
            new_lines.append(" ".join(parts))
        if file_changed:
            txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1

    print(f"Done – processed {len(txt_files)} file(s), modified {changed_files} file(s).")

def crop_by_label(base_dir: Path) -> None:
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    if not images_dir.is_dir():
        raise SystemExit(f"images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"labels folder not found: {labels_dir}")

    adult_dir = base_dir / "adult"
    child_dir = base_dir / "child"
    adult_dir.mkdir(exist_ok=True)
    child_dir.mkdir(exist_ok=True)

    import cv2

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    saved = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: cannot read {img_path}, skipping.")
            continue

        h, w = img.shape[:2]
        lines = lbl_path.read_text(encoding="utf-8").splitlines()

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
            except ValueError:
                continue

            if cls not in (0, 1):
                continue

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            out_dir = adult_dir if cls == 0 else child_dir
            out_path = out_dir / f"{img_path.stem}_box{i}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved += 1

    print(f"Done – saved {saved} crop(s) to: {adult_dir} / {child_dir}")


def clean_dataset(
    dataset_dir: Path,
    remove_ids: set[str],
    rename_map: dict[str, str],
) -> None:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    splits = ("train", "valid", "test")

    found_split = False
    total_lines_removed = 0
    total_lines_renamed = 0
    total_pairs_deleted = 0

    for split in splits:
        split_dir = dataset_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not labels_dir.is_dir():
            continue
        found_split = True

        for lbl_path in sorted(labels_dir.glob("*.txt")):
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    new_lines.append(line)
                    continue
                cls = parts[0]
                if cls in remove_ids:
                    total_lines_removed += 1
                    continue
                if cls in rename_map:
                    parts[0] = rename_map[cls]
                    total_lines_renamed += 1
                new_lines.append(" ".join(parts))

            # No annotations left -> drop the label file and its image.
            has_annotations = any(len(l.split()) >= 5 for l in new_lines)
            if not has_annotations:
                lbl_path.unlink()
                if images_dir.is_dir():
                    for ext in image_exts:
                        img_path = images_dir / f"{lbl_path.stem}{ext}"
                        if img_path.exists():
                            img_path.unlink()
                            break
                total_pairs_deleted += 1
                continue

            lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    if not found_split:
        raise SystemExit(
            f"No '{'/'.join(splits)}' split with a 'labels' folder found in: {dataset_dir}"
        )

    print(
        f"Done – removed {total_lines_removed} annotation(s), "
        f"renamed {total_lines_renamed} annotation(s), "
        f"deleted {total_pairs_deleted} empty image/label pair(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO label utilities: fix class IDs or crop bboxes by class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fix command
    fix_p = subparsers.add_parser("fix", help="Change class IDs in .txt files.")
    fix_p.add_argument("--labels_dir", required=True, help="Folder containing YOLO .txt files.")
    fix_p.add_argument("--old_id", required=True, help="Old class ID to be replaced.")
    fix_p.add_argument("--new_id", required=True, help="New class ID to replace the old one.")

    # crop command
    crop_p = subparsers.add_parser("crop", help="Crop bboxes into adult/ and child/ folders.")
    crop_p.add_argument(
        "--base_dir", required=True,
        help="Folder containing 'images' and 'labels' subfolders.",
    )

    # clean command
    clean_p = subparsers.add_parser(
        "clean",
        help="Remove/rename classes across a YOLO dataset (train/valid/test).",
    )
    clean_p.add_argument(
        "--dataset_dir", required=True,
        help="Dataset root containing train/valid/test, each with images/ and labels/.",
    )
    clean_p.add_argument(
        "--remove", nargs="*", default=[], metavar="CLASS_ID",
        help="Class IDs to remove, e.g. --remove 2 3",
    )
    clean_p.add_argument(
        "--rename", nargs="*", default=[], metavar="OLD:NEW",
        help="Class ID renames as old:new pairs, e.g. --rename 1:0 4:2",
    )

    args = parser.parse_args()

    if args.command == "fix":
        labels_dir = Path(args.labels_dir)
        if not labels_dir.is_dir():
            raise SystemExit(f"Not a directory: {labels_dir}")
        change_label_id(labels_dir, args.old_id, args.new_id)

    elif args.command == "crop":
        base_dir = Path(args.base_dir)
        if not base_dir.is_dir():
            raise SystemExit(f"Not a directory: {base_dir}")
        crop_by_label(base_dir)

    elif args.command == "clean":
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            raise SystemExit(f"Not a directory: {dataset_dir}")

        remove_ids = set(args.remove)
        rename_map: dict[str, str] = {}
        for pair in args.rename:
            if ":" not in pair:
                raise SystemExit(f"Invalid --rename value (expected old:new): {pair}")
            old, new = pair.split(":", 1)
            rename_map[old] = new

        overlap = remove_ids & rename_map.keys()
        if overlap:
            raise SystemExit(f"Class ID(s) listed in both --remove and --rename: {overlap}")
        if not remove_ids and not rename_map:
            raise SystemExit("Nothing to do: pass --remove and/or --rename.")

        clean_dataset(dataset_dir, remove_ids, rename_map)


if __name__ == "__main__":
    main()