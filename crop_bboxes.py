"""Crop YOLO bboxes from image+txt pairs into a single output folder.

Walks the input folder (optionally recursive), finds each image that has a
matching ``<stem>.txt`` next to it (YOLO format: ``class cx cy w h [conf]``,
normalized), and saves every bounding-box region as
``<stem>_box<idx>.jpg`` under the output folder.

Pairs the natural feeder for ``label_crops.py`` — run this on the frame dirs
produced by ``detect_person.py``, then label the resulting crops.
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_yolo_line(line):
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(p) for p in parts[1:5])
    except ValueError:
        return None
    return cls, cx, cy, bw, bh


def yolo_to_xyxy(cx, cy, bw, bh, w, h, pad=0.0):
    px, py = bw * pad, bh * pad
    x1 = int((cx - bw / 2 - px) * w)
    y1 = int((cy - bh / 2 - py) * h)
    x2 = int((cx + bw / 2 + px) * w)
    y2 = int((cy + bh / 2 + py) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def unique_path(out_dir, name):
    candidate = out_dir / name
    if not candidate.exists():
        return candidate
    stem, ext = candidate.stem, candidate.suffix
    i = 1
    while True:
        candidate = out_dir / f"{stem}__{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def crop_folder(input_dir, output_dir, recursive, pad, min_area):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(p for p in iterator
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f"No images found in: {input_dir}")
        return

    saved = skipped_no_txt = skipped_unreadable = skipped_small = 0

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            skipped_no_txt += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  unreadable: {img_path}")
            skipped_unreadable += 1
            continue

        h, w = img.shape[:2]
        for i, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines()):
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            _cls, cx, cy, bw, bh = parsed
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h, pad)
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < min_area:
                skipped_small += 1
                continue
            crop = img[y1:y2, x1:x2]
            out_path = unique_path(output_dir, f"{img_path.stem}_box{i}.jpg")
            cv2.imwrite(str(out_path), crop)
            saved += 1

    print(f"Done. saved={saved}  no_txt={skipped_no_txt}  "
          f"unreadable={skipped_unreadable}  too_small={skipped_small}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop YOLO bboxes from image+txt pairs into one output folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Folder containing image+txt pairs.")
    parser.add_argument("--output", "-o", required=True,
                        help="Folder to write all crops into.")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories.")
    parser.add_argument("--pad", type=float, default=0.0,
                        help="Padding fraction added around each bbox.")
    parser.add_argument("--min-area", type=int, default=0,
                        help="Skip crops smaller than this (pixels^2).")
    args = parser.parse_args()

    crop_folder(args.input, args.output, args.recursive, args.pad, args.min_area)


if __name__ == "__main__":
    main()
