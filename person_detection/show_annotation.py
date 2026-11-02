"""
Annotation Visualizer
=====================
Given a single YOLO .txt annotation file, locates its paired image in a YOLO
template dataset (train/valid/test, each with images/ and labels/), draws the
bounding boxes on the image, and displays it.

Usage
-----
    python person_detection/show_annotation.py --label path/to/<split>/labels/<name>.txt
"""

import argparse
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Repo convention (see label_person.py): 0 = adult, 1 = child. BGR colors.
CLASS_NAMES = {0: "adult", 1: "child"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}
UNKNOWN_COLOR = (160, 160, 160)


def find_image(label_path: Path) -> Path:
    """Derive the image path from a YOLO label path by swapping the last
    'labels' path segment with 'images', then trying known extensions."""
    parts = list(label_path.parts)
    images_dir = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "labels":
            parts[i] = "images"
            images_dir = Path(*parts).parent
            break
    if images_dir is None:
        # Fall back to a sibling images/ folder next to the labels folder.
        images_dir = label_path.parent.parent / "images"

    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate

    raise SystemExit(
        f"No image found for {label_path.name} in {images_dir} "
        f"(tried extensions: {', '.join(IMAGE_EXTS)})"
    )


def yolo_to_xyxy(cx, cy, bw, bh, w, h):
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def show_annotation(label_path: Path, max_width: int, max_height: int) -> None:
    if not label_path.is_file():
        raise SystemExit(f"Annotation file not found: {label_path}")

    img_path = find_image(label_path)

    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    lines = label_path.read_text(encoding="utf-8").splitlines()

    drawn = 0
    class_counts: dict[int, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
        except ValueError:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS.get(cls, UNKNOWN_COLOR)
        name = CLASS_NAMES.get(cls, f"id{cls}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, name, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
        drawn += 1
        class_counts[cls] = class_counts.get(cls, 0) + 1

    summary = ", ".join(
        f"{CLASS_NAMES.get(c, f'id{c}')}={n}"
        for c, n in sorted(class_counts.items())
    ) or "none"
    print(f"Image: {img_path}")
    print(f"Boxes drawn: {drawn} ({summary})")
    if drawn == 0:
        print("Note: no valid annotations - showing the bare image.")

    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    window = "annotation"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, img)
    print("Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw YOLO bounding boxes from a .txt annotation onto its image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label", required=True,
        help="Path to a YOLO .txt annotation file.",
    )
    parser.add_argument(
        "--max_width", type=int, default=1600,
        help="Max display width before downscaling.",
    )
    parser.add_argument(
        "--max_height", type=int, default=900,
        help="Max display height before downscaling.",
    )
    args = parser.parse_args()

    show_annotation(Path(args.label), args.max_width, args.max_height)


if __name__ == "__main__":
    main()
