"""
Auto-label a YOLO-template dataset with adult/child boxes.

Walks a dataset root (train/valid/test, each with images/ and labels/) and
runs the existing two-stage pipeline on every image:

    Stage 1: YOLO person detector  -> person bounding boxes
    Stage 2: PyTorch classifier    -> adult / child per crop

Predicted boxes are written in YOLO format (class cx cy bw bh, normalized)
and APPENDED to the matching <stem>.txt in the split's labels/ folder
(created if missing). Class IDs follow the repo convention: 0 = adult,
1 = child.

Usage
-----
    python person_detection/person_detection_images.py --dataset_dir dataset
    python person_detection/person_detection_images.py --dataset_dir dataset \\
        --detector runs/.../best.pt --classifier runs/train/best_model.pth \\
        --conf 0.6
"""

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from classify_realtime import (
    DEFAULT_CLASSIFIER,
    DEFAULT_DETECTOR,
    DEFAULT_DETECTOR_CLASSES,
    classify_batch,
    crop_with_padding,
    load_classifier,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "valid", "test")

# Repo convention (label_person.py): 0 = adult, 1 = child.
ADULT_ID, CHILD_ID = 0, 1


def class_name_to_id(name: str) -> int:
    """Map a classifier class string to the dataset's YOLO class id."""
    return CHILD_ID if name.lower().startswith("child") else ADULT_ID


def autolabel_image(
    img_path: Path,
    detector,
    model,
    classes: list[str],
    val_tf,
    device,
    detector_classes,
    conf: float,
    pad: float,
    min_area: int,
    use_amp: bool,
) -> list[str]:
    """Return YOLO label lines for one image (empty list if no detections)."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  Warning: cannot read {img_path}, skipping.")
        return []

    h, w = frame.shape[:2]
    results = detector(
        frame, conf=conf, classes=detector_classes, verbose=False
    )[0]

    boxes = []
    crops = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            crop = crop_with_padding(frame, (x1, y1, x2, y2), pad)
            if crop is None or crop.size == 0:
                continue
            boxes.append((x1, y1, x2, y2))
            crops.append(crop)

    if not crops:
        return []

    probs = classify_batch(crops, model, val_tf, device, use_amp)

    lines = []
    for (x1, y1, x2, y2), p in zip(boxes, probs):
        cls_id = class_name_to_id(classes[int(p.argmax())])
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # Clamp to the valid normalized range.
        cx, cy, bw, bh = (min(1.0, max(0.0, v)) for v in (cx, cy, bw, bh))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def append_label_file(labels_dir: Path, stem: str, new_lines: list[str]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    lbl_path = labels_dir / f"{stem}.txt"

    existing = ""
    if lbl_path.exists():
        existing = lbl_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"

    lbl_path.write_text(existing + "\n".join(new_lines) + "\n", encoding="utf-8")


def run(dataset_dir, detector_path, classifier_path, detector_classes,
        conf, pad, min_area, imgsz, use_amp, device) -> None:
    detector = YOLO(detector_path)
    print(f"Detector: {detector_path}")
    model, classes, val_tf = load_classifier(classifier_path, device, imgsz)

    found_split = False
    total_imgs = 0
    total_boxes = 0
    counts = {ADULT_ID: 0, CHILD_ID: 0}

    for split in SPLIT_NAMES:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir():
            continue
        found_split = True
        print(f"\n[{split}] {images_dir}")

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            total_imgs += 1
            lines = autolabel_image(
                img_path, detector, model, classes, val_tf, device,
                detector_classes, conf, pad, min_area, use_amp,
            )
            if lines:
                append_label_file(labels_dir, img_path.stem, lines)
                total_boxes += len(lines)
                for ln in lines:
                    counts[int(ln.split()[0])] += 1

            if total_imgs % 100 == 0:
                print(f"  Processed {total_imgs} images...")

    if not found_split:
        raise SystemExit(
            f"No '{'/'.join(SPLIT_NAMES)}' split with an 'images' folder "
            f"found in: {dataset_dir}"
        )

    print(
        f"\nDone - {total_imgs} image(s) processed, {total_boxes} box(es) "
        f"appended (adult={counts[ADULT_ID]}, child={counts[CHILD_ID]})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label a YOLO dataset with adult/child boxes "
                    "using the two-stage detector+classifier pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Dataset root with train/valid/test "
                             "(each having images/ and labels/).")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR,
                        help="YOLO person-detector weights.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help="Classifier checkpoint from train.py.")
    parser.add_argument("--detector-classes", type=int, nargs="+",
                        default=DEFAULT_DETECTOR_CLASSES,
                        help="Detector class indices to keep (0 = person).")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="Detector confidence threshold.")
    parser.add_argument("--pad", type=float, default=0.08,
                        help="Bounding-box padding fraction before crop.")
    parser.add_argument("--min-area", type=int, default=1024,
                        help="Skip detections smaller than this (px^2).")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Classifier input size.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run(
        dataset_dir=dataset_dir,
        detector_path=args.detector,
        classifier_path=args.classifier,
        detector_classes=args.detector_classes,
        conf=args.conf,
        pad=args.pad,
        min_area=args.min_area,
        imgsz=args.imgsz,
        use_amp=not args.no_amp,
        device=device,
    )


if __name__ == "__main__":
    main()
