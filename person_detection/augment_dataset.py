"""Offline augmentation for a YOLO-format person-detection dataset.

Images and their YOLO label files live in the same folder, sharing a
basename and differing only by extension (``foo.jpg`` / ``foo.txt``).
Each label line is ``class cx cy w h`` with normalised coordinates.

For every source image this writes ``--n`` augmented copies (image + an
updated label file) into ``--out``. Geometric transforms (horizontal
flip, scale, translation) recompute the boxes; photometric transforms
(brightness/contrast, HSV jitter, blur, noise) leave them unchanged.
An image with a missing or empty label file is treated as a background
(negative) sample and still augmented.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_labels(label_path):
    """Return an (N, 5) array of [class, cx, cy, w, h]; empty if no boxes."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append([float(p) for p in parts])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def write_labels(label_path, labels):
    lines = [
        f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, cx, cy, w, h in labels
    ]
    label_path.write_text("\n".join(lines))


def yolo_to_xyxy(labels, w, h):
    """Normalised [class, cx, cy, w, h] -> pixel [class, x1, y1, x2, y2]."""
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cls = labels[:, 0:1]
    cx, cy, bw, bh = labels[:, 1] * w, labels[:, 2] * h, labels[:, 3] * w, labels[:, 4] * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.concatenate([cls, np.stack([x1, y1, x2, y2], axis=1)], axis=1)


def xyxy_to_yolo(boxes, w, h, min_size=2.0, min_visibility=0.20):
    """Clip pixel boxes to the frame, drop tiny/mostly-cropped ones."""
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cls = boxes[:, 0]
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    orig_area = np.maximum(x2 - x1, 1e-6) * np.maximum(y2 - y1, 1e-6)

    x1c = np.clip(x1, 0, w)
    y1c = np.clip(y1, 0, h)
    x2c = np.clip(x2, 0, w)
    y2c = np.clip(y2, 0, h)
    bw = x2c - x1c
    bh = y2c - y1c
    clip_area = np.maximum(bw, 0) * np.maximum(bh, 0)

    keep = (bw >= min_size) & (bh >= min_size) & (clip_area / orig_area >= min_visibility)
    cls, x1c, y1c, x2c, y2c = cls[keep], x1c[keep], y1c[keep], x2c[keep], y2c[keep]
    if len(cls) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cx = ((x1c + x2c) / 2) / w
    cy = ((y1c + y2c) / 2) / h
    nw = (x2c - x1c) / w
    nh = (y2c - y1c) / h
    return np.stack([cls, cx, cy, nw, nh], axis=1).astype(np.float32)


def affine_and_flip(img, boxes, scale_range, translate, flip_p):
    """Random scale + translation + optional horizontal flip.

    ``boxes`` is pixel [class, x1, y1, x2, y2]. The canvas size is kept
    constant; areas exposed by the transform are reflect-padded.
    """
    h, w = img.shape[:2]
    s = random.uniform(*scale_range)
    max_tx = translate * w
    max_ty = translate * h
    tx = random.uniform(-max_tx, max_tx)
    ty = random.uniform(-max_ty, max_ty)
    do_flip = random.random() < flip_p

    # Scale about the image centre, then translate.
    cx, cy = w / 2, h / 2
    M = np.array([
        [s, 0, tx + cx - s * cx],
        [0, s, ty + cy - s * cy],
    ], dtype=np.float32)

    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    if do_flip:
        img = img[:, ::-1].copy()

    if len(boxes) == 0:
        return img, boxes

    cls = boxes[:, 0:1]
    corners = np.stack([
        boxes[:, [1, 2]], boxes[:, [3, 4]],
        boxes[:, [1, 4]], boxes[:, [3, 2]],
    ], axis=1)  # (N, 4, 2)
    ones = np.ones((corners.shape[0], 4, 1), dtype=np.float32)
    hom = np.concatenate([corners, ones], axis=2)  # (N, 4, 3)
    tx_corners = hom @ M.T  # (N, 4, 2)

    x1 = tx_corners[:, :, 0].min(axis=1)
    y1 = tx_corners[:, :, 1].min(axis=1)
    x2 = tx_corners[:, :, 0].max(axis=1)
    y2 = tx_corners[:, :, 1].max(axis=1)

    if do_flip:
        x1, x2 = w - x2, w - x1

    out = np.concatenate([cls, np.stack([x1, y1, x2, y2], axis=1)], axis=1)
    return img, out


def photometric(img, brightness, contrast, hue, sat, blur_p, noise_p):
    img = img.astype(np.float32)

    alpha = 1.0 + random.uniform(-contrast, contrast)   # contrast
    beta = 255.0 * random.uniform(-brightness, brightness)  # brightness
    img = np.clip(img * alpha + beta, 0, 255)

    if hue > 0 or sat > 0:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue) * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + random.uniform(-sat, sat)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if random.random() < blur_p:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < noise_p:
        img = img + np.random.normal(0, random.uniform(4, 14), img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def augment_once(img, labels, args):
    h, w = img.shape[:2]
    boxes = yolo_to_xyxy(labels, w, h)
    img, boxes = affine_and_flip(
        img, boxes,
        scale_range=(args.scale_min, args.scale_max),
        translate=args.translate,
        flip_p=args.flip_p,
    )
    img = photometric(
        img,
        brightness=args.brightness, contrast=args.contrast,
        hue=args.hue, sat=args.sat,
        blur_p=args.blur_p, noise_p=args.noise_p,
    )
    new_h, new_w = img.shape[:2]
    new_labels = xyxy_to_yolo(
        boxes, new_w, new_h,
        min_visibility=args.min_visibility,
    )
    return img, new_labels


def find_images(src):
    src = Path(src)
    return sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    )


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(args.src)
    if not images:
        raise FileNotFoundError(f"No images found under: {args.src}")
    print(f"Found {len(images)} source images. "
          f"Writing {args.n}x augmentations to {out_dir}")

    total = 0
    skipped_empty = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ! unreadable, skipping: {img_path}")
            continue
        labels = read_labels(img_path.with_suffix(".txt"))

        if args.copy_original:
            stem = f"{img_path.stem}_orig"
            cv2.imwrite(str(out_dir / f"{stem}{img_path.suffix}"), img)
            write_labels(out_dir / f"{stem}.txt", labels)
            total += 1

        for i in range(args.n):
            aug_img, aug_labels = augment_once(img.copy(), labels.copy(), args)
            if len(labels) > 0 and len(aug_labels) == 0:
                # All boxes fell outside the frame – retry would help, but
                # keeping it as a hard negative is also acceptable.
                skipped_empty += 1
            stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(out_dir / f"{stem}{img_path.suffix}"), aug_img)
            write_labels(out_dir / f"{stem}.txt", aug_labels)
            total += 1

    print(f"Done. Wrote {total} images to {out_dir} "
          f"({skipped_empty} augmentations lost all boxes).")


def main():
    p = argparse.ArgumentParser(
        description="Offline bbox-aware augmentation for a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True,
                   help="Folder with images + same-name .txt labels "
                        "(searched recursively).")
    p.add_argument("--out", required=True, help="Output folder.")
    p.add_argument("--n", type=int, default=2,
                   help="Augmented copies per source image.")
    p.add_argument("--copy-original", action="store_true",
                   help="Also copy the untouched image+label into --out.")

    p.add_argument("--scale-min", type=float, default=0.8,
                   help="Min zoom factor (<1 zooms out).")
    p.add_argument("--scale-max", type=float, default=1.2,
                   help="Max zoom factor (>1 zooms in).")
    p.add_argument("--translate", type=float, default=0.10,
                   help="Max translation as a fraction of W/H.")
    p.add_argument("--flip-p", type=float, default=0.5,
                   help="Horizontal-flip probability.")

    p.add_argument("--brightness", type=float, default=0.20,
                   help="Max +/- brightness fraction.")
    p.add_argument("--contrast", type=float, default=0.20,
                   help="Max +/- contrast fraction.")
    p.add_argument("--hue", type=float, default=0.02,
                   help="Max +/- hue shift fraction.")
    p.add_argument("--sat", type=float, default=0.25,
                   help="Max +/- saturation fraction.")
    p.add_argument("--blur-p", type=float, default=0.10,
                   help="Gaussian-blur probability.")
    p.add_argument("--noise-p", type=float, default=0.15,
                   help="Gaussian-noise probability.")

    p.add_argument("--min-visibility", type=float, default=0.20,
                   help="Drop a box if less than this fraction of its "
                        "original area survives cropping.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
