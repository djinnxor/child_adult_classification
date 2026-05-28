"""Offline augmentation for the child/adult crop-classification dataset.

The class label is carried in the image filename (the ``_adult`` /
``_child`` suffix written by ``label_crops.py``). These crops come from
person-detection boxes, so they are often small and blurry; before
augmenting, each crop is upscaled (Lanczos) to a minimum size and
sharpened with an unsharp mask to recover edge detail.

For every source image this writes ``--n`` augmented copies whose
filenames keep the same label token, so an ImageFolder/`split_dataset`
step downstream still works.
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_RE = re.compile(r"(child|adult)", re.IGNORECASE)


def label_of(path):
    """Pull 'child'/'adult' out of the filename; None if absent."""
    m = LABEL_RE.search(path.stem)
    return m.group(1).lower() if m else None


def upscale_sharpen(img, min_size, sharpen):
    """Lanczos-upscale so the short side >= min_size, then unsharp-mask."""
    h, w = img.shape[:2]
    short = min(h, w)
    if short < min_size:
        scale = min_size / short
        img = cv2.resize(
            img, (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    if sharpen > 0:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.0 + sharpen, blur, -sharpen, 0)
    return img


def resize_short(img, size):
    """Scale so the short side == `size`, preserving aspect ratio."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    return cv2.resize(
        img, (max(1, round(w * s)), max(1, round(h * s))),
        interpolation=cv2.INTER_LANCZOS4,
    )


def random_crop(img, scale=(0.7, 1.0)):
    """Sample a sub-region that keeps the source aspect ratio."""
    h, w = img.shape[:2]
    f = random.uniform(scale[0] ** 0.5, 1.0)
    ch, cw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return img[y:y + ch, x:x + cw]


def rotate(img, max_deg):
    if max_deg <= 0:
        return img
    h, w = img.shape[:2]
    ang = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )


def photometric(img, brightness, contrast, hue, sat):
    img = img.astype(np.float32)
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = 255.0 * random.uniform(-brightness, brightness)
    img = np.clip(img * alpha + beta, 0, 255)
    if hue > 0 or sat > 0:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue) * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + random.uniform(-sat, sat)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def augment_once(img, args):
    img = random_crop(img, scale=(args.scale_min, 1.0))
    img = resize_short(img, args.size)
    if random.random() < args.flip_p:
        img = img[:, ::-1].copy()
    img = rotate(img, args.rotate)
    img = photometric(
        img,
        brightness=args.brightness, contrast=args.contrast,
        hue=args.hue, sat=args.sat,
    )
    if random.random() < args.noise_p:
        img = np.clip(
            img.astype(np.float32)
            + np.random.normal(0, random.uniform(3, 10), img.shape),
            0, 255,
        ).astype(np.uint8)
    return img


def find_images(src):
    return sorted(
        p for p in Path(src).rglob("*")
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

    counts = {"child": 0, "adult": 0, None: 0}
    written = 0
    for img_path in images:
        lbl = label_of(img_path)
        counts[lbl] = counts.get(lbl, 0) + 1
        if lbl is None and not args.keep_unlabeled:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ! unreadable, skipping: {img_path}")
            continue
        base = upscale_sharpen(img, args.min_size, args.sharpen)
        tag = lbl if lbl else "unlabeled"

        if args.copy_original:
            dst = out_dir / f"{img_path.stem}_orig_{tag}{img_path.suffix}"
            cv2.imwrite(str(dst), resize_short(base, args.size))
            written += 1

        for i in range(args.n):
            aug = augment_once(base.copy(), args)
            dst = out_dir / f"{img_path.stem}_aug{i}_{tag}{img_path.suffix}"
            cv2.imwrite(str(dst), aug)
            written += 1

    print(f"Source: child={counts.get('child', 0)} "
          f"adult={counts.get('adult', 0)} "
          f"unlabeled={counts.get(None, 0)}")
    print(f"Done. Wrote {written} images to {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Upscale + augment child/adult person-crop dataset "
                    "(label read from filename).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True,
                   help="Folder of crops (searched recursively); label "
                        "taken from the '_child'/'_adult' filename token.")
    p.add_argument("--out", required=True, help="Output folder.")
    p.add_argument("--n", type=int, default=5,
                   help="Augmented copies per source image.")
    p.add_argument("--size", type=int, default=224,
                   help="Short-side output size; aspect ratio preserved.")
    p.add_argument("--copy-original", action="store_true",
                   help="Also write the upscaled, un-augmented crop.")
    p.add_argument("--keep-unlabeled", action="store_true",
                   help="Process images with no label token too.")

    p.add_argument("--min-size", type=int, default=160,
                   help="Upscale until the short side reaches this.")
    p.add_argument("--sharpen", type=float, default=0.6,
                   help="Unsharp-mask strength (0 disables).")

    p.add_argument("--scale-min", type=float, default=0.7,
                   help="Min area fraction for the random crop "
                        "(aspect ratio is kept).")
    p.add_argument("--flip-p", type=float, default=0.5,
                   help="Horizontal-flip probability.")
    p.add_argument("--rotate", type=float, default=8.0,
                   help="Max +/- rotation in degrees.")

    p.add_argument("--brightness", type=float, default=0.20,
                   help="Max +/- brightness fraction.")
    p.add_argument("--contrast", type=float, default=0.20,
                   help="Max +/- contrast fraction.")
    p.add_argument("--hue", type=float, default=0.02,
                   help="Max +/- hue shift fraction.")
    p.add_argument("--sat", type=float, default=0.25,
                   help="Max +/- saturation fraction.")
    p.add_argument("--noise-p", type=float, default=0.10,
                   help="Gaussian-noise probability.")

    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
