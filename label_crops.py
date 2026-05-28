"""Interactive labeler for person-crop images.

Walks one or more source directories (recursively), shows each crop in a cv2
window, and on keypress renames the file as ``<stem>_<adult|child><ext>`` and
moves it into a single collection folder. Designed for the case where crops
live in many sub-folders (e.g. one per source video) and you want a flat,
labeled pile to feed into ``train.py``'s ImageFolder layout afterwards.

Controls
--------
    a       Label as adult, move to <output>/
    c       Label as child, move to <output>/
    s       Skip (leave file in place, advance)
    u       Undo last labeling (restore the previous file from <output>/)
    q / Esc Quit

After labeling, sort the collected files into ``adult/`` and ``child/`` for
training (one-liner in ``--help``).
"""

import argparse
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "label_crops"
MAX_W, MAX_H = 1200, 900


def gather_images(sources):
    paths = []
    for src in sources:
        root = Path(src)
        if not root.exists():
            print(f"WARNING: source not found: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
    paths.sort()
    return paths


def fit_to_window(img):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def unique_dest(out_dir, src_path, label):
    """Return a non-colliding destination path under out_dir.

    Format: ``<stem>_<label><ext>``; on collision, append the parent folder
    name, then a numeric suffix.
    """
    stem, ext = src_path.stem, src_path.suffix
    candidate = out_dir / f"{stem}_{label}{ext}"
    if not candidate.exists():
        return candidate
    candidate = out_dir / f"{stem}_{src_path.parent.name}_{label}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = out_dir / f"{stem}_{src_path.parent.name}_{i}_{label}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def draw_overlay(img, idx, total, labeled, skipped, src_path):
    """Draw a status bar onto a copy of img and return it."""
    canvas = img.copy()
    h = canvas.shape[0]
    bar = f"[{idx + 1}/{total}]  labeled={labeled}  skipped={skipped}  {src_path.name}"
    cv2.rectangle(canvas, (0, h - 28), (canvas.shape[1], h), (0, 0, 0), -1)
    cv2.putText(canvas, bar, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "a=adult  c=child  s=skip  u=undo  q=quit",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return canvas


def run(sources, out_dir, copy_only):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = gather_images(sources)
    if not paths:
        print("No images found.")
        return
    print(f"Found {len(paths)} images across {len(sources)} source dir(s).")
    print(f"Output: {out_dir}  ({'copy' if copy_only else 'move'} mode)")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    history = []  # list of (original_src_path, dest_path) for undo
    labeled = skipped = 0
    i = 0
    while i < len(paths):
        src = paths[i]
        if not src.exists():
            # Already moved (e.g. by undo replay or external action)
            i += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"  unreadable, skipping: {src}")
            i += 1
            skipped += 1
            continue

        #view = draw_overlay(fit_to_window(img), i, len(paths), labeled, skipped, src)
        #cv2.imshow(WINDOW, view)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            skipped += 1
            i += 1
        elif key == ord("u"):
            if not history:
                print("  nothing to undo")
                continue
            orig, dest = history.pop()
            if dest.exists():
                if copy_only:
                    dest.unlink()
                else:
                    shutil.move(str(dest), str(orig))
            labeled = max(0, labeled - 1)
            # Step back to the file we just un-labeled if it's still in our list
            if orig in paths:
                i = paths.index(orig)
            print(f"  undid: {dest.name}")
        elif key in (ord("a"), ord("c")):
            label = "adult" if key == ord("a") else "child"
            dest = unique_dest(out_dir, src, label)
            if copy_only:
                shutil.copy2(str(src), str(dest))
            else:
                shutil.move(str(src), str(dest))
            history.append((src, dest))
            labeled += 1
            i += 1
            print(f"  {label:<5} -> {dest.name}")
        # any other key: redraw same image

    cv2.destroyAllWindows()
    print(f"\nDone. labeled={labeled}  skipped={skipped}  remaining={len(paths) - i}")
    print(f"Collected files in: {out_dir}")
    print("\nTo split into ImageFolder layout for train.py:")
    print(f"  mkdir -p dataset/adult dataset/child")
    print(f"  mv {out_dir}/*_adult.* dataset/adult/")
    print(f"  mv {out_dir}/*_child.* dataset/child/")


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard labeler: tag person crops as adult/child and "
                    "collect them into one folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sources", nargs="+",
                        help="One or more source directories (searched recursively).")
    parser.add_argument("--output", "-o", required=True,
                        help="Destination folder for labeled images.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of move (originals stay in place).")
    args = parser.parse_args()

    run(args.sources, args.output, args.copy)


if __name__ == "__main__":
    main()
