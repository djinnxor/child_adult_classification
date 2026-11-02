"""
Interactive YOLO Labeler (adult / scooter)
===========================================
Loads image + YOLO annotation pairs, lets the user edit bounding boxes and
toggle each box between **adult (0)** and **scooter (1)**, and overwrites
the original ``.txt`` with the corrected labels in standard YOLO format
(``class cx cy w h``).

Usage
-----
    python person_scooter_detection/label_person.py --images_dir output_frames/video_name
    python person_scooter_detection/label_person.py --images_dir images --labels_dir labels

Controls
--------
    Mouse
        Left-drag on empty area : draw a new adult box
        Left-click inside a box : select it
        Right-click inside box  : delete it

    Keyboard
        n / ->      Next image
        p / <-      Previous image
        s           Save current image's labels (overwrite .txt)
        t           Toggle selected box adult <-> scooter
        d / Del     Delete selected box
        u           Undo last edit on current image
        r           Reload from disk (discard unsaved edits)
        q / Esc     Quit
"""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ADULT, SCOOTER = 0, 1
CLASS_NAMES = {ADULT: "adult", SCOOTER: "scooter"}
CLASS_COLORS = {ADULT: (0, 200, 0), SCOOTER: (0, 165, 255)}  # BGR

MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900
MIN_BOX_PIXELS = 5

WINDOW_NAME = "label_person"


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def xyxy_px(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = int(round((self.cx - self.w / 2) * img_w))
        y1 = int(round((self.cy - self.h / 2) * img_h))
        x2 = int(round((self.cx + self.w / 2) * img_w))
        y2 = int(round((self.cy + self.h / 2) * img_h))
        return x1, y1, x2, y2

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def contains(self, nx: float, ny: float) -> bool:
        return (
            self.cx - self.w / 2 <= nx <= self.cx + self.w / 2
            and self.cy - self.h / 2 <= ny <= self.cy + self.h / 2
        )


def load_labels(path: Path) -> list[Box]:
    """Parse a YOLO ``.txt`` — tolerant of 5- or 6-field lines (drops ``conf``)."""
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        boxes.append(Box(cls, cx, cy, w, h))
    return boxes


def save_labels(path: Path, boxes: list[Box]) -> None:
    """Write standard YOLO detector format: ``class cx cy w h`` per line."""
    lines = [
        f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (image, label) pairs where the label file exists."""
    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            pairs.append((img_path, lbl))
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} image(s) without a matching .txt in {labels_dir}")
    return pairs


class Labeler:
    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        self.pairs = pairs
        self.idx = 0

        self.image = None  # original BGR image
        self.boxes: list[Box] = []
        self.history: list[list[Box]] = []  # undo stack for current image
        self.unsaved = False
        self.selected = -1

        # Mouse drag state (in image-pixel coords)
        self.dragging = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None

        # Display scaling
        self.display_scale = 1.0

        self._load_current()

    # ------------------------------------------------------------------ loading

    def _load_current(self) -> None:
        img_path, lbl_path = self.pairs[self.idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        self.image = img
        self.boxes = load_labels(lbl_path)
        self.history = []
        self.unsaved = False
        self.selected = -1
        self.dragging = False
        self.drag_start = self.drag_end = None

    def _push_history(self) -> None:
        self.history.append(copy.deepcopy(self.boxes))
        if len(self.history) > 100:
            self.history.pop(0)

    def undo(self) -> None:
        if not self.history:
            return
        self.boxes = self.history.pop()
        self.selected = -1
        self.unsaved = True

    # ---------------------------------------------------------------- saving/nav

    def save(self) -> None:
        _, lbl_path = self.pairs[self.idx]
        save_labels(lbl_path, self.boxes)
        self.unsaved = False
        print(f"Saved {len(self.boxes)} box(es) -> {lbl_path}")

    def goto(self, new_idx: int) -> None:
        new_idx = max(0, min(len(self.pairs) - 1, new_idx))
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self._load_current()

    # -------------------------------------------------------------- box editing

    def _box_at(self, nx: float, ny: float) -> int:
        """Return index of the smallest box containing the normalized point."""
        best = -1
        best_area = float("inf")
        for i, b in enumerate(self.boxes):
            if b.contains(nx, ny):
                a = b.area()
                if a < best_area:
                    best = i
                    best_area = a
        return best

    def select_at(self, nx: float, ny: float) -> None:
        self.selected = self._box_at(nx, ny)

    def delete_selected(self) -> None:
        if 0 <= self.selected < len(self.boxes):
            self._push_history()
            del self.boxes[self.selected]
            self.selected = -1
            self.unsaved = True

    def delete_at(self, nx: float, ny: float) -> None:
        i = self._box_at(nx, ny)
        if i >= 0:
            self._push_history()
            del self.boxes[i]
            if self.selected == i:
                self.selected = -1
            elif self.selected > i:
                self.selected -= 1
            self.unsaved = True

    def toggle_selected(self) -> None:
        if 0 <= self.selected < len(self.boxes):
            self._push_history()
            b = self.boxes[self.selected]
            b.cls = SCOOTER if b.cls == ADULT else ADULT
            self.unsaved = True

    def add_box_px(self, x1: int, y1: int, x2: int, y2: int) -> None:
        h, w = self.image.shape[:2]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if (x2 - x1) < MIN_BOX_PIXELS or (y2 - y1) < MIN_BOX_PIXELS:
            return
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        self._push_history()
        self.boxes.append(Box(ADULT, cx, cy, bw, bh))
        self.selected = len(self.boxes) - 1
        self.unsaved = True

    # ----------------------------------------------------------------- rendering

    def _compute_scale(self) -> float:
        h, w = self.image.shape[:2]
        scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
        return scale

    def _display_to_image(self, dx: int, dy: int) -> tuple[int, int]:
        s = self.display_scale or 1.0
        return int(round(dx / s)), int(round(dy / s))

    def render(self) -> "cv2.Mat":
        img = self.image.copy()
        h, w = img.shape[:2]

        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b.xyxy_px(w, h)
            color = CLASS_COLORS.get(b.cls, (255, 255, 255))
            thickness = 3 if i == self.selected else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            tag = CLASS_NAMES.get(b.cls, str(b.cls))
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tag_y = max(0, y1 - th - 6)
            cv2.rectangle(
                img, (x1, tag_y), (x1 + tw + 6, tag_y + th + 6), color, -1
            )
            cv2.putText(
                img,
                tag,
                (x1 + 3, tag_y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw in-progress drag rectangle
        if self.dragging and self.drag_start and self.drag_end:
            cv2.rectangle(
                img, self.drag_start, self.drag_end, (255, 255, 0), 2, cv2.LINE_AA
            )

        self._draw_hud(img)

        self.display_scale = self._compute_scale()
        if self.display_scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * self.display_scale), int(h * self.display_scale)),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _draw_hud(self, img) -> None:
        img_path, _ = self.pairs[self.idx]
        n_adult = sum(1 for b in self.boxes if b.cls == ADULT)
        n_scooter = sum(1 for b in self.boxes if b.cls == SCOOTER)
        unsaved_mark = "  *UNSAVED*" if self.unsaved else ""
        top = (
            f"{img_path.name}   [{self.idx + 1}/{len(self.pairs)}]   "
            f"adult:{n_adult}  scooter:{n_scooter}{unsaved_mark}"
        )
        footer = (
            "n/->:next  p/<-:prev  s:save  t:toggle  d:del  "
            "u:undo  r:reload  q/Esc:quit   (L-drag:new  R-click:delete)"
        )
        self._text_with_bg(img, top, (10, 10))
        h = img.shape[0]
        self._text_with_bg(img, footer, (10, h - 30))

    @staticmethod
    def _text_with_bg(img, text: str, origin: tuple[int, int]) -> None:
        x, y = origin
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x - 4, y), (x + tw + 4, y + th + 10), (0, 0, 0), -1)
        cv2.putText(
            img,
            text,
            (x, y + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------- mouse

def make_mouse_callback(labeler: Labeler):
    def on_mouse(event, x, y, flags, _param):
        ix, iy = labeler._display_to_image(x, y)
        h, w = labeler.image.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))

        if event == cv2.EVENT_LBUTTONDOWN:
            labeler.dragging = True
            labeler.drag_start = (ix, iy)
            labeler.drag_end = (ix, iy)

        elif event == cv2.EVENT_MOUSEMOVE and labeler.dragging:
            labeler.drag_end = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP and labeler.dragging:
            labeler.dragging = False
            x1, y1 = labeler.drag_start or (ix, iy)
            x2, y2 = (ix, iy)
            labeler.drag_start = labeler.drag_end = None
            if abs(x2 - x1) < MIN_BOX_PIXELS and abs(y2 - y1) < MIN_BOX_PIXELS:
                # Treat as a click → select
                labeler.select_at(ix / w, iy / h)
            else:
                labeler.add_box_px(x1, y1, x2, y2)

        elif event == cv2.EVENT_RBUTTONDOWN:
            labeler.delete_at(ix / w, iy / h)

    return on_mouse


# ----------------------------------------------------------------- unsaved gate

def confirm_discard(labeler: Labeler) -> bool:
    """Show a blocking prompt when trying to leave with unsaved changes.

    Returns True if the caller should proceed (saved or discarded),
    False if cancelled.
    """
    if not labeler.unsaved:
        return True

    while True:
        img = labeler.render()
        overlay = img.copy()
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        msg1 = "Unsaved changes on this image."
        msg2 = "s: save    n: discard    Esc: cancel"
        cv2.putText(
            img, msg1, (40, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, msg2, (40, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        if k in (ord("s"), ord("S")):
            labeler.save()
            return True
        if k in (ord("n"), ord("N")):
            return True
        if k == 27:  # Esc
            return False


# ------------------------------------------------------------------------- main

def run(images_dir: Path, labels_dir: Path) -> None:
    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        raise SystemExit(
            f"No image/label pairs found.\n"
            f"  images_dir: {images_dir}\n"
            f"  labels_dir: {labels_dir}"
        )
    print(f"Loaded {len(pairs)} image/label pair(s) from {images_dir}")

    labeler = Labeler(pairs)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(labeler))

    # Arrow key codes vary by platform / OpenCV build; accept common variants.
    LEFT_KEYS = {65361, 2424832, 81}
    RIGHT_KEYS = {65363, 2555904, 83}
    DEL_KEYS = {65535, 3014656}

    while True:
        cv2.imshow(WINDOW_NAME, labeler.render())
        key = cv2.waitKeyEx(30)
        if key == -1:
            # Allow closing via the window X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                if confirm_discard(labeler):
                    break
            continue

        if key in (ord("q"), ord("Q"), 27):  # Esc
            if confirm_discard(labeler):
                break
        elif key in (ord("n"), ord("N")) or key in RIGHT_KEYS:
            if confirm_discard(labeler):
                labeler.goto(labeler.idx + 1)
        elif key in (ord("p"), ord("P")) or key in LEFT_KEYS:
            if confirm_discard(labeler):
                labeler.goto(labeler.idx - 1)
        elif key in (ord("s"), ord("S")):
            labeler.save()
        elif key in (ord("t"), ord("T")):
            labeler.toggle_selected()
        elif key in (ord("d"), ord("D")) or key in DEL_KEYS:
            labeler.delete_selected()
        elif key in (ord("u"), ord("U")):
            labeler.undo()
        elif key in (ord("r"), ord("R")):
            if confirm_discard(labeler):
                labeler._load_current()

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive labeler for YOLO person annotations (adult/child).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images_dir", required=True, help="Folder containing images."
    )
    parser.add_argument(
        "--labels_dir",
        default=None,
        help="Folder containing matching YOLO .txt files (defaults to --images_dir).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else images_dir
    if not images_dir.is_dir():
        raise SystemExit(f"images_dir is not a directory: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"labels_dir is not a directory: {labels_dir}")

    run(images_dir, labels_dir)


if __name__ == "__main__":
    main()
