"""
YOLOv8 Person Detection and Frame Extraction
=============================================
Processes a video file, detects and tracks persons using a model, saves one
frame per second (1 FPS) whenever at least one person is present, and saves
the best (largest) crop per track ID into a 'cropped' subfolder.

Output structure
----------------
    <output_dir>/
        <video_stem>/
            <video_stem>_<frame_index>.jpg   – 1-FPS frame
            <video_stem>_<frame_index>.txt   – YOLO-format detections
                                               (class cx cy w h conf)
            cropped/
                <video_stem>_id<track_id>.jpg – best crop per person track

Usage
-----
    python detect_person.py --video_path path/to/video.mp4
    python detect_person.py --video_path path/to/video.mp4 --output_dir frames --conf 0.4
    python detect_person.py --video_path path/to/video.mp4 --model yolov12n.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class index for "person"
PERSON_CLASS_ID = 0
SCOOTER_CLASS_ID = 1
SAVE_CLASS_IDS = {PERSON_CLASS_ID, SCOOTER_CLASS_ID}


def detect_and_track(
    video_path: str,
    output_dir: str = "output_frames",
    model_path: str = "yolov12n.pt",
    conf_threshold: float = 0.5,
    interval_seconds: float = 5.0,
) -> Path:
    """Run YOLOv12n person detection+tracking on *video_path* in a single pass.

    Saves 1-FPS annotated frames with YOLO .txt annotations, and writes the
    largest-bbox crop seen per track ID into a 'cropped' subfolder.

    Parameters
    ----------
    video_path:      Path to the source video file.
    output_dir:      Root directory that will contain per-video sub-folders.
    model_path:      YOLOv12n model file (e.g. ``yolov12n.pt``).
    conf_threshold:  Minimum detection confidence to accept a person box.

    Returns
    -------
    Path to the folder where the extracted frames were saved.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = YOLO(model_path)
    print(f"Using YOLO model: {model_path}")

    video_name = video_path.stem
    save_dir = Path(output_dir) / video_name
    crops_dir = save_dir / "cropped"
    save_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(fps * interval_seconds))

    print(f"Video     : {video_path.name}")
    print(f"FPS       : {fps:.2f}  |  Total frames: {total_frames}")
    print(f"Saving 1 frame every {interval_seconds}s (every {frame_interval} raw frames)")
    print(f"Output dir: {save_dir}\n")

    frame_idx = 0
    saved_count = 0
    # Keep only the best area per track; write crop files immediately when improved.
    best_crop_areas: dict[int, float] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=conf_threshold)[0]

        save_boxes = [
            box
            for box in results.boxes
            if int(box.cls[0]) in SAVE_CLASS_IDS
            and float(box.conf[0]) >= conf_threshold
        ]

        person_boxes = [box for box in save_boxes if int(box.cls[0]) == PERSON_CLASS_ID]

        # --- track_crop logic: update best crop per track ID every frame ---
        if results.boxes.id is not None:
            h, w = frame.shape[:2]
            for box in person_boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    continue
                if track_id not in best_crop_areas or area > best_crop_areas[track_id]:
                    best_crop_areas[track_id] = area
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(str(crops_dir / f"{video_name}_id{track_id}.jpg"), crop)

        # --- detect_and_save_frames logic: save 1-FPS frame + .txt ---
        if frame_idx % frame_interval == 0 and save_boxes:
            img_filename = f"{video_name}_{frame_idx:06d}.jpg"
            txt_filename = f"{video_name}_{frame_idx:06d}.txt"

            cv2.imwrite(str(save_dir / img_filename), frame)

            h, w = frame.shape[:2]
            with open(save_dir / txt_filename, "w", encoding="utf-8") as fh:
                for box in save_boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")

            saved_count += 1
            if saved_count % 20 == 0:
                print(f"  Saved {saved_count} frames …")

        frame_idx += 1

    cap.release()

    print(f"\nDone – saved {saved_count} frame(s) to: {save_dir}")
    print(f"       saved {len(best_crop_areas)} crop(s)  to: {crops_dir}")
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and track persons in a video with YOLOv8n."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output_dir",
        default="output_frames",
        help="Root output directory (default: output_frames)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8n weights file or model name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Person detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between saved frames (default: 5.0)",
    )
    args = parser.parse_args()
    detect_and_track(args.video_path, args.output_dir, args.model, args.conf, args.interval)


if __name__ == "__main__":
    main()
