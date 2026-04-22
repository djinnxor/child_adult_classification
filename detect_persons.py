"""
YOLOv8 Person Detection and Frame Extraction
=============================================
Processes a video file, detects persons using YOLOv8, and saves one frame per
second (1 FPS) whenever at least one person is present.  A YOLO-format
annotation text file is written alongside every saved image so the detections
can be used as a starting point for manual labelling.

Output structure
----------------
    <output_dir>/
        <video_stem>/
            <video_stem>_<frame_index>.jpg   – annotated frame
            <video_stem>_<frame_index>.txt   – YOLO-format detections
                                               (class cx cy w h conf)

Usage
-----
    python detect_persons.py path/to/video.mp4
    python detect_persons.py path/to/video.mp4 --output_dir frames --conf 0.4
    python detect_persons.py path/to/video.mp4 --model yolov8s.pt
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class index for "person"
PERSON_CLASS_ID = 0


def detect_and_save_frames(
    video_path: str,
    output_dir: str = "output_frames",
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.3,
) -> Path:
    """Run YOLOv8 person detection on *video_path* and persist 1-FPS frames.

    Parameters
    ----------
    video_path:      Path to the source video file.
    output_dir:      Root directory that will contain per-video sub-folders.
    model_path:      YOLOv8 model file or Hub shorthand (e.g. ``yolov8n.pt``).
    conf_threshold:  Minimum detection confidence to accept a person box.

    Returns
    -------
    Path to the folder where the extracted frames were saved.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = YOLO(model_path)

    video_name = video_path.stem
    save_dir = Path(output_dir) / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # How many raw frames to skip between saves so we get ~1 saved frame/second
    frame_interval = max(1, round(fps))

    print(f"Video     : {video_path.name}")
    print(f"FPS       : {fps:.2f}  |  Total frames: {total_frames}")
    print(f"Saving 1 frame per second (every {frame_interval} raw frames)")
    print(f"Output dir: {save_dir}\n")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, verbose=False)[0]

            # Keep only confident person detections
            person_boxes = [
                box
                for box in results.boxes
                if int(box.cls[0]) == PERSON_CLASS_ID
                and float(box.conf[0]) >= conf_threshold
            ]

            if person_boxes:
                img_filename = f"{video_name}_{frame_idx:06d}.jpg"
                txt_filename = f"{video_name}_{frame_idx:06d}.txt"

                # Save annotated frame (bounding boxes drawn by ultralytics)
                annotated = results.plot()
                cv2.imwrite(str(save_dir / img_filename), annotated)

                # Save YOLO-format annotation  (class cx cy w h conf)
                h, w = frame.shape[:2]
                with open(save_dir / txt_filename, "w", encoding="utf-8") as fh:
                    for box in person_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cx = (x1 + x2) / 2.0 / w
                        cy = (y1 + y2) / 2.0 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        # class 0 = unlabelled person; update to adult(0)/child(1)
                        # during the manual annotation step
                        fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")

                saved_count += 1
                if saved_count % 20 == 0:
                    print(f"  Saved {saved_count} frames …")

        frame_idx += 1

    cap.release()
    print(f"\nDone – saved {saved_count} frames to: {save_dir}")
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect persons in a video with YOLOv8 and save one annotated "
            "frame per second alongside a YOLO-format prediction text file."
        )
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--output_dir",
        default="output_frames",
        help="Root output directory (default: output_frames)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 weights file or model name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Person detection confidence threshold (default: 0.3)",
    )
    args = parser.parse_args()
    detect_and_save_frames(args.video_path, args.output_dir, args.model, args.conf)


if __name__ == "__main__":
    main()
