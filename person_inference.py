import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Multi-class detector: 0 -> person, 1 -> scooter (finetuned from person-only weights
# on a person + scooter mix dataset).
MODEL_PATH = "/home/rookie/Documents/Minje/child_adult_classification/runs/detect/runs/train/yolov8n_person-7/weights/best.pt"

# BGR colors per class id; falls back to white for unknown ids.
CLASS_COLORS = {
    0: (0, 255, 0),     # person  -> green
    1: (0, 165, 255),   # scooter -> orange
}
DEFAULT_COLOR = (255, 255, 255)
DISPLAY_SCALE = 0.6


def _draw_detections(frame, boxes, names, conf_threshold):
    count_by_class = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = names.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        count_by_class[label] = count_by_class.get(label, 0) + 1
    return count_by_class


def run_inference(
    video_path: str,
    output_path: str = None,
    model_path: str = MODEL_PATH,
    conf_threshold: float = 0.5,
    show: bool = False,
) -> None:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)[0]
        _draw_detections(frame, results.boxes, model.names, conf_threshold)

        if writer:
            writer.write(frame)

        if show:
            display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Inference", display_frame)
            wait_ms = max(1, int(1000 / fps))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Done. Total frames processed: {frame_idx}")


def run_image_inference(
    image_path: str,
    output_path: str = None,
    model_path: str = MODEL_PATH,
    conf_threshold: float = 0.5,
    show: bool = True,
) -> None:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    results = model(img, conf=conf_threshold, verbose=False)[0]
    counts = _draw_detections(img, results.boxes, model.names, conf_threshold)

    if counts:
        summary = ", ".join(f"{n} {k}" for k, n in counts.items())
        print(f"Detected in {image_path.name}: {summary}")
    else:
        print(f"No detections in {image_path.name}")

    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Output image: {output_path}")

    if show:
        display_img = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Inference", display_img)
        print("Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run person + scooter detection inference on a video or image."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", help="Path to input video")
    src.add_argument("--image_path", help="Path to input image")
    parser.add_argument("--output_path", default=None, help="Path to save annotated output (optional)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display result while processing")
    args = parser.parse_args()

    if args.image_path:
        # For a single image, show by default unless an output path is given.
        show = args.show or not args.output_path
        run_image_inference(args.image_path, args.output_path, args.model, args.conf, show)
    else:
        run_inference(args.video_path, args.output_path, args.model, args.conf, args.show)


if __name__ == "__main__":
    main()
