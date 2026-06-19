"""Real-time two-stage child/adult classification.

Stage 1: YOLO person detection + ByteTrack (Ultralytics built-in).
Stage 2: per-frame batched classification of every tracked crop, with
per-track temporal smoothing of softmax probabilities.
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from train_flat import build_model, get_transforms


DEFAULT_DETECTOR = "runs/detect/runs/train/yolov8n_person-11/weights/best.pt"
DEFAULT_CLASSIFIER = "runs/train/best_model.pth"
DEFAULT_DETECTOR_CLASSES = [0]  # 0 = person (works for COCO yolov8n.pt and the fine-tuned single-class detector)


def load_classifier(ckpt_path, device, img_size=224):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["backbone"], num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _, val_tf = get_transforms(img_size)
    print(f"Classifier: {ckpt['backbone']}  classes={classes}  "
          f"val_acc={ckpt.get('val_acc', float('nan')):.2f}%")
    return model, classes, val_tf


def crop_with_padding(frame, xyxy, pad=0.08):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


@torch.no_grad()
def classify_batch(crops_bgr, model, val_tf, device, use_amp=True):
    tensors = []
    for c in crops_bgr:
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        tensors.append(val_tf(Image.fromarray(rgb)))
    batch = torch.stack(tensors).to(device, non_blocking=True)

    if device.type == "cuda" and use_amp:
        with torch.amp.autocast(device_type="cuda"):
            logits = model(batch)
    else:
        logits = model(batch)

    return torch.softmax(logits.float(), dim=1).cpu().numpy()


class TrackSmoother:
    """Per-track-ID rolling softmax buffer with mean-of-softmax decision."""

    def __init__(self, classes, window=15, min_votes=5):
        self.classes = classes
        self.window = window
        self.min_votes = min_votes
        self.buf = {}
        self.last_seen = {}

    def update(self, track_id, probs, frame_idx):
        d = self.buf.get(track_id)
        if d is None:
            d = deque(maxlen=self.window)
            self.buf[track_id] = d
        d.append(probs)
        self.last_seen[track_id] = frame_idx

    def predict(self, track_id):
        d = self.buf.get(track_id)
        if d is None or len(d) < self.min_votes:
            return None, 0.0
        mean = np.mean(np.stack(d), axis=0)
        idx = int(np.argmax(mean))
        return self.classes[idx], float(mean[idx])

    def gc(self, frame_idx):
        cutoff = frame_idx - self.window
        stale = [tid for tid, seen in self.last_seen.items() if seen < cutoff]
        for tid in stale:
            self.buf.pop(tid, None)
            self.last_seen.pop(tid, None)


def color_for(label):
    if label is None:
        return (160, 160, 160)
    if label.lower().startswith("child"):
        return (255, 0, 255)
    return (0, 255, 0)


def run(video, output, detector_path, classifier_path, detector_classes,
        conf, pad, window, min_votes, min_area, imgsz, use_amp, show, device):
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = YOLO(detector_path)
    print(f"Detector: {detector_path}")

    model, classes, val_tf = load_classifier(classifier_path, device, imgsz)

    # Warm up CUDA kernels so the first real frame doesn't stall.
    if device.type == "cuda":
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    model(dummy)
            else:
                model(dummy)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
        print(f"Output: {output}")

    smoother = TrackSmoother(classes, window=window, min_votes=min_votes)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.track(
            frame,
            persist=True,
            verbose=False,
            conf=conf,
            classes=detector_classes,
        )[0]

        triples = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                crop = crop_with_padding(frame, (x1, y1, x2, y2), pad)
                if crop is None or crop.size == 0:
                    continue
                triples.append((tid, (x1, y1, x2, y2), crop))

        if triples:
            probs = classify_batch(
                [c for _, _, c in triples], model, val_tf, device, use_amp
            )
            for (tid, xyxy, _), p in zip(triples, probs):
                smoother.update(tid, p, frame_idx)
                label, score = smoother.predict(tid)
                x1, y1, x2, y2 = xyxy
                col = color_for(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                text = f"id{tid} {label or '...'} {score:.2f}"
                cv2.putText(
                    frame, text, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                )

        smoother.gc(frame_idx)

        if writer:
            writer.write(frame)
        if show:
            display = cv2.resize(frame, (1200, 700))
            cv2.imshow("classify_realtime", display)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames  "
                  f"(active tracks: {len(smoother.buf)})")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done. Total frames: {frame_idx}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time two-stage child/adult classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", default=None, help="Annotated output path.")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR,
                        help="YOLO person-detector weights.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help="Classifier checkpoint from train.py.")
    parser.add_argument("--detector-classes", type=int, nargs="+",
                        default=DEFAULT_DETECTOR_CLASSES,
                        help="Detector class indices to keep. "
                             "Use 0 for COCO yolov8n.pt or the single-class "
                             "fine-tuned person detector. Pass multiple to "
                             "filter (e.g. --detector-classes 0 1).")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detector confidence threshold.")
    parser.add_argument("--pad", type=float, default=0.08,
                        help="Bounding-box padding fraction before crop.")
    parser.add_argument("--window", type=int, default=15,
                        help="Smoothing window (frames).")
    parser.add_argument("--min-votes", type=int, default=5,
                        help="Frames required before showing a label.")
    parser.add_argument("--min-area", type=int, default=1024,
                        help="Skip detections smaller than this (px^2).")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Classifier input size.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision.")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated frames live.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run(
        video=args.video,
        output=args.output,
        detector_path=args.detector,
        classifier_path=args.classifier,
        detector_classes=args.detector_classes,
        conf=args.conf,
        pad=args.pad,
        window=args.window,
        min_votes=args.min_votes,
        min_area=args.min_area,
        imgsz=args.imgsz,
        use_amp=not args.no_amp,
        show=args.show,
        device=device,
    )


if __name__ == "__main__":
    main()
