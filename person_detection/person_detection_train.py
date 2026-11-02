"""
YOLOv12n Person Detection Training
====================================
Trains a YOLOv12n model for person detection (adult/child) using a YAML
dataset config file.

Expected dataset structure
--------------------------
    <dataset_root>/
        images/
            train/   *.jpg / *.png
            val/     *.jpg / *.png
        labels/
            train/   *.txt  (YOLO format: class cx cy w h)
            val/     *.txt

YAML file format (dataset.yaml)
--------------------------------
    path: /absolute/path/to/dataset_root
    train: images/train
    val:   images/val

    nc: 2
    names: ['adult', 'child']

Usage
-----
    python person_detection/person_detection_train.py --data dataset.yaml
    python person_detection/person_detection_train.py --data dataset.yaml --epochs 100 --imgsz 640
    python person_detection/person_detection_train.py --data dataset.yaml --resume   # resume from last.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data: str,
    model_path: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/train",
    name: str = "yolov8n_person-3_best_weight",
    resume: bool = False,
) -> None:
    """Train YOLOv8n on the dataset described by *data* YAML.

    Parameters
    ----------
    data:       Path to the dataset YAML file.
    model_path: Pretrained weights to fine-tune from.
    epochs:     Number of training epochs.
    imgsz:      Input image size (square).
    batch:      Batch size (-1 for auto-batch).
    device:     Device string: '' for auto, '0' for GPU 0, 'cpu' for CPU.
    project:    Root directory for saving run outputs.
    name:       Sub-directory name inside *project* for this run.
    resume:     Resume training from last checkpoint.
    """
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if resume:
        # Resume from the last saved checkpoint of the named run
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        print(f"Resuming from: {last_ckpt}")
    else:
        model = YOLO(model_path)
        print(f"Starting training with model: {model_path}")

    print(f"Dataset : {data_path}")
    print(f"Epochs  : {epochs}  |  Image size: {imgsz}  |  Batch: {batch}")
    print(f"Output  : {project}/{name}\n")

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=project,
        name=name,
        resume=resume,
        pretrained=True,
        verbose=True,
    )

    # Evaluate on val set after training
    print("\nRunning validation on best weights …")
    best_ckpt = Path(project) / name / "weights" / "best.pt"
    if best_ckpt.exists():
        best_model = YOLO(str(best_ckpt))
        metrics = best_model.val(data=str(data_path), imgsz=imgsz)
        print(f"mAP50     : {metrics.box.map50:.4f}")
        print(f"mAP50-95  : {metrics.box.map:.4f}")

    print(f"\nDone. Weights saved to: {Path(project) / name / 'weights'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv12n for adult/child person detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to dataset YAML file.")
    parser.add_argument(
        "--model", default="yolov12n.pt", help="Pretrained weights to start from."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (-1 for auto-batch)."
    )
    parser.add_argument(
        "--device", default="", help="Device: '' auto, '0' GPU 0, 'cpu' CPU."
    )
    parser.add_argument(
        "--project", default="runs/train", help="Directory to save training runs."
    )
    parser.add_argument(
        "--name", default="yolov8n_person", help="Name of this training run."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint."
    )
    args = parser.parse_args()

    train(
        data=args.data,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
