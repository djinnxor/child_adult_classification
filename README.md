# Child / Adult Classification

Real-time classification of adults and children from low-resolution CCTV footage.
The pipeline has two stages:

1. **Frame extraction** (`detect_persons.py`) – runs YOLOv8 person detection on a video and saves annotated frames at 1 FPS for manual labelling.
2. **Classifier training** (`train.py`) – fine-tunes EfficientNet-B0, ResNet-18/50, or a YOLOv8 classification head on the labelled frames.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note:** for CUDA support install the matching `torch` wheel from <https://pytorch.org/get-started/locally/> before running `pip install -r requirements.txt`.

---

## 1 – Frame Extraction with YOLOv8 Person Detection

`detect_persons.py` processes a video file, detects every person using YOLOv8, and saves **one annotated frame per second** whenever at least one person is present.  A YOLO-format `.txt` file (one detection per line: `class cx cy w h conf`) is written next to each image so the files can serve as a starting point for manual annotation.

### Output layout

```
output_frames/
└── <video_name>/
    ├── <video_name>_000000.jpg   ← annotated frame
    ├── <video_name>_000000.txt   ← detections (YOLO format)
    ├── <video_name>_000025.jpg
    ├── <video_name>_000025.txt
    └── …
```

The `.txt` annotation format is:

```
<class_id> <cx> <cy> <w> <h> <conf>
```

`class_id` is `0` (unlabelled person) at extraction time. During the manual labelling step, update it to `0` (adult) or `1` (child).

### Usage

```bash
# Minimal – uses yolov8n.pt, saves to ./output_frames/
python detect_persons.py path/to/video.mp4

# Custom options
python detect_persons.py path/to/video.mp4 \
    --output_dir frames \
    --model yolov8s.pt \
    --conf 0.4
```

| Argument | Default | Description |
|---|---|---|
| `video_path` | — | Path to the input video file |
| `--output_dir` | `output_frames` | Root directory for saved frames |
| `--model` | `yolov8n.pt` | YOLOv8 weights (downloaded automatically) |
| `--conf` | `0.3` | Minimum detection confidence |

---

## 2 – Classifier Training

`train.py` trains a binary classifier (adult / child) using one of three backbone options.

### Dataset layout (ImageFolder-compatible)

```
dataset/
├── train/
│   ├── adult/   ← cropped person images labelled as adult
│   └── child/   ← cropped person images labelled as child
└── val/
    ├── adult/
    └── child/
```

> **Tip:** use the bounding-box coordinates saved by `detect_persons.py` to crop individual persons from the extracted frames, then sort them into `adult/` and `child/` folders.

### Backbones

| `--backbone` | Architecture | Notes |
|---|---|---|
| `efficientnet` | EfficientNet-B0 (default) | Best accuracy / size trade-off |
| `resnet50` | ResNet-50 | Strong baseline, larger |
| `resnet18` | ResNet-18 | Fastest PyTorch option |
| `yolov8` | YOLOv8-cls | End-to-end ultralytics pipeline |

### Usage

```bash
# EfficientNet-B0 (recommended)
python train.py --data_dir dataset --backbone efficientnet --epochs 50

# ResNet-50
python train.py --data_dir dataset --backbone resnet50 --epochs 50 --batch_size 16

# ResNet-18 (fastest)
python train.py --data_dir dataset --backbone resnet18 --epochs 50

# YOLOv8 classification head
python train.py --data_dir dataset --backbone yolov8 --yolo_model yolov8n-cls.pt --epochs 50
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | — | Dataset root (must contain `train/` and `val/`) |
| `--backbone` | `efficientnet` | Model backbone (see table above) |
| `--yolo_model` | `yolov8n-cls.pt` | YOLOv8 weights (only for `--backbone yolov8`) |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--img_size` | `224` | Input resolution |
| `--lr` | `0.001` | Initial learning rate |
| `--output_dir` | `runs/train` | Checkpoint output directory |

The best checkpoint is saved to `<output_dir>/best_model.pth` (PyTorch backends) or `<output_dir>/yolov8_cls/` (YOLOv8 backend).

---

## Training augmentations

The PyTorch training pipeline applies augmentations designed for low-resolution CCTV footage:

- Random crop after a slight upscale
- Random horizontal flip
- Colour jitter (brightness, contrast, saturation) – handles variable lighting
- Random rotation ±10°
- Gaussian blur – simulates motion / compression artefacts
