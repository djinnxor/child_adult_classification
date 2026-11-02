# Child / Adult Classification

This repo grew two **distinct but related** purposes over time. Both start from
video/images of people and end up deciding "child or adult" — but they solve
it differently, on different data, and each has its own labeling + training
pipeline:

| | **A. CCTV person detection + classification** | **B. Face age / gender regression (MiVOLO)** |
|---|---|---|
| Input | Full CCTV frames (people at a distance, low-res) | Face crops (eKYC-style, close-up) |
| Approach | YOLO detects people, a classifier (or the detector itself) labels each box adult/child | A MiVOLO (VOLO-D1) model predicts a continuous **age** + **gender** per face; "child" is then just "predicted age < threshold" |
| Labeling scripts | `detect_person.py`, `person_detection_images.py`, `label_person.py`, `label_crops.py`, `label_fixer.py`, `crop_bboxes.py`, `show_annotation.py`, `split_dataset.py` | `ekyc_age_gender.py` (`predict` / `review` / `export` / `package`) |
| Training scripts | `person_detection_train.py` (YOLO detector), `train.py` / `train_flat.py` (classifier), `augment_classify.py`, `eval_classifier.py` | `train_mivolo.py` |
| Inference scripts | `person_inference.py`, `classify_realtime.py` | `infer_mivolo.py` |

If you're picking this repo up fresh: figure out which of the two problems
you actually have (sparse/distant CCTV people → **A**; close-up face photos
where you want an actual age estimate, not just a binary label → **B**)
before reaching for a script, since the two pipelines are not interchangeable
and expect different dataset layouts.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note:** for CUDA support install the matching `torch` wheel from <https://pytorch.org/get-started/locally/> before running `pip install -r requirements.txt`.

---

# Part A — CCTV Person Detection & Adult/Child Classification

Real-time classification of adults and children from low-resolution CCTV
footage. Two stages: a YOLO person/box detector, then a classifier (or the
detector itself, trained with adult/child classes directly) decides adult vs
child per box.

## A.1 – Labeling pipeline

| Script | Purpose |
|---|---|
| `detect_person.py` | Runs YOLOv8 person detection/tracking on a video, saves 1 annotated frame per second (whenever a person is present) plus a YOLO `.txt` per frame, as a starting point for manual labelling. |
| `person_detection_images.py` | Auto-labels a YOLO-template dataset (`train/valid/test`, each with `images/` + `labels/`) with adult/child boxes using the existing two-stage pipeline, instead of labelling by hand. |
| `label_person.py` | Interactive labeler: load image + YOLO annotation pairs, edit boxes, toggle each box between **adult (0)** and **child (1)**. |
| `label_crops.py` | Interactive labeler for already-cropped person images: shows each crop, keypress renames it `<stem>_adult<ext>` / `<stem>_child<ext>`. |
| `label_fixer.py` | Bulk-sets the class ID in a folder of YOLO `.txt` files (and can crop boxes into `adult/` / `child/` folders). |
| `crop_bboxes.py` | Crops YOLO boxes from image+txt pairs into a single output folder (feeds `label_crops.py` / `train_flat.py`). |
| `show_annotation.py` | Visualizes a single YOLO `.txt` annotation against its paired image, for sanity-checking a dataset. |
| `split_dataset.py` | Builds YOLO `train/valid/test` splits. |

### Output layout of `detect_person.py`

```
output_frames/
└── <video_name>/
    ├── <video_name>_000000.jpg   ← annotated frame
    ├── <video_name>_000000.txt   ← detections (YOLO format)
    ├── <video_name>_000025.jpg
    ├── <video_name>_000025.txt
    └── …
```

`.txt` format: `<class_id> <cx> <cy> <w> <h> <conf>`. `class_id` is `0`
(unlabelled person) at extraction time — update it to `0` (adult) or `1`
(child) during manual labelling (`label_person.py`), or let
`person_detection_images.py` pre-fill it.

### Usage

```bash
python detect_person.py path/to/video.mp4 --output_dir frames --model yolov8s.pt --conf 0.4
```

## A.2 – Training

Two training options exist for this half of the repo, depending on whether
you want a **detector** that outputs adult/child boxes directly, or a
**classifier** that labels pre-cropped person images.

### Detector (`person_detection_train.py`)

Trains a YOLOv12n model to detect `adult` / `child` boxes directly from a
YAML dataset config.

```
<dataset_root>/
    images/{train,val}/   *.jpg / *.png
    labels/{train,val}/   *.txt   (YOLO format: class cx cy w h)
```

```yaml
# dataset.yaml
path: /absolute/path/to/dataset_root
train: images/train
val:   images/val
nc: 2
names: ['adult', 'child']
```

```bash
python person_detection_train.py --data dataset.yaml --epochs 100 --imgsz 640
```

### Classifier on cropped persons (`train.py` / `train_flat.py`)

`train.py` trains a binary classifier (adult / child) on an
ImageFolder-style dataset:

```
dataset/
├── train/
│   ├── adult/
│   └── child/
└── val/
    ├── adult/
    └── child/
```

| `--backbone` | Architecture |
|---|---|
| `efficientnet` (default) | EfficientNet-B0 |
| `resnet50` / `resnet18` | ResNet |
| `yolov8` | YOLOv8-cls |

```bash
python train.py --data_dir dataset --backbone efficientnet --epochs 50
```

`train_flat.py` is the alternative for a **flat** folder of crops where the
label is carried in the filename (`..._adult.jpg` / `..._child.jpg`, the
convention `label_crops.py` writes):

```bash
python train_flat.py --data_dir crops/ --backbone efficientnet_v2_s --epochs 40
```

`augment_classify.py` offline-augments that same filename-labelled crop
dataset (upscale + augmentation) before training; `eval_classifier.py`
evaluates a trained classifier checkpoint on a val split.

### Training augmentations

The PyTorch training pipeline applies augmentations designed for
low-resolution CCTV footage: random crop after a slight upscale, random
horizontal flip, colour jitter, random rotation ±10°, and Gaussian blur
(simulates motion/compression artefacts).

## A.3 – Inference

- `classify_realtime.py` — real-time two-stage pipeline: YOLO person
  detection + ByteTrack, then per-frame batched adult/child classification of
  every tracked crop.
- `person_inference.py` — runs a trained detector on a video and draws boxes.
  **Note:** its current `MODEL_PATH` points at a person/scooter detector, not
  an adult/child one — repoint it at your adult/child weights
  (`runs/train/<name>/weights/best.pt`) before using it for this purpose.

---

# Part B — Face Age & Gender Prediction (MiVOLO, eKYC)

A face-only MiVOLO (VOLO-D1) model that predicts a **continuous age** and
**gender** from a face crop, fine-tuned on eKYC customer photos. "Child" here
is derived from the predicted age (see `CHILD_AGE` / `child_recall` /
`child_precision` metrics in `train_mivolo.py`), not a separate class — this
is a regression model, not a binary classifier.

## B.1 – Labeling pipeline (`ekyc_age_gender.py`)

Works on an eKYC results folder of small (~112×112) face crops, one
sub-folder per customer ID:

```
<root>/matched/6000000/2026-02-02_16-15-15_image1.jpg
<root>/unmatched/6000004/2026-02-02_16-46-44_image1.jpg
```

Three steps (plus `package` to hand the result to another machine):

```bash
python ekyc_age_gender.py predict --root ~/Documents/Minje/ekyc_results   # pre-annotate with a MiVOLO checkpoint
python ekyc_age_gender.py review  --root ~/Documents/Minje/ekyc_results   # OpenCV window, accept/correct each label
python ekyc_age_gender.py export  --valid 0.15 --test 0.15                # -> ekyc_{train,valid,test}.csv, IMDB-clean layout
```

`review` keys: Enter/Space confirm, `0-9`/Backspace type a new age, `m`/`f`
set gender, `x` reject (unusable), `n`/`b` next/back, `u` undo, `q`/Esc save
& quit. Labels (`annotations.csv`, journal, per-person folders) contain
customer IDs and are gitignored (`ekyc_labels/`) — never commit them.

## B.2 – Training (`train_mivolo.py`)

Trains / fine-tunes / evaluates the face-only VOLO-D1 model on IMDB-clean-
or eKYC-format CSVs (`filename,age,gender,x_min,y_min,x_max,y_max,...`).

```bash
# fine-tune the IMDB-pretrained model on reviewed eKYC images
python train_mivolo.py --data-root ekyc_labels --images <ekyc_root> --csv-pattern ekyc_{split}.csv \
    --init models/mivolo_volo_d1_face_imdb.pth.tar --epochs 30 --warmup-epochs 3 --lr 1e-5 \
    --batch-size 32 --output runs/ekyc_finetune_1

# evaluate a checkpoint on the test split
python train_mivolo.py --data-root ekyc_labels --images <ekyc_root> --csv-pattern ekyc_{split}.csv \
    --eval-only --split test --init runs/ekyc_finetune_1/best.pth.tar
```

Checkpoints:

- `models/mivolo_volo_d1_face_imdb.pth.tar` — official IMDB-clean-pretrained starting point.
- `runs/mivolo_face/best.pth.tar` — **current best checkpoint**, fine-tuned on the eKYC dataset (see `runs/mivolo_face/args.json` for the exact run config; data lived at `/mnt/Transcend/ekyc_age_gender_20261002/`).

Checkpoint format: `{state_dict, min_age, max_age, avg_age, no_gender,
with_persons_model, epoch, metrics, args}`. Predicted
age = `out[:, 2] * (max_age - min_age) + avg_age`; gender =
`argmax(out[:, :2])` (0 = male, 1 = female).

## B.3 – Inference (`infer_mivolo.py`)

Runs a checkpoint on a single image (defaults to
`runs/mivolo_face/best.pth.tar`). By default the whole image is treated as
the face crop (same convention as the eKYC folders); pass `--box` if it's a
full photo and you already know the face's pixel box.

```bash
python infer_mivolo.py path/to/face_crop.jpg
python infer_mivolo.py path/to/photo.jpg --box x0 y0 x1 y1
```

---

## Shared infrastructure

- `tracking.py` — experiment tracking used by both `train_mivolo.py` and
  `ekyc_age_gender.py`: the personal mlops tracker (lineage tree under
  `$MLOPS_DOCS`) plus optional Weights & Biases.
- `mov_mp4.py` — video format conversion helper.
