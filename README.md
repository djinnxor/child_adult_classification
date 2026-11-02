# Child / Adult Classification

This repo holds **two main, distinct purposes**, plus one smaller/legacy
piece that shares tooling with the first. Both main purposes start from
video/images of people and end up deciding "child or adult" — but they solve
it differently, on different data, and each has its own labeling + training
pipeline, now split into its own folder:

| | **[`person_detection/`](person_detection/) — A. CCTV person detection + classification** | **[`face_age_gender/`](face_age_gender/) — B. Face age/gender regression (MiVOLO)** |
|---|---|---|
| Input | Full CCTV frames (people at a distance, low-res) | Face crops (eKYC-style, close-up) |
| Approach | A classifier (or a YOLO detector trained directly with adult/child classes) labels each person box adult/child | A MiVOLO (VOLO-D1) model predicts a continuous **age** + **gender** per face; "child" is then just "predicted age < threshold" |
| Labeling | `person_detection_images.py`, `label_crops.py`, `label_fixer.py`, `crop_bboxes.py`, `show_annotation.py`, `split_dataset.py` | `ekyc_age_gender.py` (`predict` / `review` / `export` / `package`) |
| Training | `person_detection_train.py` (detector), `train.py` / `train_flat.py` (classifier), `augment_classify.py`, `eval_classifier.py` | `train_mivolo.py` |
| Inference | `classify_realtime.py` | `infer_mivolo.py` |

There's also **[`person_scooter_detection/`](person_scooter_detection/) — a
separate, smaller side project** (detecting people vs. scooters, not
adult/child) that reuses this repo's generic video/box-labeling tooling. It
is **not** one of the two main purposes above — see its own section below —
but if you're hunting for `detect_person.py` or `label_person.py`, they live
there now, because that's what they're currently wired for (see the note in
that section).

If you're picking this repo up fresh: figure out which problem you actually
have (sparse/distant CCTV people → **A**; close-up face photos where you want
an actual age estimate, not just a binary label → **B**) before reaching for
a script, since the two pipelines expect different dataset layouts and are
not interchangeable. All commands below assume you run `python` **from the
repo root** (so relative defaults like `runs/...` and `models/...` resolve
correctly) and just give the subfolder path to the script.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note:** for CUDA support install the matching `torch` wheel from <https://pytorch.org/get-started/locally/> before running `pip install -r requirements.txt`.

---

# A — [`person_detection/`](person_detection/): CCTV Person Detection & Adult/Child Classification

Real-time classification of adults and children from low-resolution CCTV
footage. Two stages: a person/box detector, then a classifier (or the
detector itself, trained with adult/child classes directly) decides adult vs
child per box.

## A.1 – Labeling pipeline

| Script | Purpose |
|---|---|
| `person_detection_images.py` | Auto-labels a YOLO-template dataset (`train/valid/test`, each with `images/` + `labels/`) with adult/child boxes using the existing two-stage pipeline, instead of labelling by hand. |
| `label_crops.py` | Interactive labeler for already-cropped person images: shows each crop, keypress renames it `<stem>_adult<ext>` / `<stem>_child<ext>`. |
| `label_fixer.py` | Bulk-sets the class ID in a folder of YOLO `.txt` files (and can crop boxes into `adult/` / `child/` folders). |
| `crop_bboxes.py` | Crops YOLO boxes from image+txt pairs into a single output folder (feeds `label_crops.py` / `train_flat.py`). Class-agnostic. |
| `show_annotation.py` | Visualizes a single YOLO `.txt` annotation against its paired image, for sanity-checking a dataset. |
| `split_dataset.py` | Builds YOLO `train/valid/test` splits. Class-agnostic. |
| `augment_dataset.py` | Offline augmentation for a YOLO-format detection dataset (image + `.txt` label pairs): geometric transforms recompute boxes, photometric ones leave them unchanged. Class-agnostic — also reusable for Part C's dataset. |

> **Note:** this repo used to also have a frame-extraction + interactive
> box-editing step for fresh CCTV video (`detect_person.py` +
> `label_person.py`). Both have since been repurposed for the unrelated
> person/scooter side project (see Part C below) and are currently hardcoded
> for that class pair, not adult/child — they now live in
> [`person_scooter_detection/`](person_scooter_detection/). If you need to
> label a brand-new video for the adult/child task, start from those two
> scripts and change their class IDs back, or write a fresh extraction step;
> the labeling scripts above all assume you already have images + boxes (or
> crops) to start from.

### Usage

```bash
python person_detection/person_detection_images.py --dataset_dir dataset
```

## A.2 – Training

Two training options exist for this half of the repo, depending on whether
you want a **detector** that outputs adult/child boxes directly, or a
**classifier** that labels pre-cropped person images.

### Detector (`person_detection_train.py`)

Trains a YOLO model to detect `adult` / `child` boxes directly from a YAML
dataset config. This script is class-agnostic — it just trains whatever
2-class YAML you give it — so it's also the same script used to train the
person/scooter detector in Part C, pointed at a different YAML.

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
python person_detection/person_detection_train.py --data dataset.yaml --epochs 100 --imgsz 640
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
python person_detection/train.py --data_dir dataset --backbone efficientnet --epochs 50
```

`train_flat.py` is the alternative for a **flat** folder of crops where the
label is carried in the filename (`..._adult.jpg` / `..._child.jpg`, the
convention `label_crops.py` writes):

```bash
python person_detection/train_flat.py --data_dir crops/ --backbone efficientnet_v2_s --epochs 40
```

`augment_classify.py` offline-augments that same filename-labelled crop
dataset (upscale + augmentation) before training; `eval_classifier.py`
evaluates a trained classifier checkpoint on a val split (both import
`build_model`/`get_transforms` from `train.py`, so they must stay in this
same folder).

### Training augmentations

The PyTorch training pipeline applies augmentations designed for
low-resolution CCTV footage: random crop after a slight upscale, random
horizontal flip, colour jitter, random rotation ±10°, and Gaussian blur
(simulates motion/compression artefacts).

## A.3 – Inference

`classify_realtime.py` — real-time two-stage pipeline: YOLO person detection
+ ByteTrack, then per-frame batched adult/child classification of every
tracked crop.

---

# B — [`face_age_gender/`](face_age_gender/): Face Age & Gender Prediction (MiVOLO, eKYC)

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
python face_age_gender/ekyc_age_gender.py predict --root ~/Documents/Minje/ekyc_results   # pre-annotate with a MiVOLO checkpoint
python face_age_gender/ekyc_age_gender.py review  --root ~/Documents/Minje/ekyc_results   # OpenCV window, accept/correct each label
python face_age_gender/ekyc_age_gender.py export  --valid 0.15 --test 0.15                # -> ekyc_{train,valid,test}.csv, IMDB-clean layout
```

`review` keys: Enter/Space confirm, `0-9`/Backspace type a new age, `m`/`f`
set gender, `x` reject (unusable), `n`/`b` next/back, `u` undo, `q`/Esc save
& quit. Labels (`annotations.csv`, journal, per-person folders) contain
customer IDs and live in `face_age_gender/ekyc_labels/`, which is gitignored
— never commit them.

## B.2 – Training (`train_mivolo.py`)

Trains / fine-tunes / evaluates the face-only VOLO-D1 model on IMDB-clean-
or eKYC-format CSVs (`filename,age,gender,x_min,y_min,x_max,y_max,...`).

```bash
# fine-tune the IMDB-pretrained model on reviewed eKYC images
python face_age_gender/train_mivolo.py --data-root face_age_gender/ekyc_labels --images <ekyc_root> --csv-pattern ekyc_{split}.csv \
    --init models/mivolo_volo_d1_face_imdb.pth.tar --epochs 30 --warmup-epochs 3 --lr 1e-5 \
    --batch-size 32 --output runs/ekyc_finetune_1

# evaluate a checkpoint on the test split
python face_age_gender/train_mivolo.py --data-root face_age_gender/ekyc_labels --images <ekyc_root> --csv-pattern ekyc_{split}.csv \
    --eval-only --split test --init runs/ekyc_finetune_1/best.pth.tar
```

Checkpoints (shared location, not moved — see `models/` and `runs/` below):

- `models/mivolo_volo_d1_face_imdb.pth.tar` — official IMDB-clean-pretrained starting point.
- `runs/mivolo_face/best.pth.tar` — **current best checkpoint**, fine-tuned on the eKYC dataset (see `runs/mivolo_face/args.json` for the exact run config; data lived at `/mnt/Transcend/ekyc_age_gender_20261002/`).

Checkpoint format: `{state_dict, min_age, max_age, avg_age, no_gender,
with_persons_model, epoch, metrics, args}`. Predicted
age = `out[:, 2] * (max_age - min_age) + avg_age`; gender =
`argmax(out[:, :2])` (0 = male, 1 = female).

`train_mivolo.py` and `ekyc_age_gender.py` both `import tracking` (mlops/W&B
experiment tracking) — `tracking.py` lives in this same folder for that
reason; don't move it out on its own.

## B.3 – Inference (`infer_mivolo.py`)

Runs a checkpoint on a single image (defaults to
`runs/mivolo_face/best.pth.tar`). By default the whole image is treated as
the face crop (same convention as the eKYC folders); pass `--box` if it's a
full photo and you already know the face's pixel box.

```bash
python face_age_gender/infer_mivolo.py path/to/face_crop.jpg
python face_age_gender/infer_mivolo.py path/to/photo.jpg --box x0 y0 x1 y1
```

---

# C — [`person_scooter_detection/`](person_scooter_detection/): Person vs. Scooter Detection (minor, legacy)

**Not** one of the repo's two main purposes — a separate side task
(detecting a person vs. a scooter, not adult vs. child) that happens to reuse
some of the same generic YOLO tooling as Part A. Called out in its own
folder so it doesn't get confused with the adult/child pipeline.

| Script | Purpose |
|---|---|
| `detect_person.py` | Runs a YOLO detector + tracker on a video, saves 1 frame/sec plus crops. **Currently hardcoded** to only keep `person` (class 0) and `scooter` (class 1) detections — not the generic adult/child extractor it once was (see the note in Part A). |
| `label_person.py` | Interactive box labeler/editor; currently toggles each box between **adult (0)** and **scooter (1)** (the class names its own code uses — yes, "adult" is the literal label for the person/non-scooter class here). |
| `person_inference.py` | Runs a trained person+scooter detector on a video and draws boxes. Its `MODEL_PATH` points at `runs/detect/runs/train/yolov8n_person-16/weights/best.pt`. |
| `dataset.yaml` | YOLO dataset config, `names: [person, scooter]`. Its `train:`/`val:`/`test:` entries (`../train/images`, etc.) are relative to wherever this file sits — **the actual dataset isn't present on this machine right now** (it was last known to live at `/mnt/transcend/person detection dataset/Person/{train,valid}/`, as a sibling of this repo, not inside it); if you recover or re-acquire that dataset, either put a copy of this YAML inside its root, or edit the paths. |
| `Dockerfile.label_person` / `docker-run-label.sh` | Runs `label_person.py` in a container with X11 forwarding. Build/run **from inside this folder** (`cd person_scooter_detection && docker build -f Dockerfile.label_person -t label-person .`). |

The YOLO detector itself is trained with the same generic
`person_detection/person_detection_train.py` used in Part A, just pointed at
this folder's YAML:

```bash
python person_detection/person_detection_train.py --data person_scooter_detection/dataset.yaml
```

---

## Shared infrastructure (stays at repo root)

- `models/` — shared model weights (currently the MiVOLO IMDB-pretrained checkpoint used by Part B).
- `runs/` — shared training/eval outputs for all three parts (`detect/`, `eval/`, `mivolo_face/`, `train/`, `wandb/`).
- `mov_mp4.py` — generic video format conversion helper, not tied to any one part.
- `requirements.txt`, `out.jpg` (a stray sample image, unused by any script).
