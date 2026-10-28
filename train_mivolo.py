"""Train MiVOLO's face-only age + gender model (VOLO-D1) on IMDB-clean.

Data (IMDB-clean, github.com/yiminglin-ai/imdb-clean)
-----------------------------------------------------
    <data-root>/
        imdb-clean-1024/imdb-clean-1024/01/nm0271501_rm2204216064_1949-8-17_2010.jpg
        imdb_train_new_1024.csv   filename,age,gender,x_min,y_min,x_max,y_max,head_*
        imdb_valid_new_1024.csv
        imdb_test_new_1024.csv

Recipe (paper, section 5.2)
---------------------------
    init       ImageNet VOLO-D1 (timm volo_d1_224.sail_in1k)
    optimiser  AdamW, lr 1.5e-5 at batch 192 (sqrt-scaled for other batch
               sizes, as timm does for Adam), weight decay 5e-5
    schedule   220 epochs, 25 warmup epochs from 1e-6, cosine decay
    aug        RandAugment m22 (bilinear), bbox jitter 0.45,
               RandomErasing p=0.5, hflip p=0.5, dropout = drop-path = 0.32
    loss       LDS-weighted MSE on normalised age + 3e-2 * CE on gender
    input      face box cropped, letterboxed to 224x224 (as MiVOLO inference)

Setup
-----
    pip install "timm>=1.0"

Usage
-----
    # full training
    python train_mivolo.py --data-root ~/Downloads/archive

    # quick smoke test on a small subset
    python train_mivolo.py --data-root ~/Downloads/archive --limit 512 \\
        --epochs 2 --warmup-epochs 1 --batch-size 32 --output runs/mivolo_face_smoke

    # continue after a crash / reboot
    python train_mivolo.py --data-root ~/Downloads/archive \\
        --resume runs/mivolo_face/last.pth.tar

    # evaluate on the test split
    python train_mivolo.py --data-root ~/Downloads/archive --eval-only \\
        --init runs/mivolo_face/best.pth.tar --split test

Tracking (see tracking.py)
    mlops   on by default: dataset versions, the starting checkpoint, one
            experiment node per run (updated every --mlops-every epochs) and
            best.pth.tar as an artifact, in $MLOPS_DOCS/<--mlops-project>.json.
            --no-mlops for smoke tests.
    W&B     --wandb [--wandb-project mivolo --wandb-entity TEAM --wandb-name NAME
            --wandb-group G --wandb-tags a,b --wandb-notes TEXT --wandb-mode offline]
    --resume continues the same mlops node and the same W&B run.

Output folder (--output)
    train.log      timestamped progress: setup steps, % of epoch, time left,
                   GPU memory / temperature / power, validation, saves
    log.jsonl      one JSON record of metrics per epoch ("epoch" counts from 0)
    last.pth.tar   every epoch, for --resume;  best.pth.tar  lowest val MAE

Checkpoint (``best.pth.tar`` / ``last.pth.tar``)
    {state_dict, min_age, max_age, avg_age, no_gender, with_persons_model,
     epoch, metrics, args [, optimizer, state_dict_ema, best_mae]}
    Predicted age = out[:, 2] * (max_age - min_age) + avg_age;
    gender = argmax(out[:, :2]) (0 = male, 1 = female).
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import tracking

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUBDIR = "imdb-clean-1024/imdb-clean-1024"
MIN_FACE = 5  # px; MiVOLO's reader drops smaller faces too
GENDERS = {"M": 0, "F": 1}

log = logging.getLogger("train_mivolo")
ARCHITECTURE = "VOLO-D1 face-only MiVOLO (timm volo_d1_224, outputs male/female/age)"


# --------------------------------------------------------------------------
# Progress logging
# --------------------------------------------------------------------------
def setup_logging(log_file=None):
    """Timestamped messages to the console and, if given, appended to log_file
    (flushed per line, so the log survives a crash or freeze)."""
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        log.addHandler(console)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        log.addHandler(handler)


def fmt_duration(seconds):
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return (f"{days}d " if days else "") + f"{hours}:{minutes:02d}:{secs:02d}"


def gpu_status(device):
    """e.g. '16.0GB (peak 14.1) 71C 312W 98%': GPU memory held by this process
    (and its peak use so far), then temperature, power draw and load."""
    if device.type != "cuda":
        return "cpu"
    status = (f"{torch.cuda.memory_reserved(device) / 1e9:.1f}GB "
              f"(peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f})")
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        out = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--format=csv,noheader,nounits",
             "--query-gpu=temperature.gpu,power.draw,utilization.gpu"],
            capture_output=True, text=True, timeout=2).stdout
        temp, power, util = (v.strip() for v in out.split(","))
        status += f" {temp}C {float(power):.0f}W {util}%"
    except Exception:  # nvidia-smi missing or slow: memory only
        pass
    return status


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_split(data_root, split, limit=None, csv_pattern="imdb_{split}_new_1024.csv"):
    """[(filename, age, gender, [x0, y0, x1, y1])] from one annotation CSV in
    either layout:
      IMDB-clean   filename,age,gender,x_min,y_min,x_max,y_max,...
      MiVOLO       img_name,age,gender,face_x0,face_y0,face_x1,face_y1,person_*
                   (-1 = missing; rows without an age or a face are skipped)
    """
    samples = []
    with open(Path(data_root) / csv_pattern.format(split=split), newline="") as f:
        for row in csv.DictReader(f):
            if "img_name" in row:
                fname, keys = row["img_name"], ("face_x0", "face_y0", "face_x1", "face_y1")
            else:
                fname, keys = row["filename"], ("x_min", "y_min", "x_max", "y_max")
            try:
                age = round(float(row["age"]))
            except ValueError:  # an age range such as "(25;32)"
                continue
            box = [float(row[k]) for k in keys]
            if age < 0 or box[0] < 0 or box[2] - box[0] < MIN_FACE or box[3] - box[1] < MIN_FACE:
                continue
            samples.append((fname, age, GENDERS.get(row["gender"], -1), box))
    if limit and limit < len(samples):
        samples = random.Random(0).sample(samples, limit)
    return samples


def letterbox(img, size, color=(0, 0, 0)):
    """Aspect-preserving resize + centre pad (MiVOLO's class_letterbox)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def jitter_bbox(box, magnitude):
    """Random shift and rescale of an xyxy box. ``magnitude`` bounds the
    centre shift (fraction of half the box side) and the relative size
    change; the paper gives 0.45 without defining it further."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2 + random.uniform(-magnitude, magnitude) * w / 2
    cy = (y0 + y1) / 2 + random.uniform(-magnitude, magnitude) * h / 2
    w *= 1 + random.uniform(-magnitude, magnitude)
    h *= 1 + random.uniform(-magnitude, magnitude)
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def clip_box(box, w, h):
    x0, y0, x1, y1 = box
    return (int(min(max(x0, 0), w)), int(min(max(y0, 0), h)),
            int(min(max(x1, 0), w)), int(min(max(y1, 0), h)))


def lds_weights(ages, kernel_size=5, sigma=2.0):
    """Label Distribution Smoothing (Yang et al. 2021, 'Delving into Deep
    Imbalanced Regression'): sqrt-inverse frequency, Gaussian-smoothed over
    neighbouring ages, rescaled to mean 1 over the training samples."""
    lo, hi = min(ages), max(ages)
    counts = np.sqrt(np.bincount(np.asarray(ages) - lo, minlength=hi - lo + 1).astype(np.float64))
    half = (kernel_size - 1) // 2
    kernel = np.exp(-np.arange(-half, half + 1) ** 2 / (2 * sigma ** 2))
    kernel /= kernel.max()
    smoothed = np.convolve(np.pad(counts, half, mode="edge"), kernel, mode="valid")
    per_age = {lo + i: 1.0 / max(v, 1e-6) for i, v in enumerate(smoothed)}
    scale = len(ages) / sum(per_age[a] for a in ages)
    return {a: w * scale for a, w in per_age.items()}


class FaceAgeGenderDataset(Dataset):
    """Face crop -> normalised tensor; target = (normalised age, gender, weight).

    Age is normalised like MiVOLO: (age - avg_age) / (max_age - min_age).
    """

    def __init__(self, samples, image_dir, min_age, max_age, img_size=224,
                 train=False, rand_aug="", jitter=0.0, reprob=0.0, hflip=0.0,
                 age_weights=None):
        from timm.data.auto_augment import rand_augment_transform
        from timm.data.random_erasing import RandomErasing

        self.samples = samples
        self.image_dir = Path(image_dir)
        self.min_age, self.max_age = min_age, max_age
        self.avg_age = (max_age + min_age) / 2.0
        self.img_size = img_size
        self.train = train
        self.jitter = jitter
        self.hflip = hflip
        self.age_weights = age_weights or {}
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        self.rand_aug = None
        if train and rand_aug:
            self.rand_aug = rand_augment_transform(rand_aug, dict(
                translate_const=int(img_size * 0.45),
                img_mean=tuple(round(255 * x) for x in IMAGENET_MEAN),
                interpolation=Image.BILINEAR,
            ))
        self.erase = None
        if train and reprob > 0:
            self.erase = RandomErasing(probability=reprob, mode="pixel", device="cpu")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, age, gender, box = self.samples[index]
        img = cv2.imread(str(self.image_dir / fname))
        if img is None:
            raise IOError(f"Can not read {self.image_dir / fname}")
        h, w = img.shape[:2]

        x0, y0, x1, y1 = clip_box(box, w, h)
        if self.train and self.jitter > 0:
            j = clip_box(jitter_bbox(box, self.jitter), w, h)
            if j[2] - j[0] >= MIN_FACE and j[3] - j[1] >= MIN_FACE:
                x0, y0, x1, y1 = j
        crop = letterbox(img[y0:y1, x0:x1], self.img_size)

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if self.train:
            if self.rand_aug is not None:
                pil = self.rand_aug(pil)
            if random.random() < self.hflip:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - self.mean) / self.std
        if self.erase is not None:
            x = self.erase(x)

        age_norm = (age - self.avg_age) / (self.max_age - self.min_age)
        weight = self.age_weights.get(age, 1.0)
        return x, torch.tensor([age_norm, gender, weight], dtype=torch.float32)


def seed_worker(_):
    # torch seeds python `random` per worker but not numpy, and timm's
    # RandAugment picks its ops with np.random
    np.random.seed(torch.initial_seed() % 2**32)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(args, num_classes):
    import timm

    pretrained = args.init == "imagenet" and not args.resume
    # timm 0.8 (used by MiVOLO) applied `drop` right after the pos-embed;
    # newer timm calls that pos_drop_rate
    model = timm.create_model("volo_d1_224.sail_in1k" if pretrained else "volo_d1_224",
                              pretrained=pretrained, num_classes=num_classes,
                              pos_drop_rate=args.drop, drop_path_rate=args.drop_path)
    if pretrained:
        log.info("Initialised from ImageNet VOLO-D1 (head re-initialised)")
    return model


def load_weights(model, path):
    """Load a checkpoint (ours or an official MiVOLO face-only one);
    tensors whose shape differs (e.g. age-only head) stay at init."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.startswith("fds.")}
    if "patch_embed.conv1.0.weight" in sd:
        raise ValueError(f"{path} is a face+body MiVOLO checkpoint, not face-only")
    own = model.state_dict()
    skipped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in sd.items() if k not in skipped}, strict=False)
    log.info(f"Loaded {path}")
    for name, keys in (("shape mismatch, left at init", skipped),
                       ("missing", missing), ("unused", unexpected)):
        if keys:
            log.info(f"  {name}: {len(keys)} tensors, e.g. {keys[:4]}")
    return ckpt


# --------------------------------------------------------------------------
# Loss / metrics
# --------------------------------------------------------------------------
def compute_loss(out, target, only_age, gender_weight):
    age_t, gender_t, weight = target.unbind(1)
    age_out = out[:, 0] if only_age else out[:, 2]
    age_loss = (weight * (age_out - age_t) ** 2).mean()
    if only_age:
        return age_loss, age_loss.detach(), torch.zeros(())
    gender_loss = F.cross_entropy(out[:, :2], gender_t.long(), ignore_index=-1)
    return age_loss + gender_weight * gender_loss, age_loss.detach(), gender_loss.detach()


CHILD_AGE = 18  # under this age = child (positive class of the child / adult metrics)
AGE_GROUPS = ((0, 12, "0-12"), (13, 17, "13-17"), (18, 29, "18-29"), (30, 49, "30-49"),
              (50, 200, "50+"))


# Metrics stored in mlops: only those whose better direction the dashboard
# infers correctly from the name ("mae" = lower is better, anything else =
# higher), so its compare view marks the right winner. bias (best near 0) and
# the counts stay in the report, the predictions CSV, log.jsonl and W&B.
KEY_METRICS = ("mae", "cs@5", "mae_under18", "child_recall", "child_precision", "child_f1",
               "gender_acc", "gender_acc_under18")


def mlops_metrics(m):
    """KEY_METRICS plus the MAE of every age group that has images."""
    keep = {k: m[k] for k in KEY_METRICS if k in m}
    keep.update({f"mae_{name}": m[f"mae_{name}"] for _, _, name in AGE_GROUPS
                 if m.get(f"n_{name}")})
    return keep


def _pct(num, den):
    return 100 * num / den if den else float("nan")


def age_gender_metrics(pred_age, gt_age, g_pred=None, g_gt=None):
    """Evaluation numbers as one flat dict (rates in %).

    Age        mae, cs@5 (% within 5 years), bias (mean predicted - true,
               > 0 = too old), mae_under18, and n / mae / bias per age group
    Child      child = true age < 18, decided from the predicted age:
               child_recall (% of children found), child_precision (% of
               "child" calls that are right), child_f1, and the counts
               child_found / child_missed / adult_flagged
    Gender     gender_acc, gender_acc_under18, and the error counts
               gender_m_as_f / gender_f_as_m
    """
    pred_age, gt_age = pred_age.float(), gt_age.float()
    diff = pred_age - gt_age
    err = diff.abs()
    minors = gt_age < CHILD_AGE
    m = {
        "n": len(err),
        "mae": err.mean().item(),
        "cs@5": _pct(int((err <= 5 + 1e-6).sum()), len(err)),
        "bias": diff.mean().item(),
        "mae_under18": err[minors].mean().item() if minors.any() else float("nan"),
    }
    for lo, hi, name in AGE_GROUPS:
        sel = (gt_age >= lo) & (gt_age <= hi)
        m[f"n_{name}"] = int(sel.sum())
        m[f"mae_{name}"] = err[sel].mean().item() if sel.any() else float("nan")
        m[f"bias_{name}"] = diff[sel].mean().item() if sel.any() else float("nan")

    called_child = pred_age < CHILD_AGE
    found = int((called_child & minors).sum())
    missed = int((~called_child & minors).sum())
    flagged = int((called_child & ~minors).sum())
    m.update({
        "child_recall": _pct(found, found + missed),
        "child_precision": _pct(found, found + flagged),
        # 2TP / (2TP + FP + FN): 0 when no child is found, undefined only when
        # there are no children and none are predicted
        "child_f1": _pct(2 * found, 2 * found + flagged + missed),
        "child_found": found, "child_missed": missed, "adult_flagged": flagged,
    })

    if g_pred is not None:
        valid = g_gt >= 0
        gp, gt, young = g_pred[valid], g_gt[valid], minors[valid]
        m["gender_acc"] = _pct(int((gp == gt).sum()), len(gt))
        m["gender_acc_under18"] = _pct(int((gp[young] == gt[young]).sum()), int(young.sum()))
        m["gender_m_as_f"] = int(((gt == 0) & (gp == 1)).sum())
        m["gender_f_as_m"] = int(((gt == 1) & (gp == 0)).sum())
    return m


def format_report(m, title=""):
    """Readable summary of age_gender_metrics() output."""
    def f(v, fmt=".2f"):
        return "-" if v != v else format(v, fmt)

    lines = [f"===== evaluation{' - ' + title if title else ''}: {m['n']} images",
             f"age      MAE {f(m['mae'])} years   within 5 years {f(m['cs@5'], '.1f')}%   "
             f"bias {f(m['bias'], '+.2f')} (+ = predicted too old)",
             "         true age    " + "".join(f"{name:>8s}" for _, _, name in AGE_GROUPS),
             "           images    " + "".join(f"{m[f'n_{name}']:>8d}" for _, _, name in AGE_GROUPS),
             "           MAE       " + "".join(f"{f(m[f'mae_{name}']):>8s}" for _, _, name in AGE_GROUPS),
             "           bias      " + "".join(f"{f(m[f'bias_{name}'], '+.2f'):>8s}"
                                            for _, _, name in AGE_GROUPS),
             f"child    (under {CHILD_AGE}, from the predicted age)  recall {f(m['child_recall'], '.1f')}%  "
             f"precision {f(m['child_precision'], '.1f')}%  F1 {f(m['child_f1'], '.1f')}%",
             f"         {m['child_found']} children found, {m['child_missed']} missed (called adult), "
             f"{m['adult_flagged']} adults wrongly called child"]
    if "gender_acc" in m:
        lines.append(f"gender   accuracy {f(m['gender_acc'])}%  (under 18: {f(m['gender_acc_under18'], '.1f')}%)"
                     f"   errors: {m['gender_m_as_f']} male called female, "
                     f"{m['gender_f_as_m']} female called male")
    return lines


@torch.no_grad()
def evaluate(model, loader, device, only_age, min_age, max_age, amp_ctx, log_every=0,
             return_predictions=False):
    """Metrics of age_gender_metrics(); with ``return_predictions`` also the
    per-image predictions (in loader order) as a dict of tensors."""
    model.eval()
    avg_age = (max_age + min_age) / 2.0
    pred_ages, gt_ages, g_pred, g_gt, g_conf = [], [], [], [], []
    t0, n_batches = time.time(), len(loader)
    for b, (x, target) in enumerate(loader):
        if log_every and b and b % log_every == 0:
            done = b / n_batches
            log.info(f"  validating: batch {b}/{n_batches} ({100 * done:.0f}%), "
                     f"ETA {fmt_duration((time.time() - t0) / done * (1 - done))}")
        x = x.to(device, non_blocking=True)
        with amp_ctx():
            out = model(x).float().cpu()
        age_out = out[:, 0] if only_age else out[:, 2]
        pred_ages.append((age_out * (max_age - min_age) + avg_age).clamp(min=0))
        gt_ages.append(target[:, 0] * (max_age - min_age) + avg_age)
        if not only_age:
            probs = out[:, :2].softmax(1)
            conf, cls = probs.max(1)
            g_pred.append(cls)
            g_conf.append(conf)
            g_gt.append(target[:, 1].long())
    pred = {"age": torch.cat(pred_ages), "true_age": torch.cat(gt_ages).round()}
    if not only_age:
        pred.update(gender=torch.cat(g_pred), true_gender=torch.cat(g_gt),
                    gender_conf=torch.cat(g_conf))
    metrics = age_gender_metrics(pred["age"], pred["true_age"], pred.get("gender"),
                                 pred.get("true_gender"))
    return (metrics, pred) if return_predictions else metrics


def write_predictions(path, samples, pred):
    """Per-image predictions, worst age error first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (fname, _, _, _) in enumerate(samples):
        row = {"filename": fname, "true_age": int(pred["true_age"][i]),
               "pred_age": round(pred["age"][i].item(), 1),
               "age_error": round(pred["age"][i].item() - pred["true_age"][i].item(), 1)}
        row["child_correct"] = int((row["pred_age"] < CHILD_AGE) == (row["true_age"] < CHILD_AGE))
        if "gender" in pred:
            g, t = int(pred["gender"][i]), int(pred["true_gender"][i])
            row.update(true_gender="MF"[t] if t >= 0 else "", pred_gender="MF"[g],
                       gender_conf=round(pred["gender_conf"][i].item(), 3),
                       gender_correct=int(g == t) if t >= 0 else "")
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["age_error"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(path, model, args, epoch, metrics, ages, **extra):
    min_age, max_age = ages
    state = {
        "state_dict": model.state_dict(),
        "min_age": min_age,
        "max_age": max_age,
        "avg_age": (max_age + min_age) / 2.0,
        "no_gender": args.only_age,
        "with_persons_model": False,
        "epoch": epoch,
        "metrics": metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **extra,
    }
    # fsync before the atomic rename: a hard freeze right after saving must
    # not leave a truncated checkpoint behind
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="folder with imdb_*_new_1024.csv and imdb-clean-1024/")
    p.add_argument("--images", default=None, help=f"default: <data-root>/{IMAGE_SUBDIR}")
    p.add_argument("--csv-pattern", default="imdb_{split}_new_1024.csv",
                   help="annotation file name in --data-root; e.g. 'ekyc_{split}.csv' "
                        "for ekyc_age_gender.py exports")
    p.add_argument("--output", default="runs/mivolo_face")
    p.add_argument("--train-split", default="train")
    p.add_argument("--split", default="valid", help="split to validate / evaluate on")
    p.add_argument("--limit", type=int, default=None,
                   help="use a random subset of N samples per split (smoke tests)")
    p.add_argument("--init", default="imagenet",
                   help="'imagenet', 'none', or a face-only checkpoint path")
    p.add_argument("--resume", default=None, help="last.pth.tar to continue from")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--predictions", default=None,
                   help="--eval-only: per-image predictions CSV (default: runs/eval/"
                        "<checkpoint>_<split>_predictions.csv; 'none' to skip)")
    p.add_argument("--only-age", action="store_true", help="age-only head (no gender)")

    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1.5e-5, help="lr at --lr-base-batch")
    p.add_argument("--lr-base-batch", type=int, default=192,
                   help="lr is scaled by sqrt(batch_size / this)")
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--warmup-epochs", type=int, default=25)
    p.add_argument("--warmup-lr", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--clip-grad", type=float, default=None)
    p.add_argument("--gender-weight", type=float, default=3e-2)
    p.add_argument("--no-lds", action="store_true", help="plain MSE instead of LDS-weighted")

    p.add_argument("--drop", type=float, default=0.32)
    p.add_argument("--drop-path", type=float, default=0.32)
    p.add_argument("--rand-aug", default="rand-m22-n2-mstd0.5-mmax30",
                   help="timm RandAugment config, '' to disable (paper: magnitude 22)")
    p.add_argument("--jitter", type=float, default=0.45, help="bbox jitter magnitude")
    p.add_argument("--reprob", type=float, default=0.5, help="RandomErasing probability")
    p.add_argument("--hflip", type=float, default=0.5)

    p.add_argument("--ema", action="store_true", help="keep an EMA of the weights, validate with it")
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--grad-checkpointing", action="store_true", help="less GPU memory, slower")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=50, help="progress line every N steps")
    p.add_argument("--max-steps", type=int, default=None, help="debug: steps per epoch")
    p.add_argument("--dataset-name", default=None,
                   help="mlops dataset name (default: from --csv-pattern, e.g. 'ekyc', "
                        "'imdb-clean'); nodes are named <name>-<split>")
    p.add_argument("--mlops-every", type=int, default=5,
                   help="update the mlops run node every N epochs (and at the end)")
    tracking.add_args(p, wandb=True)
    return p.parse_args()


def track_dataset(tracker, args, split, n_items):
    """mlops node id of the CSV a split is read from (new version if it changed)."""
    if args.dataset_name:
        base = args.dataset_name
    elif args.csv_pattern == "imdb_{split}_new_1024.csv":
        base = "imdb-clean"
    else:
        base = Path(args.csv_pattern).stem.replace("{split}", "").strip("_-. ") or "dataset"
    return tracker.dataset(
        f"{base}-{split}", Path(args.data_root) / args.csv_pattern.format(split=split),
        tracking.dataset_split(split), n_items=n_items,
        note=f"random subset of {args.limit} (--limit)" if args.limit else "",
        key=f"limit={args.limit}" if args.limit else "")


def run_parameters(args, **derived):
    params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if not k.startswith(("mlops_", "wandb", "no_mlops"))}
    return {**params, **derived}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    image_dir = Path(args.images or Path(args.data_root) / IMAGE_SUBDIR)
    out_dir = Path(args.output)
    if args.eval_only:
        setup_logging()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(out_dir / "train.log")
        log.info(f"===== {'resuming' if args.resume else 'starting'} training -> {out_dir}")
        log.info("arguments: " + json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    if device.type == "cuda":
        log.info(f"device: {torch.cuda.get_device_name(device)}  ({gpu_status(device)})")
    tracker = tracking.Tracker(args, log)
    if not image_dir.is_dir():
        raise SystemExit(f"image folder not found: {image_dir}")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    def amp_ctx():
        if amp_dtype is None or device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    def make_loader(ds, train):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, drop_last=train,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=train and args.workers > 0,
                          worker_init_fn=seed_worker)

    # ---- model ----
    log.info("building VOLO-D1 model"
             + (" (downloading ImageNet weights if not cached)"
                if args.init == "imagenet" and not args.resume else ""))
    model = build_model(args, num_classes=1 if args.only_age else 3)
    src_ckpt = None
    if args.resume:
        src_ckpt = load_weights(model, args.resume)
    elif args.init not in ("imagenet", "none"):
        src_ckpt = load_weights(model, args.init)
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)
    model.to(device)

    # ---- age normalisation: keep the stats the loaded age head was trained
    # with; otherwise take them from the train split
    if src_ckpt is not None and "min_age" in src_ckpt:
        min_age, max_age = float(src_ckpt["min_age"]), float(src_ckpt["max_age"])
    elif args.eval_only:
        raise SystemExit("--eval-only needs --init <checkpoint>")
    else:
        min_age = max_age = None

    log.info(f"reading {args.csv_pattern.format(split=args.split)}")
    val_samples = read_split(args.data_root, args.split, args.limit, args.csv_pattern)
    if args.eval_only:
        ds = FaceAgeGenderDataset(val_samples, image_dir, min_age, max_age)
        loader = make_loader(ds, False)
        init = Path(args.init)
        title = f"eval {init.parent.name}/{init.name} on {args.split}"
        ds_id = track_dataset(tracker, args, args.split, len(val_samples))
        ckpt_id = tracker.checkpoint(init)
        params = run_parameters(args, n_eval=len(val_samples))
        tracker.start_run(title, init.name, "evaluation", params,
                          datasets={ds_id: tracking.dataset_split(args.split)} if ds_id else None,
                          parents=[ckpt_id] if ckpt_id else None, architecture=ARCHITECTURE)
        tracker.init_wandb(args, params, "eval", "runs",
                           name=f"eval-{init.parent.name}-{args.split}")
        log.info(f"evaluating on {len(val_samples)} images ({len(loader)} batches)")
        try:
            m, pred = evaluate(model, loader, device, args.only_age, min_age, max_age, amp_ctx,
                               log_every=args.log_every, return_predictions=True)
        except BaseException as e:
            tracker.finish_run("failed", error=f"{type(e).__name__}: {e}")
            tracker.finish_wandb(exit_code=1)
            raise
        for line in format_report(m, f"{init.parent.name}/{init.name} on {args.split}"):
            log.info(line)
        log.info(f"[{args.split}] " + json.dumps(m))
        if args.predictions != "none":
            pred_path = Path(args.predictions or Path("runs") / "eval" /
                             f"{init.parent.name}_{init.stem.split('.')[0]}_{args.split}_predictions.csv")
            rows = write_predictions(pred_path, val_samples, pred)
            log.info(f"per-image predictions (worst age error first) -> {pred_path}")
            for r in rows[:5]:
                log.info(f"  {r['filename']}: true {r['true_age']}, predicted {r['pred_age']}"
                         + (f", gender {r['true_gender']}->{r['pred_gender']}" if "pred_gender" in r else ""))
            tracker.attach_path(tracker.run_id, pred_path, "per-image predictions")
        tracker.finish_run("completed",
                           metrics={f"{args.split}_{k}": v for k, v in mlops_metrics(m).items()})
        tracker.wb_summary({f"{args.split}/{k}": v for k, v in m.items()})
        tracker.finish_wandb()
        return

    log.info(f"reading {args.csv_pattern.format(split=args.train_split)}")
    train_samples = read_split(args.data_root, args.train_split, args.limit, args.csv_pattern)
    train_ages = [s[1] for s in train_samples]
    if min_age is None:
        min_age, max_age = float(min(train_ages)), float(max(train_ages))
    log.info(f"{len(train_samples)} train / {len(val_samples)} {args.split} images; "
             f"age normalisation range [{min_age:g}, {max_age:g}]")

    if not args.no_lds:
        log.info("computing LDS age weights")
    train_ds = FaceAgeGenderDataset(
        train_samples, image_dir, min_age, max_age, train=True,
        rand_aug=args.rand_aug, jitter=args.jitter, reprob=args.reprob, hflip=args.hflip,
        age_weights=None if args.no_lds else lds_weights(train_ages))
    train_loader = make_loader(train_ds, train=True)
    val_loader = make_loader(FaceAgeGenderDataset(val_samples, image_dir, min_age, max_age), False)

    # ---- optimisation ----
    from timm.optim import create_optimizer_v2
    from timm.scheduler import CosineLRScheduler
    from timm.utils import ModelEmaV2

    lr = args.lr * math.sqrt(args.batch_size / args.lr_base_batch)
    optimizer = create_optimizer_v2(model, "adamw", lr=lr, weight_decay=args.weight_decay)
    # warmup epochs come first, then cosine decay over the remaining epochs
    scheduler = CosineLRScheduler(optimizer, t_initial=max(1, args.epochs - args.warmup_epochs),
                                  lr_min=args.min_lr,
                                  warmup_t=args.warmup_epochs, warmup_lr_init=args.warmup_lr,
                                  warmup_prefix=True)
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
    ema = ModelEmaV2(model, decay=args.ema_decay) if args.ema else None

    start_epoch, best_mae, best_epoch = 0, float("inf"), None
    if args.resume:
        if "optimizer" in src_ckpt:
            optimizer.load_state_dict(src_ckpt["optimizer"])
        if ema is not None and "state_dict_ema" in src_ckpt:
            ema.module.load_state_dict(src_ckpt["state_dict_ema"])
        start_epoch = src_ckpt["epoch"] + 1
        best_mae = src_ckpt.get("best_mae", best_mae)
        best_epoch = src_ckpt.get("best_epoch")
        log.info(f"resuming at epoch {start_epoch + 1}/{args.epochs}, "
                 f"best val MAE so far {best_mae:.3f}")

    if args.compile:
        log.info("torch.compile enabled: the first steps will be slow while it compiles")
    train_model = torch.compile(model) if args.compile else model
    with open(out_dir / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    steps = len(train_loader) if not args.max_steps else min(len(train_loader), args.max_steps)
    log.info(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
             f"peak lr {lr:.2e}, batch {args.batch_size}, {steps} steps/epoch, "
             f"epochs {start_epoch + 1}-{args.epochs}, validation every {args.val_every} epoch(s)")

    # ---- tracking: datasets + starting checkpoint -> this run's node; W&B
    previous = (src_ckpt or {}).get("tracking", {}) if args.resume else {}
    params = run_parameters(args, peak_lr=lr, steps_per_epoch=steps,
                            n_train=len(train_samples), n_val=len(val_samples))
    if not (previous.get("mlops_run") and tracker.resume_run(previous["mlops_run"])):
        roles = {track_dataset(tracker, args, args.train_split, len(train_samples)): "train",
                 track_dataset(tracker, args, args.split, len(val_samples)): "validation"}
        roles.pop(None, None)
        init_id = None
        if args.init not in ("imagenet", "none"):
            init_id = tracker.checkpoint(args.init)
        tracker.start_run(out_dir.name, "VOLO-D1 face (MiVOLO)",
                          "training" if args.init in ("imagenet", "none") else "finetune",
                          params, datasets=roles or None, parents=[init_id] if init_id else None,
                          architecture=ARCHITECTURE)
        if tracker.run_id:
            tracker.attach_path(tracker.run_id, out_dir, "run folder (train.log, checkpoints)")
    tracker.init_wandb(args, params, "train", out_dir, resume_id=previous.get("wandb_id"))
    best_record = {}

    def run_metrics(record):
        if best_record:
            best = {f"best_{k}": v for k, v in best_record.items() if k.startswith("val_")}
        else:
            best = {}
        return {**best, "best_epoch": best_epoch, "epochs_done": record["epoch"] + 1,
                "last_val_mae": record.get("val_mae", float("nan")),
                "last_train_loss": record["train_loss"]}

    run_t0, epoch_times = time.time(), []
    try:
        for epoch in range(start_epoch, args.epochs):
            scheduler.step(epoch)
            model.train()
            warm = "  (warmup)" if epoch < args.warmup_epochs else ""
            log.info(f"=== epoch {epoch + 1}/{args.epochs}  lr {optimizer.param_groups[0]['lr']:.2e}{warm}")
            t0 = time.time()
            if epoch == start_epoch:
                log.info(f"waiting for the first batch ({args.workers} data workers starting)")
            sums = {"loss": 0.0, "age": 0.0, "gender": 0.0}
            n_steps = 0
            for step, (x, target) in enumerate(train_loader):
                if step == 0 and epoch == start_epoch:
                    log.info(f"first batch ready after {time.time() - t0:.1f}s")
                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with amp_ctx():
                    out = train_model(x)
                loss, age_l, gender_l = compute_loss(out.float(), target, args.only_age,
                                                     args.gender_weight)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

                sums["loss"] += loss.item()
                sums["age"] += age_l.item()
                sums["gender"] += gender_l.item()
                n_steps += 1
                if tracker.wb is not None and (n_steps % 10 == 0 or n_steps == 1):
                    tracker.wb_log({"global_step": epoch * steps + n_steps,
                                    "train/loss": loss.item(), "train/age_loss": age_l.item(),
                                    "train/gender_loss": gender_l.item(),
                                    "train/lr": optimizer.param_groups[0]["lr"],
                                    "train/img_per_s": n_steps * args.batch_size / (time.time() - t0)})
                if step % args.log_every == 0 or n_steps == steps:
                    elapsed = time.time() - t0
                    epoch_left = elapsed / n_steps * (steps - n_steps)
                    per_epoch = (sum(epoch_times[-5:]) / len(epoch_times[-5:]) if epoch_times
                                 else elapsed / n_steps * steps)
                    run_left = epoch_left + per_epoch * (args.epochs - epoch - 1)
                    log.info(f"ep {epoch + 1}/{args.epochs} step {n_steps}/{steps} "
                             f"({100 * n_steps / steps:.0f}%)  loss {loss.item():.4f} "
                             f"(avg {sums['loss'] / n_steps:.4f})  age {age_l.item():.4f}  "
                             f"gender {gender_l.item():.4f}  "
                             f"{n_steps * args.batch_size / elapsed:.0f} img/s  "
                             f"epoch left {fmt_duration(epoch_left)}  "
                             f"run left ~{fmt_duration(run_left)}  gpu {gpu_status(device)}")
                if args.max_steps and n_steps >= args.max_steps:
                    break

            train_time = time.time() - t0
            record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                      "train_min": round(train_time / 60, 2),
                      **{f"train_{k}": v / max(n_steps, 1) for k, v in sums.items()}}
            if device.type == "cuda":
                record["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            log.info(f"epoch {epoch + 1} training done in {fmt_duration(train_time)}: "
                     f"avg loss {record['train_loss']:.4f}  age {record['train_age']:.4f}  "
                     f"gender {record['train_gender']:.4f}")

            is_best = False
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                tv = time.time()
                log.info(f"validating on {len(val_samples)} images ({len(val_loader)} batches)")
                eval_model = ema.module if ema is not None else model
                m = evaluate(eval_model, val_loader, device, args.only_age, min_age, max_age,
                             amp_ctx, log_every=args.log_every)
                record.update({f"val_{k}": v for k, v in m.items()})
                if m["mae"] < best_mae:
                    best_mae, best_epoch, is_best = m["mae"], epoch + 1, True
                    best_record = {f"val_{k}": m[k] for k in KEY_METRICS if k in m}
                gender = f"  gender acc {m['gender_acc']:.2f}%" if "gender_acc" in m else ""
                best = "  <- new best" if is_best else f"  (best {best_mae:.3f} at epoch {best_epoch})"
                log.info(f"validation done in {fmt_duration(time.time() - tv)}: "
                         f"MAE {m['mae']:.3f}  CS@5 {m['cs@5']:.1f}%  "
                         f"MAE under 18 {m['mae_under18']:.2f}  child F1 {m['child_f1']:.1f}%"
                         f"{gender}{best}")

            with open(out_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

            ts = time.time()
            extra = {"optimizer": optimizer.state_dict(), "best_mae": best_mae,
                     "best_epoch": best_epoch,
                     "tracking": {"mlops_project": args.mlops_project,
                                  "mlops_run": tracker.run_id,
                                  "wandb_id": tracker.wb.id if tracker.wb is not None else None}}
            if ema is not None:
                extra["state_dict_ema"] = ema.module.state_dict()
            save_checkpoint(out_dir / "last.pth.tar", model, args, epoch, record,
                            (min_age, max_age), **extra)
            if is_best:
                save_checkpoint(out_dir / "best.pth.tar", ema.module if ema is not None else model,
                                args, epoch, record, (min_age, max_age))
            log.info(f"saved last.pth.tar{' and best.pth.tar' if is_best else ''} "
                     f"in {time.time() - ts:.1f}s")

            epoch_times.append(time.time() - t0)
            tracker.wb_log({
                "epoch": epoch + 1, "epoch/lr": record["lr"], "epoch/train_loss": record["train_loss"],
                "epoch/train_age_loss": record["train_age"],
                "epoch/train_gender_loss": record["train_gender"],
                "epoch/minutes": epoch_times[-1] / 60,
                **{f"val/{k[4:]}": v for k, v in record.items() if k.startswith("val_") and k != "val_n"},
                **({"val/best_mae": best_mae} if best_epoch else {})})
            if (epoch + 1) % args.mlops_every == 0:
                tracker.log_run(metrics=run_metrics(record))
            left = args.epochs - epoch - 1
            if left:
                eta = sum(epoch_times[-5:]) / len(epoch_times[-5:]) * left
                finish = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + eta))
                log.info(f"epoch {epoch + 1}/{args.epochs} finished in "
                         f"{fmt_duration(epoch_times[-1])}; {left} epochs left, "
                         f"~{fmt_duration(eta)} (done around {finish})")
    except KeyboardInterrupt:
        log.info(f"interrupted. Continue with --resume {out_dir / 'last.pth.tar'} "
                 f"(it holds the last finished epoch)")
        tracker.finish_run("failed", error=f"interrupted (Ctrl+C); resumable with --resume "
                                           f"{out_dir / 'last.pth.tar'}")
        tracker.finish_wandb(exit_code=1)
        return
    except Exception as e:
        log.exception("training crashed")
        tracker.finish_run("failed", error=f"{type(e).__name__}: {e}")
        tracker.finish_wandb(exit_code=1)
        raise

    log.info(f"Done in {fmt_duration(time.time() - run_t0)}. best val MAE {best_mae:.3f} "
             f"at epoch {best_epoch} -> {out_dir / 'best.pth.tar'}")
    if epoch_times:
        tracker.finish_run("completed", metrics=run_metrics(record))
    else:
        tracker.finish_run("completed")
    if best_epoch and (out_dir / "best.pth.tar").exists():
        tracker.artifact(f"{out_dir.name}/best.pth.tar", out_dir / "best.pth.tar",
                         note=f"best val MAE {best_mae:.3f} at epoch {best_epoch}")
    tracker.wb_summary({"best_val_mae": best_mae, "best_epoch": best_epoch})
    tracker.finish_wandb()


if __name__ == "__main__":
    main()
