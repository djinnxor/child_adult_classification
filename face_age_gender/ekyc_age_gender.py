"""Pre-annotate, review and export age / gender labels for eKYC face crops.

The eKYC results folder has one sub-folder per customer ID holding small
(~112x112) face crops:

    <root>/matched/6000000/2026-02-02_16-15-15_image1.jpg
    <root>/matched/6000000/2026-02-02_16-15-15_image2.jpg
    <root>/unmatched/6000004/2026-02-02_16-46-44_image1.jpg
    ...

Three steps, all local (these are customer photos, nothing is uploaded),
plus ``package`` to carry the result to another machine:

1. predict  Run a MiVOLO face-only checkpoint on every image and write
            annotations.csv: one row per image, the label pre-filled from the
            prediction and marked ``pending``. Re-running only adds images
            that are not in the CSV yet; labels are never overwritten.
2. review   OpenCV window to accept or correct each label. Every decision is
            appended to a journal file at once (survives a crash or freeze);
            annotations.csv is rewritten when you quit.
3. export   Reviewed rows -> ekyc_{train,valid,test}.csv in the layout of
            IMDB-clean's *_1024.csv (box = whole image), split by customer ID
            so one person never lands in two splits. The split is a hash of
            the ID, so exporting again after more review keeps every person
            in the same split.

Labels are per image, not per person: the two images of one customer can be
taken years apart (e.g. an ID-card photo and a selfie).

Every step is also recorded in the mlops project (see tracking.py; --no-mlops
to skip): predict -> an experiment from the checkpoint; predict / review /
export -> a new version of the "ekyc-annotations" dataset (a content-addressed
copy in <labels folder>/versions/); export -> "ekyc-train/valid/test"
datasets; package -> an export artifact.

Every step prints timestamped progress and appends it to ekyc.log next to
annotations.csv (predict: % done, speed, time left, file problems, age
summary; review: each decision and your pace; export: what went where).

Usage
-----
    python face_age_gender/ekyc_age_gender.py predict --root ~/Documents/Minje/ekyc_results
    python face_age_gender/ekyc_age_gender.py predict --root ... --limit 1000   # a first batch only;
                                          # run again to add the next 1000
    python face_age_gender/ekyc_age_gender.py review  --root ~/Documents/Minje/ekyc_results
    python face_age_gender/ekyc_age_gender.py review  --root ~/Documents/Minje/ekyc_results \\
        --sort pred_age --max-age 25          # likely children first
    python face_age_gender/ekyc_age_gender.py export --valid 0.15 --test 0.15

    # everything needed to train elsewhere in one archive: reviewed images,
    # MiVOLO-format CSVs, train_mivolo.py, the starting checkpoint, README
    python face_age_gender/ekyc_age_gender.py package --root ~/Documents/Minje/ekyc_results

    # fine-tune the IMDB model on the reviewed images (a few thousand images
    # are far too few to train from ImageNet), then compare on the same test
    # split: the untouched model first, the fine-tuned one after
    D="--data-root ekyc_labels --images ~/Documents/Minje/ekyc_results --csv-pattern ekyc_{split}.csv"
    python face_age_gender/train_mivolo.py $D --eval-only --split test \\
        --init models/mivolo_volo_d1_face_imdb.pth.tar
    python face_age_gender/train_mivolo.py $D --init models/mivolo_volo_d1_face_imdb.pth.tar \\
        --epochs 30 --warmup-epochs 3 --lr 1e-5 --batch-size 32 --log-every 10 \\
        --output runs/ekyc_finetune_1
    python face_age_gender/train_mivolo.py $D --eval-only --split test \\
        --init runs/ekyc_finetune_1/best.pth.tar

    # if it is better, let it pre-annotate the next batch (fewer corrections)
    python face_age_gender/ekyc_age_gender.py predict --root ... --limit 1000 \\
        --checkpoint runs/ekyc_finetune_1/best.pth.tar

Review keys
-----------
    Enter / Space   confirm the shown age + gender, go to the next image
    0-9, Backspace  type a new age (applied on Enter)
    m / f           set gender to male / female
    x               reject: unusable (no face, unreadable, can't judge), next
    n / b           next / back without changing anything
    u               undo the last decision
    q / Esc         save and quit

annotations.csv columns
-----------------------
    path             image path relative to --root
    group            first folder level (matched / unmatched)
    person_id        customer folder name
    image_kind       image1 / image2 (from the file name)
    captured         capture time parsed from the file name
    width, height    image size in pixels
    file_status      ok / truncated (JPEG cut short) / unreadable
    pred_age, pred_gender, pred_gender_conf     model output
    age, gender      the label (M / F): pre-filled from the model, set in review
    review           pending / accepted / corrected / rejected
    reviewed_at      time of the last review decision
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import tracking
from train_mivolo import (IMAGENET_MEAN, IMAGENET_STD, fmt_duration, gpu_status, letterbox,
                          load_weights)

COLUMNS = ["path", "group", "person_id", "image_kind", "captured", "width", "height",
           "file_status", "pred_age", "pred_gender", "pred_gender_conf",
           "age", "gender", "review", "reviewed_at"]
LABEL_FIELDS = ("age", "gender", "review", "reviewed_at")
REVIEWED = ("accepted", "corrected")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")
DEFAULT_LABELS = "ekyc_labels/annotations.csv"
DEFAULT_CHECKPOINT = "models/mivolo_volo_d1_face_imdb.pth.tar"
AGE_GROUPS = ((0, "0-12"), (13, "13-17"), (18, "18-29"), (30, "30-49"), (50, "50+"))

log = logging.getLogger("ekyc_age_gender")


# --------------------------------------------------------------------------
# progress logging
# --------------------------------------------------------------------------
def setup_logging(labels):
    """Timestamped messages to the console and appended to ekyc.log next to
    the annotations (also catches train_mivolo's messages)."""
    log_file = Path(labels).parent / "ekyc.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    return log_file


def age_groups(ages):
    """'0-12: 3, 13-17: 10, 18-29: 250, ...' for a list of ages."""
    counts = Counter(next(name for lo, name in reversed(AGE_GROUPS) if a >= lo) for a in ages)
    return ", ".join(f"{name}: {counts[name]}" for _, name in AGE_GROUPS)


def fmt_counts(counter):
    return ", ".join(f"{k} {v}" for k, v in sorted(counter.items())) or "-"


def track_annotations(tracker, labels, rows, note, parents=None):
    """Record annotations.csv as a version of the 'ekyc-annotations' dataset.
    The version points to a copy named by its content, so it stays readable
    after annotations.csv changes again."""
    if not tracker.mlops:
        return None
    labels = Path(labels)
    snapshot = labels.parent / "versions" / f"{labels.stem}_{tracking.file_sha256(labels)[:12]}.csv"
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels, snapshot)
    counts = fmt_counts(Counter(r["review"] for r in rows))
    return tracker.dataset("ekyc-annotations", snapshot, "raw", n_items=len(rows),
                           note=f"{note}; {counts}", parents=parents)


# --------------------------------------------------------------------------
# annotations.csv
# --------------------------------------------------------------------------
def read_annotations(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_annotations(path, rows):
    """Atomic + fsynced, so a crash never leaves a half-written CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def journal_path(labels):
    return Path(labels).with_suffix(".review_log.jsonl")


def replay_journal(labels, rows):
    """Apply review decisions logged since annotations.csv was last written."""
    journal = journal_path(labels)
    if not journal.exists():
        return 0
    by_path = {r["path"]: r for r in rows}
    n = 0
    with open(journal) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:  # last line cut off by a crash
                continue
            row = by_path.get(entry["path"])
            if row is not None:
                row.update({k: entry[k] for k in LABEL_FIELDS})
                n += 1
    return n


def describe(rel):
    """Path relative to --root -> the descriptive CSV columns."""
    parts = Path(rel).parts
    stem = Path(rel).stem
    m = TIME_RE.search(stem)
    return {
        "path": rel,
        "group": parts[0] if len(parts) >= 3 else "",
        "person_id": parts[-2] if len(parts) >= 2 else "",
        "image_kind": stem.rsplit("_", 1)[-1] if "_" in stem else "",
        "captured": f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else "",
    }


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
class FaceFiles(Dataset):
    """Whole image = face crop -> letterboxed, normalised 224x224 tensor."""

    def __init__(self, root, rels, img_size=224):
        self.root = Path(root)
        self.rels = rels
        self.img_size = img_size
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        empty = torch.zeros(3, self.img_size, self.img_size)
        path = self.root / self.rels[i]
        try:
            data = path.read_bytes()
        except OSError:
            return empty, i, "unreadable", 0, 0
        # imread (unlike imdecode) still decodes a cut-short JPEG, as the
        # review window does
        img = cv2.imread(str(path))
        if img is None:
            return empty, i, "unreadable", 0, 0
        # a complete JPEG ends with the EOI marker; truncated ones decode
        # with a grey bottom part
        status = "ok"
        if data[:2] == b"\xff\xd8" and b"\xff\xd9" not in data[-64:]:
            status = "truncated"
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(letterbox(img, self.img_size), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), i, status, w, h


def load_model(path, device):
    import timm

    meta = torch.load(path, map_location="cpu", weights_only=False)
    if meta.get("with_persons_model"):
        raise SystemExit(f"{path} is a face+body model; a face-only checkpoint is needed")
    no_gender = bool(meta.get("no_gender", False))
    model = timm.create_model("volo_d1_224", num_classes=1 if no_gender else 3)
    load_weights(model, path)
    return (model.eval().to(device), float(meta["min_age"]), float(meta["max_age"]),
            no_gender)


def scan_images(root, stop_after=None, wanted=lambda rel: True):
    """Relative paths of the images under root, folder by folder in sorted
    order. Unreadable folders are logged and skipped. With ``stop_after``,
    scanning stops at the end of the folder in which that many ``wanted``
    images have been found, so a customer's images are never split."""
    rels, bad_dirs, n_wanted = [], [], 0
    log.info(f"scanning {root} for images"
             + (f" (until {stop_after} new ones are found)" if stop_after else ""))
    for dirpath, dirnames, filenames in os.walk(root, onerror=bad_dirs.append):
        dirnames.sort()
        found = sorted((Path(dirpath) / name).relative_to(root).as_posix()
                       for name in filenames if Path(name).suffix.lower() in IMAGE_EXTS)
        if len(rels) // 20000 != (len(rels) + len(found)) // 20000:
            log.info(f"  found {len(rels) + len(found)} images so far")
        rels.extend(found)
        n_wanted += sum(map(wanted, found))
        if stop_after and n_wanted >= stop_after:
            log.info(f"  stopped scanning at {Path(dirpath).relative_to(root)}: limit reached")
            break
    if bad_dirs:
        log.info(f"WARNING: {len(bad_dirs)} folders could not be read and were skipped, e.g. "
                 + "; ".join(f"{e.filename}: {e.strerror}" for e in bad_dirs[:3]))
    rels.sort()
    info = [describe(r) for r in rels]
    customers = len({(d["group"], d["person_id"]) for d in info})
    log.info(f"found {len(rels)} images of {customers} customers "
             f"({fmt_counts(Counter(d['group'] for d in info))}; "
             f"{fmt_counts(Counter(d['image_kind'] for d in info))})")
    return rels


def cmd_predict(args):
    root = Path(args.root).expanduser()
    device = torch.device(args.device)
    log.info(f"===== predict: {root} -> {args.labels}")
    tracker = tracking.Tracker(args, log)
    log.info(f"loading model {args.checkpoint} on {device}")
    model, min_age, max_age, no_gender = load_model(args.checkpoint, device)
    avg_age = (min_age + max_age) / 2
    log.info(f"model predicts age{'' if no_gender else ' + gender'}, "
             f"age range [{min_age:g}, {max_age:g}]"
             + (f"; gpu {gpu_status(device)}" if device.type == "cuda" else ""))

    rows = []
    if Path(args.labels).exists():
        rows = read_annotations(args.labels)
        recovered = replay_journal(args.labels, rows)
        log.info(f"existing {args.labels}: {len(rows)} rows "
                 f"({fmt_counts(Counter(r['review'] for r in rows))})"
                 + (f"; recovered {recovered} decisions from an unsaved review" if recovered else ""))
    by_path = {r["path"]: r for r in rows}

    def wanted(rel):
        return args.repredict or rel not in by_path

    if args.limit and args.sample == "first":
        rels = scan_images(root, stop_after=args.limit, wanted=wanted)
        todo = [r for r in rels if wanted(r)]
    else:
        rels = scan_images(root)
        todo = [r for r in rels if wanted(r)]
        if args.limit:  # whole customers picked at random
            per_folder = defaultdict(list)
            for r in todo:
                per_folder[str(Path(r).parent)].append(r)
            folders = sorted(per_folder)
            random.Random(args.seed).shuffle(folders)
            todo = []
            for folder in folders:
                if len(todo) >= args.limit:
                    break
                todo.extend(per_folder[folder])
            todo.sort()
    skipped = sum(not wanted(r) for r in rels)
    customers = len({str(Path(r).parent) for r in todo})
    log.info(f"predicting {len(todo)} images of {customers} customers"
             + (f" ({args.sample} {args.limit}, whole customer folders)" if args.limit else "")
             + (f"; skipping {skipped} already in the CSV" if skipped else ""))
    if not todo:
        log.info("nothing to do")
        return

    loader = DataLoader(FaceFiles(root, todo), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    log.info(f"{len(loader)} batches of {args.batch_size}; "
             f"waiting for the first batch ({args.workers} data workers starting)")
    use_amp = device.type == "cuda"
    t0 = time.time()
    done = 0
    file_counts = Counter()
    for step, (x, idx, status, width, height) in enumerate(loader):
        if step == 0:
            log.info(f"first batch ready after {time.time() - t0:.1f}s")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            out = model(x.to(device, non_blocking=True)).float().cpu()
        ages = (out[:, 0 if no_gender else 2] * (max_age - min_age) + avg_age).clamp(min=0)
        probs = None if no_gender else out[:, :2].softmax(1)

        for j, i in enumerate(idx.tolist()):
            rel = todo[i]
            row = by_path.get(rel)
            if row is None:
                row = {**describe(rel), "age": "", "gender": "", "review": "pending",
                       "reviewed_at": ""}
                by_path[rel] = row
                rows.append(row)
            row.update(width=int(width[j]), height=int(height[j]), file_status=status[j])
            file_counts[status[j]] += 1
            if status[j] == "unreadable":
                row.update(pred_age="", pred_gender="", pred_gender_conf="")
                if row["review"] == "pending":
                    row.update(review="rejected", reviewed_at="auto: unreadable")
                continue
            row["pred_age"] = f"{ages[j].item():.1f}"
            if probs is not None:
                g = int(probs[j].argmax())
                row["pred_gender"] = "MF"[g]
                row["pred_gender_conf"] = f"{probs[j, g].item():.3f}"
            if row["review"] == "pending":  # never overwrite a reviewed label
                row["age"] = str(round(float(row["pred_age"])))
                row["gender"] = row.get("pred_gender", "")

        done += len(idx)
        if step % args.log_every == 0 or done == len(todo):
            elapsed = time.time() - t0
            log.info(f"predicted {done}/{len(todo)} ({100 * done / len(todo):.0f}%)  "
                     f"{done / elapsed:.0f} img/s  left {fmt_duration((len(todo) - done) * elapsed / done)}  "
                     f"files: {fmt_counts(file_counts)}"
                     + (f"  gpu {gpu_status(device)}" if device.type == "cuda" else ""))
        if step % 200 == 199:  # checkpoint progress on long runs
            write_annotations(args.labels, sorted(rows, key=lambda r: r["path"]))
            log.info(f"progress saved to {args.labels} ({len(rows)} rows)")

    rows.sort(key=lambda r: r["path"])
    write_annotations(args.labels, rows)
    journal_path(args.labels).unlink(missing_ok=True)

    elapsed = time.time() - t0
    new = [by_path[r] for r in todo]
    ages = [round(float(r["pred_age"])) for r in new if r["pred_age"]]
    minors = sum(a < 18 for a in ages)
    log.info(f"done: {len(todo)} images in {fmt_duration(elapsed)} ({len(todo) / elapsed:.0f} img/s)")
    log.info(f"file status: {fmt_counts(file_counts)}"
             + ("  (unreadable ones are auto-rejected)" if file_counts["unreadable"] else ""))
    if ages:
        log.info(f"predicted age groups: {age_groups(ages)}; "
                 f"under 18: {minors} ({100 * minors / len(ages):.1f}%)")
    if not no_gender:
        low = sum(1 for r in new if r["pred_gender_conf"] and float(r["pred_gender_conf"]) < 0.9)
        log.info(f"predicted gender: {fmt_counts(Counter(r['pred_gender'] for r in new if r['pred_gender']))}; "
                 f"low confidence (<0.9): {low}")
    log.info(f"wrote {len(rows)} rows to {args.labels}")

    ckpt_id = tracker.checkpoint(args.checkpoint)
    metrics = {"images": len(todo), "customers": customers,
               "img_per_s": round(len(todo) / elapsed, 1),
               **{f"files_{k}": v for k, v in file_counts.items()},
               "pred_under18": minors,
               "pred_under18_pct": round(100 * minors / len(ages), 2) if ages else None}
    if not no_gender:
        metrics["pred_gender_low_conf"] = low
    exp_id = tracker.experiment(
        f"pre-annotate {len(todo)} eKYC images", Path(args.checkpoint).name, "evaluation",
        parameters={"root": str(root), "checkpoint": args.checkpoint, "limit": args.limit,
                    "sample": args.sample, "seed": args.seed, "repredict": args.repredict},
        metrics=metrics, parents=[ckpt_id] if ckpt_id else None,
        notes=f"model predictions written to {args.labels} as pending labels")
    track_annotations(tracker, args.labels, rows,
                      note=f"pre-labelled {len(todo)} more images with {Path(args.checkpoint).name}",
                      parents=[exp_id] if exp_id else None)
    log.info(f"next: python face_age_gender/ekyc_age_gender.py review --root {args.root}")


# --------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------
KEY_ENTER, KEY_SPACE, KEY_BACKSPACE, KEY_ESC = (13, 10), 32, (8, 127), 27
BG, WHITE, GREY = (30, 30, 30), (235, 235, 235), (150, 150, 150)
STATUS_COLOR = {"pending": (0, 200, 255), "accepted": (80, 200, 80),
                "corrected": (255, 180, 60), "rejected": (80, 80, 255)}
FACE_PX, THUMB_PX = 336, 104
CANVAS_W, CANVAS_H = 860, 540
MAX_THUMBS = 7


class Reviewer:
    """Review state and key handling, separate from the window so it can be
    driven by tests. ``queue`` holds indices into ``rows``."""

    def __init__(self, rows, queue, root, journal):
        self.rows = rows
        self.queue = queue
        self.root = Path(root)
        self.journal = journal
        self.pos = next((k for k, i in enumerate(queue) if rows[i]["review"] == "pending"), 0)
        self.typed = ""
        self.gender_edit = None
        self.history = []  # (row index, previous label fields, previous pos)
        self.message = ""
        self.quit = False
        self.by_person = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_person[(r["group"], r["person_id"])].append(i)
        self._cache = {}
        self.session = Counter()  # decisions made in this session
        self.t0 = time.time()

    # ---- state changes ----
    @property
    def current(self):
        return self.queue[self.pos] if self.pos < len(self.queue) else None

    def _log(self, row):
        self.journal.write(json.dumps({k: row[k] for k in ("path",) + LABEL_FIELDS}) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def _set(self, i, **fields):
        row = self.rows[i]
        self.history.append((i, {k: row[k] for k in LABEL_FIELDS}, self.pos))
        row.update(fields, reviewed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._log(row)
        self.session[row["review"]] += 1
        log.info(f"{row['review']:9s} {row['path']}  age {row['age']} {row['gender']}  "
                 f"(model {row['pred_age'] or '?'} {row['pred_gender'] or '?'})")
        if sum(self.session.values()) % 25 == 0:
            self.log_progress()

    def log_progress(self):
        n = sum(self.session.values())
        elapsed = time.time() - self.t0
        pending = sum(1 for i in self.queue if self.rows[i]["review"] == "pending")
        reviewed = sum(1 for r in self.rows if r["review"] != "pending")
        pace = n / elapsed * 60 if elapsed else 0
        left = f", ~{fmt_duration(pending / pace * 60)} left at this pace" if pace else ""
        log.info(f"session: {n} decisions ({fmt_counts(self.session)}) in "
                 f"{fmt_duration(elapsed)}, {pace:.1f}/min; queue: {pending} of "
                 f"{len(self.queue)} still pending{left}; overall reviewed "
                 f"{reviewed}/{len(self.rows)} ({100 * reviewed / len(self.rows):.1f}%)")

    def move(self, step):
        self.pos = min(max(self.pos + step, 0), len(self.queue))
        self.typed, self.gender_edit, self.message = "", None, ""

    def confirm(self):
        i = self.current
        if i is None:
            return
        row = self.rows[i]
        age = self.typed or row["age"]
        gender = self.gender_edit or row["gender"]
        if not age.isdigit() or not 0 <= int(age) <= 100:
            self.message = "type an age 0-100 first"
            return
        if gender not in ("M", "F"):
            self.message = "set gender with m / f first"
            return
        predicted = (str(round(float(row["pred_age"]))) if row["pred_age"] else "",
                     row["pred_gender"])
        review = "accepted" if (str(int(age)), gender) == predicted else "corrected"
        self._set(i, age=str(int(age)), gender=gender, review=review)
        self.move(+1)

    def reject(self):
        if self.current is not None:
            self._set(self.current, review="rejected")
            self.move(+1)

    def undo(self):
        if not self.history:
            self.message = "nothing to undo"
            return
        i, previous, pos = self.history.pop()
        self.session[self.rows[i]["review"]] -= 1
        self.rows[i].update(previous)
        self._log(self.rows[i])
        log.info(f"undo      {self.rows[i]['path']} -> {self.rows[i]['review']}")
        self.pos = pos
        self.typed, self.gender_edit = "", None
        self.message = f"undone: {self.rows[i]['path']}"

    def press(self, key):
        ch = chr(key) if 32 < key < 127 else ""
        if key == KEY_ESC or ch == "q":
            self.quit = True
        elif ch.isdigit():
            self.typed = (self.typed + ch)[-3:]
        elif key in KEY_BACKSPACE:
            self.typed = self.typed[:-1]
        elif ch in ("m", "f"):
            self.gender_edit = ch.upper()
        elif key in KEY_ENTER or key == KEY_SPACE:
            self.confirm()
        elif ch == "x":
            self.reject()
        elif ch == "n":
            self.move(+1)
        elif ch == "b":
            self.move(-1)
        elif ch == "u":
            self.undo()

    # ---- drawing ----
    def _image(self, i, size):
        key = (i, size)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            img = cv2.imread(str(self.root / self.rows[i]["path"]))
            if img is None:
                img = np.full((size, size, 3), 80, np.uint8)
                cv2.putText(img, "unreadable", (8, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            size / 300, WHITE, 1, cv2.LINE_AA)
            else:
                img = letterbox(img, size, color=BG)
            self._cache[key] = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return self._cache[key]

    def render(self):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)

        def text(s, x, y, scale=0.6, color=WHITE, thick=1):
            cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                        cv2.LINE_AA)

        help_line = "Enter=ok  0-9=age  m/f=gender  x=reject  n/b=next/back  u=undo  q=quit"
        text(help_line, 16, CANVAS_H - 14, 0.5, GREY)

        i = self.current
        if i is None:
            text(f"End of queue ({len(self.queue)} images).", 16, 60, 0.9)
            text("b = go back, q = save and quit", 16, 100, 0.7, GREY)
            return canvas

        row = self.rows[i]
        canvas[16:16 + FACE_PX, 16:16 + FACE_PX] = self._image(i, FACE_PX)

        x0 = FACE_PX + 40
        text(f"{self.pos + 1} / {len(self.queue)}", x0, 40, 0.7)
        text(f"{row['group']} / {row['person_id']} / {row['image_kind']}", x0, 72, 0.55, GREY)
        text(f"captured {row['captured'] or '?'}", x0, 98, 0.55, GREY)
        file_color = WHITE if row["file_status"] == "ok" else (80, 80, 255)
        text(f"file {row['file_status']}  {row['width']}x{row['height']}", x0, 124, 0.55,
             file_color)

        conf = f" ({float(row['pred_gender_conf']):.2f})" if row["pred_gender_conf"] else ""
        pred = f"{row['pred_age'] or '?'}  {row['pred_gender'] or '?'}{conf}"
        text(f"model:  {pred}", x0, 170, 0.65, GREY)

        age = self.typed + "_" if self.typed else row["age"] or "?"
        gender = self.gender_edit or row["gender"] or "?"
        edited = self.typed or self.gender_edit
        text("label:", x0, 222, 0.65, GREY)
        text(f"{age}  {gender}", x0 + 80, 226, 1.3, (0, 230, 255) if edited else WHITE, 2)
        text(row["review"], x0, 268, 0.7, STATUS_COLOR.get(row["review"], WHITE), 2)
        if self.message:
            text(self.message, x0, 300, 0.55, (80, 80, 255))

        # other images of the same customer, for a consistent judgement
        siblings = self.by_person[(row["group"], row["person_id"])][:MAX_THUMBS]
        y = FACE_PX + 36
        text("same customer:", 16, y - 6, 0.5, GREY)
        for k, s in enumerate(siblings):
            x = 16 + k * (THUMB_PX + 12)
            canvas[y:y + THUMB_PX, x:x + THUMB_PX] = self._image(s, THUMB_PX)
            if s == i:
                cv2.rectangle(canvas, (x - 2, y - 2), (x + THUMB_PX + 1, y + THUMB_PX + 1),
                              (0, 230, 255), 2)
            r = self.rows[s]
            text(f"{r['image_kind']} {r['age']}{r['gender']}", x, y + THUMB_PX + 16, 0.42,
                 STATUS_COLOR.get(r["review"], WHITE))
        return canvas


def build_queue(rows, args):
    def keep(r):
        if args.status == "reviewed":
            ok = r["review"] in REVIEWED
        else:
            ok = args.status == "all" or r["review"] == args.status
        if args.group and r["group"] != args.group:
            return False
        if args.kind and r["image_kind"] != args.kind:
            return False
        if (args.min_age is not None or args.max_age is not None) and not r["pred_age"]:
            return False
        if args.min_age is not None and float(r["pred_age"]) < args.min_age:
            return False
        if args.max_age is not None and float(r["pred_age"]) > args.max_age:
            return False
        return ok

    queue = [i for i, r in enumerate(rows) if keep(r)]
    if args.sort == "pred_age":
        queue.sort(key=lambda i: float(rows[i]["pred_age"] or 999))
    elif args.sort == "gender_conf":  # least confident first
        queue.sort(key=lambda i: float(rows[i]["pred_gender_conf"] or 0))
    return queue


def cmd_review(args):
    log.info(f"===== review: {args.labels}")
    rows = read_annotations(args.labels)
    recovered = replay_journal(args.labels, rows)
    if recovered:
        write_annotations(args.labels, rows)  # recover decisions from a crashed session
        log.info(f"recovered {recovered} decisions from the last session's journal")
    journal_path(args.labels).unlink(missing_ok=True)

    queue = build_queue(rows, args)
    filters = {k: getattr(args, k) for k in ("status", "group", "kind", "min_age", "max_age", "sort")
               if getattr(args, k) is not None}
    log.info(f"{len(rows)} images: {fmt_counts(Counter(r['review'] for r in rows))}")
    log.info(f"queue: {len(queue)} images for {filters}")
    if not queue:
        log.info("nothing to review with these filters")
        return

    with open(journal_path(args.labels), "a") as journal:
        reviewer = Reviewer(rows, queue, Path(args.root).expanduser(), journal)
        cv2.namedWindow("ekyc_review", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ekyc_review", CANVAS_W, CANVAS_H)
        try:
            while not reviewer.quit:
                cv2.imshow("ekyc_review", reviewer.render())
                reviewer.press(cv2.waitKey(0) & 0xFF)
        finally:
            cv2.destroyAllWindows()
            write_annotations(args.labels, rows)
            reviewer.log_progress()
    journal_path(args.labels).unlink(missing_ok=True)
    log.info(f"saved {args.labels}: {fmt_counts(Counter(r['review'] for r in rows))}")

    decisions = +reviewer.session  # drop undone (zero) counts
    if decisions:
        minutes = (time.time() - reviewer.t0) / 60
        track_annotations(tracking.Tracker(args, log), args.labels, rows,
                          note=f"review session: {sum(decisions.values())} decisions "
                               f"({fmt_counts(decisions)}) in {minutes:.0f} min")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def split_of(person_id, valid_frac, test_frac):
    """Deterministic split from a hash of the customer ID."""
    u = int(hashlib.md5(person_id.encode()).hexdigest()[:8], 16) / 16 ** 8
    return "test" if u < test_frac else "valid" if u < test_frac + valid_frac else "train"


def cmd_export(args):
    log.info(f"===== export: {args.labels}")
    rows = read_annotations(args.labels)
    if replay_journal(args.labels, rows):
        log.info("note: included decisions from an unsaved review session")
    ok_files = ("ok", "truncated") if args.include_truncated else ("ok",)
    keep = [r for r in rows if r["review"] in REVIEWED and r["file_status"] in ok_files]
    skipped_files = sum(1 for r in rows if r["review"] in REVIEWED
                        and r["file_status"] not in ok_files)
    log.info(f"{len(rows)} images: {fmt_counts(Counter(r['review'] for r in rows))}")
    log.info(f"exporting {len(keep)} reviewed images; left out: "
             f"{sum(r['review'] == 'pending' for r in rows)} pending, "
             f"{sum(r['review'] == 'rejected' for r in rows)} rejected, "
             f"{skipped_files} reviewed but truncated/unreadable"
             + ("" if args.include_truncated or not skipped_files
                else " (add --include-truncated to keep truncated ones)"))

    out_dir = Path(args.out or Path(args.labels).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "age", "gender", "x_min", "y_min", "x_max", "y_max",
              "person_id", "group", "image_kind", "captured"]
    splits = defaultdict(list)
    for r in keep:
        splits[split_of(r["person_id"] or r["path"], args.valid, args.test)].append({
            "filename": r["path"], "age": r["age"], "gender": r["gender"],
            "x_min": 0, "y_min": 0, "x_max": r["width"], "y_max": r["height"],
            "person_id": r["person_id"], "group": r["group"],
            "image_kind": r["image_kind"], "captured": r["captured"],
        })

    exported = {}
    for split in ("train", "valid", "test"):
        items = splits[split]
        path = out_dir / f"ekyc_{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
        ages = [int(r["age"]) for r in items]
        minors = sum(a < 18 for a in ages)
        people = len({r["person_id"] for r in items})
        genders = Counter(r["gender"] for r in items)
        log.info(f"{split:5s} -> {path}: {len(items)} images, {people} customers, "
                 f"under 18: {minors}, M/F {genders['M']}/{genders['F']}")
        if ages:
            log.info(f"        age groups: {age_groups(ages)}")
        exported[split] = (path, len(items), people, minors)

    tracker = tracking.Tracker(args, log)
    ann_id = track_annotations(tracker, args.labels, rows, note="annotations as exported")
    for split, (path, n, people, minors) in exported.items():
        tracker.dataset(f"ekyc-{split}", path, tracking.dataset_split(split), n_items=n,
                        note=f"{people} customers, {minors} under 18; split by customer-ID hash "
                             f"(valid {args.valid:g}, test {args.test:g})",
                        parents=[ann_id] if ann_id else None)


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------
MIVOLO_COLUMNS = ["img_name", "age", "gender", "face_x0", "face_y0", "face_x1", "face_y1",
                  "person_x0", "person_y0", "person_x1", "person_y1"]
PACKAGE_README = """# eKYC face age / gender dataset for MiVOLO

Created {created} from `{labels}`: reviewed (accepted or corrected) images only.
These are customer eKYC photos: keep this archive on trusted machines and
delete copies you no longer need.

| split | images | customers | under 18 | male / female |
|---|---|---|---|---|
{table}

Customers are split by a hash of their ID (valid {valid:.0%}, test {test:.0%}), so one
person is never in two splits. Each image is a whole face crop; the face box
is the full image.

## Contents

    images/<group>/<customer id>/<capture time>_image<N>.jpg
    annotations/ekyc_train.csv, ekyc_valid.csv, ekyc_test.csv
        MiVOLO annotation format: img_name,age,gender,face_x0,face_y0,face_x1,face_y1,
        person_x0..person_y1 (-1 = no body box); gender M / F; img_name relative to images/
    metadata/reviewed_annotations.csv   review details (model prediction, review status,
                                        split) for every packaged image
    code/train_mivolo.py                face-only MiVOLO (VOLO-D1) trainer
    code/tracking.py                    mlops / W&B tracking used by the trainer
    code/requirements.txt               packages (versions this was tested with)
{checkpoint_line}    SHA256SUMS                          checksums of every file above

## On the training machine

    tar -xzf {tar_name}
    cd {top}
    sha256sum -c --quiet SHA256SUMS && echo "all files OK"

    python3 -m venv .venv && . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the machine's CUDA
    pip install -r code/requirements.txt

    D="--data-root annotations --images images --csv-pattern ekyc_{{split}}.csv"

    # 1. score the original IMDB model on the test split (the baseline)
    python code/train_mivolo.py $D --eval-only --split test --init {init}

    # 2. fine-tune it on the training split (progress: runs/ekyc_finetune/train.log)
    python code/train_mivolo.py $D --init {init} \\
        --epochs 30 --warmup-epochs 3 --lr 1e-5 --batch-size 32 --log-every 10 \\
        --output runs/ekyc_finetune

    # 3. score the fine-tuned model on the same test split
    python code/train_mivolo.py $D --eval-only --split test --init runs/ekyc_finetune/best.pth.tar

    # continue an interrupted run
    python code/train_mivolo.py $D --init {init} ... --resume runs/ekyc_finetune/last.pth.tar

`best.pth.tar` is a standard MiVOLO face-only checkpoint. To evaluate it with the
official MiVOLO repository (github.com/WildChlamydia/MiVOLO, in its own environment):

    python eval_pretrained.py --dataset_images images --dataset_annotations annotations \\
        --dataset_name imdb --split test --checkpoint runs/ekyc_finetune/best.pth.tar

To train on IMDB-clean and this data together, see the docstring of code/train_mivolo.py.

## Tracking

Live training curves in Weights & Biases: run `wandb login` once, then add to step 2

    --wandb --wandb-project mivolo --wandb-name ekyc_finetune --wandb-tags ekyc

(`--wandb-mode offline` without internet; upload later with `wandb sync runs/ekyc_finetune/wandb`).

The mlops lineage tree lives on the machine that made this package; without the
mlops package here, train_mivolo.py says so and trains anyway. Back on that
machine, record the W&B run under this package's node ({pkg_node}):

    python3 -m mlops.integrations.wandb <entity>/mivolo/<run id> \\
        --project {project} --parents {pkg_node} --mode finetune
"""


def _tar_add_bytes(tar, name, data, mtime):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = mtime
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _csv_bytes(fieldnames, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def _requirements():
    from importlib.metadata import PackageNotFoundError, version

    lines = ["# torch: install the build matching the machine's CUDA first (pytorch.org)"]
    for name, spec in (("torch", "torch>=2.1"), ("timm", "timm>=1.0"),
                       ("opencv-python", "opencv-python"), ("numpy", "numpy"),
                       ("pillow", "pillow"), ("wandb", "wandb")):
        try:
            lines.append(f"{spec}  # tested with {version(name)}"
                         + ("; optional, for --wandb" if name == "wandb" else ""))
        except PackageNotFoundError:
            lines.append(spec)
    return "\n".join(lines) + "\n"


def cmd_package(args):
    root = Path(args.root).expanduser()
    log.info(f"===== package: {args.labels} + images from {root}")
    rows = read_annotations(args.labels)
    if replay_journal(args.labels, rows):
        log.info("note: included decisions from an unsaved review session")
    ok_files = ("ok", "truncated") if args.include_truncated else ("ok",)
    keep = [r for r in rows if r["review"] in REVIEWED and r["file_status"] in ok_files]
    log.info(f"{len(rows)} images in the CSV: {fmt_counts(Counter(r['review'] for r in rows))}; "
             f"packaging the {len(keep)} reviewed ones with readable files")
    if not keep:
        raise SystemExit("nothing to package: review some images first")

    stamp = time.strftime("%Y%m%d")
    out = Path(args.out or Path(args.labels).parent / f"ekyc_age_gender_{stamp}.tar.gz")
    top = out.name.removesuffix(".gz").removesuffix(".tar").removesuffix(".tgz")
    checkpoint = None if args.no_checkpoint else Path(args.checkpoint)
    if checkpoint is not None and not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint} (or use --no-checkpoint)")
    code = Path(__file__).parent

    # the package's mlops node is created first so its id can go into the README
    tracker = tracking.Tracker(args, log)
    ann_id = track_annotations(tracker, args.labels, rows, note="annotations as packaged")
    pkg_id = tracker.artifact(out.name, out, kind="export", note="being written",
                              parents=[ann_id] if ann_id else None, from_run=False)
    try:
        digest, n_files = _write_package(args, root, out, top, keep, checkpoint, code, pkg_id)
    except BaseException:
        if pkg_id:
            tracker.remove(pkg_id)
        raise
    tracker.artifact(out.name, out, kind="export", parents=[ann_id] if ann_id else None,
                     from_run=False,
                     note=f"{n_files} files, {out.stat().st_size / 1e6:.1f} MB, sha256 "
                          f"{digest[:12]}; images from {root}; valid {args.valid:g}, "
                          f"test {args.test:g}")


def _write_package(args, root, out, top, keep, checkpoint, code, pkg_id):

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    now = int(time.time())
    sums = {}  # path inside the package -> sha256
    splits = defaultdict(list)
    metadata = []
    skipped = []
    t0 = time.time()
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        def add(name, data):
            sums[name] = hashlib.sha256(data).hexdigest()
            _tar_add_bytes(tar, f"{top}/{name}", data, now)

        # ---- images: read once, check they decode, add straight to the tar
        def add_image(r):
            try:
                data = (root / r["path"]).read_bytes()
            except OSError as e:
                skipped.append((r["path"], e.strerror or str(e)))
                return
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None and not args.include_truncated:
                skipped.append((r["path"], "does not decode"))
                return
            h, w = (img.shape[:2] if img is not None
                    else (int(r["height"]), int(r["width"])))
            add(f"images/{r['path']}", data)
            split = split_of(r["person_id"] or r["path"], args.valid, args.test)
            splits[split].append({
                "img_name": r["path"], "age": int(r["age"]), "gender": r["gender"],
                "face_x0": 0, "face_y0": 0, "face_x1": w, "face_y1": h,
                "person_x0": -1, "person_y0": -1, "person_x1": -1, "person_y1": -1,
            })
            metadata.append({**r, "split": split})

        for n, r in enumerate(keep, 1):
            add_image(r)
            if n % args.log_every == 0 or n == len(keep):
                log.info(f"  images {n}/{len(keep)} ({100 * n / len(keep):.0f}%)"
                         f"{f', skipped {len(skipped)}' if skipped else ''}  "
                         f"left {fmt_duration((len(keep) - n) * (time.time() - t0) / n)}")
        for path, why in skipped[:10]:
            log.info(f"  WARNING: skipped {path}: {why}")
        if len(skipped) > 10:
            log.info(f"  WARNING: ... {len(skipped) - 10} more skipped")

        # ---- annotations (MiVOLO format), review metadata, code
        table = []
        for split in ("train", "valid", "test"):
            items = splits[split]
            add(f"annotations/ekyc_{split}.csv", _csv_bytes(MIVOLO_COLUMNS, items))
            minors = sum(r["age"] < 18 for r in items)
            people = len({Path(r["img_name"]).parent for r in items})
            g = Counter(r["gender"] for r in items)
            table.append(f"| {split} | {len(items)} | {people} | {minors} | {g['M']} / {g['F']} |")
            log.info(f"  {split:5s}: {len(items)} images, {people} customers, under 18: {minors}, "
                     f"M/F {g['M']}/{g['F']}"
                     + (f"; age groups {age_groups([r['age'] for r in items])}" if items else ""))
            if not items:
                log.info(f"  WARNING: the {split} split is empty")
        add("metadata/reviewed_annotations.csv", _csv_bytes(COLUMNS + ["split"], metadata))
        add("code/train_mivolo.py", (code / "train_mivolo.py").read_bytes())
        add("code/tracking.py", (code / "tracking.py").read_bytes())
        add("code/requirements.txt", _requirements().encode())

        init = "models/mivolo_volo_d1_face_imdb.pth.tar"
        if checkpoint is not None:
            log.info(f"  adding the starting checkpoint {checkpoint} "
                     f"({checkpoint.stat().st_size / 1e6:.0f} MB)")
            add(init, checkpoint.read_bytes())
            checkpoint_line = f"    {init}   MiVOLO face-only IMDB checkpoint to start from\n"
        else:
            init = "<MiVOLO face-only checkpoint>"
            checkpoint_line = ""

        readme = PACKAGE_README.format(
            created=time.strftime("%Y-%m-%d %H:%M"), labels=args.labels, table="\n".join(table),
            valid=args.valid, test=args.test, checkpoint_line=checkpoint_line,
            tar_name=out.name, top=top, init=init, project=args.mlops_project,
            pkg_node=pkg_id or "<id of this package's node in the mlops project>")
        add("README.md", readme.encode())
        _tar_add_bytes(tar, f"{top}/SHA256SUMS",
                       "".join(f"{h}  {name}\n" for name, h in sorted(sums.items())).encode(), now)

    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    tmp.replace(out)
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n")

    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(sums) + 1} files) "
             f"in {fmt_duration(time.time() - t0)}")
    log.info(f"archive checksum in {out.name}.sha256; after copying run: "
             f"sha256sum -c {out.name}.sha256")
    log.info(f"on the other machine: tar -xzf {out.name} && cat {top}/README.md")
    return digest, len(sums) + 1


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="eKYC age/gender pre-annotation, review, export.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict", help="run MiVOLO on every image, write annotations.csv",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pp.add_argument("--root", required=True, help="eKYC results folder")
    pp.add_argument("--labels", default=DEFAULT_LABELS)
    pp.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="face-only MiVOLO checkpoint (official or from train_mivolo.py)")
    pp.add_argument("--repredict", action="store_true",
                    help="refresh predictions of images already in the CSV (labels kept)")
    pp.add_argument("--limit", type=int, default=None,
                    help="predict only about N images not yet in the CSV (whole customer "
                         "folders); run again to add the next N")
    pp.add_argument("--sample", choices=["first", "random"], default="first",
                    help="with --limit: first folders in path order, or random customers")
    pp.add_argument("--seed", type=int, default=0, help="for --sample random")
    pp.add_argument("--batch-size", type=int, default=64)
    pp.add_argument("--log-every", type=int, default=20, help="progress line every N batches")
    pp.add_argument("--workers", type=int, default=6)
    pp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    rp = sub.add_parser("review", help="accept / correct labels in an OpenCV window",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    rp.add_argument("--root", required=True, help="eKYC results folder")
    rp.add_argument("--labels", default=DEFAULT_LABELS)
    rp.add_argument("--status", default="pending",
                    choices=["pending", "all", "reviewed", "accepted", "corrected", "rejected"])
    rp.add_argument("--group", default=None, help="e.g. matched or unmatched")
    rp.add_argument("--kind", default=None, help="e.g. image1 or image2")
    rp.add_argument("--min-age", type=float, default=None, help="filter on predicted age")
    rp.add_argument("--max-age", type=float, default=None, help="filter on predicted age")
    rp.add_argument("--sort", default="path", choices=["path", "pred_age", "gender_conf"])

    ep = sub.add_parser("export", help="reviewed labels -> ekyc_{train,valid,test}.csv",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ep.add_argument("--labels", default=DEFAULT_LABELS)
    ep.add_argument("--out", default=None, help="default: folder of --labels")
    ep.add_argument("--valid", type=float, default=0.1, help="fraction of customers")
    ep.add_argument("--test", type=float, default=0.1, help="fraction of customers")
    ep.add_argument("--include-truncated", action="store_true",
                    help="also export reviewed images whose JPEG is cut short")

    kp = sub.add_parser("package",
                        help="reviewed images + MiVOLO-format CSVs + trainer -> one .tar.gz",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    kp.add_argument("--root", required=True, help="eKYC results folder (the images)")
    kp.add_argument("--labels", default=DEFAULT_LABELS)
    kp.add_argument("--out", default=None,
                    help="default: <labels folder>/ekyc_age_gender_<date>.tar.gz")
    kp.add_argument("--valid", type=float, default=0.15, help="fraction of customers")
    kp.add_argument("--test", type=float, default=0.15, help="fraction of customers")
    kp.add_argument("--include-truncated", action="store_true",
                    help="also package reviewed images whose JPEG is cut short")
    kp.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="checkpoint to fine-tune from, added to the archive")
    kp.add_argument("--no-checkpoint", action="store_true",
                    help="leave the checkpoint out (about 100 MB smaller)")
    kp.add_argument("--log-every", type=int, default=200, help="progress line every N images")

    for sp in (pp, rp, ep, kp):
        tracking.add_args(sp)
    args = p.parse_args()
    setup_logging(args.labels)
    {"predict": cmd_predict, "review": cmd_review, "export": cmd_export,
     "package": cmd_package}[args.command](args)


if __name__ == "__main__":
    main()
