"""Run the best MiVOLO face-only checkpoint on a single image.

By default the whole image is treated as the face crop, the same convention
``ekyc_age_gender.py`` uses for the eKYC face-crop folders. Pass --box if the
image is a full photo and you already know the face's pixel box.

Usage
-----
    python infer_mivolo.py path/to/face_crop.jpg
    python infer_mivolo.py path/to/photo.jpg --box x0 y0 x1 y1
    python infer_mivolo.py path/to/face_crop.jpg --checkpoint runs/mivolo_face/best.pth.tar
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_mivolo import IMAGENET_MEAN, IMAGENET_STD, letterbox, load_weights

DEFAULT_CHECKPOINT = "runs/mivolo_face/best.pth.tar"


def load_model(path, device):
    import timm

    meta = torch.load(path, map_location="cpu", weights_only=False)
    if meta.get("with_persons_model"):
        raise SystemExit(f"{path} is a face+body model; a face-only checkpoint is needed")
    no_gender = bool(meta.get("no_gender", False))
    model = timm.create_model("volo_d1_224", num_classes=1 if no_gender else 3)
    load_weights(model, path)
    return model.eval().to(device), float(meta["min_age"]), float(meta["max_age"]), no_gender


def preprocess(img, box=None, img_size=224):
    if box is not None:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    rgb = cv2.cvtColor(letterbox(img, img_size), cv2.COLOR_BGR2RGB)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("image")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                   help="face box in --image, if it is a full photo rather than a face crop")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, min_age, max_age, no_gender = load_model(args.checkpoint, device)
    avg_age = (min_age + max_age) / 2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"can not read {args.image}")
    x = preprocess(img, args.box).to(device)

    with torch.no_grad():
        out = model(x).float().cpu()[0]
    age = (out[0 if no_gender else 2] * (max_age - min_age) + avg_age).clamp(min=0).item()

    if no_gender:
        print(f"{args.image}: age={age:.1f}")
    else:
        probs = out[:2].softmax(0)
        g = int(probs.argmax())
        print(f"{args.image}: age={age:.1f}  gender={'MF'[g]}  gender_conf={probs[g].item():.3f}")


if __name__ == "__main__":
    main()
