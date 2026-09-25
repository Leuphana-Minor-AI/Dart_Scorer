"""Convert the DeepDarts dataset (cropped_images + labels.pkl) to YOLO format.

Label layout per image in labels.pkl: normalized xy points, the first 4 are
the board calibration points (top/bottom/left/right on the double ring), any
further points are dart tips.

We emit a 5-class DETECTION dataset (small fixed-size boxes around each
keypoint), which is the same trick the DeepDarts paper uses:
  0 cal_top, 1 cal_bottom, 2 cal_left, 3 cal_right, 4 dart_tip

Split is done per session folder (not per image!) so near-duplicate frames of
the same throw never end up in both train and val.

Usage:
  python tools/prepare_dataset.py --images <pfad>/cropped_images/800 --out dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

CLASS_NAMES = ["cal_top", "cal_bottom", "cal_left", "cal_right", "dart_tip"]
BOX_SIZE = 0.025  # box width/height as fraction of image size


def yolo_lines(xy: list[list[float]]) -> list[str]:
    lines = []
    for i, (x, y) in enumerate(xy):
        cls = i if i < 4 else 4
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue  # a few labels lie slightly outside the crop
        lines.append(f"{cls} {x:.6f} {y:.6f} {BOX_SIZE} {BOX_SIZE}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Pfad zu .../cropped_images/800")
    p.add_argument("--labels", default="data/labels.pkl")
    p.add_argument("--out", default="dataset")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    images_root = Path(args.images)
    out = Path(args.out)
    df = pd.read_pickle(args.labels)

    folders = sorted(df.img_folder.unique())
    rng = random.Random(args.seed)
    rng.shuffle(folders)
    n_val = max(1, int(len(folders) * args.val_fraction))
    val_folders = set(folders[:n_val])
    print(f"{len(folders)} Sessions -> {len(folders) - n_val} train, "
          f"{n_val} val: {sorted(val_folders)}")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for row in df.itertuples():
        split = "val" if row.img_folder in val_folders else "train"
        stem = f"{row.img_folder}_{Path(row.img_name).stem}"
        src = images_root / row.img_folder / row.img_name
        dst = out / "images" / split / f"{stem}.jpg"
        if not dst.exists():
            shutil.copyfile(src, dst)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(yolo_lines(row.xy)) + "\n")
        counts[split] += 1

    yaml = out / "dataset.yaml"
    yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    print(f"Fertig: {counts['train']} train / {counts['val']} val Bilder")
    print(f"Dataset-Config: {yaml.resolve()}")


if __name__ == "__main__":
    main()
