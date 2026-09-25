"""Evaluate a trained detector on the val split: Percent Correct Score (PCS).

For each val image we compare the sum of predicted dart scores against the
ground truth from the YOLO label files (scored through the same geometry).

Usage:
  python tools/eval_model.py --model models/dart_yolo11n_best.pt [--limit 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dart_geometry import (CAL_CLASSES, CAL_REFERENCE_MM,
                               DART_CLASS, score_at)
except ImportError:
    from dartscorer.model_detection import (CAL_CLASSES, CAL_REFERENCE_MM,
                                            DART_CLASS)
    from dartscorer.geometry import score_at


def truth_points(label_file: Path) -> tuple[dict[int, tuple[float, float]], list]:
    cal, darts = {}, []
    for line in label_file.read_text().splitlines():
        c, x, y, *_ = line.split()
        c, x, y = int(c), float(x), float(y)
        if c == DART_CLASS:
            darts.append((x, y))
        else:
            cal[c] = (x, y)
    return cal, darts


def truth_score(cal: dict, darts: list, w: int, h: int) -> int | None:
    if len(cal) < 4:
        return None
    src = np.array([[cal[c][0] * w, cal[c][1] * h] for c in CAL_CLASSES])
    dst = np.array([CAL_REFERENCE_MM[c] for c in CAL_CLASSES])
    m, _ = cv2.findHomography(src, dst)
    total = 0
    for x, y in darts:
        pt = cv2.perspectiveTransform(np.array([[[x * w, y * h]]]), m)
        total += score_at(float(pt[0, 0, 0]), float(pt[0, 0, 1])).points
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    images = sorted(Path(args.dataset, "images", "val").glob("*.jpg"))
    if args.limit:
        images = images[:args.limit]

    correct = failed = 0
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        cal, darts = truth_points(
            Path(args.dataset, "labels", "val", img_path.stem + ".txt"))
        expected = truth_score(cal, darts, w, h)
        if expected is None:
            continue
        try:
            got = sum(s.points for s in score_image(model, img, conf=args.conf).scores)
        except RuntimeError:
            failed += 1
            continue
        correct += (got == expected)
        if i % 100 == 0:
            print(f"  {i}/{len(images)}  PCS bisher: {correct / i:.1%}")

    n = len(images)
    print(f"\nPCS (Percent Correct Score): {correct}/{n} = {correct / n:.1%}")
    print(f"Bilder ohne vollstaendige Kalibrierpunkte: {failed}")


if __name__ == "__main__":
    main()
