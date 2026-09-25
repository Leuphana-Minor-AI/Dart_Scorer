#!/usr/bin/env python3
"""Evaluate a trained YOLO11n dart scorer model.

Computes:
1. Detection metrics (mAP50, mAP50-95) on the validation split.
2. Percent Correct Score (PCS): compares predicted turn score sum vs ground truth score.
3. Optional visualization: draws calibration points, dart tips, and score text onto images.

Usage:
  python evaluate.py --model models/dart_yolo11n_best.pt
  python evaluate.py --model models/dart_yolo11n_best.pt --image dataset/images/val/sample.jpg --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dart_geometry import (
    CAL_CLASSES,
    CAL_REFERENCE_MM,
    DART_CLASS,
    DartScore,
    compute_homography,
    project_point_to_mm,
    score_at,
)


def read_truth_from_label_file(
    label_path: Path,
) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[float, float]]]:
    """Parse ground truth calibration points and dart tips from YOLO label file."""
    cal: Dict[int, Tuple[float, float]] = {}
    darts: List[Tuple[float, float]] = []

    if not label_path.exists():
        return cal, darts

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        c = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        if c == DART_CLASS:
            darts.append((x, y))
        elif c in CAL_CLASSES:
            cal[c] = (x, y)
    return cal, darts


def compute_ground_truth_score(
    cal: Dict[int, Tuple[float, float]],
    darts: List[Tuple[float, float]],
    w: int,
    h: int,
) -> Optional[int]:
    """Calculate the expected total score from ground truth coordinates."""
    homography = compute_homography(cal, w, h)
    if homography is None:
        return None

    total_score = 0
    for x_norm, y_norm in darts:
        x_mm, y_mm = project_point_to_mm(x_norm * w, y_norm * h, homography)
        total_score += score_at(x_mm, y_mm).points
    return total_score


def predict_scores_for_image(
    model,
    img: np.ndarray,
    conf: float = 0.25,
) -> Tuple[Optional[List[DartScore]], Optional[np.ndarray], Dict[int, Tuple[float, float]]]:
    """Run model prediction on an image, extract calibration points and score each dart."""
    h, w = img.shape[:2]
    results = model(img, conf=conf, verbose=False)[0]

    # Find highest confidence detection for each calibration class
    cal_points: Dict[int, Tuple[float, float]] = {}
    cal_confs: Dict[int, float] = {}
    detected_darts: List[Tuple[float, float]] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        # Box center normalized
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h

        if cls_id in CAL_CLASSES:
            if cls_id not in cal_confs or score > cal_confs[cls_id]:
                cal_confs[cls_id] = score
                cal_points[cls_id] = (cx, cy)
        elif cls_id == DART_CLASS:
            detected_darts.append((cx, cy))

    homography = compute_homography(cal_points, w, h)
    if homography is None:
        return None, None, cal_points

    scores: List[DartScore] = []
    for cx, cy in detected_darts:
        x_mm, y_mm = project_point_to_mm(cx * w, cy * h, homography)
        scores.append(score_at(x_mm, y_mm))

    return scores, homography, cal_points


def draw_overlay(
    img: np.ndarray,
    scores: Optional[List[DartScore]],
    cal_points: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """Draw calibration points, dart scores, and turn total onto the image."""
    vis = img.copy()
    h, w = vis.shape[:2]

    cal_colors = {
        0: (0, 255, 0),    # top: green
        1: (255, 0, 0),    # bottom: blue
        2: (0, 255, 255),  # left: yellow
        3: (255, 165, 0),  # right: orange
    }

    for c, (nx, ny) in cal_points.items():
        px, py = int(nx * w), int(ny * h)
        cv2.circle(vis, (px, py), 6, cal_colors.get(c, (0, 255, 0)), -1)
        cv2.putText(vis, f"C{c}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if scores is not None:
        total = sum(s.points for s in scores)
        desc_list = [s.description for s in scores]
        text = f"Darts: {len(scores)} | Score: {total} ({', '.join(desc_list)})"
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "Calibration Incomplete (<4 points)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


def evaluate_val_dataset(model, dataset_dir: Path, conf: float, limit: int = 0) -> None:
    val_images = sorted((dataset_dir / "images" / "val").glob("*.jpg"))
    if limit > 0:
        val_images = val_images[:limit]

    total_images = len(val_images)
    print(f"Evaluating PCS on {total_images} validation images (conf={conf})...")

    correct = 0
    cal_failures = 0
    total_evaluated = 0

    for i, img_path in enumerate(val_images, 1):
        label_path = dataset_dir / "labels" / "val" / f"{img_path.stem}.txt"
        cal_gt, darts_gt = read_truth_from_label_file(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        expected_score = compute_ground_truth_score(cal_gt, darts_gt, w, h)
        if expected_score is None:
            continue

        scores, _, _ = predict_scores_for_image(model, img, conf=conf)
        if scores is None:
            cal_failures += 1
            got_score = 0
        else:
            got_score = sum(s.points for s in scores)

        if got_score == expected_score:
            correct += 1
        total_evaluated += 1

        if i % 100 == 0 or i == total_images:
            pcs = (correct / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
            print(f"  [{i}/{total_images}] Current PCS: {pcs:.2f}% ({correct}/{total_evaluated})")

    pcs_final = (correct / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
    print("\n" + "=" * 60)
    print("PCS (Percent Correct Score) Evaluation Finished:")
    print(f"  Total images evaluated:  {total_evaluated}")
    print(f"  Correct score matches:   {correct}")
    print(f"  PCS Accuracy:            {pcs_final:.2f}%")
    print(f"  Calibration failures:    {cal_failures}")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate YOLO11n Dart Scorer Model.")
    p.add_argument("--model", type=str, required=True, help="Path to best.pt")
    p.add_argument("--dataset", type=str, default="dataset", help="Path to dataset folder")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--limit", type=int, default=0, help="Limit number of validation images (0 = all)")
    p.add_argument("--image", type=str, default=None, help="Path to single image to evaluate")
    p.add_argument("--save-vis", type=str, default=None, help="Save visualization output image path")
    args = p.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}...")
    model = YOLO(str(model_path))

    if args.image:
        img_path = Path(args.image)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"ERROR: Could not load image {img_path}")
            sys.exit(1)

        scores, _, cal_points = predict_scores_for_image(model, img, conf=args.conf)
        if scores is not None:
            total = sum(s.points for s in scores)
            print(f"\nResult for {img_path.name}:")
            print(f"  Darts detected: {len(scores)}")
            for idx, s in enumerate(scores, 1):
                print(f"    Dart {idx}: {s.description} ({s.points} pts) [radius: {s.radius_mm:.1f} mm]")
            print(f"  TOTAL SCORE: {total}")
        else:
            print(f"\nResult for {img_path.name}: Could not detect all 4 calibration points.")

        if args.save_vis:
            out_vis = draw_overlay(img, scores, cal_points)
            cv2.imwrite(args.save_vis, out_vis)
            print(f"Visualization saved to: {args.save_vis}")
    else:
        evaluate_val_dataset(model, Path(args.dataset), conf=args.conf, limit=args.limit)


if __name__ == "__main__":
    main()
