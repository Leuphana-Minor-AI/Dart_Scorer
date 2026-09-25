#!/usr/bin/env python3
"""Train a YOLO11n model for Dart Scoreboard Detection (DeepDarts dataset).

This script trains Ultralytics YOLO11n on the 5-class dartboard dataset:
  0: cal_top     (outer double ring wire intersection 5|20)
  1: cal_bottom  (outer double ring wire intersection 3|17)
  2: cal_left    (outer double ring wire intersection 11|8)
  3: cal_right   (outer double ring wire intersection 6|13)
  4: dart_tip    (point of each thrown dart)

Usage:
  python train.py --epochs 100 --batch 16 --imgsz 800 --device 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def check_hardware(requested_device: str | int | None) -> str | int:
    """Check PyTorch, CUDA, and GPU hardware availability."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed in the current Python environment.")
        print("Install it via: pip install torch torchvision")
        sys.exit(1)

    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"  [{i}] {props.name} ({vram_gb:.2f} GB VRAM, Compute {props.major}.{props.minor})")
        if requested_device is None:
            return 0
        return requested_device
    else:
        print("\nWARNING: No CUDA GPU detected! Training on CPU will be significantly slower.")
        if requested_device is not None and str(requested_device) != "cpu":
            print(f"Requested device '{requested_device}' not available, falling back to 'cpu'.")
        return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on the DeepDarts 5-class dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model weights or config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (-1 for AutoBatch)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Input image resolution (DeepDarts images are 800x800)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. '0', '0,1', or 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader worker threads",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without validation improvement)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory to save training runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_darts",
        help="Experiment name",
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="models",
        help="Directory where dart_yolo11n_best.pt will be saved",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint in project/name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Hardware check
    device = check_hardware(args.device)
    print(f"Active Device for Training: {device}")
    print("=" * 60)

    # Verify dataset yaml exists
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"ERROR: Dataset YAML not found at: {data_path}")
        sys.exit(1)

    # Ensure dataset.yaml points to the correct images directory
    try:
        yaml_content = data_path.read_text(encoding="utf-8")
        dataset_dir = data_path.parent.resolve().as_posix()
        # If path is '.' or hardcoded to another user path, fix to dataset directory
        lines = []
        for line in yaml_content.splitlines():
            if line.strip().startswith("path:"):
                lines.append(f"path: {dataset_dir}")
            else:
                lines.append(line)
        fixed_yaml = "\n".join(lines) + "\n"
        if fixed_yaml != yaml_content:
            data_path.write_text(fixed_yaml, encoding="utf-8")
            print(f"Updated dataset.yaml path to: {dataset_dir}")
    except Exception as e:
        print(f"Warning: Could not auto-adjust dataset.yaml: {e}")

    print(f"Using dataset configuration: {data_path}")

    # Import Ultralytics and force workspace output directory
    try:
        from ultralytics import YOLO, settings
        workspace_dir = Path(__file__).resolve().parent
        runs_dir = workspace_dir / "runs"
        models_dir = workspace_dir / "models"
        runs_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        settings.update({"runs_dir": str(runs_dir)})
        print(f"Ultralytics runs_dir configured to: {runs_dir}")
    except ImportError:
        print("ERROR: Ultralytics is not installed. Install via: pip install ultralytics")
        sys.exit(1)

    # Initialize model
    print(f"Loading base model: {args.model}...")
    model = YOLO(args.model)

    project_dir = str(runs_dir / "train")

    # Run training
    print("\nStarting YOLO11n training...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        project=project_dir,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # Detection hyperparameter fine-tuning for small keypoint-like bounding boxes
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training finished successfully!")

    # Robustly find best.pt and copy to workspace models/
    target_pt = models_dir / "dart_yolo11n_best.pt"
    saved = False

    candidate_paths = []
    if hasattr(model, "trainer") and model.trainer:
        if getattr(model.trainer, "best", None):
            candidate_paths.append(Path(model.trainer.best))
        if getattr(model.trainer, "save_dir", None):
            candidate_paths.append(Path(model.trainer.save_dir) / "weights" / "best.pt")

    candidate_paths.extend([
        runs_dir / "train" / args.name / "weights" / "best.pt",
        Path("/ultralytics/runs/detect/runs/train") / args.name / "weights" / "best.pt",
    ])
    # Also glob search in case ultralytics nested it
    candidate_paths.extend(list(Path("/ultralytics").glob("**/best.pt")))
    candidate_paths.extend(list(runs_dir.glob("**/best.pt")))

    for cand in candidate_paths:
        if cand.exists() and cand.is_file():
            shutil.copyfile(cand, target_pt)
            print(f"✓ Best model weights successfully saved to: {target_pt.resolve()}")
            saved = True
            # Also copy results.png if available
            cand_dir = cand.parent.parent
            res_png = cand_dir / "results.png"
            if res_png.exists():
                shutil.copyfile(res_png, models_dir / "results.png")
                print(f"✓ Training curves saved to: {(models_dir / 'results.png').resolve()}")
            break

    if not saved:
        print(f"⚠️ Warning: Could not locate best.pt automatically. Checked: {candidate_paths[:4]}")

    # Run final validation
    print("\nRunning final validation evaluation...")
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=device)
    print("\nValidation Summary:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
