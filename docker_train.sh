#!/usr/bin/env bash
# ==============================================================================
# Run YOLO11n Training inside official Ultralytics Docker container on Nvidia Spark
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🎯 Dart Scorer YOLO11n Training via Docker on Nvidia Spark"
echo "============================================================"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed on this system. Please install Docker first."
    exit 1
fi

# Parameters (can be passed as arguments or use defaults)
EPOCHS="${1:-100}"
BATCH="${2:-16}"

echo "Starting Docker container with NVIDIA GPU support..."
echo "  - Epochs:  $EPOCHS"
echo "  - Batch:   $BATCH"
echo "  - Mount:   $SCRIPT_DIR -> /workspace"
echo ""

# Run official Ultralytics image with NVIDIA GPU support and current directory mounted
docker run --ipc=host --gpus all --rm \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    ultralytics/ultralytics:latest \
    python3 train.py --epochs "$EPOCHS" --batch "$BATCH" --device 0

echo ""
echo "============================================================"
echo "🎉 Training in Docker completed!"
echo "Trained model saved to: $SCRIPT_DIR/models/dart_yolo11n_best.pt"
echo "============================================================"
