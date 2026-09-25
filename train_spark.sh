#!/usr/bin/env bash
# ==============================================================================
# YOLO11n Dart Score Training Setup & Runner for Nvidia Spark (Linux)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🎯 Dart Scorer YOLO11n Training on Nvidia Spark"
echo "============================================================"

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 was not found. Please install Python 3.10+ (e.g., sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# 2. Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA Driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠️  WARNING: nvidia-smi not found. CUDA acceleration might not be available."
fi

# 3. Create or activate Virtual Environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated."

# 4. Install / Update dependencies
echo ""
echo "📦 Installing required dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify PyTorch with CUDA
echo ""
echo "🔍 Checking PyTorch CUDA status..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 6. Parse CLI options or use defaults
EPOCHS="${1:-100}"
BATCH="${2:-16}"
DEVICE="${3:-0}"

echo ""
echo "============================================================"
echo "🚀 Starting YOLO11n Training:"
echo "   - Epochs:   $EPOCHS"
echo "   - Batch:    $BATCH"
echo "   - Device:   $DEVICE"
echo "   - ImgSize:  800"
echo "============================================================"
echo ""

# Tip for persistent background execution:
# To keep training running if SSH disconnects, run inside tmux:
#   tmux new -s train
#   ./train_spark.sh
# Detach with: Ctrl+B then D. Reattach with: tmux attach -t train

python3 train.py \
    --data dataset/dataset.yaml \
    --model yolo11n.pt \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz 800 \
    --device "$DEVICE" \
    --project runs/train \
    --name yolo11n_darts \
    --save-model-dir models

echo ""
echo "============================================================"
echo "🎉 Training complete! Best weights saved to: models/dart_yolo11n_best.pt"
echo "============================================================"
