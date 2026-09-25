#!/bin/zsh
# Startet den Live Dart Scorer auf dem Mac (Pendant zu run_dart_scorer.bat)
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "[FEHLER] .venv nicht gefunden. Einmalig anlegen mit:"
    echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt opencv-python"
    exit 1
fi
exec .venv/bin/python live_scorer.py --model models/dart_sense_yolov8n.pt "$@"
