#!/bin/zsh
# Startet den Live Dart Scorer (muss aus Terminal.app laufen,
# sonst verweigert macOS den Kamerazugriff).
# Kamera-Index ggf. anpassen: GoPro war bisher Index 0, MacBook-Kamera Index 1.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
./run_dart_scorer.sh --mode PRACTICE 2>&1 | tee /tmp/scorer.log
