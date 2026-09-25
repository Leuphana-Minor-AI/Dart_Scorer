#!/bin/zsh
# Ausrichtungshilfe: zeigt das Kamerabild mit Fadenkreuz und Ziel-Ring.
# Aus Terminal.app starten (Kamerarechte). Tasten 0-4 wechseln die Kamera, Q beendet.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
.venv/bin/python test_gopro.py
