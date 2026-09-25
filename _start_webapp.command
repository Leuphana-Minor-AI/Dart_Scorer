#!/bin/zsh
# Startet die Dart-Scorer-Web-App (Erkennung + Browser-Oberflaeche).
# Aus Terminal.app starten (Kamerarechte). Beenden mit Ctrl+C.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
( sleep 8; open "http://localhost:20744" ) &
.venv/bin/python dart_app.py "$@" 2>&1 | tee /tmp/dartapp.log
