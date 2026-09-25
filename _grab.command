#!/bin/zsh
# Hilfsskript: ein Einzelbild der GoPro nach /tmp/probe_raw2.jpg speichern (fuer Analyse)
cd "$(dirname "$0")"
.venv/bin/python - > /tmp/grab.log 2>&1 <<'PY'
import cv2
from camera_stream import _CAPTURE_BACKEND
cap = cv2.VideoCapture(1, _CAPTURE_BACKEND)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
frame = None
for _ in range(15):
    r, f = cap.read()
    if r and f is not None: frame = f
cap.release()
if frame is None: print("kein Frame"); raise SystemExit
cv2.imwrite("/tmp/probe_raw2.jpg", frame); print("saved", frame.shape)
PY
