#!/usr/bin/env python3
"""Zwei Dart-Modelle auf Testbildern vergleichen.

Prueft je Modell und Eingabegroesse: gefundene Kalibrierpunkte (Konfidenz) und
Pfeil-Erkennungen. Fuer Bilder mit bekannten Spitzenpositionen (GROUND_TRUTH)
wird zusaetzlich der Abstand der naechsten Erkennung zur echten Spitze ausgegeben.

Beispiel:
  python tools/compare_models.py models/dart_yolo11n_best.pt models/dart_sense_yolov8n.pt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Visuell abgelesene Eintrittspunkte (Vollbild-Pixel) fuer test_images/gopro_3darts.jpg
GROUND_TRUTH = {
    "gopro_3darts.jpg": {"A": (920, 325), "B": (1015, 345), "C": (818, 428)},
}
# Board-Ausschnitt fuer die GoPro-Bilder (x, y, seite) - wie vom Scorer bestimmt
ROI = (488, 68, 762)
CAL_NAMES = {0: "top", 1: "bottom", 2: "left", 3: "right"}


def run(model, img, imgsz, conf, off):
    res = model(img, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    cals, darts = {}, []
    for b in res.boxes:
        c = int(b.cls[0])
        s = float(b.conf[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1]
        if c in CAL_NAMES:
            if c not in cals or s > cals[c][0]:
                cals[c] = (s, (cx, cy))
        elif c == 4:
            darts.append((s, (cx, cy)))
    return cals, sorted(darts, key=lambda d: -d[0])


def main() -> None:
    from ultralytics import YOLO
    import torch

    global DEVICE
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    weights = sys.argv[1:] or [str(ROOT / "models/dart_yolo11n_best.pt")]
    images = sorted((ROOT / "test_images").glob("gopro_*.jpg"))
    for w in weights:
        model = YOLO(w)
        print(f"\n{'=' * 70}\nModell: {w}\n  Klassen: {model.names}")
        for img_path in images:
            frame = cv2.imread(str(img_path))
            x, y, s = ROI
            crop = frame[y:y + s, x:x + s]
            gt = GROUND_TRUTH.get(img_path.name, {})
            print(f"\n  Bild {img_path.name} (Ausschnitt {s}px):")
            for imgsz in (640, 800, 960, 1120):
                cals, darts = run(model, crop, imgsz, 0.25, (x, y))
                cal_txt = ", ".join(f"{CAL_NAMES[c]} {v[0]:.2f}" for c, v in sorted(cals.items()))
                line = f"    @{imgsz:4d}: Kal-Punkte [{cal_txt}] | Pfeile: {len(darts)}"
                if gt:
                    errs = []
                    for name, tip in gt.items():
                        best = min((math.hypot(d[1][0] - tip[0], d[1][1] - tip[1]) for d in darts), default=float("nan"))
                        errs.append(f"{name}:{best:.0f}px")
                    line += " | Abstand naechste Erkennung zur Spitze: " + " ".join(errs)
                print(line)
            # Annotiertes Bild speichern
            out = ROOT / "snapshots" / f"compare_{Path(w).stem}_{img_path.stem}.jpg"
            out.parent.mkdir(exist_ok=True)
            res = model(crop, conf=0.25, imgsz=960, device=DEVICE, verbose=False)[0]
            cv2.imwrite(str(out), res.plot())
            print(f"    -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
