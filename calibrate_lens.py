#!/usr/bin/env python3
"""Einmalige Linsenkalibrierung mit einem Schachbrettmuster.

Ablauf:
  1. Schachbrett ausdrucken (Standard: 9x6 innere Ecken = 10x7 Felder) und
     auf eine feste Unterlage kleben.
  2. python calibrate_lens.py --camera 1
  3. Schachbrett vor die Kamera halten: Wenn die Ecken farbig markiert werden,
     LEERTASTE druecken. Mindestens 10 Aufnahmen aus verschiedenen Positionen
     und Neigungen machen (Mitte, Raender, Ecken, gekippt).
  4. ENTER berechnet die Kalibrierung und speichert camera_calib.npz.
     live_scorer.py nutzt die Datei automatisch, wenn sie vorhanden ist.

Tasten: LEERTASTE = Aufnahme, ENTER = berechnen & speichern, Q/ESC = abbrechen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from camera_stream import _CAPTURE_BACKEND
from lens_undistort import LensUndistorter


def main() -> None:
    p = argparse.ArgumentParser(description="Linsenkalibrierung mit Schachbrett")
    p.add_argument("--camera", type=int, default=0, help="Kamera-Index")
    p.add_argument("--cols", type=int, default=9, help="Innere Ecken horizontal")
    p.add_argument("--rows", type=int, default=6, help="Innere Ecken vertikal")
    p.add_argument("--square", type=float, default=25.0, help="Feldgroesse in mm (nur Massstab)")
    p.add_argument("--out", type=str, default="camera_calib.npz", help="Ausgabedatei")
    p.add_argument("--min-shots", type=int, default=10, help="Mindestanzahl Aufnahmen")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    cap = cv2.VideoCapture(args.camera, _CAPTURE_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise SystemExit(f"Kamera [{args.camera}] konnte nicht geoeffnet werden.")

    pattern = (args.cols, args.rows)
    # 3D-Koordinaten der Schachbrettecken in der Musterebene (z = 0)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    obj_points = []
    img_points = []
    frame_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    win = "Linsenkalibrierung"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    print(f"Schachbrett {args.cols}x{args.rows} innere Ecken. LEERTASTE = Aufnahme, ENTER = fertig, Q = Abbruch")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        vis = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, pattern, corners, found)

        status = f"Aufnahmen: {len(img_points)}/{args.min_shots}   Muster: {'GEFUNDEN' if found else '---'}"
        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if found else (0, 165, 255), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("Abgebrochen.")
            break
        if key == ord(" ") and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"Aufnahme {len(img_points)} gespeichert.")
        if key in (13, 10):  # ENTER
            if len(img_points) < args.min_shots:
                print(f"Noch zu wenige Aufnahmen ({len(img_points)}/{args.min_shots}).")
                continue
            print("Berechne Kalibrierung ...")
            rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, frame_size, None, None)
            print(f"RMS-Reprojektionsfehler: {rms:.3f} px (gut: < 1.0)")
            LensUndistorter(K, dist, frame_size).save(out_path)
            print(f"Gespeichert: {out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
