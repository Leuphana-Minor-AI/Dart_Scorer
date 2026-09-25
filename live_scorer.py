#!/usr/bin/env python3
"""Live Dart Scorer Application with GoPro/Webcam & YOLO11n AI Model.

Features:
- Real-time video ingestion from GoPro / Webcam / Fallback test images.
- YOLO11n detection of 4 board calibration points & dart tips.
- Projective homography mapping pixel coordinates to official WDF millimeter board coordinates.
- Calibration Lock / EMA stabilization: prevents calibration loss when players throw or stand in front.
- Dart debounce tracking across frames to avoid flicker.
- Game modes: Free Practice & 501 / 301 X01 Countdown with checkout suggestions.
- Live HUD with camera overlay, 2D Dartboard Radar widget and turn stats.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_stream import CameraStream, list_available_cameras
from lens_undistort import LensUndistorter
from dart_geometry import (
    BOARD_RADIUS_MM,
    CAL_BOTTOM,
    CAL_CLASSES,
    CAL_EXTRA_CLASSES,
    CAL_LEFT,
    CAL_REFERENCE_MM,
    CAL_RIGHT,
    CAL_TOP,
    DART_CLASS,
    SECTORS_CLOCKWISE,
    DartScore,
    compute_homography,
    project_point_to_mm,
    score_at,
    wire_distance_mm,
)

# Pfeile naeher als so viel mm an einem Draht werden im HUD als unsicher markiert
UNCERTAIN_WIRE_MM = 3.0

# Pfeiltasten-Codes von cv2.waitKeyEx (macOS / Windows / Linux)
KEY_LEFT = {63234, 2424832, 65361}
KEY_RIGHT = {63235, 2555904, 65363}
KEY_UP = {63232, 2490368, 65362}
KEY_DOWN = {63233, 2621440, 65364}
ARROW_KEYS = KEY_LEFT | KEY_RIGHT | KEY_UP | KEY_DOWN

# Checkout routes for standard 501 finishes
CHECKOUTS: Dict[int, str] = {
    170: "T20 T20 Bull",
    167: "T20 T19 Bull",
    164: "T20 T18 Bull",
    161: "T20 T17 Bull",
    160: "T20 T20 D20",
    158: "T20 T20 D19",
    157: "T20 T19 D20",
    156: "T20 T20 D18",
    155: "T20 T19 D19",
    154: "T20 T18 D20",
    153: "T20 T19 D18",
    152: "T20 T20 D16",
    151: "T20 T17 D20",
    150: "T20 T18 D18",
    149: "T20 T19 D16",
    148: "T20 T20 D14",
    147: "T20 T17 D18",
    146: "T20 T18 D16",
    145: "T20 T15 D20",
    144: "T20 T20 D12",
    143: "T20 T17 D16",
    142: "T20 T14 D20",
    141: "T20 T19 D12",
    140: "T20 T20 D10",
    130: "T20 T20 D5",
    128: "T18 T14 D16",
    124: "T20 T16 D8",
    121: "T20 T15 D8",
    120: "T20 S20 D20",
    110: "T20 S10 D20",
    100: "T20 D20",
    80: "T20 D10",
    60: "S20 D20",
    50: "Bullseye",
    40: "D20",
    36: "D18",
    32: "D16",
    24: "D12",
    20: "D10",
    16: "D8",
    8: "D4",
    4: "D2",
    2: "D1",
}


@dataclass
class DartTrack:
    """Ein ueber mehrere Frames verfolgter Pfeil (Bildposition geglaettet)."""
    px: float
    py: float
    hits: int
    first_seen: float
    last_seen: float
    override: Optional[DartScore] = None  # manuell korrigierte Wertung
    id: int = 0


@dataclass
class TrackedDart:
    px: float
    py: float
    x_mm: float
    y_mm: float
    score: DartScore
    frames_seen: int = 1
    uncertain: bool = False           # nahe an einem Draht
    track: Optional[DartTrack] = None


# Pfeile ausserhalb dieses Radius (Double-Ring = 170 mm) ignorieren: dort sitzen
# nur die Metallklammern des Zahlenrings, die das Modell gern fuer Spitzen haelt.
MAX_DART_RADIUS_MM = 180.0


# Bildausschnitt (Region of Interest) als (x, y, breite, hoehe) in Vollbild-Pixeln
Roi = Tuple[int, int, int, int]


def _roi_iou(a: Roi, b: Roi) -> float:
    """Überlappung zweier Ausschnitte (Intersection over Union)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class LiveDartScorer:
    def __init__(
        self,
        model_path: str,
        camera_idx: Optional[int] = None,
        conf_thresh: float = 0.25,
        target_score: int = 501,
        mode: str = "PRACTICE",  # "PRACTICE" or "501"
        calib_path: Optional[str] = None,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.conf_thresh = conf_thresh
        # Pfeile duerfen unsicherer sein als Kalibrierpunkte: Pfeile, die sich gegenseitig
        # verdecken, bekommen oft nur 0.15-0.25. Die Spurverfolgung (3 Treffer) und der
        # Radiusfilter halten Fehltreffer trotzdem draussen.
        self.dart_conf = min(conf_thresh, 0.15)
        self.game_mode = mode
        self.start_score = target_score
        self.current_score = target_score

        # YOLO Model
        from ultralytics import YOLO
        import torch
        print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        # Ultralytics waehlt Apple-GPU (MPS) nicht automatisch -> explizit setzen (8-14x schneller als CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Inferenz-Device: {self.device}")

        # Camera
        self.camera_idx = camera_idx
        self.stream = CameraStream(camera_index=camera_idx).start()

        # Optionale Linsenentzerrung (Kalibrierdatei aus calibrate_lens.py)
        self.undistorter = LensUndistorter.load(calib_path) if calib_path else None
        if self.undistorter is not None:
            print(f"Linsenentzerrung aktiv: {calib_path}")

        # Board-Ausschnitt (zweistufige Erkennung): Das Board wird zuerst im
        # Vollbild gesucht, danach wird nur noch dieser Ausschnitt in voller
        # Aufloesung ausgewertet - so bleibt das Board fuer YOLO immer gross,
        # egal wie weit die Kamera entfernt steht.
        self.roi: Optional[Roi] = None
        self.roi_manual = False
        self.board_px: Optional[float] = None  # geschaetzter Board-Durchmesser in Pixeln
        self.last_cal_seen = 0.0
        self.roi_lost_timeout = 5.0   # Sekunden ohne Kalibrierpunkt -> neue Suche
        self.search_interval = 10     # jede n-te Frame Kachelsuche in voller Aufloesung
        # Das Modell erkennt die Kalibrierpunkte am besten, wenn das Board in der
        # Modell-Eingabe ~500px gross ist; staerker als 1.5x hochskalieren bringt
        # nichts mehr (Unschaerfe).
        self.target_board_px = 500
        self.max_upscale = 1.5

        # Calibration state
        self.homography: Optional[np.ndarray] = None
        self.inv_homography: Optional[np.ndarray] = None
        self.cal_points_norm: Dict[int, Tuple[float, float]] = {}
        self.cal_locked = False
        self.cal_history: Dict[int, deque] = {c: deque(maxlen=15) for c in CAL_CLASSES}
        self.stable_cal_frames = 0
        # Punktgedaechtnis: Board und Kamera stehen still, daher darf ein kurz
        # nicht erkannter Kalibrierpunkt einige Sekunden nachwirken. So reicht
        # es, wenn jeder Punkt irgendwann gesehen wird - nicht alle gleichzeitig.
        self.cal_memory: Dict[int, Tuple[Tuple[float, float], float]] = {}
        self.cal_memory_ttl = 3.0
        # Auto-Rekalibrierung, wenn sich Board oder Kamera nach dem Einrasten bewegen
        self._drift_frames = 0
        self.drift_frames_needed = 30
        self._nocal_since = 0.0
        self.drift_nocal_timeout = 15.0

        # Dart tracking: Erkennungen ueber Frames hinweg verfolgen, damit ein Pfeil
        # erst nach mehreren Treffern zaehlt (kein Flackern, keine Einzel-Fehltreffer)
        self.dart_tracks: List[DartTrack] = []
        self._track_seq = 0
        # Web-Modus: Spiellogik laeuft ausserhalb (engine/ + game/), HUD ohne Seitenleiste
        self.external_game = False
        self.draw_sidebar = True
        self.track_match_px = 25.0
        self.track_merge_px = 30.0        # Duplikat Spitze/Schaft: bis 30 px entlang der Schaftrichtung
        self.track_merge_close_px = 6.0   # unter 6 px immer derselbe Punkt
        self.track_merge_align = 0.7      # cos(45 Grad): Duplikate liegen entlang der Schaftrichtung
        # Richtung von der Spitze zum Flight im Bild (Einheitsvektor). Kamera unterhalb des
        # Boards: Flights zeigen nach oben und leicht nach rechts. Aus Messungen an
        # mehreren Pfeilen bestimmt; bei anderer Kameraposition anpassen.
        self.shaft_dir = (0.45, -0.89)
        self.track_min_hits = 3
        self.track_expire_s = 0.8
        # Konstante Parallaxen-Korrektur in Board-mm (Taste B: Pfeil im Bull -> Offset messen)
        self.tip_offset_mm: Tuple[float, float] = (0.0, 0.0)
        self.offset_path = Path(__file__).resolve().parent / "dart_offset.json"
        self._load_tip_offset()

        # Stoerstellen: Punkte auf dem leeren Board, die das Modell fuer Pfeilspitzen haelt
        # (z. B. die Bull-Mitte, Klammern, Flecken). Werden nach dem Einrasten ~2 s lang
        # automatisch gelernt (Board muss dann leer sein) und danach ausgeblendet.
        self.artifacts: List[Tuple[float, float]] = []
        self.artifact_radius_px = 14.0
        self.artifact_learn_frames = 40
        self.artifact_max_conf = 0.45
        self._artifact_samples: Optional[Dict[Tuple[int, int], int]] = None
        self._artifact_frames_left = 0

        self.active_darts: List[TrackedDart] = []
        self._last_darts: List[TrackedDart] = []
        self.selected_dart = 0  # fuer manuelle Korrektur (TAB wechselt)
        # Zusaetzliche Kalibrierpunkte (dart-sense Klassen 5/6): Position, Zeitpunkt
        self._extra_seen: Dict[int, Tuple[Tuple[float, float], float]] = {}
        self.dart_history: deque = deque(maxlen=10)
        self.turn_darts: List[DartScore] = []
        self.turn_history: List[List[DartScore]] = []
        self.last_turn_sum = 0
        self.prev_detected_count = 0
        self.empty_frames_count = 0

        # Performance
        self.fps = 0.0
        self.prev_time = time.time()
        self.frame_count = 0

    def _resolve_model_path(self, path_str: str) -> Path:
        base = Path(__file__).resolve().parent
        candidates = [
            Path(path_str),
            base / path_str,
            # dart-sense (YOLOv8n, ~24k Bilder aus vielen Kamerawinkeln, CC BY-NC 4.0):
            # findet Eintrittspunkte auch aus schraeger Perspektive - bevorzugt
            base / "models" / "dart_sense_yolov8n.pt",
            base / "models" / "dart_yolo11n_best.pt",
            base / "dart_yolo11n_best.pt",
            base / "runs" / "training" / "yolo11n_darts" / "weights" / "best.pt",
            base / "yolo11n.pt",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError(f"Model file not found. Checked: {[str(c) for c in candidates]}")

    @staticmethod
    def _plausible_cal_set(pts_px: Dict[int, Tuple[float, float]]) -> Tuple[bool, Optional[int]]:
        """Pruefen, ob vier Kalibrierpunkte geometrisch zu einem Board passen.

        oben-unten und links-rechts sind Durchmesser des Boards: ihre Mittelpunkte
        muessen (auch in Perspektive) nahezu zusammenfallen, die Laengen aehnlich
        sein und die Punkte im Uhrzeigersinn oben-rechts-unten-links liegen.
        Liefert (plausibel, verdaechtigste Klasse).
        """
        P = {c: np.array(pts_px[c], dtype=np.float64) for c in CAL_CLASSES}
        tb = P[CAL_BOTTOM] - P[CAL_TOP]
        lr = P[CAL_RIGHT] - P[CAL_LEFT]
        len_tb, len_lr = float(np.linalg.norm(tb)), float(np.linalg.norm(lr))
        if len_tb < 20 or len_lr < 20:
            return False, None
        radius = (len_tb + len_lr) / 4.0

        # Verdaechtigster Punkt: groesster Abstand zum Schwerpunkt der drei anderen
        dist_to_others = {
            c: float(np.linalg.norm(P[c] - np.mean([P[o] for o in CAL_CLASSES if o != c], axis=0)))
            for c in CAL_CLASSES
        }
        suspect = max(dist_to_others, key=dist_to_others.get)

        mid_gap = float(np.linalg.norm((P[CAL_TOP] + P[CAL_BOTTOM]) / 2 - (P[CAL_LEFT] + P[CAL_RIGHT]) / 2))
        if mid_gap > 0.2 * radius:
            return False, suspect
        if not (0.5 <= len_tb / len_lr <= 2.0):
            return False, suspect
        # Uhrzeigersinn (Bildkoordinaten, y nach unten): alle Kreuzprodukte > 0
        order = [CAL_TOP, CAL_RIGHT, CAL_BOTTOM, CAL_LEFT]
        for i in range(4):
            e1 = P[order[(i + 1) % 4]] - P[order[i]]
            e2 = P[order[(i + 2) % 4]] - P[order[(i + 1) % 4]]
            if e1[0] * e2[1] - e1[1] * e2[0] <= 0:
                return False, suspect
        return True, None

    def update_calibration(
        self,
        detected_cals: Dict[int, Tuple[float, float]],
        w: int,
        h: int,
    ) -> bool:
        """Update and smooth calibration homography matrix."""
        if self.cal_locked and self.homography is not None:
            return True

        # Frisch erkannte Punkte merken, fehlende aus dem Gedaechtnis ergaenzen
        now = time.time()
        for c, pt in detected_cals.items():
            self.cal_memory[c] = (pt, now)
        detected_cals = dict(detected_cals)
        for c in CAL_CLASSES:
            if c not in detected_cals and c in self.cal_memory:
                pt, seen = self.cal_memory[c]
                if now - seen <= self.cal_memory_ttl:
                    detected_cals[c] = pt

        # Fehltreffer (z. B. ein "Kalibrierpunkt" auf dem Fussboden) aussortieren
        if len(detected_cals) == 4:
            px = {c: (nx * w, ny * h) for c, (nx, ny) in detected_cals.items()}
            ok, suspect = self._plausible_cal_set(px)
            if not ok:
                if suspect is not None:
                    detected_cals.pop(suspect, None)
                    self.cal_memory.pop(suspect, None)
                    self.cal_history[suspect].clear()
                else:
                    detected_cals.clear()
                    self.cal_memory.clear()

        if len(detected_cals) == 4:
            for c in CAL_CLASSES:
                self.cal_history[c].append(detected_cals[c])

            # Average calibration points over recent frames for jitter-free homography
            avg_cal = {}
            for c in CAL_CLASSES:
                xs = [pt[0] for pt in self.cal_history[c]]
                ys = [pt[1] for pt in self.cal_history[c]]
                avg_cal[c] = (float(np.mean(xs)), float(np.mean(ys)))

            self.cal_points_norm = avg_cal
            H = compute_homography(avg_cal, w, h)
            # Zusatzpunkte (dart-sense) einbeziehen, wenn sie zur 4-Punkt-Loesung passen
            if H is not None and self._extra_seen:
                pts = dict(avg_cal)
                for c, (pt, seen) in self._extra_seen.items():
                    if now - seen > self.cal_memory_ttl:
                        continue
                    mx, my = project_point_to_mm(pt[0], pt[1], H)
                    rx, ry = CAL_REFERENCE_MM[c]
                    if math.hypot(mx - rx, my - ry) <= 8.0:
                        pts[c] = (pt[0] / w, pt[1] / h)
                if len(pts) > 4:
                    H = compute_homography(pts, w, h)
            if H is not None:
                self.homography = H
                try:
                    self.inv_homography = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    self.inv_homography = None

                self.stable_cal_frames += 1
                # Auto-lock after 15 solid frames
                if self.stable_cal_frames >= 15 and not self.cal_locked:
                    self.cal_locked = True
                    print("🎯 Dartboard calibration automatically LOCKED!")
                    self.learn_artifacts_now()
                return True
        else:
            self.stable_cal_frames = max(0, self.stable_cal_frames - 1)

        return self.homography is not None

    WEAK_CAL_CONF = 0.08  # schwache Kalibrier-Kandidaten, nur mit Ortsvorhersage verwendbar

    def _infer(
        self,
        img: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        imgsz: int = 640,
        weak: Optional[List[Tuple[int, float, Tuple[float, float]]]] = None,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float], List[Tuple[float, float]]]:
        """YOLO auf einem Bild(ausschnitt) ausfuehren.

        Liefert Kalibrierpunkte {klasse: (x, y)}, deren Konfidenzen und Pfeilspitzen,
        alle in Vollbild-Pixeln (offset = Position des Ausschnitts im Vollbild).
        Wird `weak` uebergeben, werden dort zusaetzlich Kalibrier-Kandidaten unterhalb
        der normalen Schwelle als (klasse, konfidenz, (x, y)) gesammelt.
        """
        ox, oy = offset
        conf = min(self.conf_thresh, self.dart_conf)
        if weak is not None:
            conf = min(conf, self.WEAK_CAL_CONF)
        results = self.model(img, conf=conf, imgsz=imgsz, device=self.device, verbose=False)[0]

        cals: Dict[int, Tuple[float, float]] = {}
        confs: Dict[int, float] = {}
        tips: List[Tuple[float, float]] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0 + ox
            cy = (y1 + y2) / 2.0 + oy

            if cls_id in CAL_CLASSES:
                if score < self.conf_thresh:
                    if weak is not None:
                        weak.append((cls_id, score, (cx, cy)))
                elif cls_id not in confs or score > confs[cls_id]:
                    confs[cls_id] = score
                    cals[cls_id] = (cx, cy)
            elif cls_id in CAL_EXTRA_CLASSES and score >= self.conf_thresh:
                self._extra_seen[cls_id] = ((cx, cy), time.time())
            elif cls_id == DART_CLASS and score >= self.dart_conf:
                tips.append((cx, cy, score))
        return cals, confs, tips

    @staticmethod
    def _predict_point(
        known: Dict[int, Tuple[float, float]], target: int
    ) -> Tuple[Tuple[float, float], float]:
        """Lage eines Kalibrierpunkts aus den anderen vorhersagen.

        Die vier Punkte liegen in fester Geometrie (Kreis, 9 Grad gedreht). Aus je zwei
        bekannten Punkten folgt eine Aehnlichkeitstransformation mm -> Pixel, mit der
        der Zielpunkt berechnet wird; ueber alle Paare gemittelt.
        Liefert (vorhergesagter Punkt, Board-Radius in Pixeln).
        """
        keys = [k for k in known if k != target]
        preds, radii = [], []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                za, zb = complex(*known[a]), complex(*known[b])
                zA, zB = complex(*CAL_REFERENCE_MM[a]), complex(*CAL_REFERENCE_MM[b])
                s = (zb - za) / (zB - zA)      # Skalierung + Rotation
                t = za - s * zA                # Verschiebung
                zt = s * complex(*CAL_REFERENCE_MM[target]) + t
                preds.append((zt.real, zt.imag))
                radii.append(abs(s) * BOARD_RADIUS_MM)
        pred = (float(np.mean([p[0] for p in preds])), float(np.mean([p[1] for p in preds])))
        return pred, float(np.mean(radii))

    def _complete_from_weak(
        self,
        cals: Dict[int, Tuple[float, float]],
        confs: Dict[int, float],
        weak: List[Tuple[int, float, Tuple[float, float]]],
    ) -> None:
        """Fehlenden vierten Punkt aus schwachen Kandidaten nahe der Vorhersage ergaenzen."""
        if len(cals) != 3:
            return
        missing = next(c for c in CAL_CLASSES if c not in cals)
        pred, radius = self._predict_point(cals, missing)
        best = None
        for cls_id, score, pt in weak:
            if cls_id != missing:
                continue
            if math.hypot(pt[0] - pred[0], pt[1] - pred[1]) <= 0.4 * radius:
                if best is None or score > best[0]:
                    best = (score, pt)
        if best is not None:
            confs[missing] = best[0]
            cals[missing] = best[1]

    def _tiled_search(
        self, frame: np.ndarray, tile: int = 640, stride: int = 320
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float]]:
        """Vollbild in ueberlappende Kacheln zerlegen und jede in nativer Aufloesung pruefen.

        Findet auch ein kleines, weit entferntes Board, das im verkleinerten Vollbild untergeht.
        """
        h, w = frame.shape[:2]
        ys = list(range(0, max(1, h - tile + 1), stride))
        xs = list(range(0, max(1, w - tile + 1), stride))
        if ys[-1] != max(0, h - tile):
            ys.append(max(0, h - tile))
        if xs[-1] != max(0, w - tile):
            xs.append(max(0, w - tile))

        cals: Dict[int, Tuple[float, float]] = {}
        confs: Dict[int, float] = {}
        tile_sz = int(tile * self.max_upscale) // 32 * 32
        for y in ys:
            for x in xs:
                c, cf, _ = self._infer(frame[y:y + tile, x:x + tile], (x, y), imgsz=tile_sz)
                for k, v in c.items():
                    if k not in confs or cf[k] > confs[k]:
                        confs[k] = cf[k]
                        cals[k] = v
        return cals, confs

    @staticmethod
    def _roi_from_points(points, n_points: int, w: int, h: int) -> Roi:
        """Quadratischen Ausschnitt um die gefundenen Kalibrierpunkte legen.

        Je weniger Punkte bekannt sind, desto grosszuegiger der Rand, damit das
        ganze Board sicher enthalten ist.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        factor = 1.5 if n_points >= 4 else (1.9 if n_points == 3 else 2.6)
        side = int(round(min(max(extent * factor, 240.0), max(w, h))))
        x0 = int(round(cx - side / 2))
        y0 = int(round(cy - side / 2))
        x0 = max(0, min(x0, w - side))
        y0 = max(0, min(y0, h - side))
        return (x0, y0, min(side, w - x0), min(side, h - y0))

    @staticmethod
    def _estimate_board_px(points, n_points: int) -> float:
        """Board-Durchmesser aus der Ausdehnung der Kalibrierpunkte schaetzen."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        # 4 Punkte spannen fast den ganzen Durchmesser auf, 2-3 Punkte oft nur eine Diagonale
        return extent / (0.99 if n_points >= 4 else 0.85)

    @staticmethod
    def _enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Lokale Kontrastanhebung (CLAHE auf dem Helligkeitskanal) gegen ungleichmaessiges Licht."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _calibration_variants(self, crop_side: int, primary_sz: int) -> List[Tuple[bool, int]]:
        """Skalierungs-/Kontrastvarianten fuer die Kalibrierphase: (CLAHE?, imgsz).

        Das Modell erkennt die vier Punkte je nach Board-Groesse in der Eingabe
        (ca. 450-750px) und Beleuchtung unterschiedlich gut.
        """
        board = self.board_px or crop_side * 0.7
        sizes = set()
        for target in (450, 550, 650, 750):
            wanted = crop_side * target / board
            size = int(max(640, min(wanted, crop_side * self.max_upscale, 1600)))
            sizes.add((size // 32) * 32)
        variants = [(True, primary_sz)]
        for size in sorted(sizes):
            variants.append((True, size))
            if size != primary_sz:
                variants.append((False, size))
        return variants

    def _crop_imgsz(self, crop_side: int, target: Optional[int] = None, max_upscale: Optional[float] = None) -> int:
        """Modell-Eingabegroesse so waehlen, dass das Board ~target Pixel gross ist."""
        board = self.board_px or crop_side * 0.7
        wanted = crop_side * (target or self.target_board_px) / board
        cap = crop_side * (max_upscale or self.max_upscale)
        size = int(max(640, min(wanted, cap, 1600)))
        return (size // 32) * 32

    def _drop_outliers(
        self, cals: Dict[int, Tuple[float, float]], confs: Dict[int, float]
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float]]:
        """Kalibrierpunkte entfernen, die geometrisch nicht zu den anderen passen.

        Bei 4 Punkten entscheidet die Plausibilitaetspruefung, bei 3 Punkten die
        paarweisen Abstaende: Punkte auf dem Board liegen hoechstens einen
        Durchmesser auseinander, ein Fehltreffer abseits ist um ein Vielfaches weiter weg.
        """
        cals, confs = dict(cals), dict(confs)
        if len(cals) == 4:
            ok, suspect = self._plausible_cal_set(cals)
            if not ok:
                if suspect is None:
                    return {}, {}
                cals.pop(suspect)
                confs.pop(suspect)
        if len(cals) == 3:
            # Jeden Punkt aus den beiden anderen vorhersagen; wer weit daneben liegt, fliegt raus
            errors = {}
            for c in cals:
                others = {k: v for k, v in cals.items() if k != c}
                pred, radius = self._predict_point(others, c)
                errors[c] = math.hypot(cals[c][0] - pred[0], cals[c][1] - pred[1]) / max(radius, 1.0)
            worst = max(errors, key=errors.get)
            if errors[worst] > 0.6:
                cals.pop(worst)
                confs.pop(worst)
        return cals, confs

    def _detect_board(
        self, frame: np.ndarray
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float], List[Tuple[float, float]]]:
        """Zweistufige Erkennung: Board im Vollbild suchen, dann nur den Ausschnitt auswerten."""
        h, w = frame.shape[:2]
        now = time.time()

        if self.roi is None:
            # Stufe 1: Suche im Vollbild in nativer Aufloesung ...
            full_sz = min(1920, (max(h, w) // 32) * 32)
            cals, confs, tips = self._infer(frame, (0, 0), imgsz=full_sz)
            # ... und regelmaessig kachelweise, leicht hochskaliert (kleines Board)
            if len(cals) < 2 and self.frame_count % self.search_interval == 0:
                t_cals, t_confs = self._tiled_search(frame)
                for k, v in t_cals.items():
                    if k not in confs or t_confs[k] > confs[k]:
                        confs[k] = t_confs[k]
                        cals[k] = v
            cals, confs = self._drop_outliers(cals, confs)
            if len(cals) >= 2:
                self.board_px = self._estimate_board_px(cals.values(), len(cals))
                self.roi = self._roi_from_points(cals.values(), len(cals), w, h)
                self.last_cal_seen = now
                print(
                    f"🔍 Board gefunden ({len(cals)} Kalibrierpunkte, ~{self.board_px:.0f}px) "
                    f"-> Ausschnitt {self.roi}, Modell-Eingabe {self._crop_imgsz(self.roi[2])}px"
                )
            return cals, confs, tips

        # Stufe 2: nur den Board-Ausschnitt auswerten, Eingabegroesse an Board-Groesse angepasst.
        # Die Kalibrierpunkte werden je nach Beleuchtung und Skalierung unterschiedlich
        # gut erkannt - daher drei Durchlaeufe (Original, kontrastverstaerkt, zweite
        # Skalierung) und je Punkt die beste Konfidenz uebernehmen.
        x, y, rw, rh = self.roi
        crop = frame[y:y + rh, x:x + rw]
        sz = self._crop_imgsz(max(rw, rh))
        if self.cal_locked:
            # Nach dem Einrasten zaehlen nur noch die Pfeilspitzen. Zwischen zwei
            # Eingabegroessen wechseln (Board ~500 und ~700 px); die Spurverfolgung
            # sammelt die Treffer ueber die Frames ein. Auswertung auf gespeicherten
            # Bildern (dart-sense): Board ~500px findet die meisten Pfeile, ~700px
            # ist am fehlerfreisten, staerkeres Hochskalieren bringt Fehltreffer.
            targets = (500, 700)
            tip_sz = self._crop_imgsz(max(rw, rh), target=targets[self.frame_count % 2], max_upscale=1.6)
            cals, confs, tips = self._infer(crop, (x, y), imgsz=tip_sz)
            self._check_drift(cals, w, h, now)
            if self.roi is None:  # Kalibrierung wurde soeben verworfen
                return {}, {}, []
        else:
            weak: List[Tuple[int, float, Tuple[float, float]]] = []
            cals, confs, tips = self._infer(crop, (x, y), imgsz=sz, weak=weak)
            # Pro Frame eine weitere Variante (rotierend); das Punktgedaechtnis
            # sammelt die Treffer der verschiedenen Varianten ueber wenige Frames ein.
            variants = self._calibration_variants(max(rw, rh), sz)
            use_clahe, isz = variants[self.frame_count % len(variants)]
            img = self._enhance_contrast(crop) if use_clahe else crop
            c2, cf2, _ = self._infer(img, (x, y), imgsz=isz, weak=weak)
            for k, v in c2.items():
                if k not in confs or cf2[k] > confs[k]:
                    confs[k] = cf2[k]
                    cals[k] = v
            cals, confs = self._drop_outliers(cals, confs)
            # Fehlt genau ein Punkt: gemerkte Punkte hinzuziehen und gezielt an der
            # vorhergesagten Stelle nach einem schwachen Kandidaten suchen
            now_ = time.time()
            known = dict(cals)
            for c, (pt, seen) in self.cal_memory.items():
                if c not in known and now_ - seen <= self.cal_memory_ttl:
                    known[c] = pt
            if len(known) == 3:
                self._complete_from_weak(known, confs, weak)
                for c in known:
                    if c not in cals and c in confs:
                        cals[c] = known[c]

        cals, confs = self._drop_outliers(cals, confs)
        if cals:
            self.last_cal_seen = now
            # Ausschnitt und Groessenschaetzung nachfuehren, sobald >= 3 konsistente Punkte da sind
            if len(cals) >= 3 and not self.cal_locked and not self.roi_manual and self.roi is not None:
                self.board_px = self._estimate_board_px(cals.values(), len(cals))
                desired = self._roi_from_points(cals.values(), len(cals), w, h)
                current = self.roi  # kann aus einem anderen Thread auf None gesetzt werden
                if current is not None and _roi_iou(desired, current) < 0.7:
                    self.roi = desired
                    print(f"   Ausschnitt nachgefuehrt ({len(cals)} Punkte, ~{self.board_px:.0f}px) -> {self.roi}")
        elif (
            not self.cal_locked
            and not self.roi_manual
            and now - self.last_cal_seen > self.roi_lost_timeout
        ):
            print("Board im Ausschnitt verloren -> neue Suche im Vollbild")
            self.roi = None
        return cals, confs, tips

    def _save_snapshot(self, vis: np.ndarray, tag: str) -> None:
        """HUD-Bild nach snapshots/ speichern (automatisch beim Einrasten, manuell mit S)."""
        out_dir = Path(__file__).resolve().parent / "snapshots"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"📸 Schnappschuss gespeichert: {path.name}")

    def reset_calibration(self) -> None:
        """Kalibrierung komplett verwerfen und Board neu suchen (Taste C / Rekalibrieren)."""
        self.cal_locked = False
        self.stable_cal_frames = 0
        for c in CAL_CLASSES:
            self.cal_history[c].clear()
        self.cal_memory.clear()
        self._extra_seen.clear()
        self.homography = None
        self.inv_homography = None
        self.cal_points_norm = {}
        self.roi = None
        self.roi_manual = False
        self.board_px = None
        self.dart_tracks = []
        self._drift_frames = 0
        self._nocal_since = 0.0
        self.artifacts = []
        self._artifact_samples = None

    def _check_drift(self, cals: Dict[int, Tuple[float, float]], w: int, h: int, now: float) -> None:
        """Nach dem Einrasten: Haben sich Board oder Kamera bewegt? Dann neu kalibrieren.

        Kriterium: mindestens 3 Kalibrierpunkte liegen ueber viele Frames deutlich
        neben ihren eingerasteten Positionen - oder ueber laengere Zeit wird gar
        kein Kalibrierpunkt mehr gefunden.
        """
        if not self.cal_locked or not self.cal_points_norm:
            return
        if not cals:
            if self._nocal_since == 0.0:
                self._nocal_since = now
            elif now - self._nocal_since > self.drift_nocal_timeout:
                print("⚠️  Keine Kalibrierpunkte mehr sichtbar -> Kalibrierung neu")
                self.reset_calibration()
            return
        self._nocal_since = 0.0
        known = {c: (nx * w, ny * h) for c, (nx, ny) in self.cal_points_norm.items()}
        common = [c for c in cals if c in known]
        if len(common) < 3:
            return
        tol = max(12.0, 0.04 * (self.board_px or 400))
        moved = sum(1 for c in common if math.hypot(cals[c][0] - known[c][0], cals[c][1] - known[c][1]) > tol)
        if moved >= 3:
            self._drift_frames += 1
            if self._drift_frames >= self.drift_frames_needed:
                print("⚠️  Board/Kamera hat sich bewegt -> Kalibrierung neu")
                self.reset_calibration()
        else:
            self._drift_frames = 0

    def select_roi_manually(self, frame: np.ndarray) -> None:
        """Board per Maus markieren (Rueckfallebene, wenn die automatische Suche scheitert)."""
        win = "Board markieren: Rechteck ziehen, ENTER = OK, C = Abbruch"
        r = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
        x, y, rw, rh = [int(v) for v in r]
        if rw > 20 and rh > 20:
            self.roi = (x, y, rw, rh)
            self.roi_manual = True
            self.board_px = None  # unbekannt -> Annahme: Board fuellt ~70% der Markierung
            self.last_cal_seen = time.time()
            self.cal_locked = False
            self.stable_cal_frames = 0
            print(f"✏️  Board manuell markiert: {self.roi}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO inference, track darts, calculate scores, and draw HUD."""
        h, w = frame.shape[:2]
        self.frame_count += 1
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            # gleitender Mittelwert, damit die Anzeige nicht flackert
            self.fps = 1.0 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
        self.prev_time = now

        # Run YOLO inference (two-stage: full frame search, then board crop)
        detected_px, cal_confs, raw_dart_tips = self._detect_board(frame)
        detected_cals = {c: (px / w, py / h) for c, (px, py) in detected_px.items()}

        # Update board calibration
        cal_ok = self.update_calibration(detected_cals, w, h)

        current_frame_darts: List[TrackedDart] = []
        if cal_ok and self.homography is not None:
            # Nur Erkennungen auf dem Board verfolgen (Klammern am Zahlenring aussortieren)
            on_board = []
            for px, py, conf in raw_dart_tips:
                x_mm, y_mm = project_point_to_mm(px, py, self.homography)
                if math.hypot(x_mm, y_mm) <= MAX_DART_RADIUS_MM:
                    on_board.append((px, py, conf))
            if self._artifact_step(on_board):
                on_board = []  # Lernphase: Board gilt als leer, nichts werten
            else:
                on_board = [d for d in on_board if not self._is_artifact(d[0], d[1])]
            confirmed = self._update_tracks(on_board, now)

            # Bestaetigte Pfeile bewerten (mit Parallaxen-Offset)
            for tr in confirmed[:3]:
                x_mm, y_mm = project_point_to_mm(tr.px, tr.py, self.homography)
                x_mm += self.tip_offset_mm[0]
                y_mm += self.tip_offset_mm[1]
                current_frame_darts.append(
                    TrackedDart(
                        px=tr.px, py=tr.py, x_mm=x_mm, y_mm=y_mm,
                        score=tr.override or score_at(x_mm, y_mm),
                        frames_seen=tr.hits,
                        # knapp ausserhalb des Double-Rings (170-178 mm) ist fast immer ein
                        # Double mit leichtem Versatz nach aussen -> immer als unsicher markieren
                        uncertain=(wire_distance_mm(x_mm, y_mm) < UNCERTAIN_WIRE_MM
                                   or BOARD_RADIUS_MM < math.hypot(x_mm, y_mm) < BOARD_RADIUS_MM + 8.0),
                        track=tr,
                    )
                )
            self.selected_dart = min(self.selected_dart, max(0, len(current_frame_darts) - 1))
            if not self.external_game:
                self._update_turn_state(current_frame_darts)

        # Draw overlays and sidebar
        self._last_darts = current_frame_darts
        return self._render_display(frame, current_frame_darts, cal_ok)

    def learn_artifacts_now(self) -> None:
        """Stoerstellen neu lernen: die naechsten ~2 s gilt das Board als leer."""
        self.artifacts = []
        self._artifact_samples = {}
        self._artifact_frames_left = self.artifact_learn_frames
        self.dart_tracks = []
        print("🧹 Lerne Stoerstellen auf dem leeren Board ...")

    def _artifact_step(self, detections: List[Tuple[float, float, float]]) -> bool:
        """Lernphase fortschreiben. Liefert True, solange gelernt wird (dann keine Pfeilwertung)."""
        if self._artifact_samples is None:
            return False
        for px, py, conf in detections:
            # Echte Pfeile werden meist sicher erkannt (> 0.45); Stoerstellen sind schwach.
            # So werden Pfeile, die beim Start noch stecken, nicht als Stoerstelle gelernt.
            if conf >= self.artifact_max_conf:
                continue
            key = (int(px // 8), int(py // 8))
            self._artifact_samples[key] = self._artifact_samples.get(key, 0) + 1
        self._artifact_frames_left -= 1
        if self._artifact_frames_left > 0:
            return True
        need = max(3, self.artifact_learn_frames // 2)
        self.artifacts = [((kx + 0.5) * 8, (ky + 0.5) * 8) for (kx, ky), n in self._artifact_samples.items() if n >= need]
        self._artifact_samples = None
        if self.artifacts:
            print(f"🧹 {len(self.artifacts)} Stoerstelle(n) ausgeblendet: {[(round(x), round(y)) for x, y in self.artifacts]}")
        else:
            print("🧹 Keine Stoerstellen gefunden.")
        return False

    def _is_artifact(self, px: float, py: float) -> bool:
        return any(math.hypot(px - ax, py - ay) <= self.artifact_radius_px for ax, ay in self.artifacts)

    def _update_tracks(self, detections: List[Tuple[float, float, float]], now: float) -> List[DartTrack]:
        """Erkennungen den bestehenden Spuren zuordnen; bestaetigte Spuren zurueckgeben."""
        # Doppelte Erkennungen desselben Pfeils (Spitze + Punkt auf dem Schaft) zusammenfassen.
        # Kriterium: nahe beieinander UND entlang der Schaftrichtung (bei Kamera von unten
        # zeigen die Flights nach oben). Zwei echte Pfeile dicht nebeneinander liegen meist
        # quer dazu und bleiben getrennt. Von einem Duplikat bleibt der Punkt erhalten, der
        # am weitesten Richtung Spitze liegt - nicht der mit der hoechsten Konfidenz, denn
        # der Schaftpunkt bekommt oft die hoehere.
        sx, sy = self.shaft_dir
        ordered = sorted(detections, key=lambda d: -d[2])
        merged: List[Tuple[float, float]] = []
        for px, py, conf in ordered:
            target = None
            for i, (mx, my) in enumerate(merged):
                dx, dy = px - mx, py - my
                dist = math.hypot(dx, dy)
                if dist <= self.track_merge_close_px:  # praktisch derselbe Punkt
                    target = i
                    break
                if dist <= self.track_merge_px:
                    align = abs((dx * sx + dy * sy) / dist)
                    if align >= self.track_merge_align:
                        target = i
                        break
            if target is None:
                merged.append((px, py))
            else:
                mx, my = merged[target]
                if (px * sx + py * sy) < (mx * sx + my * sy):  # weiter entgegen der Schaftrichtung
                    merged[target] = (px, py)
        unmatched = merged
        for tr in self.dart_tracks:
            best, best_d = None, self.track_match_px
            for det in unmatched:
                d = math.hypot(det[0] - tr.px, det[1] - tr.py)
                if d < best_d:
                    best, best_d = det, d
            if best is not None:
                unmatched.remove(best)
                tr.px = 0.7 * tr.px + 0.3 * best[0]
                tr.py = 0.7 * tr.py + 0.3 * best[1]
                tr.hits += 1
                tr.last_seen = now
        for det in unmatched:
            self._track_seq += 1
            self.dart_tracks.append(
                DartTrack(px=det[0], py=det[1], hits=1, first_seen=now, last_seen=now, id=self._track_seq)
            )
        self.dart_tracks = [tr for tr in self.dart_tracks if now - tr.last_seen <= self.track_expire_s]
        confirmed = [tr for tr in self.dart_tracks if tr.hits >= self.track_min_hits]
        confirmed.sort(key=lambda tr: tr.first_seen)
        return confirmed

    def _load_tip_offset(self) -> None:
        try:
            import json
            data = json.loads(self.offset_path.read_text())
            self.tip_offset_mm = (float(data["x_mm"]), float(data["y_mm"]))
            print(f"Parallaxen-Offset geladen: ({self.tip_offset_mm[0]:.0f}, {self.tip_offset_mm[1]:.0f}) mm")
        except (OSError, ValueError, KeyError):
            self.tip_offset_mm = (0.0, 0.0)

    def calibrate_bull_offset(self, darts: List[TrackedDart]) -> None:
        """Taste B: genau ein Pfeil steckt im Bull -> gemessene Abweichung als Offset uebernehmen.

        Ohne Pfeil auf dem Board wird der Offset zurueckgesetzt.
        """
        import json
        if len(darts) == 1:
            d = darts[0]
            raw_x = d.x_mm - self.tip_offset_mm[0]
            raw_y = d.y_mm - self.tip_offset_mm[1]
            self.tip_offset_mm = (-raw_x, -raw_y)
            print(f"🎯 Bull-Kalibrierung: Offset = ({-raw_x:.0f}, {-raw_y:.0f}) mm")
        elif len(darts) == 0:
            self.tip_offset_mm = (0.0, 0.0)
            print("Parallaxen-Offset zurueckgesetzt (0, 0).")
        else:
            print("Bull-Kalibrierung: bitte genau EINEN Pfeil ins Bull stecken.")
            return
        self.offset_path.write_text(json.dumps({"x_mm": self.tip_offset_mm[0], "y_mm": self.tip_offset_mm[1]}))

    def adjust_selected_dart(self, key: int) -> None:
        """Manuelle Korrektur des gewaehlten Pfeils: links/rechts = Nachbarsektor,
        hoch/runter = Single -> Double -> Triple."""
        if not self._last_darts:
            return
        dart = self._last_darts[self.selected_dart]
        if dart.track is None:
            return
        cur = dart.score
        sector, mult = cur.sector, cur.multiplier
        if key in KEY_LEFT or key in KEY_RIGHT:
            step = -1 if key in KEY_LEFT else 1
            if sector in SECTORS_CLOCKWISE:
                sector = SECTORS_CLOCKWISE[(SECTORS_CLOCKWISE.index(sector) + step) % 20]
            else:  # aus Bull/Miss heraus: naechstgelegener Sektor
                sector = score_at(dart.x_mm * 3 or 1.0, dart.y_mm * 3 or -1.0).sector or 20
            mult = mult if mult in (1, 2, 3) else 1
        else:
            if sector == 25:  # Bull <-> Bullseye
                mult = 2 if mult == 1 else 1
            else:
                mult = (mult % 3) + 1 if key in KEY_UP else ((mult - 2) % 3) + 1
                if sector == 0:
                    sector = 20
        if sector == 25:
            desc = "BULLSEYE" if mult == 2 else "BULL"
            points = 50 if mult == 2 else 25
        else:
            desc = f"{'SDT'[mult - 1]}{sector}"
            points = sector * mult
        dart.track.override = DartScore(sector=sector, multiplier=mult, points=points,
                                        description=desc, radius_mm=cur.radius_mm)
        print(f"✏️  Dart {self.selected_dart + 1} manuell: {desc} ({points})")

    def _update_turn_state(self, current_darts: List[TrackedDart]) -> None:
        """Track turns and auto-detect when darts are removed from the board."""
        cnt = len(current_darts)
        if cnt == 0:
            self.empty_frames_count += 1
            # If board was occupied and now empty for 12+ frames -> player pulled darts!
            if self.prev_detected_count >= 1 and self.empty_frames_count >= 12 and self.turn_darts:
                self.commit_turn()
        else:
            self.empty_frames_count = 0
            # Update current live turn darts
            self.turn_darts = [d.score for d in current_darts]

        self.prev_detected_count = cnt

    def commit_turn(self) -> None:
        """Submit the current 3-dart turn and update game score."""
        if not self.turn_darts:
            return

        turn_sum = sum(d.points for d in self.turn_darts)
        self.last_turn_sum = turn_sum
        self.turn_history.append(list(self.turn_darts))

        # Game mode logic
        if self.game_mode in ["501", "301", "X01"]:
            rem = self.current_score - turn_sum
            if rem == 0:
                self.current_score = 0
            elif rem >= 2:
                self.current_score = rem
            # rem < 2: Bust bei Double-Out, Rest bleibt unveraendert

        print(f"Turn finished: {turn_sum} pts ({', '.join(d.description for d in self.turn_darts)})")
        self.turn_darts = []

    def undo_last_turn(self) -> None:
        """Undo last committed turn."""
        if not self.turn_history:
            return
        last_turn = self.turn_history.pop()
        pts = sum(d.points for d in last_turn)
        if self.game_mode in ["501", "301", "X01"]:
            self.current_score += pts
        print(f"Undid turn: +{pts} pts")

    def _render_display(
        self,
        frame: np.ndarray,
        darts: List[TrackedDart],
        cal_ok: bool,
    ) -> np.ndarray:
        """Build the combined live view + modern dark HUD."""
        h, w = frame.shape[:2]
        vis_frame = frame.copy()

        # 0. Board-Ausschnitt (ROI) einzeichnen
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            roi_col = (255, 200, 0)
            cv2.rectangle(vis_frame, (rx, ry), (rx + rw, ry + rh), roi_col, 1)
            roi_tag = "BOARD (manuell)" if self.roi_manual else "BOARD"
            cv2.putText(vis_frame, roi_tag, (rx + 4, max(14, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_col, 1)

        # 1. Draw Dartboard Calibration & Outer Rings
        if cal_ok and self.inv_homography is not None:
            self._draw_projected_board_rings(vis_frame)

        # 2. Draw Detected Calibration Markers
        cal_colors = {
            0: (0, 255, 0),    # Top: Green
            1: (255, 0, 0),    # Bottom: Blue
            2: (0, 255, 255),  # Left: Yellow
            3: (0, 165, 255),  # Right: Orange
        }
        for c, (nx, ny) in self.cal_points_norm.items():
            px, py = int(nx * w), int(ny * h)
            col = cal_colors.get(c, (0, 255, 0))
            cv2.circle(vis_frame, (px, py), 6, col, -1)
            cv2.circle(vis_frame, (px, py), 10, col, 2)

        # 3. Draw Detected Darts on Camera Feed
        for idx, dart in enumerate(darts, 1):
            ipx, ipy = int(dart.px), int(dart.py)
            # Glowing concentric target rings
            cv2.circle(vis_frame, (ipx, ipy), 14, (0, 0, 255), 2)
            cv2.circle(vis_frame, (ipx, ipy), 5, (0, 255, 255), -1)
            # Label tag
            tag = f"D{idx}: {dart.score.description} ({dart.score.points})"
            if dart.uncertain:
                tag += " ?"
            tag_col = (0, 200, 255) if dart.uncertain else (0, 255, 255)
            cv2.rectangle(vis_frame, (ipx + 12, ipy - 24), (ipx + 170, ipy + 4), (20, 20, 20), -1)
            cv2.putText(vis_frame, tag, (ipx + 16, ipy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_col, 2)

        if not self.draw_sidebar:
            return vis_frame

        # 4. Create Side HUD Panel (Width: 420px)
        sidebar_w = 420
        sidebar = np.full((h, sidebar_w, 3), 28, dtype=np.uint8)

        self._draw_sidebar_hud(sidebar, darts, cal_ok)

        # Combine camera feed and sidebar side-by-side
        combined = np.hstack([vis_frame, sidebar])
        return combined

    def _draw_projected_board_rings(self, img: np.ndarray) -> None:
        """Project ideal board rings from mm onto the camera image for perfect alignment check."""
        if self.inv_homography is None:
            return

        rings_mm = [
            (6.35, (0, 0, 255)),     # Double Bull (red)
            (15.9, (0, 255, 0)),     # Outer Bull (green)
            (99.0, (0, 255, 255)),   # Treble inner
            (107.0, (0, 255, 255)),  # Treble outer
            (162.0, (0, 255, 0)),    # Double inner
            (170.0, (0, 0, 255)),    # Double outer
        ]

        for radius_mm, color in rings_mm:
            pts_mm = []
            for angle_deg in range(0, 360, 10):
                rad = math.radians(angle_deg)
                pts_mm.append([radius_mm * math.sin(rad), -radius_mm * math.cos(rad)])

            pts_mm_arr = np.array([pts_mm], dtype=np.float32)
            pts_px = cv2.perspectiveTransform(pts_mm_arr, self.inv_homography)
            pts_px_int = np.int32(pts_px[0])
            cv2.polylines(img, [pts_px_int], isClosed=True, color=color, thickness=1)

    def _draw_sidebar_hud(
        self,
        sidebar: np.ndarray,
        darts: List[TrackedDart],
        cal_ok: bool,
    ) -> None:
        """Draw scoreboards, 2D dartboard radar, and stats on the sidebar."""
        sb_h, sb_w = sidebar.shape[:2]

        # Top Header
        cv2.rectangle(sidebar, (0, 0), (sb_w, 60), (45, 45, 45), -1)
        header_title = f"🎯 DART SCORER [{self.game_mode}]"
        cv2.putText(sidebar, header_title, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 200), 2)

        # Game Score / Leg Counter
        y_pos = 90
        if self.game_mode in ["501", "301", "X01"]:
            cv2.putText(sidebar, "REMAINING", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            y_pos += 45
            score_str = f"{self.current_score}"
            cv2.putText(sidebar, score_str, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

            # Checkout suggestion if <= 170
            checkout = CHECKOUTS.get(self.current_score)
            y_pos += 25
            if checkout:
                chk_text = f"Checkout: {checkout}"
                cv2.putText(sidebar, chk_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
            else:
                cv2.putText(sidebar, "Setup turn", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            y_pos += 30
        else:
            # Practice mode
            cv2.putText(sidebar, "PRACTICE MODE", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            y_pos += 35
            tot_turns = len(self.turn_history)
            tot_pts = sum(sum(d.points for d in t) for t in self.turn_history)
            avg_3 = (tot_pts / tot_turns) if tot_turns > 0 else 0.0
            avg_text = f"3-Dart Avg: {avg_3:.1f}  |  Turns: {tot_turns}"
            cv2.putText(sidebar, avg_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_pos += 30

        # Separator line
        cv2.line(sidebar, (15, y_pos), (sb_w - 15, y_pos), (60, 60, 60), 1)
        y_pos += 25

        # Current Turn Score
        curr_turn_sum = sum(d.score.points for d in darts)
        cv2.putText(sidebar, "CURRENT TURN", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        y_pos += 35

        turn_sum_col = (0, 255, 255) if curr_turn_sum < 100 else (0, 255, 0)
        cv2.putText(sidebar, f"{curr_turn_sum} pts", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 1.1, turn_sum_col, 2)
        y_pos += 30

        # Individual dart slots
        for i in range(3):
            slot_y = y_pos + (i * 26)
            if i < len(darts):
                d = darts[i].score
                marker = ">" if i == self.selected_dart else " "
                flag = ""
                if darts[i].track is not None and darts[i].track.override is not None:
                    flag = "  [manuell]"
                elif darts[i].uncertain:
                    flag = "  ?"
                slot_txt = f"{marker}Dart {i+1}:  {d.description}  ({d.points} pts){flag}"
                col = (0, 200, 255) if darts[i].uncertain and not flag.startswith("  [") else (255, 255, 255)
            else:
                slot_txt = f" Dart {i+1}:  ---"
                col = (90, 90, 90)
            cv2.putText(sidebar, slot_txt, (25, slot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        y_pos += 95
        # Separator
        cv2.line(sidebar, (15, y_pos), (sb_w - 15, y_pos), (60, 60, 60), 1)
        y_pos += 20

        # 2D Dartboard Radar Widget
        radar_center = (sb_w // 2, y_pos + 85)
        radar_radius = 80
        self._draw_2d_radar(sidebar, radar_center, radar_radius, darts)
        y_pos += 185

        # Calibration status banner
        if self.cal_locked:
            status_text = "CALIBRATION: LOCKED [OK]"
            status_col = (0, 255, 0)
        elif cal_ok:
            status_text = f"CALIBRATION: TRACKING ({self.stable_cal_frames}/15)"
            status_col = (0, 255, 255)
        else:
            status_text = "CALIBRATION: SEARCHING (<4 pts)"
            status_col = (0, 0, 255)

        cv2.putText(sidebar, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_col, 1)
        y_pos += 22

        # Board-Suche / Ausschnitt
        if self.roi is None:
            roi_text = "BOARD: SUCHE IM VOLLBILD..."
            roi_col = (0, 165, 255)
        else:
            roi_text = f"BOARD: AUSSCHNITT {self.roi[2]}x{self.roi[3]}px"
            if self.roi_manual:
                roi_text += " (manuell)"
            roi_col = (255, 200, 0)
        cv2.putText(sidebar, roi_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_col, 1)
        y_pos += 22

        if self.tip_offset_mm != (0.0, 0.0):
            cv2.putText(sidebar, f"OFFSET: ({self.tip_offset_mm[0]:+.0f}, {self.tip_offset_mm[1]:+.0f}) mm",
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 120), 1)
            y_pos += 22

        fps_text = f"FPS: {self.fps:.1f}  |  Source: {'Camera [' + str(self.stream.active_index) + ']' if self.stream.is_camera else 'Test Images'}"
        cv2.putText(sidebar, fps_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        # Bottom Shortcut Legend
        legend_y = sb_h - 15
        legends = [
            "[SPACE] Submit Turn  |  [U] Undo",
            "[C] Recalibrate  |  [L] Lock/Unlock",
            "[R] Board markieren  |  [B] Bull-Offset",
            "[TAB] Pfeil waehlen  |  Pfeiltasten: korrigieren",
            "[S] Schnappschuss  |  [M] Mode  |  [Q] Exit",
        ]
        for line in reversed(legends):
            cv2.putText(sidebar, line, (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
            legend_y -= 18

    def _draw_2d_radar(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        darts: List[TrackedDart],
    ) -> None:
        """Render a digital 2D mini-dartboard with live hit markers."""
        cx, cy = center
        scale = radius / BOARD_RADIUS_MM

        # Background circle
        cv2.circle(img, (cx, cy), radius + 2, (10, 10, 10), -1)
        cv2.circle(img, (cx, cy), radius + 2, (70, 70, 70), 2)

        # Draw sectors
        for idx, sec in enumerate(SECTORS_CLOCKWISE):
            start_ang = (idx * 18.0) - 9.0 - 90.0
            end_ang = start_ang + 18.0
            col_dark = (30, 30, 30) if idx % 2 == 0 else (220, 220, 220)
            cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_ang, end_ang, col_dark, -1)

        # Treble ring (99 to 107 mm)
        r_tr_in = int(99.0 * scale)
        r_tr_out = int(107.0 * scale)
        cv2.circle(img, (cx, cy), r_tr_out, (0, 150, 0), 2)
        cv2.circle(img, (cx, cy), r_tr_in, (0, 0, 150), 1)

        # Double ring (162 to 170 mm)
        r_db_in = int(162.0 * scale)
        r_db_out = int(170.0 * scale)
        cv2.circle(img, (cx, cy), r_db_out, (0, 0, 150), 2)
        cv2.circle(img, (cx, cy), r_db_in, (0, 150, 0), 1)

        # Bull rings (Outer 15.9mm, Inner 6.35mm)
        cv2.circle(img, (cx, cy), max(2, int(15.9 * scale)), (0, 180, 0), -1)
        cv2.circle(img, (cx, cy), max(1, int(6.35 * scale)), (0, 0, 220), -1)

        # Plot landed darts
        for d in darts:
            # Map (x_mm, y_mm) to radar px
            rx = int(cx + (d.x_mm * scale))
            ry = int(cy + (d.y_mm * scale))
            cv2.circle(img, (rx, ry), 5, (0, 255, 255), -1)
            cv2.circle(img, (rx, ry), 7, (0, 0, 255), 2)

    def run(self) -> None:
        """Main game loop."""
        window_name = "Live Dart Scorer AI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 850)

        print("\n" + "=" * 60)
        print("🎯 Live Dart Scorer AI is running!")
        print("Controls:")
        print("  [SPACE]   : Submit turn (manual next player)")
        print("  [U]       : Undo last turn")
        print("  [C]       : Recalibrate / Unlock board")
        print("  [L]       : Lock / Freeze calibration")
        print("  [M]       : Switch game mode (Practice <-> 501)")
        print("  [R]       : Board manuell markieren (Rechteck ziehen)")
        print("  [B]       : Bull-Kalibrierung (ein Pfeil im Bull -> Parallaxen-Offset)")
        print("  [S]       : Schnappschuss speichern")
        print("  [TAB]     : Pfeil fuer manuelle Korrektur waehlen")
        print("  [<-] [->] : Nachbarsektor  |  [^] [v] : Single/Double/Triple")
        print("  [0 - 4]   : Switch camera device index")
        print("  [N / P]   : Next / Previous image (in test image mode)")
        print("  [Q / ESC] : Quit")
        print("=" * 60 + "\n")

        while True:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            if self.undistorter is not None:
                frame = self.undistorter.apply(frame)

            was_locked = self.cal_locked
            vis = self.process_frame(frame)
            cv2.imshow(window_name, vis)
            if self.cal_locked and not was_locked:
                self._save_snapshot(vis, "lock")

            raw = cv2.waitKeyEx(1)
            key = raw if raw in ARROW_KEYS or raw == -1 else raw & 0xFF
            if key in ARROW_KEYS:
                self.adjust_selected_dart(key)
                continue
            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key == 9:  # TAB: naechsten Pfeil fuer manuelle Korrektur waehlen
                if self._last_darts:
                    self.selected_dart = (self.selected_dart + 1) % len(self._last_darts)
            elif key in [ord("s"), ord("S")]:
                self._save_snapshot(vis, "manual")
            elif key in [ord("b"), ord("B")]:
                self.calibrate_bull_offset(self._last_darts)
            elif key == ord(" "):
                self.commit_turn()
            elif key in [ord("u"), ord("U")]:
                self.undo_last_turn()
            elif key in [ord("c"), ord("C")]:
                self.reset_calibration()
                print("🔄 Calibration reset! Point camera at the dartboard.")
            elif key in [ord("r"), ord("R")]:
                self.select_roi_manually(frame)
            elif key in [ord("l"), ord("L")]:
                self.cal_locked = not self.cal_locked
                status = "LOCKED" if self.cal_locked else "UNLOCKED"
                print(f"🔒 Calibration {status}!")
            elif key in [ord("m"), ord("M")]:
                self.game_mode = "501" if self.game_mode == "PRACTICE" else "PRACTICE"
                print(f"🎮 Game mode switched to: {self.game_mode}")
            elif key in [ord("n"), ord("N")]:
                self.stream.next_test_image()
            elif key in [ord("p"), ord("P")]:
                self.stream.prev_test_image()
            elif ord("0") <= key <= ord("4"):
                cam_idx = key - ord("0")
                print(f"Switching to camera index {cam_idx}...")
                self.stream.switch_camera(cam_idx)

        self.stream.stop()
        cv2.destroyAllWindows()
        print("Dart Scorer closed gracefully.")


def main() -> None:
    # Kamera-Meldungen (welcher Index, welche Auflösung) sichtbar machen
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Live Dart Scorer with GoPro & YOLO11n")
    parser.add_argument(
        "--model",
        type=str,
        default="models/dart_sense_yolov8n.pt",
        help="Modellgewichte (Standard: dart-sense; Rueckfall: models/dart_yolo11n_best.pt)",
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g. 0 or 1)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--mode", type=str, default="501", choices=["501", "301", "PRACTICE"], help="Game mode")
    parser.add_argument(
        "--calib",
        type=str,
        default="camera_calib.npz",
        help="Linsen-Kalibrierdatei aus calibrate_lens.py (wird ignoriert, wenn nicht vorhanden)",
    )
    args = parser.parse_args()

    calib = Path(args.calib)
    if not calib.is_absolute():
        calib = Path(__file__).resolve().parent / calib

    scorer = LiveDartScorer(
        model_path=args.model,
        camera_idx=args.camera,
        conf_thresh=args.conf,
        mode=args.mode,
        calib_path=str(calib) if calib.exists() else None,
    )
    scorer.run()


if __name__ == "__main__":
    main()
