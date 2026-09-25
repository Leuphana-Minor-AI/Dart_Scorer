"""Dartboard geometry and scoring calculations according to official WDF regulations.

Board dimensions (radius from bullseye center in mm):
  - Inner bull (Double Bull / 50 pts): 0 - 6.35 mm
  - Outer bull (Single Bull / 25 pts): 6.35 - 15.9 mm
  - Inner single ring: 15.9 - 99.0 mm
  - Treble (Triple) ring: 99.0 - 107.0 mm (3x score)
  - Outer single ring: 107.0 - 162.0 mm
  - Double ring: 162.0 - 170.0 mm (2x score)
  - Miss / Out of board: > 170.0 mm (0 pts)

Calibration points in DeepDarts:
  The 4 board calibration points are outer double ring wire intersections (r = 170.0 mm)
  rotated 9 degrees from the 4 cardinal axes:
    cal_top    (0): wire between 5 and 20   (angle: 90 + 9 = 99 deg)
    cal_bottom (1): wire between 3 and 17   (angle: 270 + 9 = 279 deg)
    cal_left   (2): wire between 11 and 8   (angle: 180 + 9 = 189 deg)
    cal_right  (3): wire between 6 and 13   (angle: 0 + 9 = 9 deg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Class indices matching YOLO dataset
CAL_TOP = 0
CAL_BOTTOM = 1
CAL_LEFT = 2
CAL_RIGHT = 3
DART_TIP = 4

CAL_CLASSES = [CAL_TOP, CAL_BOTTOM, CAL_LEFT, CAL_RIGHT]
DART_CLASS = DART_TIP

# Zusaetzliche Kalibrierpunkte des dart-sense-Modells (Klassen 5 und 6):
# ebenfalls Drahtkreuzungen am aeusseren Double-Ring, Draht 14|9 (297 Grad)
# und Draht 10|15 (117 Grad). Sie gehen als weitere Stuetzpunkte in die
# Homographie ein (kleinste Quadrate), sind aber nicht erforderlich.
CAL_EXTRA_9 = 5
CAL_EXTRA_15 = 6
CAL_EXTRA_CLASSES = [CAL_EXTRA_9, CAL_EXTRA_15]

# Outer double wire radius in mm
BOARD_RADIUS_MM = 170.0

# 9 degrees offset in radians
OFFSET_RAD = math.radians(9.0)
COS_9 = BOARD_RADIUS_MM * math.cos(OFFSET_RAD)  # ~167.905 mm
SIN_9 = BOARD_RADIUS_MM * math.sin(OFFSET_RAD)  # ~26.595 mm

# Reference coordinates in mm for (x, y), where (0,0) is center, +X is right, +Y is DOWN (image standard)
# Top: wire 5|20: X = -sin(9), Y = -cos(9)
# Bottom: wire 3|17: X = +sin(9), Y = +cos(9)
# Left: wire 11|8: X = -cos(9), Y = +sin(9)
# Right: wire 6|13: X = +cos(9), Y = -sin(9)
CAL_REFERENCE_MM: Dict[int, Tuple[float, float]] = {
    CAL_TOP: (-SIN_9, -COS_9),
    CAL_BOTTOM: (SIN_9, COS_9),
    CAL_LEFT: (-COS_9, SIN_9),
    CAL_RIGHT: (COS_9, -SIN_9),
    # Winkel im Uhrzeigersinn ab 12 Uhr: x = r*sin, y = -r*cos
    CAL_EXTRA_9: (BOARD_RADIUS_MM * math.sin(math.radians(297.0)), -BOARD_RADIUS_MM * math.cos(math.radians(297.0))),
    CAL_EXTRA_15: (BOARD_RADIUS_MM * math.sin(math.radians(117.0)), -BOARD_RADIUS_MM * math.cos(math.radians(117.0))),
}

# Dartboard sector arrangement clockwise starting from 20 (top)
SECTORS_CLOCKWISE = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


@dataclass
class DartScore:
    sector: int          # 1-20 or 25 (bull) or 0 (miss)
    multiplier: int      # 1 (single), 2 (double), 3 (triple)
    points: int          # Total score: sector * multiplier
    description: str     # e.g. "T20", "D16", "S1", "BULL", "BULLSEYE", "MISS"
    radius_mm: float     # Distance from center in mm


def score_at(x_mm: float, y_mm: float) -> DartScore:
    """Calculate the dart score for a hit at board coordinate (x_mm, y_mm).

    Coordinate system: (0,0) is Bullseye center, +X right, +Y down.
    """
    r = math.hypot(x_mm, y_mm)

    # Bullseye (Double Bull)
    if r <= 6.35:
        return DartScore(sector=25, multiplier=2, points=50, description="BULLSEYE", radius_mm=r)

    # Outer Bull (Single Bull)
    if r <= 15.9:
        return DartScore(sector=25, multiplier=1, points=25, description="BULL", radius_mm=r)

    # Off the scoring board
    if r > 170.0:
        return DartScore(sector=0, multiplier=0, points=0, description="MISS", radius_mm=r)

    # Determine multiplier by ring radius
    if 99.0 <= r <= 107.0:
        multiplier = 3
        mult_str = "T"
    elif 162.0 <= r <= 170.0:
        multiplier = 2
        mult_str = "D"
    else:
        multiplier = 1
        mult_str = "S"

    # Angle calculation:
    # 20 is vertically top (x=0, y < 0 in +Y down coordinates).
    # Angle from top: math.atan2(x, -y) gives 0 at top (12 o'clock), positive clockwise!
    angle_from_top = math.atan2(x_mm, -y_mm)  # [-pi, pi]
    deg = math.degrees(angle_from_top)
    if deg < 0:
        deg += 360.0  # [0, 360) clockwise from 12 o'clock

    # Each sector is 18 degrees wide.
    # 20 spans [-9, +9] degrees -> [351, 9] degrees.
    sector_idx = int((deg + 9.0) // 18) % 20
    sector = SECTORS_CLOCKWISE[sector_idx]
    points = sector * multiplier
    description = f"{mult_str}{sector}"

    return DartScore(
        sector=sector,
        multiplier=multiplier,
        points=points,
        description=description,
        radius_mm=r,
    )


def wire_distance_mm(x_mm: float, y_mm: float) -> float:
    """Abstand eines Punkts zum naechsten Draht (Ring- oder Sektorgrenze) in mm.

    Kleine Werte bedeuten: die Wertung haengt an wenigen Millimetern.
    """
    r = math.hypot(x_mm, y_mm)
    radial = min(abs(r - b) for b in (6.35, 15.9, 99.0, 107.0, 162.0, 170.0))
    if r <= 15.9:
        return radial  # im Bull gibt es keine Sektordraehte
    deg = (math.degrees(math.atan2(x_mm, -y_mm)) + 360.0) % 360.0
    off = (deg + 9.0) % 18.0  # Winkelabstand zur letzten Sektorgrenze
    ang = min(off, 18.0 - off)
    return min(radial, r * math.sin(math.radians(ang)))


def compute_homography(
    cal_points: Dict[int, Tuple[float, float]],
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Compute the 3x3 homography matrix mapping image pixels to board millimeters.

    cal_points should map class_id to normalized (x, y) coordinates [0, 1].
    The four main points (0..3) are required; the extra points (5, 6) are used as
    additional least-squares constraints when present.
    Returns None if any of the four main calibration points is missing.
    """
    if any(c not in cal_points for c in CAL_CLASSES):
        return None

    classes = list(CAL_CLASSES) + [c for c in CAL_EXTRA_CLASSES if c in cal_points]
    src = np.array(
        [[cal_points[c][0] * img_w, cal_points[c][1] * img_h] for c in classes],
        dtype=np.float32,
    )
    dst = np.array([CAL_REFERENCE_MM[c] for c in classes], dtype=np.float32)

    homography, _ = cv2.findHomography(src, dst, 0)  # 0 = kleinste Quadrate ueber alle Punkte
    return homography


def project_point_to_mm(
    x_px: float,
    y_px: float,
    homography: np.ndarray,
) -> Tuple[float, float]:
    """Transform pixel coordinates (x_px, y_px) to board millimeter coordinates."""
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
