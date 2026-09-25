"""Linsenentzerrung fuer Weitwinkelkameras (z. B. GoPro).

Die Homographie im Scorer setzt eine verzerrungsfreie (Lochkamera-)Abbildung
voraus. Weitwinkelobjektive kruemmen gerade Linien tonnenfoermig, wodurch die
Punkteberechnung zum Rand hin ungenau wird. Mit einer einmaligen
Schachbrett-Kalibrierung (calibrate_lens.py) wird jedes Kamerabild vor der
Auswertung entzerrt.

Kalibrierdatei (npz): K (3x3 Kameramatrix), dist (Verzerrungskoeffizienten),
width/height (Aufloesung, bei der kalibriert wurde).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class LensUndistorter:
    def __init__(self, K: np.ndarray, dist: np.ndarray, calib_size: Tuple[int, int]) -> None:
        self.K = K.astype(np.float64)
        self.dist = dist.astype(np.float64)
        self.calib_size = calib_size  # (width, height)
        self._maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._map_size: Optional[Tuple[int, int]] = None

    @classmethod
    def load(cls, path: str | Path) -> Optional["LensUndistorter"]:
        """Kalibrierdatei laden; None, wenn sie nicht existiert."""
        p = Path(path)
        if not p.exists():
            return None
        data = np.load(p)
        size = (int(data["width"]), int(data["height"]))
        return cls(data["K"], data["dist"], size)

    def save(self, path: str | Path) -> None:
        np.savez(path, K=self.K, dist=self.dist, width=self.calib_size[0], height=self.calib_size[1])

    def _build_maps(self, w: int, h: int) -> None:
        # Kameramatrix auf die aktuelle Aufloesung skalieren, falls sie von der
        # Kalibrier-Aufloesung abweicht
        sx = w / self.calib_size[0]
        sy = h / self.calib_size[1]
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        # alpha=1: alle Originalpixel behalten (schwarze Raender statt Beschnitt)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, self.dist, (w, h), alpha=1)
        self._maps = cv2.initUndistortRectifyMap(K, self.dist, None, new_K, (w, h), cv2.CV_16SC2)
        self._map_size = (w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._maps is None or self._map_size != (w, h):
            self._build_maps(w, h)
        assert self._maps is not None
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
