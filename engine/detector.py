"""Erkennungs-Engine fuer die Web-Oberflaeche.

Laeuft in einem eigenen Thread: liest die Kamera, laesst `LiveDartScorer` Kalibrierung und
Pfeil-Verfolgung rechnen und leitet daraus Ereignisse fuer die Spiellogik ab:

  dart_added     {field, points, sector, multiplier, uncertain, x_mm, y_mm, px, py, track_id}
  darts_removed  {}            alle Pfeile wurden vom Board genommen
  calibration    {locked, searching, roi}
  camera         {connected, index, width, height}

Das aktuelle Kamerabild mit Overlay steht als JPEG fuer den MJPEG-Stream bereit.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2

from live_scorer import LiveDartScorer

EventHandler = Callable[[str, dict], None]


class DetectionEngine:
    def __init__(
        self,
        model_path: str,
        camera_idx: Optional[int],
        conf_thresh: float,
        calib_path: Optional[str],
        on_event: EventHandler,
        jpeg_width: int = 1280,
        jpeg_quality: int = 75,
    ) -> None:
        self.on_event = on_event
        self.jpeg_width = jpeg_width
        self.jpeg_quality = jpeg_quality
        self.scorer = LiveDartScorer(
            model_path=model_path,
            camera_idx=camera_idx,
            conf_thresh=conf_thresh,
            mode="PRACTICE",
            calib_path=calib_path,
        )
        self.scorer.external_game = True
        self.scorer.draw_sidebar = False

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._jpeg_seq = 0
        self._known_ids: Set[int] = set()
        # Positionen aller Pfeile, die seit dem letzten Ziehen gezaehlt wurden. Eine neue
        # Spur dicht an so einer Position ist derselbe Pfeil (Spur kurz verloren oder
        # Pfeil blieb ueber die Aufnahme hinaus stecken) und wird nicht erneut gemeldet.
        self._added_positions: List[Tuple[float, float]] = []
        self.readd_radius_px = 12.0
        self._last_cal_state: Optional[Tuple[bool, bool]] = None
        self._accept_darts = True
        self.frame_count = 0

    # ---- Lebenszyklus ---------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="DetectionEngine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.scorer.stream.stop()

    # ---- Steuerung ------------------------------------------------------
    def set_accept_darts(self, accept: bool) -> None:
        """Waehrend keine Aufnahme offen ist (z. B. Spiel vorbei), keine Pfeile melden."""
        self._accept_darts = accept

    def reset_turn(self) -> None:
        """Neues Spiel: Positionsgedaechtnis leeren. Pfeile, die bereits im Board stecken,
        bleiben bekannt und werden nicht nachtraeglich gewertet."""
        with self._lock:
            self._added_positions = []

    def recalibrate(self) -> None:
        self.scorer.reset_calibration()

    def learn_empty_board(self) -> None:
        """Board ist leer: Stoerstellen (Phantom-Pfeile) neu lernen."""
        self.scorer.learn_artifacts_now()
        with self._lock:
            self._known_ids = set()
            self._added_positions = []

    def toggle_lock(self) -> bool:
        self.scorer.cal_locked = not self.scorer.cal_locked
        return self.scorer.cal_locked

    def set_bull_offset(self) -> str:
        darts = self.scorer._last_darts
        self.scorer.calibrate_bull_offset(darts)
        ox, oy = self.scorer.tip_offset_mm
        return f"({ox:+.0f}, {oy:+.0f}) mm"

    def switch_camera(self, index: int) -> bool:
        ok = self.scorer.stream.switch_camera(index)
        self.recalibrate()
        self.on_event("camera", self.camera_info())
        return ok

    def snapshot(self) -> Optional[str]:
        with self._lock:
            jpeg = self._jpeg
        if not jpeg:
            return None
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, dtype="uint8"), cv2.IMREAD_COLOR)
        self.scorer._save_snapshot(img, "web")
        return "snapshots/"

    # ---- Abfragen -------------------------------------------------------
    def latest_jpeg(self) -> Tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq

    def camera_info(self) -> dict:
        st = self.scorer.stream
        return {
            "connected": bool(st.is_camera),
            "index": st.active_index,
            "source": "camera" if st.is_camera else ("test_images" if st.test_images else "none"),
        }

    def status(self) -> dict:
        s = self.scorer
        return {
            "locked": s.cal_locked,
            "calibrated": s.homography is not None,
            "searching": s.roi is None,
            "roi": list(s.roi) if s.roi else None,
            "board_px": round(s.board_px) if s.board_px else None,
            "fps": round(s.fps, 1),
            "camera": self.camera_info(),
            "tip_offset_mm": list(s.tip_offset_mm),
            "darts_on_board": len(s._last_darts),
            "artifacts": len(s.artifacts),
            "learning_empty": s._artifact_samples is not None,
            "model": s.model_path.name,
        }

    # ---- Hauptschleife --------------------------------------------------
    def _loop(self) -> None:
        s = self.scorer
        self.on_event("camera", self.camera_info())
        errors = 0
        while self._running:
            try:
                ret, frame = s.stream.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                if s.undistorter is not None:
                    frame = s.undistorter.apply(frame)

                vis = s.process_frame(frame)
                self.frame_count += 1
                self._emit_calibration()
                self._emit_darts()
                self._encode(vis)
                errors = 0
            except Exception:  # Die Erkennung darf nie sterben - Fehler loggen, weitermachen
                import traceback
                errors += 1
                traceback.print_exc()
                if errors >= 5:
                    print("Erkennung: wiederholte Fehler -> Kalibrierung zuruecksetzen")
                    s.reset_calibration()
                    errors = 0
                time.sleep(0.1)

    def _emit_calibration(self) -> None:
        s = self.scorer
        state = (bool(s.cal_locked), s.roi is None)
        if state != self._last_cal_state:
            self._last_cal_state = state
            self.on_event("calibration", {"locked": state[0], "searching": state[1], "roi": list(s.roi) if s.roi else None})

    def _emit_darts(self) -> None:
        s = self.scorer
        darts = list(s._last_darts)
        ids = {d.track.id for d in darts if d.track is not None}

        with self._lock:
            known = set(self._known_ids)
            self._known_ids = set(ids)
            added_positions = list(self._added_positions)

        # Alle Pfeile weg -> Aufnahme beendet; das leere Board kurz auf Stoerstellen pruefen
        if known and not ids:
            with self._lock:
                self._added_positions = []
            self.on_event("darts_removed", {})
            s.learn_artifacts_now()
            return

        new_darts = [d for d in sorted(darts, key=lambda d: d.track.first_seen) if d.track.id not in known]
        for d in new_darts:
            if not self._accept_darts:
                continue
            # Bereits gezaehlter Pfeil (Spur kurz verloren / steckt noch von vorher)?
            if any(math.hypot(d.px - ax, d.py - ay) < self.readd_radius_px for ax, ay in added_positions):
                continue
            with self._lock:
                self._added_positions.append((d.px, d.py))
            added_positions.append((d.px, d.py))
            sc = d.score
            self.on_event(
                "dart_added",
                {
                    "field": sc.description,
                    "points": sc.points,
                    "sector": sc.sector,
                    "multiplier": sc.multiplier,
                    "uncertain": bool(d.uncertain),
                    "x_mm": round(d.x_mm, 1),
                    "y_mm": round(d.y_mm, 1),
                    "px": round(d.px),
                    "py": round(d.py),
                    "track_id": d.track.id,
                },
            )

    def _encode(self, vis) -> None:
        h, w = vis.shape[:2]
        if w > self.jpeg_width:
            scale = self.jpeg_width / w
            vis = cv2.resize(vis, (self.jpeg_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._jpeg_seq += 1
