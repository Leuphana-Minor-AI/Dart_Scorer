"""Camera stream management for GoPro, webcams, and test image sources.

Supports:
- Plattformabhaengiges Backend: DirectShow (Windows), AVFoundation (macOS), V4L2 (Linux).
- Automatic device enumeration to find connected cameras.
- Resolution negotiation (1080p, 720p).
- Fallback test mode using static validation images if no camera is connected.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("DartCamera")

# Capture-Backend je Plattform: DirectShow gibt es nur unter Windows,
# macOS nutzt AVFoundation, Linux V4L2.
if sys.platform == "win32":
    _CAPTURE_BACKEND = cv2.CAP_DSHOW
elif sys.platform == "darwin":
    _CAPTURE_BACKEND = cv2.CAP_AVFOUNDATION
else:
    _CAPTURE_BACKEND = cv2.CAP_ANY


def list_available_cameras(max_tested: int = 4) -> List[int]:
    """Probe camera indices to find working video capture devices."""
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, _CAPTURE_BACKEND)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


class CameraStream:
    """Threaded camera capture to prevent buffer lag and ensure real-time frames."""

    def __init__(
        self,
        camera_index: Optional[int] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fallback_dir: Optional[str] = "dataset/images/val",
    ) -> None:
        self.requested_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = False
        self.active_index = -1

        # Threading state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps_measured = 0.0
        self._last_fps_time = time.time()
        self._fps_counter = 0

        # Fallback images
        self.test_images: List[Path] = []
        self.test_img_idx = 0

        self._init_source()

    def _init_source(self) -> None:
        """Initialize camera or fallback to dataset images."""
        # Probe all candidate cameras and check for actual live video
        active_candidates = []
        for idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(idx, _CAPTURE_BACKEND)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Mehrere Leseversuche: virtuelle Kameras (GoPro Webcam) liefern das
                # erste Bild oft erst nach einigen hundert ms, eingebaute Kameras
                # anfangs ein schwarzes Bild (Belichtung noch nicht eingeregelt).
                ret, frame = False, None
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    ret, frame = cap.read()
                    if ret and frame is not None and float(np.mean(frame)) > 1.0:
                        break
                    time.sleep(0.05)
                if ret and frame is not None:
                    brightness = float(np.mean(frame))
                    fh, fw = frame.shape[:2]
                    logger.info(f"Kamera [{idx}] gefunden: {fw}x{fh}, Helligkeit {brightness:.1f}")
                    active_candidates.append((idx, cap, brightness, frame))
                else:
                    cap.release()

        if not active_candidates:
            logger.warning("Keine Kamera liefert ein Bild (Indizes 0-3 geprueft).")

        # If user explicitly requested an index
        chosen = None
        if self.requested_index is not None:
            for idx, cap, bright, f in active_candidates:
                if idx == self.requested_index:
                    chosen = (idx, cap, bright, f)
                else:
                    cap.release()
        else:
            # Auto-select: prefer camera with actual live light (brightness > 2.0)
            # If camera 1 (GoPro) is delivering real video (> 2.0), prefer camera 1!
            # If camera 1 is pitch black (GoPro app virtual camera idle), fall back to camera 0!
            candidates_with_light = [c for c in active_candidates if c[2] > 2.0]
            if candidates_with_light:
                # If GoPro (index >= 1) has light, use it, otherwise use index 0
                gopro_with_light = [c for c in candidates_with_light if c[0] >= 1]
                chosen = gopro_with_light[0] if gopro_with_light else candidates_with_light[0]
            elif active_candidates:
                chosen = active_candidates[0]

            # Release unused cameras
            for idx, cap, bright, f in active_candidates:
                if chosen is None or idx != chosen[0]:
                    cap.release()

        if chosen is not None:
            self.cap = chosen[1]
            self.latest_frame = chosen[3]
            self.is_camera = True
            self.active_index = chosen[0]
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Connected to Camera [{self.active_index}] ({actual_w}x{actual_h}) - Brightness: {chosen[2]:.1f}")
            return

        # If no camera found, check for test images across common locations
        base = Path(__file__).resolve().parent
        fallback_dirs = [
            self.fallback_dir,
            base / "dataset" / "images" / "val",
            base / "test_images",
            base / "runs" / "validation" / "val",
            base / "runs" / "training" / "yolo11n_darts",
            base,
        ]
        for fdir in fallback_dirs:
            if fdir and fdir.exists():
                # Fotos (jpg) zuerst, Diagramme (png) nur als letzte Reserve
                imgs = sorted(fdir.glob("*.jpg")) or sorted(fdir.glob("*.png"))
                if imgs:
                    self.test_images = imgs
                    break

        if self.test_images:
            self.is_camera = False
            self.active_index = -1
            logger.warning(
                f"No camera detected! Falling back to test images mode ({len(self.test_images)} images found)."
            )
        else:
            logger.warning("No camera detected and no test images found. Generating virtual synthetic board.")

    def start(self) -> "CameraStream":
        """Start background capture thread."""
        self.running = True
        if self.is_camera and self.cap is not None:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
        return self

    def _capture_loop(self) -> None:
        """Continuously grab frames to always have the freshest one (zero latency)."""
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.frame_count += 1
                    self._fps_counter += 1

                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self.fps_measured = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._last_fps_time = now
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame."""
        if self.is_camera:
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    if float(np.mean(frame)) < 1.0:
                        fh, fw = frame.shape[:2]
                        cv2.putText(
                            frame,
                            f"Kamera [{self.active_index}] liefert kein Bild (Standby / schwarz)",
                            (40, fh // 2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 165, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Druecke Taste [0] fuer Laptop-Kamera oder schliesse GoPro an!",
                            (40, fh // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 255),
                            2,
                        )
                    return True, frame
            return False, None

        # Test image mode
        if self.test_images:
            img_path = self.test_images[self.test_img_idx % len(self.test_images)]
            frame = cv2.imread(str(img_path))
            if frame is not None:
                self.frame_count += 1
                return True, frame

        # Fallback synthetic frame
        synth = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            synth,
            "No Camera / No Images Found",
            (50, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return True, synth

    def next_test_image(self) -> None:
        """Switch to next test image (for test mode)."""
        if self.test_images:
            self.test_img_idx = (self.test_img_idx + 1) % len(self.test_images)

    def prev_test_image(self) -> None:
        """Switch to previous test image (for test mode)."""
        if self.test_images:
            self.test_img_idx = (self.test_img_idx - 1) % len(self.test_images)

    def switch_camera(self, new_index: int) -> bool:
        """Switch to a different camera device index."""
        self.stop()
        self.requested_index = new_index
        self._init_source()
        if self.is_camera:
            self.start()
            return True
        return False

    def stop(self) -> None:
        """Stop capture thread and release hardware resources."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        self.stop()
