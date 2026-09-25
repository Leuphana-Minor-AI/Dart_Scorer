#!/usr/bin/env python3
"""GoPro & Webcam Diagnostic Tool for Dart Scorer.

Features:
- Scans all camera devices (0..4) and reports available devices.
- Tests camera connection, frame rate and resolution.
- Opens an alignment view with bullseye target crosshair so you can perfectly position your GoPro.
- Press [0..9] to switch camera index on the fly.
- Press [Q] or [ESC] to quit.
"""

from __future__ import annotations

import sys
import time
import cv2
import numpy as np

from camera_stream import _CAPTURE_BACKEND, list_available_cameras


def test_cameras() -> None:
    print("=" * 60)
    print("📹 GoPro & Camera Hardware Diagnostic")
    print("=" * 60)

    print("Scanning connected cameras...")
    cam_details = []
    for idx in range(5):
        c = cv2.VideoCapture(idx, _CAPTURE_BACKEND)
        if c.isOpened():
            ret, f = c.read()
            if ret and f is not None:
                bright = float(np.mean(f))
                cam_details.append((idx, bright, f.shape[1], f.shape[0]))
            c.release()

    if not cam_details:
        print("\n⚠️  No active cameras detected!")
        print("  Tips for GoPro setup:")
        print("  1. Connect GoPro via USB-C cable to your laptop.")
        print("  2. Make sure the 'GoPro Webcam' app is running in the Windows taskbar.")
        print("  3. Turn on the GoPro and wait for the blue icon in the taskbar.")
        print("  4. Alternatively, plug in a USB capture card / Cam Link.")
        print("\nYou can still test with test images or try camera index 0 manually.")
        active_cam = 0
    else:
        print("\n✅ Detected cameras:")
        live_cams = []
        for idx, bright, cw, ch in cam_details:
            if bright > 2.0:
                print(f"  - Kamera [{idx}]: {cw}x{ch} -> LIVE BILD (Helligkeit: {bright:.0f})")
                live_cams.append(idx)
            else:
                print(f"  - Kamera [{idx}]: {cw}x{ch} -> SCHWARZ / STANDBY (GoPro noch nicht eingeschaltet)")

        # Select camera with live feed, otherwise first camera
        if live_cams:
            active_cam = live_cams[0]
        else:
            active_cam = cam_details[0][0]

        print(f"\nOpening camera index [{active_cam}]...")

    cap = cv2.VideoCapture(active_cam, _CAPTURE_BACKEND)
    if not cap.isOpened():
        print(f"Could not open camera [{active_cam}]. Trying index 0...")
        cap = cv2.VideoCapture(0, _CAPTURE_BACKEND)
        active_cam = 0

    # Try setting HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Stream resolution: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print("\nControls in preview window:")
    print("  [0 - 4] : Switch to camera index 0..4")
    print("  [Q/ESC] : Exit diagnostic")
    print("=" * 60)

    cv2.namedWindow("GoPro / Camera Alignment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GoPro / Camera Alignment", 1280, 720)

    fps_count = 0
    fps_start = time.time()
    measured_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Show waiting frame
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"Waiting for Camera [{active_cam}]... (Press 0..4 to switch)",
                (60, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
            )
            cv2.imshow("GoPro / Camera Alignment", blank)
        else:
            h, w = frame.shape[:2]
            fps_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                measured_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            # Draw alignment helper (center crosshair & guide circles for dartboard alignment)
            cx, cy = w // 2, h // 2
            # Crosshair
            cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (0, 255, 255), 2)
            # Bullseye alignment ring
            cv2.circle(frame, (cx, cy), 20, (0, 255, 255), 1)
            # Outer board alignment ring (approximate guide)
            approx_radius = int(min(h, w) * 0.38)
            cv2.circle(frame, (cx, cy), approx_radius, (0, 255, 0), 1)

            # Top HUD banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Check if camera is in black standby
            if float(np.mean(frame)) < 1.0:
                cv2.putText(
                    frame,
                    f"Kamera [{active_cam}] ist im Standby (schwarzes Bild)!",
                    (30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Schliesse GoPro an / schalte sie ein ODER druecke Taste [0] fuer Laptop-Kamera!",
                    (30, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            info_text = f"CAM [{active_cam}] | {w}x{h} | {measured_fps:.1f} FPS | [0-4] Cam wechseln"
            cv2.putText(frame, info_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            keys_text = "Keys: [0-4] Switch Cam  |  [Q/ESC] Quit"
            cv2.putText(frame, keys_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("GoPro / Camera Alignment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            break
        elif ord("0") <= key <= ord("4"):
            new_cam = key - ord("0")
            if new_cam != active_cam:
                print(f"Switching to camera [{new_cam}]...")
                cap.release()
                active_cam = new_cam
                cap = cv2.VideoCapture(active_cam, _CAPTURE_BACKEND)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.release()
    cv2.destroyAllWindows()
    print("Camera diagnostic closed.")


if __name__ == "__main__":
    test_cameras()
