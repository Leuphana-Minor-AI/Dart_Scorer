#!/usr/bin/env python3
"""Dart Scorer Web-App: Erkennung + Spiellogik + Browser-Oberflaeche.

  python dart_app.py [--camera 1] [--port 20744]

Danach im Browser: http://localhost:20744  (im WLAN: http://<IP des Rechners>:20744)
Auf macOS aus Terminal.app starten (Kamerarechte), z. B. per _start_webapp.command.
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

import uvicorn

from engine.detector import DetectionEngine
from server.app import create_app

ROOT = Path(__file__).resolve().parent


def lan_addresses() -> list[tuple[str, str]]:
    """Alle IPv4-Adressen des Rechners als (Schnittstelle, IP) - ohne Loopback und Link-Local.

    Es wird bewusst nicht nur die Standardroute gezeigt: Haengt das Handy z. B. per
    USB-Tethering oder Hotspot an einer anderen Schnittstelle, ist das die richtige Adresse.
    """
    result: list[tuple[str, str]] = []
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                    result.append((name, a.address))
    except Exception:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            result.append(("default", s.getsockname()[0]))
            s.close()
        except OSError:
            pass
    # Wi-Fi (en0) zuerst
    result.sort(key=lambda t: (t[0] != "en0", t[0]))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Dart Scorer Web-App")
    p.add_argument("--model", default="models/dart_sense_yolov8n.pt")
    p.add_argument("--camera", type=int, default=None, help="Kamera-Index (Standard: automatisch)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--calib", default="camera_calib.npz")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=20744)
    args = p.parse_args()

    calib = ROOT / args.calib
    engine = DetectionEngine(
        model_path=args.model,
        camera_idx=args.camera,
        conf_thresh=args.conf,
        calib_path=str(calib) if calib.exists() else None,
        on_event=lambda *_: None,
    )
    app = create_app(engine)
    app.state.dart.addresses = [(iface, f"http://{ip}:{args.port}") for iface, ip in lan_addresses()]
    print("\n" + "=" * 60)
    print("🎯 Dart Scorer laeuft:")
    print(f"   Laptop:  http://localhost:{args.port}")
    for iface, ip in lan_addresses():
        print(f"   Handy:   http://{ip}:{args.port}   ({iface})")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
