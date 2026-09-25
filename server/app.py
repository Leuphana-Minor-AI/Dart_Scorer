"""FastAPI-Server: Spielzustand per WebSocket, Kamerabild als MJPEG, Aktionen per REST."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine.detector import DetectionEngine
from game.x01 import Dart, X01Game, parse_field

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SETTINGS_PATH = ROOT / "web_settings.json"


class AppState:
    def __init__(self, engine: DetectionEngine) -> None:
        self.engine = engine
        engine.on_event = self.on_event
        self.game: Optional[X01Game] = None
        self.lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: Set[WebSocket] = set()
        self.detection: Dict[str, Any] = {"calibration": {}, "camera": {}}
        self.settings = self._load_settings()
        self.events: list = []  # letzte Ereignisse fuer die Anzeige
        self.addresses: list = []  # (Schnittstelle, URL) fuer die Anzeige auf der Einrichtungsseite

    # ---- Einstellungen --------------------------------------------------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (OSError, ValueError):
            return {"players": ["Spieler 1", "Spieler 2"], "start_score": 501, "double_out": True, "legs_to_win": 1}

    def save_settings(self) -> None:
        try:
            SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2))
        except OSError:
            pass

    # ---- Zustand --------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            game = self.game.to_dict() if self.game else None
        return {
            "game": game,
            "detection": {**self.engine.status(), **self.detection},
            "settings": self.settings,
            "events": self.events[-8:],
            "addresses": self.addresses,
            "time": time.time(),
        }

    def _log(self, text: str) -> None:
        self.events.append({"time": time.time(), "text": text})
        self.events = self.events[-50:]

    # ---- Ereignisse aus der Engine (fremder Thread) ----------------------
    def on_event(self, name: str, data: dict) -> None:
        with self.lock:
            if name == "dart_added":
                if self.game and not self.game.over:
                    dart = Dart(
                        field=data["field"], points=data["points"], sector=data["sector"],
                        multiplier=data["multiplier"], manual=False, uncertain=data["uncertain"],
                        x_mm=data.get("x_mm"), y_mm=data.get("y_mm"),
                    )
                    if self.game.add_dart(dart):
                        self._log(f"{self.game.player.name}: {dart.field} ({dart.points})")
            elif name == "darts_removed":
                if self.game and self.game.turn.darts:
                    name_before = self.game.player.name
                    pts = self.game.turn.points
                    if self.game.end_turn():
                        self._log(f"{name_before}: Aufnahme {pts}")
            elif name in ("calibration", "camera"):
                self.detection[name] = data
        self.broadcast()

    # ---- Aktionen (Server-Thread) --------------------------------------
    def new_game(self, players, start_score, double_out, legs_to_win) -> dict:
        with self.lock:
            self.game = X01Game(players, start_score, double_out, legs_to_win)
            self.settings.update({"players": players, "start_score": start_score,
                                  "double_out": double_out, "legs_to_win": legs_to_win})
            self.events = []
            self._log(f"Neues Spiel: {start_score}, {'Double-Out' if double_out else 'Single-Out'}, "
                      f"Best of {2 * legs_to_win - 1}")
        self.save_settings()
        self.engine.reset_turn()
        self.engine.set_accept_darts(True)
        self.broadcast()
        return {"ok": True}

    def end_game(self) -> dict:
        with self.lock:
            self.game = None
        self.broadcast()
        return {"ok": True}

    def turn_action(self, action: str, payload: dict) -> dict:
        with self.lock:
            g = self.game
            if g is None:
                return {"ok": False, "error": "Kein Spiel"}
            if action == "next":
                ok = g.end_turn()
            elif action == "undo":
                ok = g.undo()
            elif action == "miss":
                ok = g.add_miss()
            elif action == "add":
                try:
                    ok = g.add_dart(parse_field(str(payload.get("field", "")), manual=True))
                except ValueError as e:
                    return {"ok": False, "error": str(e)}
            elif action == "correct":
                try:
                    ok = g.correct_dart(int(payload.get("index", -1)), str(payload.get("field", "")))
                except ValueError as e:
                    return {"ok": False, "error": str(e)}
            else:
                return {"ok": False, "error": f"Unbekannte Aktion {action}"}
            if ok:
                self._log({"next": "Weiter", "undo": "Undo", "miss": "Miss nachgetragen",
                           "add": f"Pfeil nachgetragen: {payload.get('field')}",
                           "correct": f"Korrektur -> {payload.get('field')}"}[action])
        self.broadcast()
        return {"ok": bool(ok)}

    def detection_action(self, action: str, payload: dict) -> dict:
        e = self.engine
        if action == "recalibrate":
            e.recalibrate(); msg = "Kalibrierung neu gestartet"
        elif action == "lock":
            msg = "Kalibrierung " + ("eingefroren" if e.toggle_lock() else "freigegeben")
        elif action == "bull_offset":
            msg = "Bull-Offset " + e.set_bull_offset()
        elif action == "snapshot":
            msg = "Schnappschuss gespeichert" if e.snapshot() else "Kein Bild"
        elif action == "learn_empty":
            e.learn_empty_board(); msg = "Leeres Board wird gelernt (2 s) ..."
        elif action == "camera":
            ok = e.switch_camera(int(payload.get("index", 0)))
            msg = f"Kamera {payload.get('index')} " + ("aktiv" if ok else "nicht verfuegbar")
        else:
            return {"ok": False, "error": f"Unbekannte Aktion {action}"}
        with self.lock:
            self._log(msg)
        self.broadcast()
        return {"ok": True, "message": msg}

    # ---- WebSocket -----------------------------------------------------
    def broadcast(self) -> None:
        if self.loop is None or not self.clients:
            return
        asyncio.run_coroutine_threadsafe(self._send_all(), self.loop)

    async def _send_all(self) -> None:
        payload = json.dumps(self.snapshot(), ensure_ascii=False)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(engine: DetectionEngine) -> FastAPI:
    state = AppState(engine)
    app = FastAPI(title="Dart Scorer")
    app.state.dart = state

    @app.on_event("startup")
    async def _startup() -> None:
        state.loop = asyncio.get_running_loop()
        engine.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine.stop()

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.post("/api/game/new")
    async def api_new_game(payload: dict) -> JSONResponse:
        players = [str(p) for p in payload.get("players", [])][:4] or ["Spieler 1"]
        try:
            return JSONResponse(state.new_game(
                players,
                int(payload.get("start_score", 501)),
                bool(payload.get("double_out", True)),
                int(payload.get("legs_to_win", 1)),
            ))
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    @app.post("/api/game/end")
    async def api_end_game() -> JSONResponse:
        return JSONResponse(state.end_game())

    @app.post("/api/turn/{action}")
    async def api_turn(action: str, payload: Optional[dict] = None) -> JSONResponse:
        return JSONResponse(state.turn_action(action, payload or {}))

    @app.post("/api/detection/{action}")
    async def api_detection(action: str, payload: Optional[dict] = None) -> JSONResponse:
        return JSONResponse(state.detection_action(action, payload or {}))

    @app.get("/stream.mjpg")
    async def stream() -> StreamingResponse:
        async def gen():
            last_seq = -1
            while True:
                jpeg, seq = engine.latest_jpeg()
                if jpeg and seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(1 / 15)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        state.clients.add(ws)
        try:
            await ws.send_text(json.dumps(state.snapshot(), ensure_ascii=False))
            while True:
                # Client schickt gelegentlich Pings; Antworten mit aktuellem Zustand
                await ws.receive_text()
                await ws.send_text(json.dumps(state.snapshot(), ensure_ascii=False))
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(ws)

    return app
