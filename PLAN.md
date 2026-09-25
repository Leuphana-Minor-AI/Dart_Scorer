# Plan: Live Dart Scorer → spielbares Dart-System

Stand: 24.09.2026. Erkennung (dart-sense-Modell, 6-Punkt-Kalibrierung, Spurverfolgung)
funktioniert. Dieses Dokument beschreibt den Ausbau zu einem nutzbaren Spielsystem mit
richtiger Oberfläche.

## Ziel

Ein Dartabend mit 2–4 Spielern, 501 Double-Out, mehrere Legs – ohne Tastatur am Laptop.
Anzeige groß und vom Oche lesbar, Bedienung per Klick/Touch (auch vom Handy), Fehlerkennungen
in Sekunden korrigierbar.

## Architektur-Entscheidung: Web-Oberfläche statt OpenCV-Fenster

Das OpenCV-Fenster wird durch eine lokale Web-App ersetzt:

```
live_scorer.py  ──►  engine/   Kamera, Erkennung, Kalibrierung  → Ereignisse (Pfeil erkannt, gezogen, Board verdeckt)
                     game/     Regeln und Spielstand (X01, Practice, Spieler, Legs, Statistik)
                     server/   FastAPI: WebSocket (Zustand), MJPEG (Kamerabild), REST (Aktionen)
                     web/      HTML/JS-Frontend (Scoreboard, Board-Ansicht, Einstellungen)
```

Warum Web:
- Läuft auf dem Laptop **und** auf Handy/Tablet im selben WLAN (`http://<mac>:20744`) – Scoreboard
  am Board, Bedienung vom Oche aus.
- Große, saubere Typografie, Animationen, Dark Mode – mit HTML/CSS trivial, mit OpenCV kaum.
- (Sprachansagen: nicht gewünscht, entfernt)
- Korrektur per **Klick auf das Board-Bild** statt Pfeiltasten.
- Die Erkennung läuft weiter als reines Python ohne GUI-Abhängigkeit (Terminal.app-Start bleibt
  wegen der Kamerarechte).

Der bestehende OpenCV-Modus bleibt als `--headless off` für Debugging erhalten.

## Features nach Priorität

### M1 – Spielbar (Grundgerüst) ✅ umgesetzt am 24.09.2026
| # | Feature | Details |
|---|---|---|
| 1.1 | Trennung Engine / Spiel / Server | Erkennungscode aus `live_scorer.py` in `engine/` überführen; Engine liefert Ereignisse: `dart_added(field, mm, px, unsicher)`, `darts_removed`, `board_occluded`, `calibration(state)` |
| 1.2 | Spielmodell X01 | 301/501/701, 2–4 Spieler mit Namen, Reihenfolge, Double-Out an/aus, Bust-Regeln (Rest < 0, = 1 bei Double-Out, 0 ohne Double), Leg-Zähler (Best of n), Anwurfwechsel |
| 1.3 | Aufnahme-Logik | 3 Pfeile → Aufnahme abgeschlossen, wenn Pfeile gezogen werden (oder Klick „Weiter“); Undo letzter Pfeil / letzte Aufnahme; fehlender Pfeil (Bounce-out) per Klick „Miss“ nachtragen |
| 1.4 | Web-Server | FastAPI + WebSocket für Spielstand (JSON), MJPEG-Stream des Kamerabilds mit Overlay, REST-Aktionen (undo, next, miss, correct, new game) |
| 1.5 | Scoreboard-UI | Spielerkacheln mit Restpunkten (sehr groß), aktueller Spieler hervorgehoben, 3 Pfeilfelder der laufenden Aufnahme, Checkout-Vorschlag, Kalibrierstatus, Kamerabild klein |
| 1.6 | Start-Skript | Ein Doppelklick: Server startet, Browser öffnet sich; Adresse fürs Handy wird angezeigt |

### M2 – Zuverlässig im Spielbetrieb
| # | Feature | Details |
|---|---|---|
| 2.1 | Korrektur per Klick | Klick auf das Board-Bild (Radar oder Kamerabild) setzt das Feld des gewählten Pfeils; Unsicher-Markierung (`?`) bleibt sichtbar |
| 2.2 | Verdeckung / Hand-Erkennung | Während der Spieler vor dem Board steht oder Pfeile zieht: Erkennung pausieren (Bewegungsmaß im Ausschnitt), erst bei Ruhe werten – verhindert Geister-Pfeile beim Ziehen |
| 2.3 | Kamera-Wiederverbindung | GoPro-Abbrüche automatisch erkennen und neu verbinden, Status in der UI („Kamera getrennt“) |
| 2.4 | Auto-Rekalibrierung | Wenn nach dem Einrasten die Kalibrierpunkte dauerhaft woanders liegen (Board/Kamera bewegt): neu kalibrieren, Hinweis in der UI |
| 2.5 | Statistik im Spiel | 3-Dart-Average, First-9-Average, höchste Aufnahme, Checkout-Quote, 180er-Zähler; Leg- und Match-Ergebnis |
| 2.6 | ~~Sprachausgabe~~ | entfällt – vom Nutzer nicht gewünscht (25.09.2026) |
| 2.7 | Einstellungsseite | Kamera-Index, Spielmodus-Defaults, Double-Out, Spielernamen, Bull-Offset setzen/zurücksetzen, Board manuell markieren, Schnappschuss |

### M3 – Training und Komfort
| # | Feature | Details |
|---|---|---|
| 3.1 | Trainingsmodi | Freies Training mit Feld-Statistik, Around the Clock, Bob’s 27, Zielfeld-Training (Trefferquote je Feld/Ring) |
| 3.2 | Spielhistorie | Spiele und Aufnahmen in SQLite speichern; Spielerstatistik über Zeit (Average-Verlauf), Export als CSV |
| 3.3 | Cricket | Standard-Cricket für 2–4 Spieler (Treffer-Marken 15–20 + Bull) |
| 3.4 | Board-Heatmap | Trefferverteilung pro Spieler als Heatmap auf dem Board |
| 3.5 | Handy als Fernbedienung | Reduzierte Ansicht: Restpunkte, Undo/Weiter/Miss – für die Bedienung vom Oche |
| 3.6 | Design | Farbschema, Animationen (180, Game shot), Vollbild-Modus fürs Board-Display |

## Bestehende Bausteine, die übernommen werden

- Kalibrierung (Suche, ROI, 6 Punkte, Plausibilität, Gedächtnis), Spurverfolgung, Bull-Offset,
  Unsicher-Erkennung (`wire_distance_mm`), Checkout-Tabelle, Schnappschüsse.
- `dart_geometry.py` bleibt unverändert die Regelbasis.

## Entscheidungen (24.09.2026)

1. **UI-Technik**: Web-App im Browser (FastAPI + WebSocket + MJPEG, statisches HTML/JS-Frontend).
2. **Spielmodi in M1**: nur X01 (301/501/701) mit Double-Out. Freies Training und Cricket später.
3. **Spieleranzahl**: 1–4, konfigurierbar mit Namen und Reihenfolge.
4. **Handy-Bedienung**: erst M3 – M1 wird für den Laptop-Bildschirm gestaltet (Layout aber nicht
   dagegen bauen: flexible Breiten, keine festen Pixelmaße).

## Technische Leitplanken für M1

- Ein Prozess: Engine-Thread (Kamera + Erkennung) und Web-Server (uvicorn) im selben Python-Prozess;
  Start weiterhin über `_start_scorer.command` aus Terminal.app (Kamerarechte).
- Zustand ist die einzige Wahrheit: `GameState` (Spieler, Legs, Aufnahmen, aktueller Wurf) wird
  bei jeder Änderung als JSON per WebSocket an alle Clients gesendet; das Frontend rendert nur.
- Aktionen vom Frontend (Undo, Weiter, Miss, Korrektur, Neues Spiel) gehen als REST-POST an den
  Server und ändern den `GameState` über die Regeln in `game/`.
- Erkennung liefert der Spiellogik ausschließlich Ereignisse (`dart_added`, `darts_removed`,
  `calibration_changed`); die Spiellogik entscheidet, was ein Pfeil im Spiel bedeutet.
- Keine Build-Toolchain im Frontend: Vanilla HTML/CSS/JS, damit das Projekt ohne Node läuft.

## Reihenfolge und Aufwand

| Meilenstein | Umfang | Ergebnis |
|---|---|---|
| M1 | ~2 Arbeitssessions | 501 zu zweit spielbar im Browser, Undo/Weiter/Miss, Kamerabild, Start per Doppelklick |
| M2 | ~2 Sessions | Korrektur per Klick, Verdeckungs-Pause, Kamera-Reconnect, Statistik, Einstellungen |
| M3 | nach Bedarf | Trainingsmodi, Historie, Cricket, Heatmap, Handy-Ansicht, Design-Politur |
