# Dart Scorer – automatisches Zählen beim Steeldart mit einer Kamera

Ein Dartboard, eine Kamera (GoPro oder Webcam) und ein Laptop: Das Programm erkennt die
Dartscheibe im Kamerabild, kalibriert sich selbst, erkennt die Pfeile und zählt ein X01-Spiel
(301/501/701) für bis zu vier Spieler – mit Scoreboard im Browser, auch auf dem Handy.

## Funktionen

- **Automatische Kalibrierung** über die Drahtkreuzungen am Double-Ring (6 Stützpunkte,
  Kleinste-Quadrate-Homographie); Board darf irgendwo im Bild stehen, Auto-Rekalibrierung,
  wenn sich Kamera oder Board bewegen
- **Pfeil-Erkennung** mit einem YOLO-Modell, Spurverfolgung über mehrere Frames, Ausblenden
  von Störstellen, Duplikat-Erkennung, Markierung unsicherer Pfeile (`?`) nahe an Drähten
- **X01-Spiel**: 301/501/701, 1–4 Spieler, Double-Out/Single-Out, Bust-Regeln, Legs (Best of n),
  Anwurfwechsel, Checkout-Vorschläge, Statistik (Average, First 9, 180er, höchstes Finish, bestes Leg)
- **Web-Oberfläche**: großes Scoreboard, Aufnahme mit drei Pfeilen, Korrektur per Klick auf das Board,
  Undo, Miss/Bounce-out, Pfeil nachtragen, Kamerabild mit Overlay, Sieger-Anzeige mit Revanche;
  responsives Layout für Handy/Tablet im selben Netz
- **Diagnose**: Kalibrier-/Kamerastatus, FPS, Schnappschüsse, Bull-Offset-Kalibrierung,
  manuelles Markieren des Boards, Modellvergleich auf Testbildern

## Aufbau

```
dart_app.py            Startpunkt der Web-App (Engine-Thread + Web-Server in einem Prozess)
engine/detector.py     Kamera + Erkennung → Ereignisse (dart_added, darts_removed, calibration, camera)
live_scorer.py         Kalibrierung, Board-Ausschnitt, Pfeil-Verfolgung, Punkteberechnung (+ alter OpenCV-Modus)
camera_stream.py       Kameraerkennung (AVFoundation / DirectShow / V4L2), Threaded Capture, Testbild-Modus
dart_geometry.py       Board-Geometrie nach WDF, Homographie, Feld aus (x, y) in mm, Drahtabstand
lens_undistort.py      Optionale Linsenentzerrung (calibrate_lens.py erzeugt camera_calib.npz)
game/x01.py            Spiellogik X01 (reines Regelwerk, ohne Kamera)
server/app.py          FastAPI: Zustand per WebSocket, Kamerabild als MJPEG, Aktionen per REST
web/                   Oberfläche (HTML/CSS/JS ohne Build-Tools)
tests/test_x01.py      Tests der Spiellogik (pytest)
tools/compare_models.py  Zwei Modelle auf Testbildern vergleichen
train.py, evaluate.py  Training / Auswertung des eigenen YOLO-Modells (DeepDarts-Datensatz)
models/                Modellgewichte (siehe unten)
```

## Installation

Python ≥ 3.10. Empfohlen: [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt opencv-python
```

Windows (Anaconda): `setup_windows.bat` installiert PyTorch (CPU), Ultralytics und OpenCV.

### Modelle

Zwei Modelle werden unterstützt, beide mit derselben Klassenreihenfolge
(`cal_top` 5|20, `cal_bottom` 3|17, `cal_left` 11|8, `cal_right` 6|13, `dart_tip`):

| Datei | Herkunft | Einsatz |
|---|---|---|
| `models/dart_sense_yolov8n.pt` | [bnww/dart-sense](https://github.com/bnww/dart-sense) (Ben Willshaw), YOLOv8n, ~24.000 Bilder aus vielen Kamerawinkeln, zwei Zusatz-Kalibrierpunkte (Klassen `9`, `15`); Lizenz **CC BY-NC 4.0** | **Standard.** Findet die Eintrittspunkte der Pfeile auch bei schräger Kamera |
| `models/dart_yolo11n_best.pt` | eigenes Training (YOLO11n) auf dem DeepDarts-Datensatz | Rückfall, wenn die dart-sense-Datei fehlt; nur für frontale Kamera geeignet |

Die dart-sense-Gewichte sind nicht Teil dieses Repositories (Lizenz erlaubt keine Weitergabe).
Einmalig herunterladen:

```bash
curl -L -o models/dart_sense_yolov8n.pt https://github.com/bnww/dart-sense/raw/main/weights.pt
```

### GoPro als Webcam

GoPro Webcam-App installieren, Kamera per USB oder WLAN verbinden. Empfohlene Einstellungen:
Sichtfeld „Linear“ (weniger Verzerrung), automatisches Ausschalten deaktivieren. Alternativ
funktioniert jede Webcam. Das Programm probiert die Kamera-Indizes 0–3 durch und wählt die mit
Live-Bild; mit `--camera N` lässt sich der Index festlegen.

**macOS:** Kamerazugriff wird nur Prozessen gewährt, die aus Terminal.app gestartet wurden – daher
die `.command`-Skripte per Doppelklick oder `open -a Terminal <skript>` starten.

## Starten

```bash
open -a Terminal _start_webapp.command      # macOS: startet Server und öffnet den Browser
python dart_app.py [--camera 1] [--port 20744]
```

Im Terminal erscheinen die Adressen: `http://localhost:20744` auf dem Laptop, dazu alle
Netzwerkadressen des Rechners für Handy/Tablet im selben Netz (auch auf der Einrichtungsseite
angezeigt). In verwalteten WLANs (Uni, Firma) ist Gerät-zu-Gerät-Verkehr oft gesperrt – dann
das Handy per USB-Tethering oder Hotspot mit dem Laptop verbinden.

Ablauf im Spiel: Spieler und Modus wählen → werfen → Pfeile erscheinen in der Aufnahme →
bei Bedarf Pfeil antippen und richtiges Feld auf dem Board anklicken → Pfeile ziehen →
nächster Spieler. „Weiter“ nur, wenn früher gewechselt werden soll.

**Tastatur:** `U` Undo · `Leertaste`/`Enter` Weiter · `M` Miss · `+`/`A` Pfeil nachtragen ·
`1`–`3` Pfeil korrigieren · `Esc` schließen

**Menü ⚙︎:** Rekalibrieren · Board leer (Störstellen neu lernen, nach dem Ziehen aller Pfeile) ·
Bull-Offset (ein Pfeil im Bull → systematischen Versatz messen) · Lock · Schnappschuss ·
Kamerabild · Spiel beenden

### Alter Fenstermodus (Debugging)

`_start_scorer.command` bzw. `run_dart_scorer.bat` starten `live_scorer.py` mit OpenCV-Fenster,
Radar und Tastensteuerung (`C` Rekalibrieren, `L` Lock, `R` Board markieren, `B` Bull-Offset,
`S` Schnappschuss, `TAB`+Pfeiltasten Korrektur, `0`–`4` Kamera, `Q` Beenden).
Ohne Kamera nutzt er Bilder aus `test_images/`.

## So funktioniert die Erkennung

1. **Board suchen** – Vollbild durch das Modell, alle 10 Frames zusätzlich kachelweise in voller
   Auflösung. Ab zwei Kalibrierpunkten wird ein quadratischer Ausschnitt ums Board gelegt und nur
   noch dieser ausgewertet, skaliert auf ~500 px Boardgröße – unabhängig vom Kameraabstand.
2. **Kalibrieren** – Homographie aus den Drahtkreuzungen (mm-Referenzen nach WDF in
   `dart_geometry.py`). Punktgedächtnis (3 s), rotierende Skalierungs-/Kontrastvarianten,
   Plausibilitätsprüfung (Durchmesser-Mittelpunkte, Reihenfolge), Vorhersage fehlender Punkte.
   Nach 15 stabilen Frames eingerastet; Drift-Erkennung kalibriert bei Bewegung neu.
3. **Pfeile** – Modell auf dem Ausschnitt in zwei wechselnden Größen; Radiusfilter (> 180 mm
   verworfen); Störstellen des leeren Boards ausgeblendet; Duplikate (Spitze + Schaft) entlang der
   Schaftrichtung zusammengefasst; Spur gilt ab 3 Treffern, Position geglättet.
4. **Punkte** – Pixel → mm → Ring/Sektor. Näher als 3 mm an einem Draht (oder knapp außerhalb
   des Double-Rings) → `?`. Ein optionaler konstanter Offset (Bull-Kalibrierung) wird abgezogen.
5. **Spiel** – Die Engine meldet nur Ereignisse; `game/x01.py` entscheidet Bust, Finish, Legs.
   Ein einmal gezählter Pfeil wird bis zum Ziehen aller Pfeile nicht erneut gezählt.

Wichtige Stellschrauben in `live_scorer.py`: `dart_conf` (Schwelle für Pfeile, Standard 0,15),
`conf_thresh` (Kalibrierpunkte, 0,25), `track_min_hits`, `track_merge_px`/`shaft_dir`
(Duplikate), `artifact_*` (Störstellen), `UNCERTAIN_WIRE_MM`, `drift_*` (Auto-Rekalibrierung).

## Genauigkeit und Grenzen

- Die Kalibrierung ist sehr genau (Restfehler ≈ 0,1 mm an den Stützpunkten); die Pfeilposition
  des Modells streut nur 1–3 px. Fehler entstehen fast nur an Drahtgrenzen und durch die
  Kameraperspektive.
- **Kameraposition**: möglichst auf Board-Höhe und nahe der Wurfachse. Je steiler die Kamera von
  unten oder seitlich schaut, desto stärker verdecken Barrel und Flight die Spitze und desto
  öfter überlappen sich benachbarte Pfeile. Sich überlappende Pfeile sind die häufigste Ursache
  für einen fehlenden Pfeil → mit „+ Pfeil“ nachtragen.
- **Licht**: gleichmäßiges Licht von vorn erhöht die Erkennungssicherheit deutlich.
- Ein Modell, das mit eigenen Bildern aus der eigenen Kameraposition nachtrainiert wird, ist der
  nächste Schritt zu höherer Trefferquote (Pipeline: `train.py`, `evaluate.py`, `tools/`).

## Training und Auswertung

`train.py` trainiert YOLO11n auf dem DeepDarts-Datensatz (`dataset/dataset.yaml`, 5 Klassen);
`evaluate.py` wertet auf den Validierungsbildern aus; `docker_train.sh`, `train_spark.sh` und
`transfer_to_spark.bat` sind Hilfsskripte für Training im Container bzw. auf einem Remote-Rechner.
`tools/compare_models.py` vergleicht zwei Modelle auf Testbildern (Kalibrierpunkte, Pfeilabstände).

## Tests

```bash
python -m pytest tests/ -q
```

## Quellen und Lizenzen

- McNally et al., *DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts
  using a Single Camera*, CVPRW 2021 – Ansatz und Datensatz ([arXiv](https://arxiv.org/abs/2105.09880),
  [Code](https://github.com/wmcnally/deep-darts))
- Ben Willshaw, [dart-sense](https://github.com/bnww/dart-sense) – Standardmodell, CC BY-NC 4.0
  (nur nicht-kommerzielle Nutzung, nicht weiterverteilen)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [OpenCV](https://opencv.org),
  [FastAPI](https://fastapi.tiangolo.com)
