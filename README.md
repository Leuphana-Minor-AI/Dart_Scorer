# Dart Scorer – automatic steel-tip dart scoring with a single camera

A dartboard, a camera (GoPro or webcam) and a laptop: the program finds the dartboard in the
camera image, calibrates itself, detects the darts and scores an X01 game (301/501/701) for up to
four players – with a scoreboard in the browser, on the laptop or on a phone.

## Features

- **Automatic calibration** from the wire intersections on the double ring (6 reference points,
  least-squares homography); the board can be anywhere in the frame; automatic recalibration when
  the camera or the board moves
- **Dart detection** with a YOLO model, tracking across frames, suppression of static false
  positives, duplicate handling, marking of uncertain darts (`?`) close to a wire
- **X01 game**: 301/501/701, 1–4 players, double-out or single-out, bust rules, legs (best of n),
  alternating throw-off, checkout suggestions, statistics (average, first 9, 180s, highest finish,
  best leg)
- **Web interface**: large scoreboard, current turn with three darts, correction by tapping the
  board, undo, miss/bounce-out, adding a missed dart, camera view with overlay, winner screen with
  rematch; responsive layout for phones and tablets on the same network
- **Diagnostics**: calibration/camera status, FPS, snapshots, bull-offset calibration, manual board
  selection, model comparison on test images

## Structure

```
dart_app.py            Entry point of the web app (engine thread + web server in one process)
engine/detector.py     Camera + detection → events (dart_added, darts_removed, calibration, camera)
live_scorer.py         Calibration, board crop, dart tracking, scoring (+ legacy OpenCV window mode)
camera_stream.py       Camera discovery (AVFoundation / DirectShow / V4L2), threaded capture, test-image mode
dart_geometry.py       Board geometry per WDF rules, homography, field from (x, y) in mm, wire distance
lens_undistort.py      Optional lens undistortion (calibrate_lens.py creates camera_calib.npz)
game/x01.py            X01 game logic (pure rules, no camera)
server/app.py          FastAPI: state via WebSocket, camera image as MJPEG, actions via REST
web/                   Frontend (HTML/CSS/JS, no build tools)
tests/test_x01.py      Game-logic tests (pytest)
tools/compare_models.py  Compare two models on test images
train.py, evaluate.py  Training / evaluation of the custom YOLO model (DeepDarts dataset)
models/                Model weights (see below)
```

## Installation

Python ≥ 3.10. Recommended: [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt opencv-python
```

Windows (Anaconda): `setup_windows.bat` installs PyTorch (CPU), Ultralytics and OpenCV.

### Models

Two models are supported, both with the same class order
(`cal_top` 5|20, `cal_bottom` 3|17, `cal_left` 11|8, `cal_right` 6|13, `dart_tip`):

| File | Origin | Use |
|---|---|---|
| `models/dart_sense_yolov8n.pt` | [bnww/dart-sense](https://github.com/bnww/dart-sense) (Ben Willshaw), YOLOv8n, ~24,000 images from many camera angles, two extra calibration points (classes `9`, `15`); license **CC BY-NC 4.0** | **Default.** Finds the dart entry points even with an oblique camera |
| `models/dart_yolo11n_best.pt` | own training (YOLO11n) on the DeepDarts dataset | Fallback if the dart-sense file is missing; only suitable for a frontal camera |

The dart-sense weights are not part of this repository (the license does not allow
redistribution). Download once:

```bash
curl -L -o models/dart_sense_yolov8n.pt https://github.com/bnww/dart-sense/raw/main/weights.pt
```

### GoPro as a webcam

Install the GoPro Webcam app and connect the camera via USB or Wi-Fi. Recommended settings:
field of view "Linear" (less distortion), auto power-off disabled. Any webcam works as well.
The program probes camera indices 0–3 and picks the one with a live image; `--camera N` selects
an index explicitly.

**macOS:** camera access is only granted to processes started from Terminal.app – start the
`.command` scripts by double-clicking them or with `open -a Terminal <script>`.

## Running

```bash
open -a Terminal _start_webapp.command      # macOS: starts the server and opens the browser
python dart_app.py [--camera 1] [--port 20744]
```

The terminal prints the addresses: `http://localhost:20744` on the laptop, plus every network
address of the machine for phones/tablets on the same network (also shown on the setup page).
Managed Wi-Fi networks (university, company) often block device-to-device traffic – in that case
connect the phone to the laptop via USB tethering or a hotspot.

Game flow: choose players and mode → throw → the darts appear in the current turn → if needed tap a
dart and click the correct field on the board → pull the darts → next player. "Next" is only
needed to switch early.

**Keyboard:** `U` undo · `Space`/`Enter` next · `M` miss · `+`/`A` add a dart ·
`1`–`3` correct a dart · `Esc` close

**Menu ⚙︎:** Recalibrate · Empty board (relearn static false positives after pulling all darts) ·
Bull offset (one dart in the bull → measure systematic offset) · Lock · Snapshot · Camera view ·
End game

### Legacy window mode (debugging)

`_start_scorer.command` or `run_dart_scorer.bat` start `live_scorer.py` with an OpenCV window,
radar and keyboard control (`C` recalibrate, `L` lock, `R` mark board, `B` bull offset,
`S` snapshot, `TAB` + arrow keys correction, `0`–`4` camera, `Q` quit).
Without a camera it uses images from `test_images/`.

## How the detection works

1. **Finding the board** – full frame through the model, plus every 10 frames a tiled pass at
   native resolution. From two calibration points on, a square crop is placed around the board and
   only that crop is processed, scaled to ~500 px board size – independent of the camera distance.
2. **Calibration** – homography from the wire intersections (mm references per WDF in
   `dart_geometry.py`). Point memory (3 s), rotating scale/contrast variants, plausibility check
   (diameter midpoints, order), prediction of missing points. Locked after 15 stable frames;
   drift detection recalibrates when something moves.
3. **Darts** – model on the crop in two alternating sizes; radius filter (> 180 mm discarded);
   static false positives of the empty board suppressed; duplicates (tip + barrel) merged along
   the shaft direction; a track counts after 3 hits, position smoothed.
4. **Scoring** – pixel → mm → ring/sector. Closer than 3 mm to a wire (or just outside the double
   ring) → `?`. An optional constant offset (bull calibration) is subtracted.
5. **Game** – the engine only reports events; `game/x01.py` decides bust, finish and legs.
   A dart that has been counted is not counted again until all darts are pulled.

Main tuning knobs in `live_scorer.py`: `dart_conf` (dart threshold, default 0.15),
`conf_thresh` (calibration points, 0.25), `track_min_hits`, `track_merge_px`/`shaft_dir`
(duplicates), `artifact_*` (static false positives), `UNCERTAIN_WIRE_MM`, `drift_*`
(auto recalibration).

## Accuracy and limitations

- Calibration is very precise (residual ≈ 0.1 mm at the reference points); the model's dart
  position varies by only 1–3 px. Errors occur almost exclusively at wire boundaries and because of
  the camera perspective.
- **Camera position**: as close to board height and to the throwing axis as possible. The steeper
  the camera looks from below or from the side, the more the barrel and flight hide the tip and the
  more often neighbouring darts overlap. Overlapping darts are the most common cause of a missing
  dart → add it with "+ Dart".
- **Light**: even light from the front improves detection confidence considerably.
- A model fine-tuned with your own images from your own camera position is the next step towards
  a higher hit rate (pipeline: `train.py`, `evaluate.py`, `tools/`).

## Training and evaluation

`train.py` trains YOLO11n on the DeepDarts dataset (`dataset/dataset.yaml`, 5 classes);
`evaluate.py` evaluates on the validation images; `docker_train.sh`, `train_spark.sh` and
`transfer_to_spark.bat` are helpers for training in a container or on a remote machine.
`tools/compare_models.py` compares two models on test images (calibration points, dart distances).

## Tests

```bash
python -m pytest tests/ -q
```

## Sources and licenses

- McNally et al., *DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts
  using a Single Camera*, CVPRW 2021 – approach and dataset ([arXiv](https://arxiv.org/abs/2105.09880),
  [code](https://github.com/wmcnally/deep-darts))
- Ben Willshaw, [dart-sense](https://github.com/bnww/dart-sense) – default model, CC BY-NC 4.0
  (non-commercial use only, no redistribution)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [OpenCV](https://opencv.org),
  [FastAPI](https://fastapi.tiangolo.com)
