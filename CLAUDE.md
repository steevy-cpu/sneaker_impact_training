# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Sneaker Impact** is becoming a live shoe **data-collection and human-labeling
platform** for future AI training. It started as a YOLO/OpenCV shoe-detection
experiment (`capture.py`, `detect_test.py`) and is being restructured into a
modular system.

The near-term priority is **clean, organized dataset collection** — not the
final automated sorting AI. Human labeling accuracy matters more than automation.

## Intended human workflow (the core idea)

The operator watches a live camera feed; YOLO draws bounding boxes on shoes.

- A detected shoe defaults to **Reuse** (green box).
- If the operator **clicks inside a shoe's box**, it becomes **Recycle** (turns red).
- When the shoe leaves the frame (tracking expires), its crop + metadata JSON
  are saved automatically.

The operator only ever interacts with BAD (Recycle) shoes, which keeps labeling fast.

## Target architecture

```
LIVE CAMERA FEED -> YOLO DETECTION -> BBOX DISPLAY -> (optional COLOR) ->
DEFAULT LABEL = REUSE -> DOUBLE-CLICK IF BAD -> AUTO-SAVE IMAGE + METADATA
```

Module responsibilities:

| File | Role | Phase |
|------|------|-------|
| `config.py` | All tunables (camera, thresholds, paths, flags). Nothing else hardcodes these. | done |
| `camera_utils.py` | Cross-platform camera open (AVFoundation/DSHOW/default); routes to GigE when enabled. | done |
| `gige_camera.py` | Aravis backend for the GigE Vision camera (Photon Focus); cv2.VideoCapture look-alike. | done |
| `list_cameras.py` | Probe which camera indices are usable. | done |
| `ui_utils.py` | Green/red bounding boxes, FPS, status overlay. | done |
| `label_live.py` | Main app: camera + display + mouse + tracker + saves. | done |
| `detector_utils.py` | Async YOLO + GrabCut worker thread (DetectorThread). | done |
| `tracking_utils.py` | Lightweight IoU shoe tracking + expiry. | done |
| `save_utils.py` | Save crops + metadata JSON into dated folders. | done |
| `color_utils.py` | Broad dominant-color estimate; must fail safe. | done |
| `dataset_clean.py` | Batch quality cleaner: blur, confidence, dedup filters. | done |
| `dataset_review.py` | Interactive visual reviewer: keep / delete / relabel per shoe. | done |
| `capture.py` | Original key-driven dataset capture tool (preserved). | existing |
| `detect_test.py` | Original detector diagnostic (preserved). | existing |

## Existing scripts (preserved, still work as-is)

- **`capture.py`** — webcam loop. Keys `1/2` save a YOLO bbox crop of a shoe
  *top*; keys `3/4` bypass detection and use OpenCV **GrabCut** on a centered
  guide box to segment the *sole* (the model can't recognize soles). Saves into
  `dataset/{A,B,A2,B2}/`.
- **`detect_test.py`** — draws every detection live (shoes green, others
  orange), or `--shot` dumps `debug_frame.jpg`. Used to confirm what the
  detector can/can't see.
- Shoe classes: `SHOE_CLASS_IDS = {203, 56, 432, 249}` (Footwear/Boot/Sandal/
  High heels in Open Images V7), duplicated in both scripts — keep in sync.

> **Platform note (resolved in Phase 1):** camera access is now centralized in
> `camera_utils.open_camera()`, which selects the right backend per OS
> (AVFoundation on macOS, DirectShow on Windows, default on Linux). All scripts
> route through it — do not call `cv2.VideoCapture` with a hardcoded backend.

## Output layout & metadata

```
sneaker_impact/pictures/incoming<MMDDYYYY>/
    shoe_Reuse_black_1.jpg     shoe_Reuse_black_1.json
    shoe_Recycle_white_1.jpg   shoe_Recycle_white_1.json
```

Filename convention: `shoe_<classification>_<color>_<N>.jpg`. Color is
detected before the file is written so it becomes part of the name. Falls
back to `unknown` if color detection is disabled or fails.

Per-class counter (Reuse and Recycle have separate sequences), restart-safe
(scans existing files to pick the next number), fresh folder per day.

Metadata JSON fields: `filename, classification, shoe_number, timestamp,
detected_color, color_confidence, yolo_confidence, bbox, tracking_id,
frame_width, frame_height, model_used`.

## Engineering rules

- Never hardcode camera index, paths, or thresholds — read them from `config.py`.
- Color detection must never crash the app (return `unknown` on failure).
- If YOLO fails, the app should keep running.
- Keep files under ~300 lines; prefer small helper modules over giant scripts.
- Beginner-readable code with comments. Simple > clever; avoid over-engineering.

## Testing the camera (Phase 1)

```bash
python list_cameras.py     # probe indices 0..5, prints which work + a suggestion
python detect_test.py      # live detector view (uses config.CAMERA_INDEX)
python capture.py          # key-driven dataset capture (uses config.CAMERA_INDEX)
```

`capture.py`/`detect_test.py` accept `--camera N`; otherwise camera selection
follows `config` (see below).

**Choosing the external USB camera:** pick it by **numeric index** —
`config.CAMERA_INDEX`. Name-based selection is intentionally disabled because
on macOS the order returned by `system_profiler` doesn't reliably match
OpenCV's AVFoundation index order, so a name lookup can silently grab the
built-in webcam. Workflow:
1. Plug in the USB camera, run `python list_cameras.py` — it probes indices
   0..5 and saves a preview JPG from each working one to `/tmp/cam_N.jpg`.
2. Open the previews, identify which file shows your external camera's view,
   and set `CAMERA_INDEX` in `config.py` to that index.
3. Leave `CAMERA_NAME = ""`. On open, the app prints e.g.
   `[camera] OK: using camera index 0.` (no device name, since names from
   `system_profiler` aren't trustworthy here).

**Troubleshooting:**
- *"could not open camera index N"* — wrong index; run `list_cameras.py` and use
  a reported one. Also confirm the cable/adapter and that the camera is seated.
- *"opened but returned no frame"* — another app (Zoom, Photo Booth, etc.) is
  holding the camera, or it needs a moment to warm up. Close other apps, retry.
- *macOS first run* — grant camera permission when prompted (System Settings →
  Privacy & Security → Camera → enable for your terminal/IDE).

## Live detection + labeling (Phases 2-4)

```bash
python label_live.py       # opens "Sneaker Impact - Live Detection"
```

- Uses `config.MODEL_PATH`, `config.CAMERA_INDEX`, `config.CONFIDENCE_THRESHOLD`,
  `config.MAX_DETECTIONS`, `config.TRACK_EXPIRATION_FRAMES`,
  `config.TRACK_IOU_THRESHOLD`, `config.OUTPUT_ROOT`, and `config.DISPLAY_FPS`.
- Keeps only detections whose class name is `shoe`/`shoes`/`footwear`
  (case-insensitive, see `SHOE_CLASS_NAMES` in `label_live.py`).
- Each detected shoe gets a **green** bounding box (Reuse default) and a
  caption like `Shoe 0.87`.
- **Click** inside a shoe's box → flips to **Recycle**, box turns **red**
  permanently, crop + metadata saved immediately under
  `OUTPUT_ROOT/incoming<MMDDYYYY>/shoe_Recycle_<color>_N.jpg`.
- When a shoe leaves the frame for `TRACK_EXPIRATION_FRAMES` frames and was
  never clicked, its last good crop is auto-saved as `shoe_Reuse_<color>_N.jpg`.
- **Keyboard:** `Q` or `ESC` to quit.
- **Model requirement:** `MODEL_PATH` must point to a model that has a shoe
  class. Plain COCO (`yolov8n.pt`) has none and will detect nothing. Use
  `yolov8m-oiv7.pt` (Open Images V7, class "Footwear"). A startup warning is
  printed if the loaded model has no shoe-like class.

## Phase roadmap

1. **Foundation + camera support** — `config.py`, `camera_utils.py`,
   `list_cameras.py`; cross-platform camera access. **Done.**
2. **Live detection UI** — `label_live.py` + `ui_utils.py`, bounding boxes + confidence + FPS. **Done.**
3. **Tracking + labeling** — stable IDs, default Reuse, click → Recycle, finalize on exit. **Done.**
4. **Dataset storage** — `save_utils.py`, crops + JSON, dated folders, safe numbering. **Done.**
5. **Color detection** — broad categories only, lightweight, fail-safe. **Done.**
6. **Dataset quality tools** — blur filter, confidence filter, dedup, visual review. **Done.**
7. **Future training pipeline** — out of scope for this repository.

**Current state:** Phases 1–6 complete. The system collects, labels, and
cleans a shoe dataset ready for external training.

Color detection details: `classify_color(image, mask=None)` in
`color_utils.py` returns `(name, confidence)` for one of 11 broad
categories: black, white, gray, brown, red, orange, yellow, green, blue,
purple, pink (or "unknown" on failure). Gated by
`config.ENABLE_COLOR_DETECTION`. `save_shoe` calls it on every save,
using the GrabCut polygon when available to ignore background pixels.
Thresholds (`_V_BLACK`, `_V_WHITE`, `_S_GRAY`, `_V_BROWN`) are tunable
constants at the top of `color_utils.py`.

## Dataset quality tools (Phase 6)

```bash
python dataset_clean.py --dry-run        # preview what would be removed
python dataset_clean.py                  # remove blurry, low-conf, duplicates
python dataset_clean.py --blur 80        # stricter blur threshold
python dataset_clean.py --conf 0.5       # stricter confidence threshold
python dataset_clean.py --folder incoming05292026   # single folder only

QT_QPA_PLATFORM=xcb python dataset_review.py        # visual review
```

`dataset_review.py` controls: `SPACE` = keep, `D` = delete, `R` = flip
Reuse↔Recycle, `←` = go back, `Q`/`ESC` = quit.

## GigE Vision camera (Photon Focus, via Aravis)

The project is moving off the USB webcam to a **Photon Focus GigE Vision**
camera (`DR1-D2048x1088C`, BayerGB8). `cv2.VideoCapture` cannot open a GigE
camera, so `gige_camera.py` wraps the open-source **Aravis** library in an
`AravisCapture` object that exposes the same `.read()` / `.release()` interface.
`open_camera()` routes to it when `config.USE_GIGE_CAMERA = True`, so
`label_live.py` and friends run unchanged.

Enable it:

```python
# config.py
USE_GIGE_CAMERA = True
GIGE_CAMERA_NAME = ""      # "" = first camera Aravis finds, or pin the device id
GIGE_PACKET_SIZE = 1440    # <= 1500 on USB-Ethernet adapters; raise on jumbo NIC
```

System deps (NOT in requirements.txt -- they're OS packages, not pip):

```bash
# Raspberry Pi 5 / Linux (preferred target):
sudo apt install aravis-tools gir1.2-aravis-0.8 python3-gi
# macOS dev box (harder): brew install aravis + PyGObject so `import gi` works
```

Before running the app, prove the camera streams at the driver level:
`arv-tool-0.8` lists it, `arv-viewer-0.8` shows live frames. The camera needs
**12V external power** (not PoE) and must share a subnet with the host. Notes:
- Bayer→BGR uses OpenCV's debayer; OpenCV's Bayer naming doesn't perfectly
  match GenICam's, so if **red/blue look swapped**, change the camera's entry in
  `_BAYER_TO_BGR` (top of `gige_camera.py`).
- Fails safe: if Aravis/`gi` is missing or no camera is found, `open_camera()`
  returns `None` with an install hint and the app exits cleanly.

## Detection model: YOLO-World vs OIV7

Two model families are wired up:

- **YOLO-World** (default, `USE_YOLO_WORLD=True`) — open-vocabulary
  detector. You give it a text prompt list (`YOLO_WORLD_CLASSES`); the
  model only detects those categories. Better at uncommon shoe types
  (five-toe shoes, etc.) than the single "Footwear" bucket in OIV7.
  Default weights `yolov8s-worldv2.pt` (~28MB) are sized to run on
  Raspberry Pi 5 and Jetson Nano.
- **OIV7** (`USE_YOLO_WORLD=False`) — standard Open Images V7 YOLOv8.
  Single "Footwear" class. Use this if YOLO-World feels too slow or
  noisy on a given setup.

Tuning class prompts: edit `config.YOLO_WORLD_CLASSES`. More prompts =
broader coverage but also more false positives.

## Deployment to Pi 5 / Jetson Nano

Code is already cross-platform-aware: `detector_utils.pick_device()`
picks CUDA on Jetson, MPS on Apple Silicon, CPU on Raspberry Pi 5.

For live-FPS deployment, raw PyTorch is usually too slow on these
targets. Export the model to an accelerated runtime once classes are
locked in:

```python
# Jetson Nano -- TensorRT engine
model.export(format="engine", imgsz=416, device=0)   # produces yolov8s-worldv2.engine

# Raspberry Pi 5 -- ONNX Runtime or NCNN
model.export(format="onnx", imgsz=416, simplify=True)
# or:
model.export(format="ncnn", imgsz=416)
```

Notes:
- **YOLO-World "bakes in" the prompt classes at export time** -- call
  `model.set_classes(...)` BEFORE `model.export(...)`.
- On Pi 5, try `imgsz=320` and INT8 quantization for higher FPS.
- Jetson with TensorRT: ~15-20 FPS at `imgsz=416`.
- Pi 5 with ONNX/NCNN: ~3-5 FPS at `imgsz=416`, ~5-10 FPS at `imgsz=320`.
- Both targets benefit from the async detector thread already in place
  (display stays smooth even when inference is slow).

Extra packages needed at deploy time:
- Jetson: TensorRT is bundled with JetPack; nothing extra.
- Pi 5: `pip install onnxruntime` (CPU build) or build NCNN per docs.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No tests, linter, or build step.
