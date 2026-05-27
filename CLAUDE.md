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

- A detected shoe defaults to **Reuse**.
- If the operator **double-clicks inside a shoe's box**, it becomes **Recycle**.
- When the shoe leaves the frame (tracking expires), its crop + metadata JSON
  are saved automatically.

The operator only ever interacts with BAD (Recycle) shoes, which keeps labeling fast.

## Target architecture

```
LIVE CAMERA FEED -> YOLO DETECTION -> BBOX DISPLAY -> (optional COLOR) ->
DEFAULT LABEL = REUSE -> DOUBLE-CLICK IF BAD -> AUTO-SAVE IMAGE + METADATA
```

Module responsibilities (most are placeholders until their phase lands):

| File | Role | Phase |
|------|------|-------|
| `config.py` | All tunables (camera, thresholds, paths, flags). Nothing else hardcodes these. | done |
| `camera_utils.py` | Cross-platform camera open (AVFoundation/DSHOW/default). | done |
| `list_cameras.py` | Probe which camera indices are usable. | done |
| `ui_utils.py` | Translucent green/red masks, FPS, status. | done |
| `label_live.py` | Main app: camera + display + mouse + tracker + saves. | done |
| `detector_utils.py` | Async YOLO + GrabCut worker thread (DetectorThread). | done |
| `tracking_utils.py` | Lightweight IoU shoe tracking + expiry. | done |
| `save_utils.py` | Save crops + metadata JSON into dated folders. | done |
| `color_utils.py` | Broad dominant-color estimate; must fail safe. | 5 |
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
    shoe_Reuse_1.jpg     shoe_Reuse_1.json
    shoe_Recycle_1.jpg   shoe_Recycle_1.json
```

Per-class counter (Reuse and Recycle have separate sequences), restart-safe
(scans existing files to pick the next number), fresh folder per day.

Metadata JSON fields: `filename, classification, shoe_number, timestamp,
detected_color (null until Phase 5), color_confidence (null until Phase 5),
yolo_confidence, bbox, tracking_id, frame_width, frame_height, model_used`.

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
- Each detected shoe gets a translucent **green** mask (Reuse default) and a
  caption like `Shoe 0.87`.
- **Double-click** inside a shoe's mask → flips to **Recycle**, mask flashes
  **red** for ~0.5s, crop + metadata saved immediately under
  `OUTPUT_ROOT/incoming<MMDDYYYY>/shoe_Recycle_N.jpg`.
- When a shoe leaves the frame for `TRACK_EXPIRATION_FRAMES` frames and was
  never clicked, its last good crop is auto-saved as `shoe_Reuse_N.jpg`.
- **Keyboard:** `Q` or `ESC` to quit.
- **Model requirement:** `MODEL_PATH` must point to a model that has a shoe
  class. Plain COCO (`yolov8n.pt`) has none and will detect nothing. Use
  `yolov8m-oiv7.pt` (Open Images V7, class "Footwear"). A startup warning is
  printed if the loaded model has no shoe-like class.

## Phase roadmap

1. **Foundation + camera support** — `config.py`, `camera_utils.py`,
   `list_cameras.py`; cross-platform camera access. **Done.**
2. **Live detection UI** — `label_live.py` + `ui_utils.py`, masks + confidence + FPS. **Done.**
3. **Tracking + labeling** — stable IDs, default Reuse, double-click → Recycle, finalize on exit. **Done.**
4. **Dataset storage** — `save_utils.py`, crops + JSON, dated folders, safe numbering. **Done.**
5. **Color detection** — broad categories only, lightweight, fail-safe.
6. **Dataset quality tools** — dedup, blur detection, confidence filtering, review mode.
7. **Future training pipeline** — YOLO fine-tuning / classification (not started).

**Current state:** Phases 1–4 complete. Live detection, IoU tracking,
double-click Recycle labeling, frame-exit Reuse auto-save, and dataset
storage (crops + metadata JSON in dated folders) all work. `color_utils.py`
remains a docstring-only placeholder (Phase 5).

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
