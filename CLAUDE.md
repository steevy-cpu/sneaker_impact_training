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
| `ui_utils.py` | Draw boxes/labels, handle double-click, flash confirmation. | 2-3 |
| `label_live.py` | Main app entry point tying it all together. | 2-5 |
| `tracking_utils.py` | Lightweight centroid/IoU shoe tracking + expiry. | 3 |
| `save_utils.py` | Save crops + metadata JSON into dated folders. | 4 |
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

## Output layout & metadata (target, Phase 4)

```
sneaker_impact/pictures/incoming_YYYY-MM-DD/
    shoe_Reuse_1.jpg     shoe_Reuse_1.json
    shoe_Recycle_2.jpg   shoe_Recycle_2.json
```

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

All three default to `config.CAMERA_INDEX`; override per-run with `--camera N`.

**USB-C camera:** plug it in, run `python list_cameras.py`. Index 0 is usually
the built-in webcam, so the higher reported index is typically the external
USB-C camera — set that as `CAMERA_INDEX` in `config.py`.

**Troubleshooting:**
- *"could not open camera index N"* — wrong index; run `list_cameras.py` and use
  a reported one. Also confirm the cable/adapter and that the camera is seated.
- *"opened but returned no frame"* — another app (Zoom, Photo Booth, etc.) is
  holding the camera, or it needs a moment to warm up. Close other apps, retry.
- *macOS first run* — grant camera permission when prompted (System Settings →
  Privacy & Security → Camera → enable for your terminal/IDE).

## Phase roadmap

1. **Foundation + camera support** — `config.py`, `camera_utils.py`,
   `list_cameras.py`; cross-platform camera access. **Done.**
2. **Live detection UI** — `label_live.py` + `ui_utils.py`, boxes + confidence, no saving.
3. **Tracking + labeling** — stable IDs, default Reuse, double-click → Recycle, finalize on exit.
4. **Dataset storage** — `save_utils.py`, crops + JSON, dated folders, safe numbering.
5. **Color detection** — broad categories only, lightweight, fail-safe.
6. **Dataset quality tools** — dedup, blur detection, confidence filtering, review mode.
7. **Future training pipeline** — YOLO fine-tuning / classification (not started).

**Current state:** Phase 1 complete — cross-platform camera support is live
(`camera_utils.py`, `list_cameras.py`) and `capture.py`/`detect_test.py` route
through it using `config.CAMERA_INDEX`. The other new modules
(`ui_utils`, `tracking_utils`, `save_utils`, `color_utils`, `label_live`) remain
docstring-only placeholders. No detection-labeling, tracking, saving, or color
logic is implemented yet.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No tests, linter, or build step.
