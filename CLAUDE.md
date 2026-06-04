# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Sneaker Impact** is becoming a live shoe **data-collection and human-labeling
platform** for future AI training. It started as a YOLO/OpenCV shoe-detection
experiment (`capture.py`, `detect_test.py`) and is being restructured into a
modular system.

> **2026 pivot (in progress).** The CEO redirected the project: instead of the
> live click-to-Recycle flow, photograph the **whole table** of shoes, then in
> the **background** segment it into individual **pairs** (shoes are tied in
> pairs → one record per pair), and identify each pair's **make + model** (no
> Reuse/Recycle for now). Plan: segment with **YOLOE-26** (open-vocab, text
> prompt "pair of shoes", no training) → brand classifier → sneaker-DB/API
> lookup for the model. The old live-labeling stack (`label_live.py`, IoU
> tracking, Reuse/Recycle) is **preserved but off the main path**. License note:
> YOLO26/YOLOE-26 is **AGPL-3.0** — used while internal-only; switch to **SAM 2
> (Apache-2.0)** or buy the Enterprise license before shipping a product. The
> segmenter is built backend-swappable for exactly that reason. See the
> "Table-segmentation pipeline" section below. **Phase A works end-to-end** on a
> real photo (Mac, MPS, `yoloe-26s-seg.pt`): a single wide pass missed ~95% of a
> crowded table, so the pipeline **tiles** the photo (recall fix) then **pairs**
> the detected single shoes geometrically — a realistic 16-pair batch yields 16
> clean per-pair crops. Realistic batch size is ~16-20 pairs, not 70+.

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
DEFAULT LABEL = REUSE -> CLICK IF BAD -> AUTO-SAVE IMAGE + METADATA
```

Module responsibilities:

| File | Role | Phase |
|------|------|-------|
| `config.py` | All tunables (camera, thresholds, paths, flags). Nothing else hardcodes these. | done |
| `camera_utils.py` | Cross-platform camera open (AVFoundation/DSHOW/default). | done |
| `list_cameras.py` | Probe which camera indices are usable. | done |
| `ui_utils.py` | Green/red bounding boxes, FPS, status overlay. | done |
| `label_live.py` | Main app: camera + display + mouse + tracker + saves. | done |
| `detector_utils.py` | Async YOLO + GrabCut worker thread (DetectorThread). | done |
| `tracking_utils.py` | Lightweight IoU shoe tracking + expiry. | done |
| `save_utils.py` | Save crops + metadata JSON into dated folders. | done |
| `color_utils.py` | Broad dominant-color estimate; must fail safe. | done |
| `dataset_clean.py` | Batch quality cleaner: blur, confidence, dedup filters. | done |
| `dataset_review.py` | Interactive visual reviewer: keep / delete / relabel per shoe (filterable). | done |
| `dataset_stats.py` | Read-only dataset summary: counts, class balance, color spread. | done |
| `dataset_export.py` | Export the labeled set to one CSV/JSON manifest. | done |
| `dataset_utils.py` | Shared folder listing + entry loading for the dataset tools. | done |
| `image_utils.py` | Shared `sharpness()` (variance of Laplacian). | done |
| `log_utils.py` | Tee console output to a timestamped log file. | done |
| `dashboard_client.py` | Push collected shoes to the Sneaker Impact Dashboard (REST). | done |
| `dashboard_sync.py` | Back-fill the dashboard from collected folders (idempotent). | done |
| `dashboard_live.py` | Background live push from label_live to the dashboard. | done |
| `segment_utils.py` | Backend-swappable table segmenter (YOLOE-26 / SAM 2) + tiling for dense recall → per-shoe `Segment`s. | Phase A done |
| `pair_utils.py` | Group detected single shoes into tied pairs (one record per pair). | Phase A done |
| `split_table.py` | Whole-table photo → segment → pair → per-pair crops + metadata JSON (make/model placeholders). | Phase A done |
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
frame_width, frame_height, model_used, sharpness`.

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
  class. Plain COCO (`yolov8n.pt`) has none and will detect nothing. The default
  is `yolov8m-oiv7.onnx` (Open Images V7 medium, class "Footwear") — nano was
  tried for Pi 5 speed but dropped live detections too often, so medium is the
  reliable default. A startup warning is printed if the loaded model has no
  shoe-like class.
- **Capture resolution:** `config.CAPTURE_WIDTH`/`CAPTURE_HEIGHT` (default
  1280×720) cap what the camera streams — smaller = faster everywhere and
  smaller crops; set both to 0 for the camera's native resolution.

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
purple, pink (or `"multi"` when the top two colors are within
`config.COLOR_AMBIGUOUS_MARGIN`, or `"unknown"` on failure). Gated by
`config.ENABLE_COLOR_DETECTION`. `save_shoe` calls it on every save. With
GrabCut off there's no polygon, so it samples a centered fraction of the crop
(`config.COLOR_CENTER_FRAC`) to keep edge background out of the estimate.
HSV thresholds (`COLOR_V_BLACK`, `COLOR_V_WHITE`, `COLOR_S_GRAY`,
`COLOR_V_BROWN`) now live in `config.py`.

Other dataset-quality knobs: `config.MIN_BBOX_AREA_FRAC` drops tiny/distant
detections; `config.BLUR_SAVE_FLOOR` (off by default) can skip auto-saving
blurry Reuse crops — Recycle clicks are always saved.

## Dataset quality tools (Phase 6)

```bash
python dataset_clean.py --dry-run        # preview what would be removed
python dataset_clean.py                  # remove blurry, low-conf, duplicates
python dataset_clean.py --blur 80        # stricter blur threshold
python dataset_clean.py --conf 0.5       # stricter confidence threshold
python dataset_clean.py --folder incoming05292026   # single folder only

QT_QPA_PLATFORM=xcb python dataset_review.py        # visual review
python dataset_review.py --only Recycle             # review just one class
python dataset_review.py --color brown              # review one color
python dataset_review.py --max-sharp 60             # review only blurry shoes

python dataset_stats.py                  # counts, class balance, color spread
python dataset_export.py --out manifest.csv         # one-row-per-shoe manifest
python dataset_export.py --format json --out manifest.json
```

`dataset_review.py` controls: `SPACE` = keep, `D` = delete, `R` = flip
Reuse↔Recycle, `B` or `←` = go back, `Q`/`ESC` = quit.

`dataset_stats.py` (read-only) prints totals, Reuse/Recycle balance, color
distribution, and per-day counts, and flags class imbalance.
`dataset_export.py` packages the whole labeled set into one CSV/JSON manifest —
the handoff artifact for the (out-of-scope) training step. All dataset tools
share `dataset_utils.{find_folders,load_entries}`, which skips macOS `._`
sidecar files so they aren't miscounted as shoes.

## Dashboard integration (Phase 1: sync)

The **Sneaker Impact Dashboard** (separate repo: FastAPI + SQLite) visualizes the
collected data. Run it in `APP_MODE=actual` on the **same machine** as the
station (so crops can be copied into its `images/` folder).

`dashboard_sync.py` back-fills it from the `incoming*` folders:

```bash
python dashboard_sync.py --dry-run                 # preview, no changes
python dashboard_sync.py                           # copy crops + POST records
python dashboard_sync.py --folder incoming06012026 # one folder
```

Mapping (in `dashboard_client.py`): the operator's Reuse/Recycle label is
mirrored into both `ai_prediction` and `final_decision` (`review_status`
COMPLETED), the crop becomes `img_top`, `detected_color` → `shoe_color`, and
yolo/color confidence + sharpness go into `notes`. `ai_confidence` is left null
(there is no Reuse/Recycle *classifier* yet — the label is human; mirroring it
into `ai_prediction` is only so the dashboard's charts, which count that column,
populate). A local ledger (`dashboard_synced.json`, git-ignored) makes re-runs
push only new shoes; one dashboard batch is opened per source day-folder. Config
is in `config.py` (`DASHBOARD_URL`, `DASHBOARD_IMAGES_DIR`, `OPERATOR_ID`).

**Phase 2 (live push):** set `config.DASHBOARD_PUSH_LIVE = True` and `label_live`
pushes each shoe to the dashboard the moment it's saved, via a fail-safe
background thread (`dashboard_live.DashboardPusher`) — the capture loop never
blocks or crashes on the network. It shares the same ledger as `dashboard_sync`,
so the two never double-push; if the dashboard is down, the shoe just stays on
disk and `dashboard_sync.py` back-fills it later. Don't run `dashboard_sync.py`
while a live-push session is active (both write the ledger). An undone (U) shoe
deleted before it's pushed is skipped; if already pushed, its dashboard record
remains (the dashboard has no delete API).

## Table-segmentation pipeline (2026 pivot, Phase A)

The new flow: a whole-table photo → segment into pairs → (later) make + model.

```bash
python split_table.py path/to/table.jpg          # segment one photo into pairs
python split_table.py path/to/table.jpg --viz     # also dump an overlay JPG
python split_table.py --all                       # all photos in TABLE_INPUT_DIR
python split_table.py table.jpg --backend sam2     # override SEGMENT_BACKEND
```

- `segment_utils.build_segmenter(config)` returns a `Segmenter` whose
  `segment(image)` yields a list of `Segment(bbox, score, label, polygon)`.
  Backends: `"yoloe"` (YOLOE-26 open-vocab, text-prompted via
  `config.SEGMENT_PROMPTS`, AGPL-3.0) and `"sam2"` (Segment Anything 2,
  Apache-2.0, class-agnostic). Fail-safe: a load/inference error logs and
  returns `[]`. **Plain `yolo26-seg` is COCO-only (no shoe class)** — use the
  open-vocab `yoloe` backend (or a custom-trained seg model) to find shoes with
  zero training. `yoloe-26s-seg.pt` auto-downloads (30MB) and pulls a 242MB
  MobileCLIP text encoder on first use.
- **Tiling** (`SEGMENT_TILE`/`_OVERLAP`/`_IOU`): a single wide pass downsamples
  small distant shoes away (caught 6 of ~80). Tiling slices the photo into
  overlapping windows, each upscaled to `SEGMENT_IMGSZ`, detects per tile, and
  merges with IoU + containment NMS (`TiledSegmenter`). On a 16-pair table this
  took recall from 6 → 32 clean single-shoe boxes.
- **Pairing** (`pair_utils.pair_shoes`, gated by `SEGMENT_PAIR`): detection is
  done at the *single-shoe* level (cleanest — prompting "pair of shoes" gave
  messy overlapping boxes), then the two nearest shoes within
  `SEGMENT_PAIR_MAX_GAP`×size are merged into one pair record (union crop).
  32 shoes → 16 pairs. Leftover odd shoes stay as single records; the
  dashboard's human-confirm is the safety net for mis-pairs.
- `split_table.py` crops each pair (pad `SEGMENT_CROP_PAD`, optional polygon
  white-out via `SEGMENT_APPLY_MASK`), runs `color_utils` on the crop, and saves
  `pair_<N>.jpg` + `.json` into `TABLE_OUTPUT_ROOT/pairs<MMDDYYYY>/`. The JSON
  carries `make/model/...` as **null placeholders** for the later identify step,
  so the schema is stable from day one.
- Config block lives under "Table segmentation" in `config.py`
  (`SEGMENT_BACKEND/MODEL/PROMPTS/CONF/DEVICE/CROP_PAD/APPLY_MASK/MIN_AREA_FRAC`,
  `TABLE_INPUT_DIR`, `TABLE_OUTPUT_ROOT`).
- **exFAT gotcha (this T7 drive):** macOS scatters `._*` AppleDouble files in
  any folder on the exFAT T7, and matplotlib (pulled in by ultralytics) crashes
  parsing them (`utf-8 ... 0xb0`). If the venv lives on the T7, run
  `find venv -name '._*' -delete` after installs (or keep the venv on APFS).
  Same root cause as the earlier git-clone and dataset-count issues.
- **Next phases:** B = brand recognition (`brand_utils.py`), C = model lookup
  via a sneaker DB/API (`model_search.py`), then storage/dashboard remap and the
  Airtable intake link.

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
