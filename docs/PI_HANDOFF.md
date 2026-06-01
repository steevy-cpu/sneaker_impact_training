# Pi handoff — live debugging context (transient)

> This is a short-lived handoff note for the Claude Code session running on the
> Raspberry Pi 5. It captures context that lives in the Mac-side conversation
> (which does NOT transfer across machines — only git does). Delete it once the
> issue below is resolved.

## First: pull the latest

The Mac session just pushed speed + quality work to `main`. On the Pi:

```bash
git pull            # get commits up to 4e1ad20 (and this doc)
```

Latest commits (newest first):
- `4e1ad20` Quality: config-driven color thresholds, center-crop color, "multi" label, bbox + blur gates
- `2dffa57` Speed: lower capture res, nano model, fewer copies, less redundant work
- `44474c8` / `2b8bb3d` Remove GigE backend; back to Logitech USB webcam

## The bug being chased

Running `python label_live.py` on the Pi: a shoe is detected and boxed for a
few seconds, then the box disappears and the status shows "no shoes" **while the
shoe is still in frame**. The video itself is fine. This means the detector
thread is returning an EMPTY detection list (not a frozen/torn display).

## Prime suspects (all introduced in 2dffa57 / 4e1ad20), ranked

1. **Model downgrade to nano.** `config.MODEL_PATH` was changed from
   `yolov8m-oiv7.onnx` (medium, accurate) to `yolov8n-oiv7.pt` (nano, ~3x faster
   but less accurate). Nano may simply lose the shoe across angle/lighting
   changes. This project values labeling accuracy over speed, so medium may be
   the correct default.
2. **`config.MIN_BBOX_AREA_FRAC = 0.004`** (new) silently drops detections
   smaller than ~0.4% of the frame (~60x60 px at 720p). A not-large shoe gets
   filtered out. Filter lives in `detector_utils._collect_shoes`.
3. **No-copy frame sharing** (in `label_live.py`): the detector thread now
   receives the raw `frame` (not `frame.copy()`), relying on `cap.read()`
   allocating a fresh buffer each call. If the Pi's OpenCV/V4L2 build reuses the
   buffer, the detector can read a torn frame -> no detection.

## Fastest bisect — config-only, no code edits

In `config.py`, change these two and rerun `python label_live.py`:

```python
MODEL_PATH = "yolov8m-oiv7.onnx"   # the model that was working before
MIN_BBOX_AREA_FRAC = 0             # disable the small-shoe filter
```

- Detection now stable -> it was the model and/or the bbox filter. Re-enable one
  at a time (put `MIN_BBOX_AREA_FRAC = 0.004` back first) to find which.
- Still dropping -> test suspect #3: in `label_live.py` main loop, temporarily
  restore `detector.post_frame(frame.copy())` (and `TRACKER.update(shoes,
  frame.copy())`). If that fixes it, the Pi's OpenCV reuses the capture buffer
  and the no-copy optimization must be reverted (or guarded) on the Pi.

## Useful context on how detection/draw works

- Main loop runs at camera FPS; the detector thread runs YOLO asynchronously and
  is slow on the Pi (a few FPS). `get_detections()` returns the latest result,
  repeated between detector cycles.
- A box is only drawn for tracks matched within the last 5 main-loop frames
  (`draw_cutoff = TRACKER.frame_idx - 5` in `label_live.py`). So a single empty
  detector cycle makes the box vanish quickly. That's expected; the question is
  why the detector returns empty while the shoe is present (suspects above).
- `config.CONFIDENCE_THRESHOLD = 0.5` and `config.YOLO_IMGSZ = 320` were NOT
  changed in this work (they pre-date it), but they're also worth a glance:
  lowering imgsz back to 416 or conf to 0.4 can help nano hold a detection.

## What "good" looks like

A stationary shoe stays green-boxed continuously; clicking it turns it red and
saves immediately; when it leaves the frame it auto-saves as Reuse. Saves land
in `sneaker_impact/pictures/incoming<MMDDYYYY>/`.
