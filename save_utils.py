"""
save_utils.py -- dataset storage.

Persists one finalized shoe as an image crop plus a metadata JSON sidecar.
Layout:
    sneaker_impact/pictures/incoming<MMDDYYYY>/
        shoe_Reuse_1.jpg     shoe_Reuse_1.json
        shoe_Recycle_2.jpg   shoe_Recycle_2.json

Numbering is per-class (Reuse and Recycle each have their own counter) and is
restart-safe: it scans existing files in today's folder and picks the next
number, so re-running the app in the same day continues the sequence instead
of overwriting.

Color fields are kept in the schema as `null` until Phase 5 (color detection)
lands. Saving must never crash the live app -- all failures are caught,
logged, and the function returns None.
"""
import json
import os
import re
from datetime import datetime

import cv2

import config


def folder_for_today(root=None):
    """Return (and create if needed) today's incoming folder.

    Folder name is `incoming<MMDDYYYY>` so a new day starts a fresh folder
    automatically.
    """
    if root is None:
        root = config.OUTPUT_ROOT
    today = datetime.now().strftime("%m%d%Y")
    folder = os.path.join(root, f"incoming{today}")
    os.makedirs(folder, exist_ok=True)
    return folder


def next_number(folder, classification):
    """Return the next per-class shoe number for `folder`.

    Scans existing `shoe_<classification>_<N>.jpg` files (case-insensitive on
    the class word) and returns max(N) + 1, defaulting to 1.
    """
    pattern = re.compile(
        rf"shoe_{re.escape(classification)}_(\d+)\.jpg$",
        re.IGNORECASE,
    )
    max_n = 0
    try:
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    return max_n + 1


def _clip_bbox(bbox, frame_w, frame_h):
    """Clip (x1, y1, x2, y2) to image bounds; return None if degenerate."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_shoe(frame, bbox, classification, yolo_confidence,
              model_used=None, tracking_id=None, output_root=None,
              polygon=None):
    """Save one shoe (crop + JSON sidecar) and return the JPG path on success.

    On any failure (bad bbox, disk error, etc.) prints the error and returns
    None -- never raises, so the live loop keeps running.

    Args:
        frame:           full BGR frame (numpy array) the bbox refers to.
        bbox:            (x1, y1, x2, y2) in pixels.
        classification:  "Reuse" or "Recycle".
        yolo_confidence: float from the detector.
        model_used:      e.g. "yolov8m-oiv7.pt".
        tracking_id:     integer track ID (or None).
        output_root:     defaults to config.OUTPUT_ROOT.
        polygon:         optional GrabCut contour (in full-frame coords).
                         When color detection is enabled, restricts color
                         sampling to pixels inside the polygon so background
                         doesn't bias the answer.
    """
    try:
        if frame is None:
            print("[save] skipped: no frame provided.")
            return None

        h, w = frame.shape[:2]
        clipped = _clip_bbox(bbox, w, h)
        if clipped is None:
            print(f"[save] skipped: degenerate bbox {bbox} for {w}x{h} frame.")
            return None
        x1, y1, x2, y2 = clipped

        folder = folder_for_today(output_root)
        n = next_number(folder, classification)
        base = f"shoe_{classification}_{n}"
        jpg_path = os.path.join(folder, base + ".jpg")
        json_path = os.path.join(folder, base + ".json")

        crop = frame[y1:y2, x1:x2]
        if not cv2.imwrite(jpg_path, crop):
            print(f"[save] ERROR: cv2.imwrite returned False for {jpg_path}.")
            return None

        if getattr(config, "SAVE_FULL_FRAME", False):
            cv2.imwrite(os.path.join(folder, base + "_full.jpg"), frame)

        # Optional color detection. Translate the polygon (if provided) from
        # full-frame coords into crop coords so the mask lines up with the
        # crop. classify_color is wrapped in try/except so any failure here
        # is logged but doesn't break the save.
        detected_color = None
        color_confidence = None
        if getattr(config, "ENABLE_COLOR_DETECTION", False):
            try:
                from color_utils import classify_color
                crop_polygon = None
                if polygon is not None and len(polygon) >= 3:
                    crop_polygon = polygon.copy()
                    crop_polygon[:, :, 0] -= x1
                    crop_polygon[:, :, 1] -= y1
                detected_color, color_confidence = classify_color(
                    crop, mask=crop_polygon)
            except Exception as exc:                  # noqa: BLE001 - fail safe
                print(f"[save] color detection failed: {exc}")
                detected_color = "unknown"
                color_confidence = 0.0

        metadata = {
            "filename": base + ".jpg",
            "classification": classification,
            "shoe_number": n,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "detected_color": detected_color,
            "color_confidence": color_confidence,
            "yolo_confidence": float(yolo_confidence),
            "bbox": [x1, y1, x2, y2],
            "tracking_id": tracking_id,
            "frame_width": w,
            "frame_height": h,
            "model_used": model_used or getattr(config, "MODEL_PATH", None),
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[save] {classification} #{n} -> {jpg_path}")
        return jpg_path

    except Exception as exc:                  # noqa: BLE001 - never crash live loop
        print(f"[save] ERROR: unexpected failure: {exc}")
        return None
