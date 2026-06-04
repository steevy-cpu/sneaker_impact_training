"""
split_table.py -- turn a whole-table photo into per-pair shoe crops.

Phase A of the 2026 pivot. Loads a full-table photo, runs the (swappable)
segmenter from segment_utils, crops each detected pair, and saves crop + a
metadata JSON sidecar into a dated folder under config.TABLE_OUTPUT_ROOT --
ready for the next stages (brand recognition, then model lookup).

Make/model fields are written as null placeholders now and filled by the
later identify step, so the dataset schema is stable from the start.

Usage:
    python split_table.py path/to/table.jpg            # segment one photo
    python split_table.py path/to/table.jpg --viz      # also dump an overlay
    python split_table.py --all                        # all photos in TABLE_INPUT_DIR
    python split_table.py table.jpg --backend sam2     # override the backend

Requires `pip install ultralytics` and weights for config.SEGMENT_MODEL.
"""
import argparse
import json
import os
import re
from datetime import datetime

import cv2

import config
from segment_utils import build_segmenter


def _pairs_folder(root=None):
    """Return (creating if needed) today's pairs<MMDDYYYY> output folder."""
    root = root or config.TABLE_OUTPUT_ROOT
    today = datetime.now().strftime("%m%d%Y")
    folder = os.path.join(root, f"pairs{today}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _next_number(folder):
    """Next pair_<N> number in `folder` (scan-based, so it's restart-safe)."""
    pattern = re.compile(r"pair_(\d+)\.jpg$", re.IGNORECASE)
    max_n = 0
    try:
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    return max_n + 1


def _pad_bbox(bbox, w, h, frac):
    """Grow a bbox by `frac` of its size on each side, clipped to the image."""
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * frac)
    pad_y = int((y2 - y1) * frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return x1, y1, x2, y2


def _apply_polygon_mask(crop, polygon, x1, y1):
    """White-out everything outside the segment polygon (cleaner brand crops)."""
    import numpy as np
    mask = np.zeros(crop.shape[:2], dtype="uint8")
    pts = polygon.copy()
    pts[:, 0] -= x1
    pts[:, 1] -= y1
    cv2.fillPoly(mask, [pts.astype("int32")], 255)
    out = crop.copy()
    out[mask == 0] = 255
    return out


def _color_of(crop):
    """Best-effort dominant color for a crop (fail-safe -> ('unknown', None))."""
    if not getattr(config, "ENABLE_COLOR_DETECTION", False):
        return "unknown", None
    try:
        from color_utils import classify_color
        return classify_color(crop)
    except Exception as exc:                          # noqa: BLE001 - fail safe
        print(f"[split] color detection failed: {exc}")
        return "unknown", None


def save_pair(image, seg, source_photo, folder, n, backend, model_name):
    """Crop one segment, save jpg + json. Returns the jpg path or None."""
    h, w = image.shape[:2]
    pad = getattr(config, "SEGMENT_CROP_PAD", 0.0)
    x1, y1, x2, y2 = _pad_bbox(seg.bbox, w, h, pad)
    if x2 <= x1 or y2 <= y1:
        print(f"[split] skipped degenerate segment {seg.bbox}.")
        return None
    crop = image[y1:y2, x1:x2]

    if getattr(config, "SEGMENT_APPLY_MASK", False) and seg.polygon is not None:
        try:
            crop = _apply_polygon_mask(crop, seg.polygon, x1, y1)
        except Exception as exc:                      # noqa: BLE001 - fail safe
            print(f"[split] mask apply failed (using plain crop): {exc}")

    color, color_conf = _color_of(crop)

    base = f"pair_{n}"
    jpg_path = os.path.join(folder, base + ".jpg")
    json_path = os.path.join(folder, base + ".json")
    if not cv2.imwrite(jpg_path, crop):
        print(f"[split] ERROR: cv2.imwrite failed for {jpg_path}.")
        return None

    metadata = {
        "filename": base + ".jpg",
        "pair_number": n,
        "source_photo": os.path.basename(source_photo),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bbox": [x1, y1, x2, y2],
        "segment_score": round(seg.score, 3),
        "segment_label": seg.label,
        "segment_backend": backend,
        "segment_model": model_name,
        "frame_width": w,
        "frame_height": h,
        "detected_color": color,
        "color_confidence": color_conf,
        # Filled by the later identify step (Phases B/C):
        "make": None,
        "make_confidence": None,
        "model": None,
        "model_confidence": None,
        "model_sources": [],
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[split] pair #{n} [{seg.label} {seg.score:.2f} {color}] -> {jpg_path}")
    return jpg_path


def _draw_overlay(image, segs):
    """Return a copy of `image` with each segment's box + polygon drawn."""
    viz = image.copy()
    for i, s in enumerate(segs, 1):
        x1, y1, x2, y2 = s.bbox
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(viz, f"{i} {s.label} {s.score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if s.polygon is not None:
            import numpy as np
            cv2.polylines(viz, [s.polygon.astype("int32")], True, (0, 0, 220), 2)
    return viz


def process_photo(path, segmenter, backend, model_name, viz=False):
    """Segment one photo and save its pairs. Returns the number saved."""
    image = cv2.imread(path)
    if image is None:
        print(f"[split] could not read image: {path}")
        return 0

    segs = segmenter.segment(image)

    # Drop tiny noise segments if configured.
    min_frac = getattr(config, "SEGMENT_MIN_AREA_FRAC", 0.0)
    if min_frac > 0:
        photo_area = image.shape[0] * image.shape[1]
        segs = [s for s in segs if s.area() >= min_frac * photo_area]

    if not segs:
        print(f"[split] no pairs found in {os.path.basename(path)}. "
              "Check SEGMENT_PROMPTS / SEGMENT_CONF / the model weights.")
        return 0

    # Group the detected single shoes into tied pairs (one record per pair).
    if getattr(config, "SEGMENT_PAIR", False):
        from pair_utils import pair_shoes
        before = len(segs)
        segs = pair_shoes(segs, getattr(config, "SEGMENT_PAIR_MAX_GAP", 1.2))
        print(f"[split] paired {before} shoes -> {len(segs)} record(s).")

    folder = _pairs_folder()
    n = _next_number(folder)
    saved = 0
    for seg in segs:
        if save_pair(image, seg, path, folder, n, backend, model_name):
            n += 1
            saved += 1

    if viz:
        viz_path = os.path.join(
            folder, "viz_" + os.path.splitext(os.path.basename(path))[0] + ".jpg")
        cv2.imwrite(viz_path, _draw_overlay(image, segs))
        print(f"[split] overlay -> {viz_path}")

    print(f"[split] {os.path.basename(path)}: saved {saved} pair(s).")
    return saved


def _gather_inputs(args):
    """Resolve which photo path(s) to process from the CLI args."""
    if args.all:
        d = config.TABLE_INPUT_DIR
        if not os.path.isdir(d):
            print(f"[split] TABLE_INPUT_DIR '{d}' does not exist.")
            return []
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith(exts) and not f.startswith("._")]
    return [args.photo] if args.photo else []


def main():
    ap = argparse.ArgumentParser(
        description="Segment a whole-table photo into per-pair shoe crops.")
    ap.add_argument("photo", nargs="?", help="path to a table photo")
    ap.add_argument("--all", action="store_true",
                    help=f"process every image in TABLE_INPUT_DIR "
                         f"({config.TABLE_INPUT_DIR})")
    ap.add_argument("--backend", default=None, help="override SEGMENT_BACKEND")
    ap.add_argument("--viz", action="store_true",
                    help="also save an overlay image showing the segments")
    args = ap.parse_args()

    if args.backend:
        config.SEGMENT_BACKEND = args.backend       # honored by build_segmenter

    inputs = _gather_inputs(args)
    if not inputs:
        ap.error("give a photo path or --all (with photos in TABLE_INPUT_DIR).")

    segmenter = build_segmenter(config)
    backend = getattr(config, "SEGMENT_BACKEND", "yoloe")
    model_name = getattr(config, "SEGMENT_MODEL", "")

    total = 0
    for path in inputs:
        total += process_photo(path, segmenter, backend, model_name, viz=args.viz)
    print(f"\n[split] done. {total} pair(s) saved from {len(inputs)} photo(s).")


if __name__ == "__main__":
    main()
