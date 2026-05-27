"""
label_live.py -- live YOLO shoe detection UI (Phase 2).

Opens the camera (cross-platform, via camera_utils), runs YOLO on each frame,
keeps only shoe detections above the confidence threshold, and draws bounding
boxes with confidence on the live feed. Optionally shows FPS.

This is detection + display ONLY. Double-click labeling, Reuse/Recycle saving,
tracking, and color detection are NOT implemented yet (later phases).

Run:
    python label_live.py

Controls:
    Q or ESC -> quit
"""
import time

import cv2
from ultralytics import YOLO

import config
from camera_utils import open_camera, release_camera
from ui_utils import draw_detection_mask, draw_fps, draw_status_text

WINDOW_TITLE = "Sneaker Impact - Live Detection"

# Class names (compared lowercase) we treat as a shoe. Different YOLO models
# name their classes differently, so we accept a few common spellings:
#   - "shoe" / "shoes"  -> custom or shoe-specific models
#   - "footwear"        -> Open Images V7 models (class "Footwear")
# Plain COCO models (e.g. yolov8n.pt) have NO shoe class at all -- see the
# startup warning below.
SHOE_CLASS_NAMES = {"shoe", "shoes", "footwear"}


def is_shoe(class_name):
    """True if a YOLO class name should count as a shoe (case-insensitive)."""
    return class_name.strip().lower() in SHOE_CLASS_NAMES


def class_name(names, cls_id):
    """Look up a class name whether `names` is a dict (YOLO) or a list."""
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def all_names(names):
    """Return all class names as a list, dict-or-list safe."""
    return list(names.values()) if isinstance(names, dict) else list(names)


def load_model(path):
    """Load the YOLO model, returning None (with a clear message) on failure."""
    try:
        return YOLO(path)
    except Exception as exc:                       # noqa: BLE001 - report any load error
        print(f"[model] ERROR: could not load YOLO model '{path}': {exc}")
        return None


def main():
    # --- Load model -------------------------------------------------------
    model = load_model(config.MODEL_PATH)
    if model is None:
        raise SystemExit("Exiting: YOLO model failed to load.")

    # Warn early if this model can't ever detect a shoe (e.g. plain COCO).
    names = getattr(model, "names", {}) or {}
    if not any(is_shoe(n) for n in all_names(names)):
        print(f"[model] WARNING: '{config.MODEL_PATH}' has no shoe/footwear "
              "class, so no shoes will be detected.")
        print("[model] For shoe detection, set MODEL_PATH in config.py to an "
              "Open Images V7 model (e.g. yolov8m-oiv7.pt).")

    # --- Open camera ------------------------------------------------------
    cap = open_camera()                            # uses config.CAMERA_INDEX
    if cap is None:
        raise SystemExit("Exiting: camera failed to open.")

    print(f"Live detection running on '{config.MODEL_PATH}'. Press Q or ESC to quit.")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] Failed to read frame; stopping.")
                break

            # Run detection. `conf` lets YOLO drop low-confidence boxes for us.
            result = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                   verbose=False)[0]

            # Collect shoe detections as (confidence, bbox).
            shoes = []
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            for i in range(count):
                cls_id = int(boxes.cls[i].item())
                if not is_shoe(class_name(result.names, cls_id)):
                    continue
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                shoes.append((conf, bbox))

            # Draw only the most confident MAX_DETECTIONS shoes.
            shoes.sort(key=lambda s: s[0], reverse=True)
            for conf, bbox in shoes[:config.MAX_DETECTIONS]:
                draw_detection_mask(frame, bbox, "Shoe", conf)

            # FPS (smoothed so the number doesn't jitter).
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            if config.DISPLAY_FPS:
                draw_fps(frame, fps)

            shown = min(len(shoes), config.MAX_DETECTIONS)
            status = f"{shown} shoe(s)" if shown else "no shoes"
            draw_status_text(frame, f"{status}  |  Q/ESC to quit")

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):              # 27 = ESC
                break
    finally:
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Live detection stopped.")


if __name__ == "__main__":
    main()
