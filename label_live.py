"""
label_live.py -- live YOLO shoe detection + labeling UI.

Architecture:
  - MAIN thread reads from the camera, displays frames, handles the mouse.
    It runs at camera FPS -- never blocked by detection.
  - DETECTOR thread continuously runs YOLO + GrabCut on the most recent
    camera frame and posts the results (bbox, conf, polygon per shoe) to
    shared state. Detection FPS is naturally lower than camera FPS, and
    that's OK: the live preview keeps moving even when YOLO is slow.

Per frame in the main loop:
  1. Read a camera frame.
  2. Hand the frame to the detector worker (shared state).
  3. Read the latest detections from the worker (may be from a slightly
     older frame -- visually this is a tiny mask lag, no big deal).
  4. Update the IoU tracker (tracking_utils), which keeps stable IDs and
     also remembers the SHARPEST frame per shoe (variance of Laplacian).
  5. If the operator clicked a shoe since the last frame, save it as
     Recycle now. If any track has been gone long enough, save it as Reuse.
  6. Draw a translucent mask over each tracked shoe -- GrabCut polygon
     when available, shrunk rectangle as fallback. Green = Reuse default,
     brief red = just-clicked Recycle.

Saves land in `sneaker_impact/pictures/incoming<MMDDYYYY>/`.

Run:
    python label_live.py

Controls:
    Click on a shoe -> classify it as Recycle and save it
    Q or ESC        -> quit
"""
import threading
import time

import cv2
from ultralytics import YOLO

import config
from camera_utils import open_camera, release_camera
from save_utils import save_shoe
from tracking_utils import ShoeTracker
from ui_utils import (GREEN, RED, draw_detection_mask, draw_fps,
                      draw_status_text, grabcut_polygon)

WINDOW_TITLE = "Sneaker Impact - Live Detection"

# Class names (compared lowercase) we treat as a shoe. Different YOLO models
# name their classes differently, so we accept a few common spellings:
#   - "shoe" / "shoes"  -> custom or shoe-specific models
#   - "footwear"        -> Open Images V7 models (class "Footwear")
# Plain COCO models (e.g. yolov8n.pt) have NO shoe class at all -- see the
# startup warning below.
SHOE_CLASS_NAMES = {"shoe", "shoes", "footwear"}

FLASH_DURATION_SEC = 0.5      # how long the Recycle mask flashes red after a click

# --- Module-level state shared with the mouse callback --------------------
# cv2.setMouseCallback doesn't pass `self`, so the tracker + pending-save list
# live at module scope. They're only read/appended from the mouse callback;
# the main loop owns all actual save calls.
TRACKER = None                # ShoeTracker, created in main()
PENDING_RECYCLE_SAVES = []    # list of ShoeTrack waiting for Recycle save

# --- Detector worker shared state -----------------------------------------
# Main thread writes _LATEST_FRAME (the freshest camera frame); the detector
# thread reads it, runs YOLO+GrabCut, and writes _LATEST_DETECTIONS. Each
# direction is protected by its own lock so neither side blocks the other.
_FRAME_LOCK = threading.Lock()
_LATEST_FRAME = [None]                  # numpy frame, written by main
_DETECTIONS_LOCK = threading.Lock()
_LATEST_DETECTIONS = [[]]               # list of (bbox, conf, polygon) tuples
_WORKER_RUNNING = threading.Event()     # set while the worker should loop


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


def detector_worker(model):
    """Background loop: run YOLO + GrabCut on the latest camera frame.

    Posts a list of `(bbox, conf, polygon)` to _LATEST_DETECTIONS each cycle.
    Polygon is None if GrabCut failed or is disabled in config.
    """
    while _WORKER_RUNNING.is_set():
        with _FRAME_LOCK:
            frame = _LATEST_FRAME[0]
        if frame is None:
            time.sleep(0.005)
            continue

        try:
            result = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                   verbose=False)[0]
            shoes = []
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            for i in range(count):
                cls_id = int(boxes.cls[i].item())
                if not is_shoe(class_name(result.names, cls_id)):
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                bbox = (x1, y1, x2, y2)
                polygon = (grabcut_polygon(frame, bbox)
                           if getattr(config, "ENABLE_GRABCUT", True) else None)
                shoes.append((bbox, conf, polygon))
            # Keep only the top-confidence detections.
            shoes.sort(key=lambda s: s[1], reverse=True)
            shoes = shoes[:config.MAX_DETECTIONS]
            with _DETECTIONS_LOCK:
                _LATEST_DETECTIONS[0] = shoes
        except Exception as exc:                   # noqa: BLE001 - never crash worker
            print(f"[detector] ERROR: {exc}")
            time.sleep(0.05)


def on_mouse(event, x, y, flags, param):
    """Mouse callback. Left-click inside a shoe -> mark Recycle and save.

    We use single-click (not double-click) because cv2.EVENT_LBUTTONDBLCLK
    fires inconsistently on macOS. Since the operator only ever clicks on
    BAD shoes, single-click is unambiguous.
    """
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if TRACKER is None:
        return
    track = TRACKER.find_at(x, y)
    if track is None:
        print(f"[click] ({x}, {y}) -> no shoe here")
        return
    if track.status == "Recycle":
        print(f"[click] track #{track.id} already Recycle, ignoring")
        return
    print(f"[click] track #{track.id} -> Recycle")
    track.status = "Recycle"
    track.flash_until = time.time() + FLASH_DURATION_SEC
    PENDING_RECYCLE_SAVES.append(track)


def main():
    global TRACKER

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

    # --- Tracker + mouse callback ----------------------------------------
    TRACKER = ShoeTracker(
        expiration_frames=config.TRACK_EXPIRATION_FRAMES,
        iou_threshold=config.TRACK_IOU_THRESHOLD,
    )
    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    # --- Start detector worker -------------------------------------------
    _WORKER_RUNNING.set()
    worker = threading.Thread(target=detector_worker, args=(model,),
                              daemon=True, name="detector")
    worker.start()

    print(f"Live detection running on '{config.MODEL_PATH}' "
          f"(GrabCut={'on' if getattr(config, 'ENABLE_GRABCUT', True) else 'off'}). "
          "Click a shoe to flag it Recycle. Press Q or ESC to quit.")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] Failed to read frame; stopping.")
                break

            # Hand the newest frame to the detector worker.
            with _FRAME_LOCK:
                _LATEST_FRAME[0] = frame.copy()

            # Read the latest detections (may be from a slightly older frame).
            with _DETECTIONS_LOCK:
                shoes = list(_LATEST_DETECTIONS[0])

            # --- Tracker update -----------------------------------------
            # Hand the tracker a CLEAN copy of the frame so the saved crops
            # don't contain the mask we're about to paint onto the display.
            active = TRACKER.update(shoes, frame.copy())

            # --- Pending Recycle saves (from mouse clicks) --------------
            # Use best_frame/best_bbox (the sharpest snapshot of this shoe)
            # rather than the moment-of-click frame, which may be blurry if
            # the shoe was moving when the operator clicked.
            while PENDING_RECYCLE_SAVES:
                t = PENDING_RECYCLE_SAVES.pop()
                if t.saved:
                    continue
                save_shoe(t.best_frame, t.best_bbox, "Recycle",
                          t.last_conf,
                          model_used=config.MODEL_PATH,
                          tracking_id=t.id)
                t.saved = True

            # --- Expired tracks: auto-save Reuse ------------------------
            for ex in TRACKER.expire():
                if ex.saved or ex.status != "Reuse":
                    continue
                save_shoe(ex.best_frame, ex.best_bbox, "Reuse",
                          ex.last_conf,
                          model_used=config.MODEL_PATH,
                          tracking_id=ex.id)
                ex.saved = True

            # --- Draw masks ---------------------------------------------
            now = time.time()
            for t in active:
                color = RED if now < t.flash_until else GREEN
                draw_detection_mask(frame, t.bbox, "Shoe", t.last_conf,
                                    color=color, polygon=t.polygon)

            # FPS (smoothed so the number doesn't jitter).
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            if config.DISPLAY_FPS:
                draw_fps(frame, fps)

            shown = len(active)
            status = f"{shown} shoe(s)" if shown else "no shoes"
            draw_status_text(frame,
                             f"{status}  |  click=Recycle, Q/ESC=quit")

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):              # 27 = ESC
                break
    finally:
        _WORKER_RUNNING.clear()
        worker.join(timeout=2.0)
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Live detection stopped.")


if __name__ == "__main__":
    main()
