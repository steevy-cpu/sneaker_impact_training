"""
label_live.py -- live YOLO shoe detection + labeling UI (Phase 3).

What it does each frame:
  1. Reads a frame from the camera (camera_utils).
  2. Runs YOLO and keeps detections whose class name is in SHOE_CLASS_NAMES.
  3. Feeds those detections to a lightweight IoU tracker (tracking_utils), so
     each shoe keeps a stable ID while it's on screen.
  4. Draws a translucent mask over each tracked shoe -- GREEN for "Reuse"
     (default), briefly RED for "Recycle" right after a double-click.
  5. On a left double-click inside a shoe's mask, flips that shoe to "Recycle"
     and saves its crop + metadata JSON immediately (save_utils).
  6. When a shoe has been gone for config.TRACK_EXPIRATION_FRAMES frames and
     was never clicked, auto-saves it as "Reuse" and drops the track.

Saves land in `sneaker_impact/pictures/incoming<MMDDYYYY>/`.

Run:
    python label_live.py

Controls:
    Double-click on a shoe -> classify it as Recycle and save it
    Q or ESC               -> quit
"""
import time

import cv2
from ultralytics import YOLO

import config
from camera_utils import open_camera, release_camera
from save_utils import save_shoe
from tracking_utils import ShoeTracker
from ui_utils import GREEN, RED, draw_detection_mask, draw_fps, draw_status_text

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
# the main loop owns all actual save calls (single-threaded, no locks needed).
TRACKER = None                # ShoeTracker, created in main()
PENDING_RECYCLE_SAVES = []    # list of ShoeTrack waiting for Recycle save


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


def on_mouse(event, x, y, flags, param):
    """Mouse callback. Double-click inside a shoe -> mark Recycle and save."""
    if event != cv2.EVENT_LBUTTONDBLCLK:
        return
    if TRACKER is None:
        return
    track = TRACKER.find_at(x, y)
    if track is None:
        return
    if track.status == "Recycle":
        return                                       # already flagged
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

    print(f"Live detection running on '{config.MODEL_PATH}'. "
          "Double-click a shoe to flag it Recycle. Press Q or ESC to quit.")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] Failed to read frame; stopping.")
                break

            # --- YOLO detection -----------------------------------------
            result = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                   verbose=False)[0]
            shoes = []                              # list of (conf, bbox-tuple)
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            for i in range(count):
                cls_id = int(boxes.cls[i].item())
                if not is_shoe(class_name(result.names, cls_id)):
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                shoes.append((conf, (x1, y1, x2, y2)))

            # Keep only the most confident MAX_DETECTIONS shoes.
            shoes.sort(key=lambda s: s[0], reverse=True)
            shoes = shoes[:config.MAX_DETECTIONS]

            # --- Tracker update -----------------------------------------
            detections = [(bbox, conf) for conf, bbox in shoes]
            active = TRACKER.update(detections, frame)

            # --- Pending Recycle saves (from mouse double-clicks) -------
            while PENDING_RECYCLE_SAVES:
                t = PENDING_RECYCLE_SAVES.pop()
                if t.saved:
                    continue
                save_shoe(t.last_frame, t.bbox, "Recycle",
                          t.last_conf,
                          model_used=config.MODEL_PATH,
                          tracking_id=t.id)
                t.saved = True

            # --- Expired tracks: auto-save Reuse ------------------------
            for ex in TRACKER.expire():
                if ex.saved or ex.status != "Reuse":
                    continue
                save_shoe(ex.last_frame, ex.bbox, "Reuse",
                          ex.last_conf,
                          model_used=config.MODEL_PATH,
                          tracking_id=ex.id)
                ex.saved = True

            # --- Draw masks ---------------------------------------------
            now = time.time()
            for t in active:
                color = RED if now < t.flash_until else GREEN
                draw_detection_mask(frame, t.bbox, "Shoe", t.last_conf,
                                    color=color)

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
                             f"{status}  |  dbl-click=Recycle, Q/ESC=quit")

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
