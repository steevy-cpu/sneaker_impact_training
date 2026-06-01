"""
label_live.py -- live YOLO shoe detection + labeling UI.

Architecture:
  - MAIN thread (this file) reads from the camera, displays frames, and
    handles mouse clicks. Runs at camera FPS -- never blocked by YOLO.
  - DETECTOR thread (`detector_utils.DetectorThread`) continuously runs
    YOLO + GrabCut on the most recent camera frame in the background.
    The main thread fetches the latest detections via `get_detections()`
    without waiting for inference.

Per frame in the main loop:
  1. Read a camera frame.
  2. Hand it to the detector (`detector.post_frame`).
  3. Read the latest detections (may be a few cycles stale -- tiny lag).
  4. Update the IoU tracker, which keeps stable IDs and remembers the
     SHARPEST frame per shoe (variance of Laplacian) for the save.
  5. Save any Recycle clicks; auto-save any expired Reuse tracks.
  6. Draw a rectangle around each tracked shoe on a throwaway display copy.
     Green = Reuse default, red = Recycle (stays red once clicked).

Saves land in `sneaker_impact/pictures/incoming<MMDDYYYY>/`.

Run:
    python label_live.py

Controls:
    Click on a shoe -> classify it as Recycle and save it
    Q or ESC        -> quit
"""
import os
import shutil
import time

import cv2
from ultralytics import YOLO, YOLOWorld

import config
from camera_utils import open_camera, release_camera
from detector_utils import DetectorThread
from save_utils import save_shoe, delete_saved
from tracking_utils import ShoeTracker
from ui_utils import (GREEN, RED, YELLOW, draw_detection_mask, draw_fps,
                      draw_status_text, draw_toast)

WINDOW_TITLE = "Sneaker Impact - Live Detection"

# Class names (compared lowercase) we treat as a shoe. Different YOLO models
# name their classes differently, so we accept a few common spellings:
#   - "shoe" / "shoes"  -> custom or shoe-specific models
#   - "footwear"        -> Open Images V7 models (class "Footwear")
# Plain COCO models (e.g. yolov8n.pt) have NO shoe class at all -- see the
# startup warning below.
SHOE_CLASS_NAMES = {"shoe", "shoes", "footwear"}

FLASH_DURATION_SEC = 0.5      # how long the Recycle mask flashes red after a click
TOAST_DURATION_SEC = 1.0      # how long a save/undo confirmation banner shows

# Module-level state shared with the mouse callback. cv2.setMouseCallback
# doesn't pass `self`, so the tracker + pending-save list live here. They're
# only read/appended from the mouse callback; the main loop owns saves.
TRACKER = None
PENDING_RECYCLE_SAVES = []


def _free_mb(path):
    """Free space (MB) on the filesystem holding `path`. 0.0 on error."""
    try:
        target = path if os.path.exists(path) else "."
        return shutil.disk_usage(target).free / (1024 * 1024)
    except Exception:                              # noqa: BLE001 - never crash
        return 0.0


def is_shoe(class_name):
    """True if a YOLO class name should count as a shoe (case-insensitive).

    When USE_YOLO_WORLD is on, every detection is already constrained by
    the prompt list (see config.YOLO_WORLD_CLASSES), so any class name we
    see came from that list -- treat all of them as shoes.
    """
    if getattr(config, "USE_YOLO_WORLD", False):
        return True
    return class_name.strip().lower() in SHOE_CLASS_NAMES


def all_names(names):
    """Return all class names as a list, dict-or-list safe."""
    return list(names.values()) if isinstance(names, dict) else list(names)


def load_model():
    """Load the detection model -- YOLO-World if enabled, plain YOLO otherwise.

    Returns the loaded model object, or None on failure (with a printed
    error). For YOLO-World the configured class prompts are applied via
    set_classes() so detections are pre-filtered to shoe-related labels.
    """
    if getattr(config, "USE_YOLO_WORLD", False):
        path = config.YOLO_WORLD_MODEL
        classes = list(config.YOLO_WORLD_CLASSES)
        try:
            m = YOLOWorld(path)
            m.set_classes(classes)
            print(f"[model] loaded YOLO-World '{path}' with {len(classes)} "
                  f"prompts: {', '.join(classes)}")
            return m
        except Exception as exc:                   # noqa: BLE001
            print(f"[model] ERROR: could not load YOLO-World '{path}': {exc}")
            return None

    path = config.MODEL_PATH
    try:
        m = YOLO(path)
        print(f"[model] loaded YOLO '{path}'")
        return m
    except Exception as exc:                       # noqa: BLE001
        print(f"[model] ERROR: could not load YOLO model '{path}': {exc}")
        return None


def on_mouse(event, x, y, flags, param):
    """Mouse callback. Left-click inside a shoe -> mark Recycle and save.

    Single-click (not double-click) because cv2.EVENT_LBUTTONDBLCLK fires
    inconsistently on macOS. The operator only ever clicks on BAD shoes,
    so single-click is unambiguous.
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

    # --- File logging (optional) -----------------------------------------
    if getattr(config, "LOG_TO_FILE", False):
        from log_utils import start_file_logging
        start_file_logging(getattr(config, "LOG_DIR", "logs"))

    # --- Load model -------------------------------------------------------
    model = load_model()
    if model is None:
        raise SystemExit("Exiting: detection model failed to load.")

    # Warn early if this model can't ever detect a shoe (e.g. plain COCO).
    # Skip the check for YOLO-World since its class list is what we asked for.
    if not getattr(config, "USE_YOLO_WORLD", False):
        names = getattr(model, "names", {}) or {}
        if not any(is_shoe(n) for n in all_names(names)):
            print(f"[model] WARNING: '{config.MODEL_PATH}' has no shoe/footwear "
                  "class, so no shoes will be detected.")
            print("[model] Set MODEL_PATH in config.py to an Open Images V7 "
                  "model (e.g. yolov8m-oiv7.pt), or enable USE_YOLO_WORLD.")

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

    # --- Start detector ---------------------------------------------------
    detector = DetectorThread(model, shoe_class_predicate=is_shoe)
    detector.start()

    active_model = (config.YOLO_WORLD_MODEL
                    if getattr(config, "USE_YOLO_WORLD", False)
                    else config.MODEL_PATH)
    print(f"Live detection running on '{active_model}'. "
          "Click a shoe to flag it Recycle. Press Q or ESC to quit.")
    prev_time = time.time()
    fps = 0.0

    # Save bookkeeping for confirmation + undo.
    save_history = []                          # stack of {"jpg","track"} for undo
    saved_counts = {"Reuse": 0, "Recycle": 0}
    toast_msg = ""
    toast_until = 0.0
    toast_color = GREEN

    # Disk-space watch (an unattended station can fill up and lose saves).
    disk_warn_mb = getattr(config, "DISK_SPACE_WARN_MB", 0)
    free_mb_now = _free_mb(config.OUTPUT_ROOT)
    print(f"[disk] {free_mb_now:.0f} MB free on the output drive.")
    if disk_warn_mb and free_mb_now < disk_warn_mb:
        print(f"[disk] WARNING: low disk space (< {disk_warn_mb} MB).")
    last_disk_check = time.time()
    last_disk_warn = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] Failed to read frame; stopping.")
                break

            # `frame` is a fresh buffer from cap.read() each tick and we never
            # draw on it (all drawing goes to `display`, a copy made below), so
            # the detector thread and the tracker can both read it directly --
            # no per-frame full-frame copies needed here.
            detector.post_frame(frame)
            shoes = detector.get_detections()

            # --- Tracker update -----------------------------------------
            # The tracker only reads `frame` to score sharpness and keeps a copy
            # of the small best crop, so sharing the clean frame is safe.
            active = TRACKER.update(shoes, frame)

            # --- Pending Recycle saves (from mouse clicks) --------------
            # Use best_frame/best_bbox (the sharpest snapshot of this shoe)
            # rather than the moment-of-click frame, which may be blurry if
            # the shoe was moving when the operator clicked.
            while PENDING_RECYCLE_SAVES:
                t = PENDING_RECYCLE_SAVES.pop()
                if t.saved:
                    continue
                jpg = save_shoe(t.best_frame, t.best_bbox, "Recycle",
                                t.last_conf,
                                model_used=active_model,
                                tracking_id=t.id,
                                polygon=t.polygon,
                                sharpness=t.best_sharpness)  # clicked -> always save
                t.saved = True
                if jpg:
                    save_history.append({"jpg": jpg, "track": t})
                    saved_counts["Recycle"] += 1
                    toast_msg = f"Saved Recycle #{saved_counts['Recycle']}"
                    toast_until = time.time() + TOAST_DURATION_SEC
                    toast_color = RED

            # --- Expired tracks: auto-save Reuse ------------------------
            for ex in TRACKER.expire():
                if ex.saved or ex.status != "Reuse":
                    continue
                jpg = save_shoe(ex.best_frame, ex.best_bbox, "Reuse",
                                ex.last_conf,
                                model_used=active_model,
                                tracking_id=ex.id,
                                polygon=ex.polygon,
                                sharpness=ex.best_sharpness,
                                apply_blur_gate=True)
                ex.saved = True
                if jpg:
                    saved_counts["Reuse"] += 1
                    toast_msg = f"Saved Reuse #{saved_counts['Reuse']}"
                    toast_until = time.time() + TOAST_DURATION_SEC
                    toast_color = GREEN

            # --- Draw masks ---------------------------------------------
            # Draw on a single throwaway copy so `frame` stays pristine for the
            # detector thread and the tracker's best-crop snapshots.
            display = frame.copy()

            # Only draw boxes for tracks actively seen in recent frames.
            # Tracks linger longer internally (TRACK_EXPIRATION_FRAMES) so
            # the save logic can capture the best crop, but we don't show a
            # ghost box after the shoe leaves the camera view.
            now = time.time()

            # Disk-space watch (throttled): re-check every 5s; log at most every
            # 30s while low, and show a persistent banner (drawn below).
            if disk_warn_mb and now - last_disk_check > 5.0:
                free_mb_now = _free_mb(config.OUTPUT_ROOT)
                last_disk_check = now
                if free_mb_now < disk_warn_mb and now - last_disk_warn > 30.0:
                    print(f"[disk] WARNING: only {free_mb_now:.0f} MB free.")
                    last_disk_warn = now

            draw_cutoff = TRACKER.frame_idx - 5
            for t in active:
                if t.last_seen < draw_cutoff:
                    continue
                color = RED if t.status == "Recycle" else GREEN
                draw_detection_mask(display, t.bbox, "Shoe", t.last_conf,
                                    color=color, polygon=t.polygon)

            # FPS (smoothed so the number doesn't jitter).
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            if config.DISPLAY_FPS:
                draw_fps(display, fps, det_fps=detector.get_fps())

            shown = len(active)
            status = f"{shown} shoe(s)" if shown else "no shoes"
            draw_status_text(
                display,
                f"{status}  |  saved Reuse:{saved_counts['Reuse']} "
                f"Recycle:{saved_counts['Recycle']}  |  "
                f"click=Recycle  U=undo  Q/ESC=quit")

            if disk_warn_mb and free_mb_now < disk_warn_mb:
                cv2.putText(display, f"LOW DISK: {free_mb_now:.0f} MB", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

            if toast_msg and now < toast_until:
                draw_toast(display, toast_msg, color=toast_color)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):              # 27 = ESC
                break
            elif key == ord("u"):                  # undo the last Recycle save
                if save_history:
                    rec = save_history.pop()
                    if delete_saved(rec["jpg"]):
                        tr = rec["track"]
                        tr.status = "Reuse"
                        tr.saved = False
                        tr.flash_until = 0.0
                        saved_counts["Recycle"] = max(
                            0, saved_counts["Recycle"] - 1)
                        toast_msg = "Undid last Recycle"
                        toast_until = time.time() + TOAST_DURATION_SEC
                        toast_color = YELLOW
                        print(f"[undo] removed {rec['jpg']}; "
                              f"track #{tr.id} back to Reuse")
                else:
                    print("[undo] nothing to undo")
    finally:
        detector.stop()
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Live detection stopped.")


if __name__ == "__main__":
    main()
