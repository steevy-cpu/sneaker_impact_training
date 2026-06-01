"""
detector_utils.py -- async YOLO + GrabCut detector thread.

DetectorThread owns a daemon thread that continuously runs YOLO inference
and GrabCut polygon extraction on whatever camera frame the main thread
last handed it via `post_frame()`. The main thread reads back the latest
detections via `get_detections()` without blocking on YOLO inference.

Optimizations baked in:
  * MPS device on Apple Silicon (auto-picked) -- much faster than CPU.
  * YOLO input shrunk via `config.YOLO_IMGSZ` (default 416).
  * Polygon cache: if a detected bbox overlaps a recently-cached one by
    IoU >= 0.5 we reuse that polygon (translated to the new center)
    instead of re-running GrabCut. Cache refreshes every
    `config.GRABCUT_REFRESH_CYCLES` cycles so shape changes catch up.
  * Skip the cycle when the camera hasn't produced a new frame.

The detector is shoe-agnostic: callers pass a `shoe_class_predicate`
function that takes a class-name string and returns True if it should be
treated as a shoe. label_live.py supplies one based on SHOE_CLASS_NAMES.
"""
import threading
import time

import config
from tracking_utils import iou
from ui_utils import grabcut_polygon


def pick_device():
    """Choose the YOLO inference device, preferring GPU when available.

    Order: CUDA (Jetson, desktop NVIDIA) -> MPS (Apple Silicon) -> CPU
    (Raspberry Pi 5, fallback). Honors `config.YOLO_DEVICE` -- "auto"
    probes in that order; anything else is used verbatim ("cpu", "mps",
    "cuda:0"). Falls back to "cpu" on any probe failure so a broken torch
    install can't block startup.
    """
    pref = getattr(config, "YOLO_DEVICE", "auto")
    if pref and pref != "auto":
        return pref
    try:
        import torch                              # type: ignore
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:                              # noqa: BLE001
        pass
    return "cpu"


def translate_polygon(polygon, old_bbox, new_bbox):
    """Shift a cached polygon to follow a bbox that moved between cycles."""
    ox1, oy1, ox2, oy2 = old_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    dx = int(((nx1 + nx2) - (ox1 + ox2)) / 2)
    dy = int(((ny1 + ny2) - (oy1 + oy2)) / 2)
    if dx == 0 and dy == 0:
        return polygon
    moved = polygon.copy()
    moved[:, :, 0] += dx
    moved[:, :, 1] += dy
    return moved


def lookup_cache(cache, bbox, refresh, iou_threshold=0.5):
    """Return a cached polygon (translated to the new bbox) if there's a
    recent-enough entry whose bbox overlaps `bbox` by >= `iou_threshold`.

    The matching entry's bbox/polygon are updated in-place to the new ones;
    its `age` is unchanged so it still expires on schedule.
    """
    for ent in cache:
        if ent["age"] >= refresh:
            continue
        if iou(ent["bbox"], bbox) >= iou_threshold:
            poly = translate_polygon(ent["polygon"], ent["bbox"], bbox)
            ent["bbox"] = bbox
            ent["polygon"] = poly
            return poly
    return None


def _resolve_class_name(names, cls_id):
    """Look up a class name whether `names` is a dict (YOLO) or a list."""
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


class DetectorThread:
    """Async wrapper around YOLO + GrabCut.

    Usage:
        det = DetectorThread(model, shoe_class_predicate=is_shoe)
        det.start()
        try:
            while ...:
                det.post_frame(frame.copy())
                shoes = det.get_detections()     # list of (bbox, conf, polygon)
                ...
        finally:
            det.stop()
    """

    def __init__(self, model, shoe_class_predicate):
        self.model = model
        self.shoe_class_predicate = shoe_class_predicate
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._frame_event = threading.Event()   # set when a new frame is posted
        self._det_lock = threading.Lock()
        self._latest_detections = []
        self._fps = 0.0                          # detector inference rate
        self._running = threading.Event()
        self._thread = None

    # --- main-thread API -------------------------------------------------

    def start(self):
        """Spawn the worker thread."""
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="detector")
        self._thread.start()

    def stop(self):
        """Signal the worker to stop and wait briefly for it."""
        self._running.clear()
        self._frame_event.set()                # wake the worker so it exits now
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def post_frame(self, frame):
        """Hand a new frame to the worker. Call this once per main-loop tick."""
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id += 1
        self._frame_event.set()                # wake the worker if it's waiting

    def get_detections(self):
        """Return a snapshot of the most recent detections."""
        with self._det_lock:
            return list(self._latest_detections)

    def get_fps(self):
        """Return the detector's smoothed inference rate (cycles/sec)."""
        return self._fps

    # --- worker thread ---------------------------------------------------

    def _run(self):
        device = pick_device()
        imgsz = int(getattr(config, "YOLO_IMGSZ", 416))
        refresh = max(1, int(getattr(config, "GRABCUT_REFRESH_CYCLES", 8)))
        print(f"[detector] device={device}  imgsz={imgsz}  "
              f"grabcut={'on' if config.ENABLE_GRABCUT else 'off'} "
              f"iters={getattr(config, 'GRABCUT_ITERS', 1)} "
              f"refresh_every={refresh}")

        # poly_cache: list of {'bbox': tuple, 'polygon': ndarray, 'age': int}.
        poly_cache = []
        last_frame_id = -1
        last_fps_t = time.time()
        err_count = 0

        while self._running.is_set():
            # Block until a new frame is posted instead of busy-polling. The
            # timeout keeps us checking the running flag ~10x/sec so stop() is
            # responsive. Clear BEFORE reading so a frame posted during the
            # (slow) inference below still re-wakes us for the next round.
            self._frame_event.wait(timeout=0.1)
            self._frame_event.clear()
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
            if frame is None or frame_id == last_frame_id:
                continue
            last_frame_id = frame_id

            try:
                result = self.model.predict(
                    frame,
                    conf=config.CONFIDENCE_THRESHOLD,
                    imgsz=imgsz, device=device,
                    verbose=False,
                )[0]
                shoes = self._collect_shoes(result, frame, poly_cache, refresh)

                # Age all cache entries; drop stale ones.
                for ent in poly_cache:
                    ent["age"] += 1
                poly_cache[:] = [e for e in poly_cache if e["age"] <= refresh * 2]

                with self._det_lock:
                    self._latest_detections = shoes

                # Detector inference rate (cycles/sec), smoothed.
                t_now = time.time()
                dt = t_now - last_fps_t
                last_fps_t = t_now
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
                err_count = 0
            except Exception as exc:               # noqa: BLE001 - never crash worker
                err_count += 1
                # Throttle a repeating failure (e.g. a broken model) so it can't
                # flood the console at detector speed; keep retrying in case it's
                # transient.
                if err_count <= 3:
                    print(f"[detector] ERROR: {exc}")
                elif err_count == 4:
                    print(f"[detector] ERROR repeating; throttling and logging "
                          f"every 30th from now: {exc}")
                elif err_count % 30 == 0:
                    print(f"[detector] ERROR x{err_count}: {exc}")
                time.sleep(0.05 if err_count < 4 else 2.0)

    def _collect_shoes(self, result, frame, poly_cache, refresh):
        """Pull shoe detections out of a YOLO result and pair each with a
        polygon (cached or freshly GrabCut'd). Returns the top-confidence
        `config.MAX_DETECTIONS` of them."""
        shoes = []
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        fh, fw = frame.shape[:2]
        min_area = getattr(config, "MIN_BBOX_AREA_FRAC", 0.0) * fw * fh
        for i in range(count):
            cls_id = int(boxes.cls[i].item())
            class_name = _resolve_class_name(result.names, cls_id)
            if not self.shoe_class_predicate(class_name):
                continue
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            bbox = (x1, y1, x2, y2)

            # Skip tiny/distant shoes -- they make poor training crops.
            if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
                continue

            polygon = None
            if getattr(config, "ENABLE_GRABCUT", True):
                polygon = lookup_cache(poly_cache, bbox, refresh)
                if polygon is None:
                    polygon = grabcut_polygon(frame, bbox)
                    if polygon is not None:
                        poly_cache.append({"bbox": bbox,
                                           "polygon": polygon,
                                           "age": 0})
            shoes.append((bbox, conf, polygon))

        shoes.sort(key=lambda s: s[1], reverse=True)
        return shoes[:config.MAX_DETECTIONS]
