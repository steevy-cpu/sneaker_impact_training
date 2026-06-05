"""
segment_utils.py -- segment a whole-table photo into individual pairs of shoes.

Phase A of the 2026 pivot (whole-table photo -> per-pair crops -> make/model).
Model-agnostic: backends sit behind one small interface so the licensing call
(YOLO26 / YOLOE-26 = AGPL-3.0; SAM 2 = Apache-2.0) can be made later WITHOUT
changing callers (see config.SEGMENT_BACKEND).

Backends:
  "yoloe" -- YOLOE-26 open-vocabulary segmentation. Prompt it with text
             (config.SEGMENT_PROMPTS, e.g. "pair of shoes") and it segments
             those, no training required. Default; AGPL-3.0.
  "sam2"  -- Segment Anything 2 (Apache-2.0). Class-agnostic: returns every
             object mask, callers filter. Stub for the future commercial path.

Fail-safe: a model load / inference error logs and returns [] so the caller
keeps running. Heavy imports (ultralytics/torch) are lazy, so importing this
module stays cheap.
"""
import config


class Segment:
    """One segmented region -- intended to be one pair of shoes."""

    def __init__(self, bbox, score, label, polygon=None):
        self.bbox = bbox          # (x1, y1, x2, y2) ints, in source-image coords
        self.score = float(score)  # detection confidence
        self.label = label        # the prompt/class that matched
        self.polygon = polygon    # optional Nx2 contour (source coords) or None

    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def _resolve_device():
    """Pick the inference device, honoring config.SEGMENT_DEVICE ("auto" reuses
    the same CUDA->MPS->CPU probe label_live uses)."""
    pref = getattr(config, "SEGMENT_DEVICE", "auto")
    if pref and pref != "auto":
        return pref
    try:
        from detector_utils import pick_device   # same probe as live detection
        return pick_device()
    except Exception:                              # noqa: BLE001 - fail safe
        return "cpu"


class Segmenter:
    """Common interface. Subclasses implement segment(image) -> list[Segment]."""

    def segment(self, image):
        raise NotImplementedError


class YoloeSegmenter(Segmenter):
    """YOLOE open-vocabulary segmentation (text-prompted). AGPL-3.0."""

    def __init__(self, model_path, prompts, conf, device, imgsz=640):
        self.prompts = list(prompts)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.model = None
        try:
            # YOLOE is the open-vocab class; if a given ultralytics version
            # exposes it only via YOLO, fall back to that.
            try:
                from ultralytics import YOLOE as _Model      # type: ignore
            except ImportError:
                from ultralytics import YOLO as _Model        # type: ignore
            self.model = _Model(model_path)
            # Bake the text prompts in so the model only segments those.
            try:
                self.model.set_classes(
                    self.prompts, self.model.get_text_pe(self.prompts))
            except Exception as exc:                 # noqa: BLE001
                # A fixed-class seg model has no set_classes -- that's OK, it
                # just segments whatever classes it was trained on.
                print(f"[segment] note: set_classes unavailable ({exc}); "
                      "using the model's built-in classes.")
            print(f"[segment] YOLOE backend ready: {model_path} on {device}, "
                  f"prompts={self.prompts}")
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] ERROR loading YOLOE model '{model_path}': {exc}")
            self.model = None

    def segment(self, image):
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                image, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False)
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] inference failed: {exc}")
            return []
        return _segments_from_result(results)


class Sam2Segmenter(Segmenter):
    """Segment Anything 2 (Apache-2.0) -- class-agnostic auto-segmentation.

    Stub for the future commercial path: SAM 2 returns a mask for *every*
    object, with no labels, so callers filter by size/shape (or pass each crop
    to the brand step). Wired through ultralytics' SAM class when we switch.
    """

    def __init__(self, model_path, conf, device):
        self.conf = conf
        self.device = device
        self.model = None
        try:
            from ultralytics import SAM               # type: ignore
            self.model = SAM(model_path)
            print(f"[segment] SAM2 backend ready: {model_path} on {device}")
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] ERROR loading SAM2 model '{model_path}': {exc}")
            self.model = None

    def segment(self, image):
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                image, device=self.device, verbose=False)
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] inference failed: {exc}")
            return []
        # SAM masks are class-agnostic (no boxes/labels in the usual sense);
        # _segments_from_result reads boxes when present and falls back to mask
        # bounding boxes otherwise.
        return _segments_from_result(results, default_label="object")


def _iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _containment(a, b):
    """Fraction of the SMALLER box that lies inside the other -- catches a
    partial box from a tile seam sitting inside a full box from a neighbor."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(min(area_a, area_b))


def _tile_windows(w, h, tile, overlap):
    """Tile (x0, y0, x1, y1) windows covering a wxh image, with overlap, always
    reaching the right/bottom edges."""
    def axis(extent):
        if extent <= tile:
            return [0]
        step = max(1, int(tile * (1 - overlap)))
        starts = list(range(0, extent - tile + 1, step))
        if starts[-1] + tile < extent:
            starts.append(extent - tile)
        return starts
    windows = []
    for y0 in axis(h):
        for x0 in axis(w):
            windows.append((x0, y0, min(w, x0 + tile), min(h, y0 + tile)))
    return windows


def _merge(segments, iou_thresh, contain_thresh=0.7):
    """Greedy NMS: keep the highest-score segment, drop later ones that overlap
    it by IoU >= iou_thresh OR are mostly contained in it (tile-seam partials)."""
    kept = []
    for s in sorted(segments, key=lambda z: z.score, reverse=True):
        if all(_iou(s.bbox, k.bbox) < iou_thresh
               and _containment(s.bbox, k.bbox) < contain_thresh for k in kept):
            kept.append(s)
    return kept


class TiledSegmenter(Segmenter):
    """Wrap any Segmenter: slice the image into overlapping tiles, segment each,
    map detections back to full-image coords, and merge duplicates. The recall
    fix for dense tables of many small shoes."""

    def __init__(self, base, tile, overlap, iou_merge, include_full=True):
        self.base = base
        self.tile = tile
        self.overlap = overlap
        self.iou_merge = iou_merge
        # Also run one pass on the WHOLE image and merge it with the tile
        # detections. Tiling alone misses large, well-separated shoes on a
        # SPARSE table: each tile sees only a slice of a big shoe, so the
        # open-vocab confidence drops below threshold and nothing survives
        # (observed: a 3-pair table -> 0 tiled detections). The full pass
        # catches those; the tiles still add the small/dense shoes a single
        # wide pass downsamples away on a CROWDED table. Best of both; the
        # merge dedups the overlap, so recall only ever goes up.
        self.include_full = include_full

    def segment(self, image):
        h, w = image.shape[:2]
        windows = _tile_windows(w, h, self.tile, self.overlap)
        raw = []
        full_n = 0
        if self.include_full:
            full = self.base.segment(image)   # whole-image coords already
            raw.extend(full)
            full_n = len(full)
        for (x0, y0, x1, y1) in windows:
            for s in self.base.segment(image[y0:y1, x0:x1]):
                bx1, by1, bx2, by2 = s.bbox
                poly = None
                if s.polygon is not None:
                    poly = s.polygon.copy()
                    poly[:, 0] += x0
                    poly[:, 1] += y0
                raw.append(Segment((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                                   s.score, s.label, poly))
        merged = _merge(raw, self.iou_merge)
        print(f"[segment] tiled: {len(windows)} tiles + {full_n} full -> "
              f"{len(raw)} raw -> {len(merged)} merged")
        return merged


def _segments_from_result(results, default_label="shoe"):
    """Convert an ultralytics Results object into a list[Segment]."""
    segs = []
    if not results:
        return segs
    r = results[0]
    names = getattr(r, "names", {}) or {}
    polys = r.masks.xy if getattr(r, "masks", None) is not None else None
    boxes = getattr(r, "boxes", None)
    n = len(boxes) if boxes is not None else (len(polys) if polys else 0)
    for i in range(n):
        try:
            poly = polys[i] if polys is not None and i < len(polys) else None
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                score = float(boxes.conf[i]) if boxes.conf is not None else 1.0
                cls = int(boxes.cls[i]) if boxes.cls is not None else -1
                label = names.get(cls, default_label)
            elif poly is not None and len(poly):
                xs, ys = poly[:, 0], poly[:, 1]
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                score, label = 1.0, default_label
            else:
                continue
            segs.append(Segment((x1, y1, x2, y2), score, label, polygon=poly))
        except Exception as exc:                     # noqa: BLE001 - skip bad one
            print(f"[segment] skipped a result row: {exc}")
    return segs


def build_segmenter(cfg=None):
    """Construct the configured segmenter. Returns a Segmenter (whose segment()
    yields [] if the model failed to load -- never raises)."""
    cfg = cfg or config
    backend = getattr(cfg, "SEGMENT_BACKEND", "yoloe").lower()
    model_path = getattr(cfg, "SEGMENT_MODEL", "yoloe-26s-seg.pt")
    conf = getattr(cfg, "SEGMENT_CONF", 0.25)
    device = _resolve_device()
    if backend == "sam2":
        base = Sam2Segmenter(model_path, conf, device)
    else:
        if backend != "yoloe":
            print(f"[segment] unknown SEGMENT_BACKEND '{backend}', using yoloe.")
        prompts = getattr(cfg, "SEGMENT_PROMPTS", ["shoe"])
        imgsz = getattr(cfg, "SEGMENT_IMGSZ", 640)
        base = YoloeSegmenter(model_path, prompts, conf, device, imgsz)

    tile = getattr(cfg, "SEGMENT_TILE", 0)
    if tile and tile > 0:
        overlap = getattr(cfg, "SEGMENT_TILE_OVERLAP", 0.25)
        iou_merge = getattr(cfg, "SEGMENT_TILE_IOU", 0.4)
        include_full = getattr(cfg, "SEGMENT_TILE_INCLUDE_FULL", True)
        return TiledSegmenter(base, tile, overlap, iou_merge, include_full)
    return base
