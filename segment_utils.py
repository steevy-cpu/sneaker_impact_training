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


def _empty_cuda_cache():
    """Best-effort: hand freed VRAM back to the driver so a co-resident process
    (the ollama VLM) can use it. Never raises."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                 # noqa: BLE001 - best effort
        pass


class Segmenter:
    """Common interface. Subclasses implement segment(image) -> list[Segment]."""

    def segment(self, image):
        raise NotImplementedError

    def segment_batch(self, images):
        """Segment several images; returns one list[Segment] per input image.
        Default = sequential loop, so every backend works unchanged; backends
        that support true batching (YOLOE) override this for speed."""
        return [self.segment(im) for im in images]

    def release(self):
        """Drop model weights off the GPU once segmentation is done, so the
        per-pair VLM step isn't starved of VRAM. Default = no-op; backends that
        hold a model override it. Safe to call once, after the last segment()."""
        return None


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

    def segment_batch(self, images):
        """True batched inference: one predict() call for the whole list, so
        the per-call overhead (preprocess setup, device sync, NMS dispatch) is
        paid once instead of once per tile. Ultralytics letterboxes each image
        to imgsz independently, so the detections are the same as the
        sequential path. Fail-safe: any batch error falls back per-image."""
        if not images:
            return []
        if self.model is None:
            return [[] for _ in images]
        try:
            results = self.model.predict(
                images, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False)
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] batch inference failed ({exc}); "
                  "falling back to per-image.")
            return [self.segment(im) for im in images]
        return [_segments_from_one(r) for r in results]

    def release(self):
        self.model = None
        _empty_cuda_cache()


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

    def release(self):
        self.model = None
        _empty_cuda_cache()


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

    def __init__(self, base, tile, overlap, iou_merge, include_full=True,
                 tile_imgsz=0, tile_batch=8, max_side=0,
                 seam_px=2, rescue_contain=0.3):
        self.base = base
        self.tile = tile
        self.overlap = overlap
        self.iou_merge = iou_merge
        # Cap the DETECTION resolution. Capture sizes vary wildly (1920x1080
        # station shots vs an 8000x6000 phone photo); SEGMENT_TILE is an
        # absolute 512px, so on a giant photo a single shoe is BIGGER than a
        # tile (cut at several seams) and the tile count explodes (337 tiles
        # observed). Downscaling the long side to max_side before tiling
        # bounds both: shoes fit in tiles again and tile count stays sane.
        # Detection happens on the downscaled copy; boxes/polygons are mapped
        # back to ORIGINAL coords, so callers still crop from the full-res
        # photo (brand/cloud-ID quality is untouched). 0 = no cap.
        self.max_side = int(max_side or 0)
        # Seam-cut handling knobs (see the comment in _segment_work): a box
        # within seam_px of a tile edge that isn't an image edge is treated as
        # a truncated partial; a deduped partial is rescued only if it overlaps
        # every kept box by less than rescue_contain (containment).
        self.seam_px = int(seam_px)
        self.rescue_contain = float(rescue_contain)
        # Tiles are inferred in chunks of this many per predict() call. One
        # call per tile wastes most of the time on fixed per-call overhead
        # (a 1080p photo = 16 calls; the 8000x6000 outlier = 337). Batching
        # pays that overhead once per chunk. Kept bounded (not "all tiles at
        # once") so VRAM stays predictable on the shared GPU -- the dash box
        # also runs ollama + DINOv2, and the website must never stall behind
        # a VRAM spike. 1 = the old sequential behavior.
        self.tile_batch = max(1, int(tile_batch or 1))
        # Tiles must be inferred near their OWN size, not at SEGMENT_IMGSZ:
        # that value is sized for the full photo, and blowing a 512px tile up
        # to e.g. 1280 pushes the shoes far outside the model's training scale,
        # so confidence collapses below SEGMENT_CONF and tiles return nothing
        # (observed: 13/15 tiles -> 0 detections on a real 14-pair table).
        self.tile_imgsz = tile_imgsz
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
        if not (self.max_side and max(h, w) > self.max_side):
            return self._segment_work(image)
        # Detect on a downscaled copy, then map results back to source coords.
        import cv2                       # lazy: keep module import light
        scale = self.max_side / float(max(h, w))
        work = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA)
        print(f"[segment] capped {w}x{h} -> {work.shape[1]}x{work.shape[0]} "
              f"for detection (max_side={self.max_side})")
        merged = self._segment_work(work)
        inv = 1.0 / scale
        for s in merged:
            x1, y1, x2, y2 = s.bbox
            s.bbox = (max(0, int(round(x1 * inv))), max(0, int(round(y1 * inv))),
                      min(w, int(round(x2 * inv))), min(h, int(round(y2 * inv))))
            if s.polygon is not None:
                s.polygon = s.polygon * inv
        return merged

    def _segment_work(self, image):
        h, w = image.shape[:2]
        windows = _tile_windows(w, h, self.tile, self.overlap)
        raw = []
        full_n = 0
        if self.include_full:
            full = self.base.segment(image)   # whole-image coords already
            raw.extend(full)
            full_n = len(full)
        h, w = image.shape[:2]
        seam_cut = []
        tile_segs = self._segment_tiles(
            [image[y0:y1, x0:x1] for (x0, y0, x1, y1) in windows])
        for (x0, y0, x1, y1), segs_in_tile in zip(windows, tile_segs):
            for s in segs_in_tile:
                bx1, by1, bx2, by2 = s.bbox
                poly = None
                if s.polygon is not None:
                    poly = s.polygon.copy()
                    poly[:, 0] += x0
                    poly[:, 1] += y0
                seg = Segment((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                              s.score, s.label, poly)
                # A box touching a tile edge that is NOT an image edge is a
                # truncated shoe. Keeping it in the main pool is harmful twice
                # over: it becomes a sliver crop, and in _merge a high-scoring
                # partial can suppress the COMPLETE box of the same shoe
                # (greedy NMS keeps the higher score). But don't discard it:
                # if no clean box covers that region at all, the partial is
                # the only evidence of that shoe — re-add it as a last resort.
                m = self.seam_px  # px tolerance
                if ((bx1 <= m and x0 > 0) or (by1 <= m and y0 > 0)
                        or (bx2 >= (x1 - x0) - m and x1 < w)
                        or (by2 >= (y1 - y0) - m and y1 < h)):
                    seam_cut.append(seg)
                else:
                    raw.append(seg)
        merged = _merge(raw, self.iou_merge)
        rescued = 0
        for s in _merge(seam_cut, self.iou_merge):       # dedup partials first
            if all(_iou(s.bbox, k.bbox) < self.iou_merge
                   and _containment(s.bbox, k.bbox) < self.rescue_contain
                   for k in merged):
                merged.append(s)
                rescued += 1
        print(f"[segment] tiled: {len(windows)} tiles + {full_n} full -> "
              f"{len(raw)} raw + {len(seam_cut)} seam-cut "
              f"({rescued} rescued) -> {len(merged)} merged")
        return merged

    def _segment_tiles(self, tile_imgs):
        """Run the base segmenter on all tiles at the tile-sized resolution
        (see __init__), in chunks of tile_batch per predict() call. Returns
        one list[Segment] per tile, in order. Backends without a real batch
        path fall back to Segmenter.segment_batch's sequential loop, so
        behavior is identical either way -- only the call count changes."""
        old = getattr(self.base, "imgsz", None)
        if self.tile_imgsz and old is not None:
            self.base.imgsz = self.tile_imgsz
        try:
            out = []
            for i in range(0, len(tile_imgs), self.tile_batch):
                out.extend(self.base.segment_batch(
                    tile_imgs[i:i + self.tile_batch]))
            return out
        finally:
            if old is not None:
                self.base.imgsz = old

    def release(self):
        self.base.release()


class SahiTiledSegmenter(Segmenter):
    """SAHI-powered tiling: slice with SAHI's geometry (`get_slice_bboxes`) and
    merge cross-tile duplicates with SAHI's battle-tested Greedy NMM/NMS
    postprocess, while reusing the SAME base Segmenter as the custom tiler.

    This is a true A/B of *SAHI's windowing + merge* against `TiledSegmenter`:
    the model, prompts, masks, device, and tile inference resolution are held
    identical, so any difference in recall/precision is the tiling logic alone.

    Merge knobs (mirror SAHI):
      merge: "NMS" suppresses overlapping boxes keeping the winner's box as-is
             (closest to our greedy `_merge`); "NMM" merges overlaps into their
             union (better at stitching a shoe split across a tile seam, but can
             fuse two shoes that sit very close together).
      metric: "IOS" = intersection-over-smaller (catches tile-seam partials that
             a small partial box makes — same intent as our `_containment`);
             "IOU" = classic intersection-over-union.
    """

    def __init__(self, base, tile, overlap, match_threshold, merge="NMS",
                 metric="IOS", include_full=True, tile_imgsz=0):
        self.base = base
        self.tile = tile
        self.overlap = overlap
        self.match_threshold = match_threshold
        self.merge = (merge or "NMS").upper()
        self.metric = (metric or "IOS").upper()
        self.include_full = include_full
        self.tile_imgsz = tile_imgsz

    def _segment_tile(self, tile_img):
        """Run the base segmenter on one tile at the tile-sized resolution
        (identical to TiledSegmenter so the A/B holds the model constant)."""
        old = getattr(self.base, "imgsz", None)
        if self.tile_imgsz and old is not None:
            self.base.imgsz = self.tile_imgsz
        try:
            return self.base.segment(tile_img)
        finally:
            if old is not None:
                self.base.imgsz = old

    def segment(self, image):
        try:
            from sahi.slicing import get_slice_bboxes
            from sahi.prediction import ObjectPrediction
            from sahi.postprocess.combine import (GreedyNMMPostprocess,
                                                  NMSPostprocess)
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[segment] SAHI unavailable ({exc}); returning base segment.")
            return self.base.segment(image)

        h, w = image.shape[:2]
        windows = get_slice_bboxes(
            image_height=h, image_width=w,
            slice_height=self.tile, slice_width=self.tile,
            auto_slice_resolution=False,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap)

        # Collect every detection in full-image coords, preserving polygons, and
        # stash each source Segment on its ObjectPrediction so we can recover the
        # mask after the merge (SAHI returns the surviving source objects).
        segs = []
        full_n = 0
        if self.include_full:
            full = self.base.segment(image)
            segs.extend(full)
            full_n = len(full)
        for (x0, y0, x1, y1) in windows:
            for s in self._segment_tile(image[y0:y1, x0:x1]):
                bx1, by1, bx2, by2 = s.bbox
                poly = None
                if s.polygon is not None:
                    poly = s.polygon.copy()
                    poly[:, 0] += x0
                    poly[:, 1] += y0
                segs.append(Segment((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                                    s.score, s.label, poly))

        ops = []
        for s in segs:
            op = ObjectPrediction(
                bbox=[float(v) for v in s.bbox], category_id=0,
                category_name=str(s.label or "shoe"), score=float(s.score),
                full_shape=[h, w])
            op._src_segment = s                       # recover polygon/label later
            ops.append(op)

        PP = GreedyNMMPostprocess if self.merge == "NMM" else NMSPostprocess
        postprocess = PP(match_threshold=self.match_threshold,
                         match_metric=self.metric, class_agnostic=True)
        kept = postprocess(ops)

        out = []
        for op in kept:
            x1b, y1b, x2b, y2b = [int(v) for v in op.bbox.to_xyxy()]
            src = getattr(op, "_src_segment", None)
            poly = src.polygon if src is not None else None
            label = src.label if src is not None else "shoe"
            out.append(Segment((x1b, y1b, x2b, y2b), op.score.value, label, poly))
        print(f"[segment] sahi-tiled ({self.merge}/{self.metric}): "
              f"{len(windows)} tiles + {full_n} full -> {len(segs)} raw -> "
              f"{len(out)} merged")
        return out

    def release(self):
        self.base.release()


def _segments_from_result(results, default_label="shoe"):
    """Convert an ultralytics predict() return (list of Results) -- reads the
    FIRST Results object, i.e. the single-image path."""
    if not results:
        return []
    return _segments_from_one(results[0], default_label)


def _segments_from_one(r, default_label="shoe"):
    """Convert ONE ultralytics Results object into a list[Segment]."""
    segs = []
    if r is None:
        return segs
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


def _is_sliver(seg, w, h):
    """A degenerate detection: a tiny crop or an extreme-aspect-ratio one (the
    seam-cut partials YOLOE leaves on crowded tables). Same rule the SAM2
    validation used; only a TRIGGER signal here, not a filter."""
    x1, y1, x2, y2 = seg.bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return True
    return (bw * bh) / float(w * h) < 0.004 or max(bw, bh) / float(min(bw, bh)) > 6.0


class Sam2GateSegmenter(Segmenter):
    """SAM2-everything + a learned shoe/not-shoe gate.

    SAM2 (Apache-2.0) returns a class-agnostic mask for every object; a size
    filter drops obvious junk, then a small resnet18 gate (trained on our own
    table photos: in-bbox masks = shoe, empty-zone masks = junk) keeps only the
    masks it calls 'shoe'. Higher recall + cleaner instance masks than the YOLOE
    tiler on crowded / low-contrast tables, with the background junk removed.
    Validated 2026-06-27 (see validate_gate.py). All heavy imports are lazy; a
    load/inference error logs and returns [] so the caller keeps running.
    """

    def __init__(self, sam_model_path, gate_path, device,
                 sam_max=1536, af_lo=0.004, af_hi=0.12, ar_max=4.5):
        self.device = device
        self.sam_max = int(sam_max)
        self.af_lo, self.af_hi, self.ar_max = af_lo, af_hi, ar_max
        self.sam = None
        self.gate = None
        self._tf = None
        try:
            from ultralytics import SAM                # type: ignore
            self.sam = SAM(sam_model_path)
            import torch, timm                          # type: ignore
            from torchvision import transforms          # type: ignore
            self.gate = timm.create_model(
                "resnet18", pretrained=False, num_classes=2).to(device).eval()
            self.gate.load_state_dict(torch.load(gate_path, map_location=device))
            self._tf = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            print(f"[segment] SAM2+gate ready: {sam_model_path} + {gate_path} "
                  f"on {device}")
        except Exception as exc:                        # noqa: BLE001 - fail safe
            print(f"[segment] ERROR loading SAM2+gate ({exc}); escalation off.")
            self.sam = None

    @property
    def ready(self):
        return self.sam is not None and self.gate is not None

    def _sam_masks(self, image):
        """SAM2-everything on a size-capped copy; size-filtered bboxes mapped
        back to original coords."""
        import cv2
        h, w = image.shape[:2]
        s = self.sam_max / float(max(h, w)) if max(h, w) > self.sam_max else 1.0
        img_s = cv2.resize(image, (int(w * s), int(h * s))) if s != 1.0 else image
        inv = 1.0 / s
        try:
            r = self.sam(img_s, verbose=False, device=self.device)[0]
        except Exception as exc:                        # noqa: BLE001 - fail safe
            print(f"[segment] SAM2 inference failed: {exc}")
            return []
        out = []
        if getattr(r, "masks", None) is None:
            return out
        for p in r.masks.xy:
            if len(p) < 3:
                continue
            x1, y1 = int(p[:, 0].min() * inv), int(p[:, 1].min() * inv)
            x2, y2 = int(p[:, 0].max() * inv), int(p[:, 1].max() * inv)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            af = (bw * bh) / float(w * h)
            ar = max(bw, bh) / float(min(bw, bh))
            if af < self.af_lo or af > self.af_hi or ar > self.ar_max:
                continue
            out.append((x1, y1, x2, y2))
        return out

    def _gate_keep(self, image, boxes):
        """Keep only boxes the gate classifies as 'shoe' (class 1)."""
        import torch
        if not boxes:
            return []
        crops = []
        for (x1, y1, x2, y2) in boxes:
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                crops.append(None)
                continue
            crops.append(self._tf(crop))
        idx = [i for i, c in enumerate(crops) if c is not None]
        if not idx:
            return []
        with torch.no_grad():
            x = torch.stack([crops[i] for i in idx]).to(self.device)
            pred = self.gate(x).argmax(1).cpu().tolist()
        return [boxes[i] for i, p in zip(idx, pred) if p == 1]

    def segment(self, image):
        if not self.ready:
            return []
        boxes = self._sam_masks(image)
        kept = self._gate_keep(image, boxes)
        print(f"[segment] SAM2+gate: {len(boxes)} masks -> {len(kept)} shoes")
        return [Segment(b, 1.0, "shoe", None) for b in kept]

    def release(self):
        # SAM2 (~5GB) + the gate must come off the GPU before the per-pair VLM
        # loop, or the resident ollama model is evicted and every model-ID call
        # times out. This is the whole reason release() exists.
        self.sam = None
        self.gate = None
        _empty_cuda_cache()


class EscalatingSegmenter(Segmenter):
    """Run a fast PRIMARY segmenter (the YOLOE tiler) every time; when its result
    looks weak, ALSO run a heavier FALLBACK (SAM2+gate) and keep whichever found
    more shoes. Recall never drops below the primary, and the expensive path runs
    only when it is likely to help -- so the live site's cost is bounded.

    Fully additive: with SEGMENT_ESCALATE_SAM2=False this wrapper isn't built at
    all (build_segmenter returns the primary directly), so it is a clean off-switch.
    """

    def __init__(self, primary, fallback_builder, mode="weak",
                 max_shoes=28, min_slivers=1):
        self.primary = primary
        self._build_fallback = fallback_builder   # lazy: built on first escalation
        self._fallback = None
        self._fallback_tried = False
        self.mode = (mode or "weak").lower()
        self.max_shoes = int(max_shoes)
        self.min_slivers = int(min_slivers)

    def _should_escalate(self, segs, image):
        if self.mode == "always":
            return True
        if len(segs) <= self.max_shoes:
            return True
        h, w = image.shape[:2]
        slivers = sum(1 for s in segs if _is_sliver(s, w, h))
        return slivers >= self.min_slivers

    def _fallback_seg(self):
        """Build the SAM2+gate segmenter once, on first need (keeps SAM2/gate off
        the GPU entirely until an escalation actually fires)."""
        if not self._fallback_tried:
            self._fallback_tried = True
            try:
                fb = self._build_fallback()
                self._fallback = fb if (fb is not None and fb.ready) else None
            except Exception as exc:                    # noqa: BLE001 - fail safe
                print(f"[segment] fallback build failed ({exc}); staying primary.")
                self._fallback = None
        return self._fallback

    def segment(self, image):
        primary = self.primary.segment(image)
        if not self._should_escalate(primary, image):
            return primary
        fb = self._fallback_seg()
        if fb is None:
            return primary
        alt = fb.segment(image)
        if len(alt) > len(primary):
            print(f"[segment] ESCALATED: SAM2+gate {len(alt)} > YOLOE "
                  f"{len(primary)} -> using SAM2+gate")
            return alt
        print(f"[segment] escalated but kept YOLOE ({len(primary)} >= "
              f"{len(alt)})")
        return primary

    def release(self):
        self.primary.release()
        if self._fallback is not None:
            self._fallback.release()


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
        tile_imgsz = getattr(cfg, "SEGMENT_TILE_IMGSZ", 640)
        tile_batch = getattr(cfg, "SEGMENT_TILE_BATCH", 8)
        max_side = getattr(cfg, "SEGMENT_MAX_SIDE", 0)
        seam_px = getattr(cfg, "SEGMENT_SEAM_PX", 2)
        rescue_contain = getattr(cfg, "SEGMENT_RESCUE_CONTAIN", 0.3)
        tiler = getattr(cfg, "SEGMENT_TILER", "custom").lower()
        if tiler == "sahi":
            merge = getattr(cfg, "SEGMENT_SAHI_MERGE", "NMS")
            metric = getattr(cfg, "SEGMENT_SAHI_METRIC", "IOS")
            primary = SahiTiledSegmenter(base, tile, overlap, iou_merge, merge,
                                         metric, include_full, tile_imgsz)
        else:
            primary = TiledSegmenter(base, tile, overlap, iou_merge, include_full,
                                     tile_imgsz, tile_batch, max_side,
                                     seam_px, rescue_contain)
    else:
        primary = base

    # Optional SAM2 escalation hybrid (off by default -- see config). Wrap the
    # primary so it runs every time and SAM2+gate only kicks in on weak results.
    # When off, the primary is returned untouched (zero behavior change).
    if getattr(cfg, "SEGMENT_ESCALATE_SAM2", False) and backend != "sam2":
        def _make_fallback():
            return Sam2GateSegmenter(
                getattr(cfg, "SEGMENT_ESCALATE_SAM_MODEL", "sam2_b.pt"),
                getattr(cfg, "SEGMENT_ESCALATE_GATE",
                        "dataset/gate_clf/gate_cnn.pt"),
                device,
                sam_max=getattr(cfg, "SEGMENT_ESCALATE_SAM_MAX", 1536),
                af_lo=getattr(cfg, "SEGMENT_ESCALATE_AF_LO", 0.004),
                af_hi=getattr(cfg, "SEGMENT_ESCALATE_AF_HI", 0.12),
                ar_max=getattr(cfg, "SEGMENT_ESCALATE_AR_MAX", 4.5))
        print("[segment] SAM2 escalation ENABLED (mode="
              f"{getattr(cfg, 'SEGMENT_ESCALATE_MODE', 'weak')})")
        return EscalatingSegmenter(
            primary, _make_fallback,
            mode=getattr(cfg, "SEGMENT_ESCALATE_MODE", "weak"),
            max_shoes=getattr(cfg, "SEGMENT_ESCALATE_MAX_SHOES", 28),
            min_slivers=getattr(cfg, "SEGMENT_ESCALATE_MIN_SLIVERS", 1))
    return primary
