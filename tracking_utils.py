"""
tracking_utils.py -- lightweight shoe tracking.

Each detected shoe gets a stable integer ID while it stays in frame, so the
operator's "Recycle" click sticks to the right shoe across frames. When a
shoe disappears for `config.TRACK_EXPIRATION_FRAMES` frames it is considered
"gone" and returned from expire(), so the caller can finalize any save it
owes (e.g. auto-save a Reuse shoe).

We also remember the *sharpest* frame the track has been seen in (variance
of Laplacian on its crop). save_shoe is called with that frame so motion
blur from a shoe entering/leaving the frame doesn't ruin the saved image.

Matching is greedy by IoU (Intersection-over-Union) -- no neural net, no
heavyweight tracker. That's plenty for shoes moving through a static scene.
"""
import cv2

import config


def sharpness(image):
    """Variance of the Laplacian -- higher = sharper. Returns 0 on failure.

    Cheap blur metric used to pick the best frame to save per track.
    """
    try:
        if image is None or image.size == 0:
            return 0.0
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else image)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:                              # noqa: BLE001
        return 0.0


class ShoeTrack:
    """One tracked shoe -- everything the rest of the app needs to know about it.

    Fields:
        id              unique integer ID, stays the same across frames
        bbox            latest (x1, y1, x2, y2) in pixels
        last_seen       the frame index in which this track was last matched
        status          "Reuse" by default; flipped to "Recycle" on click
        saved           True once the crop+JSON have been written to disk
        flash_until     while time.time() < this, draw red mask
        last_conf       the most recent YOLO confidence for this track
        best_frame      the sharpest full frame this shoe appeared in
        best_bbox       the bbox that corresponds to best_frame
        best_sharpness  variance-of-Laplacian score of best_frame's crop
    """

    def __init__(self, track_id, bbox, frame_idx, frame, conf):
        self.id = track_id
        self.bbox = bbox
        self.last_seen = frame_idx
        self.status = "Reuse"
        self.saved = False
        self.flash_until = 0.0
        self.last_conf = conf
        # Best (sharpest) snapshot seen so far -- initialized to this frame
        # and updated by ShoeTracker._update_best on every match.
        self.best_frame = frame
        self.best_bbox = bbox
        self.best_sharpness = 0.0
        # Bbox center the last time sharpness was recomputed; lets the tracker
        # skip redundant Laplacian work while a shoe sits still. See _update_best.
        self._last_sharp_xy = None
        # GrabCut contour (numpy array of shape (N, 1, 2)) or None. Refreshed
        # by the detector worker thread whenever it produces a new polygon.
        self.polygon = None


def iou(a, b):
    """Intersection-over-Union of two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ShoeTracker:
    """Greedy IoU tracker with per-frame update + expiration."""

    def __init__(self, expiration_frames=15, iou_threshold=0.3):
        self.tracks = {}                 # id -> ShoeTrack
        self.next_id = 1
        self.frame_idx = 0
        self.expiration_frames = expiration_frames
        self.iou_threshold = iou_threshold

    def update(self, detections, frame):
        """Advance one frame.

        detections: list of `(bbox, confidence)` or `(bbox, confidence,
                    polygon)`. bbox is (x1, y1, x2, y2). polygon, if present,
                    is the GrabCut contour for that shoe in the latest YOLO
                    frame and is attached to the matched track for drawing.
        frame:      the current full frame. Each matched track may keep this
                    frame as its `best_frame` if its crop is sharper than any
                    previous one seen for that shoe.

        Returns the list of currently active tracks.
        """
        self.frame_idx += 1
        unmatched_det = list(range(len(detections)))

        def unpack(d):
            """Allow either (bbox, conf) or (bbox, conf, polygon)."""
            if len(d) >= 3:
                return d[0], d[1], d[2]
            return d[0], d[1], None

        # Greedy match: for each existing track, take the best-IoU detection.
        # Skip saved tracks -- they're done and must not absorb new detections.
        for track in self.tracks.values():
            if track.saved:
                continue
            best_i = -1
            best_iou = 0.0
            for i in unmatched_det:
                bbox, _, _ = unpack(detections[i])
                v = iou(track.bbox, bbox)
                if v > best_iou:
                    best_iou = v
                    best_i = i
            if best_i >= 0 and best_iou >= self.iou_threshold:
                bbox, conf, polygon = unpack(detections[best_i])
                track.bbox = bbox
                track.last_seen = self.frame_idx
                track.last_conf = conf
                if polygon is not None:
                    track.polygon = polygon
                self._update_best(track, frame, bbox)
                unmatched_det.remove(best_i)

        # Anything left is a new shoe.
        for i in unmatched_det:
            bbox, conf, polygon = unpack(detections[i])
            t = ShoeTrack(self.next_id, bbox, self.frame_idx, frame, conf)
            t.polygon = polygon
            self._update_best(t, frame, bbox)
            self.tracks[self.next_id] = t
            self.next_id += 1

        return list(self.tracks.values())

    def _update_best(self, track, frame, bbox):
        """Replace track.best_frame if this frame's crop is sharper.

        Skips the (non-trivial) sharpness computation when the shoe has barely
        moved since the last check -- a still shoe's sharpness is essentially
        constant, so re-measuring it every frame is wasted CPU.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return

        # Throttle: once we already have a best frame, only recompute sharpness
        # when the bbox center has moved more than SHARPNESS_RECHECK_MIN_MOVE px.
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        min_move = getattr(config, "SHARPNESS_RECHECK_MIN_MOVE", 0)
        if (min_move > 0 and track.best_sharpness > 0
                and track._last_sharp_xy is not None):
            dx = cx - track._last_sharp_xy[0]
            dy = cy - track._last_sharp_xy[1]
            if (dx * dx + dy * dy) < (min_move * min_move):
                return
        track._last_sharp_xy = (cx, cy)

        s = sharpness(frame[y1:y2, x1:x2])
        if s > track.best_sharpness:
            track.best_sharpness = s
            track.best_frame = frame
            track.best_bbox = bbox

    def expire(self):
        """Pop tracks not seen for `expiration_frames` frames and return them.

        The caller decides what to do with each expired track -- typically, if
        it's still "Reuse" and not yet saved, save its last_frame crop.
        """
        expired = []
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if self.frame_idx - track.last_seen > self.expiration_frames:
                expired.append(self.tracks.pop(tid))
        return expired

    def find_at(self, x, y):
        """Return the most recently-created track whose bbox contains (x, y),
        or None. Used by the mouse callback to map a click to a shoe."""
        # Iterate newest-first so overlapping shoes resolve to the topmost one.
        for tid in sorted(self.tracks.keys(), reverse=True):
            track = self.tracks[tid]
            x1, y1, x2, y2 = track.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return track
        return None
