"""
tracking_utils.py -- lightweight shoe tracking.

Each detected shoe gets a stable integer ID while it stays in frame, so the
operator's double-click "Recycle" decision sticks to the right shoe across
frames. When a shoe disappears for `config.TRACK_EXPIRATION_FRAMES` frames it
is considered "gone" and returned from expire(), so the caller can finalize
any save it owes (e.g. auto-save a Reuse shoe).

Matching is greedy by IoU (Intersection-over-Union) -- no neural net, no
heavyweight tracker. That's plenty for shoes moving through a static scene.
"""

class ShoeTrack:
    """One tracked shoe -- everything the rest of the app needs to know about it.

    Fields:
        id          unique integer ID, stays the same across frames
        bbox        latest (x1, y1, x2, y2) in pixels
        last_seen   the frame index in which this track was last matched
        status      "Reuse" by default; flipped to "Recycle" by a double-click
        saved       True once the crop+JSON have been written to disk
        flash_until time.time() value: while now() < flash_until, draw red mask
        last_frame  the most recent full frame this shoe appeared in (for save)
        last_conf   the most recent YOLO confidence for this track
    """

    def __init__(self, track_id, bbox, frame_idx, frame, conf):
        self.id = track_id
        self.bbox = bbox
        self.last_seen = frame_idx
        self.status = "Reuse"
        self.saved = False
        self.flash_until = 0.0
        self.last_frame = frame
        self.last_conf = conf


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

        detections: list of (bbox, confidence). bbox is (x1, y1, x2, y2).
        frame:      the current full frame (kept on each matched track so we
                    can save its last good crop if it later expires as Reuse).

        Returns the list of currently active tracks.
        """
        self.frame_idx += 1
        unmatched_det = list(range(len(detections)))

        # Greedy match: for each existing track, take the best-IoU detection.
        for track in self.tracks.values():
            best_i = -1
            best_iou = 0.0
            for i in unmatched_det:
                bbox, _ = detections[i]
                v = iou(track.bbox, bbox)
                if v > best_iou:
                    best_iou = v
                    best_i = i
            if best_i >= 0 and best_iou >= self.iou_threshold:
                bbox, conf = detections[best_i]
                track.bbox = bbox
                track.last_seen = self.frame_idx
                track.last_frame = frame
                track.last_conf = conf
                unmatched_det.remove(best_i)

        # Anything left is a new shoe.
        for i in unmatched_det:
            bbox, conf = detections[i]
            t = ShoeTrack(self.next_id, bbox, self.frame_idx, frame, conf)
            self.tracks[self.next_id] = t
            self.next_id += 1

        return list(self.tracks.values())

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
