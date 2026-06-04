"""
pair_utils.py -- group detected single shoes into tied pairs.

Shoes arrive tied together in pairs, so the dataset wants ONE record per pair.
The segmenter detects individual shoes cleanly (one box each); this merges the
two shoes of a pair into one combined region.

Method (simple + robust for a tidy table): consider every shoe-to-shoe pairing,
sort by center distance, and greedily lock in the closest mutual pairings whose
gap is within SEGMENT_PAIR_MAX_GAP x their average size -- so two touching shoes
pair up but the wider gap to the NEXT pair doesn't. Any leftover shoe (odd count,
or a missed detection) is kept as a single-shoe record. Heuristic by design; the
dashboard's human-confirm step is the safety net.
"""
from segment_utils import Segment


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _size(b):
    """Representative size of a box (its larger side)."""
    return max(b[2] - b[0], b[3] - b[1])


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _union(seg_a, seg_b):
    """Combine two shoe Segments into one pair Segment (union bbox)."""
    ax1, ay1, ax2, ay2 = seg_a.bbox
    bx1, by1, bx2, by2 = seg_b.bbox
    bbox = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
    # Polygon is dropped for pairs: two separate shoe contours don't combine into
    # one meaningful polygon, and the union bbox is what we crop.
    return Segment(bbox, max(seg_a.score, seg_b.score), "pair", polygon=None)


def pair_shoes(segments, max_gap_frac=1.2):
    """Return a new list where nearby shoe pairs are merged into pair Segments.

    Leftover (unpaired) shoes are returned unchanged. Order is not preserved.
    """
    n = len(segments)
    if n < 2:
        return list(segments)

    centers = [_center(s.bbox) for s in segments]
    sizes = [_size(s.bbox) for s in segments]

    # All candidate pairings within the distance threshold, closest first.
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = _dist(centers[i], centers[j])
            threshold = max_gap_frac * (sizes[i] + sizes[j]) / 2.0
            if gap <= threshold:
                candidates.append((gap, i, j))
    candidates.sort()

    used = set()
    result = []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        result.append(_union(segments[i], segments[j]))

    # Keep any shoe that didn't get paired as its own (single-shoe) record.
    for i in range(n):
        if i not in used:
            result.append(segments[i])
    return result
