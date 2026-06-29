"""
pair_utils.py -- group detected single shoes into pairs.

The segmenter detects individual shoes cleanly (one box each); this merges the
two shoes of a pair into one combined region (one record per pair).

Two methods:

  "visual"   (pair_shoes_visual) -- pair by APPEARANCE. Each shoe crop is turned
             into a DINOv2/CLIP embedding (via embedder_utils, the same embedder
             the model-ID index uses); the two shoes whose embeddings are most
             similar are matched (Hungarian assignment), with a light spatial
             tiebreak so closeness on the table only decides near-ties. Shoes do
             NOT need to be tied or placed next to each other -- workers can just
             lay them on the table. Unmatched / low-similarity shoes are kept as
             single-shoe records.

  "geometry" (pair_shoes) -- legacy nearest-neighbour: pair the two closest shoes
             whose center gap is within SEGMENT_PAIR_MAX_GAP x their average size.
             Simple and fast, but effectively needs tied / adjacent pairs.

Both leave odd / unmatched shoes as singles, and the dashboard's human-confirm
step is the safety net for mis-pairs.
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
    # The single combined polygon is dropped (two contours don't merge into one),
    # but keep BOTH members' polygons + boxes so the crop step can white-out the
    # background to just these two shoes -- even if a 3rd shoe sits in the union
    # box, it gets masked away. None entries are fine (whitening just skips them).
    merged = Segment(bbox, max(seg_a.score, seg_b.score), "pair", polygon=None)
    merged.member_polys = [seg_a.polygon, seg_b.polygon]
    merged.member_boxes = [seg_a.bbox, seg_b.bbox]
    return merged


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


def _crop(image, bbox):
    """Clip bbox to the image and return the crop, or None if degenerate."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def pair_shoes_hybrid(image, segments, embedder,
                      max_gap_frac=1.2, veto_min=0.25, rescue_min=0.80,
                      max_dist_frac=0.20, log=None):
    """Pair shoes by ADJACENCY first, with appearance as a sanity check.

    Measured reality (TBL-20260611-0015): DINOv2 cosine ranks silhouette over
    identity — two identical white mates scored 0.83 while a *different* gray
    shoe scored 0.87 against one of them, and true mates in different poses
    (stacked slip-ons / Converse) scored only 0.41-0.49. No cosine threshold
    separates mates from strangers. What does separate them: workers place a
    pair's two shoes touching/stacked on the table.

    Algorithm:
      1. Geometric matching (closest-first 1:1 within max_gap_frac x size),
         but VETO a candidate whose crops' cosine < veto_min — that catches
         a shoe sitting next to an unrelated object/box.
      2. Visual rescue for leftovers: Hungarian on cosine, accepted only at
         cos >= rescue_min — pairs split far apart must look near-identical.
    Falls back to pure geometry when the embedder is unavailable."""
    n = len(segments)
    if n < 2:
        return list(segments)
    if embedder is None or not getattr(embedder, "ok", False):
        if log:
            log("[pair] no embedder -> geometric fallback")
        return pair_shoes(segments, max_gap_frac)

    import numpy as np

    embs = [None] * n
    for i, s in enumerate(segments):
        crop = _crop(image, s.bbox)
        if crop is None:
            continue
        try:
            embs[i] = embedder.embed(crop)               # L2-normalized vector
        except Exception:                                # noqa: BLE001 - fail safe
            embs[i] = None

    def _cos(i, j):
        if embs[i] is None or embs[j] is None:
            return None                                  # no veto possible
        return float(np.dot(embs[i], embs[j]))

    centers = [_center(s.bbox) for s in segments]
    sizes = [_size(s.bbox) for s in segments]

    # Stage 1: adjacency, closest first, appearance veto.
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = _dist(centers[i], centers[j])
            if gap <= max_gap_frac * (sizes[i] + sizes[j]) / 2.0:
                candidates.append((gap, i, j))
    candidates.sort()

    used, result = set(), []
    for gap, i, j in candidates:
        if i in used or j in used:
            continue
        cos = _cos(i, j)
        if cos is not None and cos < veto_min:
            if log:
                log(f"[pair] {i}+{j} adjacent but cos={cos:.3f} < {veto_min} -> VETO")
            continue
        used.update((i, j))
        merged = _union(segments[i], segments[j])
        merged.pair_score = cos
        result.append(merged)
        if log:
            log(f"[pair] {i}+{j} gap={gap:.0f} cos={'n/a' if cos is None else f'{cos:.3f}'} -> PAIR (adjacent)")

    # Stage 2: visual rescue for separated mates — near-identical looks only.
    # Capped by distance: without it, two look-alike shoes on opposite ends of
    # the table get "rescued" into one pair whose union crop spans the whole
    # table (and is almost surely a wrong match). max_dist_frac<=0 disables it.
    diag = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5 or 1.0
    left = [i for i in range(n) if i not in used and embs[i] is not None]
    rescue = []
    for a in range(len(left)):
        for b in range(a + 1, len(left)):
            i, j = left[a], left[b]
            if max_dist_frac > 0 and _dist(centers[i], centers[j]) / diag > max_dist_frac:
                continue
            cos = _cos(i, j)
            if cos is not None and cos >= rescue_min:
                rescue.append((cos, i, j))
    rescue.sort(reverse=True)
    for cos, i, j in rescue:
        if i in used or j in used:
            continue
        used.update((i, j))
        merged = _union(segments[i], segments[j])
        merged.pair_score = cos
        result.append(merged)
        if log:
            log(f"[pair] {i}+{j} cos={cos:.3f} -> PAIR (visual rescue)")

    singles = 0
    for i in range(n):
        if i not in used:
            result.append(segments[i])
            singles += 1
    if log:
        log(f"[pair] hybrid: {n} shoes -> {len(result) - singles} pairs "
            f"+ {singles} singles (gap<={max_gap_frac}, veto<{veto_min}, "
            f"rescue>={rescue_min})")
    return result


def pair_shoes_visual(image, segments, embedder,
                      spatial_weight=0.15, min_sim=0.5, max_dist_frac=0.20,
                      log=None):
    """Pair shoes by visual similarity so they need NOT be tied/adjacent.

    For each detected shoe we embed its crop, then find the globally-best 1:1
    matching (Hungarian) on a blended score:

        score(i, j) = cosine(emb_i, emb_j) - spatial_weight * spatial_dist_norm

    where spatial_dist_norm is the center distance over the image diagonal (so a
    small, soft tiebreak). A pair is accepted only if score >= min_sim AND the two
    shoes' centers are within max_dist_frac of the image diagonal; otherwise both
    shoes fall through to singles. Shoes that can't be embedded are singles.

    The distance cap is essential: the crop of a pair is the UNION bbox of its two
    shoes, so two visually-similar shoes matched across the table produce a giant
    crop that swallows every shoe between them (and is almost certainly a wrong
    match anyway -- real mates are placed near each other even when not tied).
    max_dist_frac<=0 disables the cap (old behavior).

    Degrades safely to the geometric pair_shoes() if the embedder is missing or
    fewer than two shoes embed. Returns the same shape as pair_shoes(): a list of
    Segments (union bbox for pairs, originals for singles)."""
    n = len(segments)
    if n < 2:
        return list(segments)
    if embedder is None or not getattr(embedder, "ok", False):
        if log:
            log("[pair] no embedder -> geometric fallback")
        return pair_shoes(segments)

    import numpy as np

    # Embed each shoe crop (None where the crop is bad or embedding fails).
    embs = [None] * n
    ok_idx = []
    for i, s in enumerate(segments):
        crop = _crop(image, s.bbox)
        if crop is None:
            continue
        try:
            embs[i] = embedder.embed(crop)               # L2-normalized vector
            ok_idx.append(i)
        except Exception:                                # noqa: BLE001 - fail safe
            embs[i] = None
    if len(ok_idx) < 2:
        if log:
            log("[pair] <2 shoes embedded -> geometric fallback")
        return pair_shoes(segments)

    centers = [_center(s.bbox) for s in segments]
    diag = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5 or 1.0

    # Blended-score matrix over the embeddable shoes (local indices a,b -> ok_idx).
    m = len(ok_idx)
    NEG = -1e9
    sim = np.full((m, m), NEG, dtype=np.float64)
    cosm = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        for b in range(a + 1, m):
            i, j = ok_idx[a], ok_idx[b]
            cos = float(np.dot(embs[i], embs[j]))        # both unit vectors
            sdist = min(_dist(centers[i], centers[j]) / diag, 1.0)
            cosm[a, b] = cosm[b, a] = cos
            # Hard distance cap: leave NEG so the matcher never pairs two shoes
            # across the table (giant union crop + almost surely a wrong match).
            # Each shoe is then matched to its best partner WITHIN range instead
            # of being dropped to a single. max_dist_frac<=0 disables the cap.
            if max_dist_frac > 0 and sdist > max_dist_frac:
                continue
            sim[a, b] = sim[b, a] = cos - spatial_weight * sdist

    # Globally-best 1:1 matching (Hungarian); greedy fallback if scipy is absent.
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-sim)
        assigns = [(sim[r, c], cosm[r, c], r, c)
                   for r, c in zip(rows, cols) if r != c]
    except Exception:                                    # noqa: BLE001 - fallback
        assigns = [(sim[a, b], cosm[a, b], a, b)
                   for a in range(m) for b in range(a + 1, m)]
    assigns.sort(reverse=True)                           # best score first

    used_local, used_global, result = set(), set(), []
    for score, cos, a, b in assigns:
        if a in used_local or b in used_local:
            continue
        if score < min_sim:
            continue                # both -> singles (incl. distance-capped NEG)
        used_local.update((a, b))
        i, j = ok_idx[a], ok_idx[b]
        used_global.update((i, j))
        merged = _union(segments[i], segments[j])
        merged.pair_score = float(cos)        # visual similarity of the two shoes
        result.append(merged)
        if log:
            log(f"[pair] {i}+{j} cos={cos:.3f} score={score:.3f} -> PAIR")

    singles = 0
    for i in range(n):
        if i not in used_global:
            result.append(segments[i])
            singles += 1
    if log:
        log(f"[pair] visual: {n} shoes -> {len(result) - singles} pairs "
            f"+ {singles} singles (min_sim={min_sim}, spatial_w={spatial_weight})")
    return result
