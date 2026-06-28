"""
proto_sam2_pipeline.py -- PROTOTYPE: detect -> SAM2 -> pair, vs the current tiler.

Goal: test whether refining each detected shoe box with SAM2 (box-prompted) gives
cleaner per-shoe crops than the current fixed-tile + NMS pipeline — fewer slivers
and fewer "two-pairs-in-one-crop" merges — while keeping the detector's recall.

Pipeline compared, on the same photos:
  CURRENT : YOLOE (tiled) -> pair_shoes_visual -> crop padded bbox
  NEW     : YOLOE (tiled) detections -> SAM2 box-prompt -> tight mask bbox per
            shoe -> pair_shoes_visual -> crop

Offline, engine-only; writes overlays + crops to /tmp/proto_out. No live impact.
"""
import os, sys, time, glob
import cv2
import numpy as np
import config
import segment_utils as su
from pair_utils import pair_shoes_visual
from embedder_utils import build_image_embedder

config.SEGMENT_MODEL = "yoloe-11m-seg.pt"          # match production detector
OUT = "/tmp/proto_out"
os.makedirs(OUT, exist_ok=True)
PAD = getattr(config, "SEGMENT_CROP_PAD", 0.04)


def pad_box(b, w, h, frac=PAD):
    x1, y1, x2, y2 = b
    px, py = int((x2 - x1) * frac), int((y2 - y1) * frac)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def tiny(seg, w, h):
    """A 'sliver' = a crop that's too small or too thin to be a real shoe pair."""
    x1, y1, x2, y2 = seg.bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return True
    area_frac = (bw * bh) / float(w * h)
    ar = max(bw, bh) / float(min(bw, bh))
    return area_frac < 0.004 or ar > 6.0            # tiny OR extreme sliver


def sam_everything(sam, image, log):
    """SAM2 in segment-everything mode IS the detector here (higher recall than
    YOLOE on these tables). Filter the class-agnostic masks down to shoe-shaped
    ones by area + aspect ratio (drops reflections, the rail, the box)."""
    h, w = image.shape[:2]
    try:
        res = sam(image, verbose=False, device=su._resolve_device())
    except Exception as exc:
        log(f"  SAM2 everything failed ({exc})"); return []
    r = res[0]
    if r.masks is None:
        return []
    segs = []
    for p in r.masks.xy:
        if len(p) < 3:
            continue
        x1, y1, x2, y2 = int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        af = (bw * bh) / float(w * h)
        ar = max(bw, bh) / float(min(bw, bh))
        if af < 0.004 or af > 0.10 or ar > 4.0:        # keep shoe-shaped only
            continue
        segs.append(su.Segment((x1, y1, x2, y2), 1.0, "shoe", polygon=p))
    return segs


def sam_refine(sam, image, segs, log):
    """Box-prompt SAM2 with each detection; replace each box with its mask's
    tight bbox + polygon. Keeps detector recall, cleans boundaries."""
    if not segs:
        return segs
    boxes = [list(map(int, s.bbox)) for s in segs]
    try:
        res = sam(image, bboxes=boxes, verbose=False, device=su._resolve_device())
    except Exception as exc:
        log(f"  SAM2 prompt failed ({exc}); keeping raw boxes")
        return segs
    r = res[0]
    if r.masks is None:
        return segs
    polys = r.masks.xy
    out = []
    for i, s in enumerate(segs):
        if i < len(polys) and len(polys[i]):
            p = polys[i]
            x1, y1, x2, y2 = int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())
            out.append(su.Segment((x1, y1, x2, y2), s.score, s.label, polygon=p))
        else:
            out.append(s)                            # SAM gave nothing -> keep box
    return out


def crop_and_count(image, pairs, w, h, tag, photo):
    slivers = 0
    for i, seg in enumerate(pairs, 1):
        x1, y1, x2, y2 = pad_box(seg.bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            slivers += 1
            continue
        if tiny(seg, w, h):
            slivers += 1
        cv2.imwrite(f"{OUT}/{photo}_{tag}_{i}.jpg", image[y1:y2, x1:x2])
    return slivers


def overlay(image, singles, pairs, path):
    vis = image.copy()
    for s in singles:                                # detector singles (thin yellow)
        cv2.rectangle(vis, s.bbox[:2], s.bbox[2:], (0, 200, 255), 1)
    for s in pairs:                                  # final pairs (green) / singles (orange)
        col = (0, 230, 0) if getattr(s, "label", "") == "pair" else (0, 140, 255)
        cv2.rectangle(vis, s.bbox[:2], s.bbox[2:], col, 3)
    cv2.imwrite(path, cv2.resize(vis, (1280, 720)))


def main():
    photos = sys.argv[1:] or ["TBL-20260626-0066", "TBL-20260626-0067", "TBL-20260626-0065"]
    tp_dir = "../images/table_photos"

    print("[proto] loading detector + embedder + SAM2 ...")
    detector = su.build_segmenter(config)            # YOLOE tiled (current detection)
    embedder = build_image_embedder(config)
    from ultralytics import SAM
    sam = SAM("sam2_b.pt")

    def vpair(image, segs):
        return pair_shoes_visual(image, segs, embedder,
                                 spatial_weight=getattr(config, "SEGMENT_PAIR_SPATIAL_WEIGHT", 0.15),
                                 min_sim=getattr(config, "SEGMENT_PAIR_MIN_SIM", 0.5), log=lambda *_: None)

    print(f"\n{'photo':22} {'yoloeDet':>8} {'samDet':>7} {'CUR pairs/sl':>16} {'NEW pairs/sl':>16}")
    print("-" * 72)
    for name in photos:
        img = cv2.imread(f"{tp_dir}/{name}.jpg")
        if img is None:
            print(f"{name}: unreadable"); continue
        h, w = img.shape[:2]

        singles = detector.segment(img)              # detection stage (single shoes)

        # CURRENT: pair raw detections, crop padded bbox
        t0 = time.time()
        cur_pairs = vpair(img, [su.Segment(tuple(s.bbox), s.score, s.label, s.polygon) for s in singles])
        cur_sl = crop_and_count(img, cur_pairs, w, h, "CUR", name)
        cur_t = time.time() - t0

        # NEW: SAM2 everything-mode AS the detector (recall), then pair
        t0 = time.time()
        sam_singles = sam_everything(sam, img, log=print)
        new_pairs = vpair(img, sam_singles)
        new_sl = crop_and_count(img, new_pairs, w, h, "NEW", name)
        new_t = time.time() - t0
        overlay(img, sam_singles, new_pairs, f"{OUT}/{name}_NEW.jpg")
        overlay(img, singles, cur_pairs, f"{OUT}/{name}_CUR.jpg")

        cp = sum(1 for s in cur_pairs if getattr(s, "label", "") == "pair")
        npr = sum(1 for s in new_pairs if getattr(s, "label", "") == "pair")
        print(f"{name:22} {len(singles):>8} {len(sam_singles):>7} "
              f"{f'{cp}p/{cur_sl}sl':>16} {f'{npr}p/{new_sl}sl':>16}")
    print(f"\noverlays + crops -> {OUT}/   (*_CUR.jpg vs *_NEW.jpg)")
    print("PROTO DONE")


if __name__ == "__main__":
    main()
