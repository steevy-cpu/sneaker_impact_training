"""
ab_tiling.py -- A/B the CUSTOM tiler vs the SAHI tiler on real table photos.

Both tilers wrap the SAME YOLOE base model (same weights, prompts, device, tile
size, tile imgsz, and whole-image pass), so any difference in the number/quality
of detected shoes is the *tiling logic* alone -- exactly what we want to compare.

It runs three configurations per photo:
  custom      -- TiledSegmenter        (greedy IoU + containment NMS)
  sahi-NMS    -- SahiTiledSegmenter     (SAHI slicing + NMS suppress)
  sahi-NMM    -- SahiTiledSegmenter     (SAHI slicing + NMM union-merge)

Usage:
  python ab_tiling.py --dir ../images/table_photos -n 12
  python ab_tiling.py table1.jpg table2.jpg --viz ab_out
  python ab_tiling.py --dir ../images/table_photos -n 8 --viz ab_out

Output: a per-photo count table + a summary, and (with --viz) side-by-side
overlay JPGs so you can eyeball recall/precision, not just counts.
"""
import argparse
import glob
import os
import time

import config
import cv2
import segment_utils as su


# Variants to compare. Each is (label, builder(base) -> Segmenter).
def _build_variants(base, tile, overlap, iou, include_full, tile_imgsz):
    return [
        ("custom", su.TiledSegmenter(base, tile, overlap, iou, include_full,
                                     tile_imgsz)),
        ("sahi-NMS", su.SahiTiledSegmenter(base, tile, overlap, iou, "NMS",
                                           "IOS", include_full, tile_imgsz)),
        ("sahi-NMM", su.SahiTiledSegmenter(base, tile, overlap, iou, "NMM",
                                           "IOS", include_full, tile_imgsz)),
    ]


_COLORS = {                      # BGR per variant for the viz overlays
    "custom":   (0, 200, 255),
    "sahi-NMS": (0, 230, 0),
    "sahi-NMM": (255, 120, 0),
}


def _draw(image, segs, color):
    out = image.copy()
    for s in segs:
        x1, y1, x2, y2 = s.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    return out


def _label_bar(img, text, color):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", help="table photo path(s)")
    ap.add_argument("--dir", help="directory of table photos to sample from")
    ap.add_argument("-n", type=int, default=10, help="max photos (with --dir)")
    ap.add_argument("--model", default=None,
                    help="YOLOE weights (default: config / yoloe-11s-seg.pt to "
                         "match production)")
    ap.add_argument("--viz", default=None,
                    help="dir to write side-by-side overlay JPGs")
    args = ap.parse_args()

    # Collect images.
    paths = list(args.images)
    if args.dir:
        found = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
        paths.extend(found[: args.n])
    if not paths:
        ap.error("no images: pass paths or --dir")

    # Build ONE base segmenter (load the model once), no tiling wrapper.
    model_path = args.model or getattr(config, "ENGINE_SEGMENT_MODEL", None) \
        or "yoloe-11s-seg.pt"
    prompts = getattr(config, "SEGMENT_PROMPTS", ["shoe"])
    conf = getattr(config, "SEGMENT_CONF", 0.25)
    imgsz = getattr(config, "SEGMENT_IMGSZ", 1280)
    device = su._resolve_device()
    print(f"[ab] base: YOLOE {model_path} on {device}, prompts={prompts}, "
          f"conf={conf}, full-imgsz={imgsz}")
    base = su.YoloeSegmenter(model_path, prompts, conf, device, imgsz)
    if base.model is None:
        raise SystemExit("[ab] base model failed to load -- aborting.")

    tile = getattr(config, "SEGMENT_TILE", 512)
    overlap = getattr(config, "SEGMENT_TILE_OVERLAP", 0.25)
    iou = getattr(config, "SEGMENT_TILE_IOU", 0.4)
    include_full = getattr(config, "SEGMENT_TILE_INCLUDE_FULL", True)
    tile_imgsz = getattr(config, "SEGMENT_TILE_IMGSZ", 640)
    print(f"[ab] tiling: tile={tile} overlap={overlap} iou={iou} "
          f"tile-imgsz={tile_imgsz} include_full={include_full}\n")

    variants = _build_variants(base, tile, overlap, iou, include_full,
                               tile_imgsz)
    if args.viz:
        os.makedirs(args.viz, exist_ok=True)

    # Warm up the GPU/model once so timings below are steady-state.
    warm = cv2.imread(paths[0])
    if warm is not None:
        base.segment(warm)

    labels = [lbl for lbl, _ in variants]
    totals = {lbl: 0 for lbl in labels}
    times = {lbl: 0.0 for lbl in labels}
    rows = []
    for p in paths:
        image = cv2.imread(p)
        if image is None:
            print(f"[ab] skip unreadable: {p}")
            continue
        counts, panels = {}, []
        for lbl, seg in variants:
            t0 = time.time()
            segs = seg.segment(image)
            dt = time.time() - t0
            counts[lbl] = len(segs)
            totals[lbl] += len(segs)
            times[lbl] += dt
            if args.viz:
                panel = _draw(image, segs, _COLORS[lbl])
                _label_bar(panel, f"{lbl}: {len(segs)}  ({dt:.1f}s)",
                           _COLORS[lbl])
                panels.append(panel)
        rows.append((os.path.basename(p), counts))
        cs = "  ".join(f"{lbl}={counts[lbl]}" for lbl in labels)
        print(f"  {os.path.basename(p):28} {cs}")
        if args.viz and panels:
            side = cv2.hconcat([cv2.resize(pn, (960, 540)) for pn in panels])
            out = os.path.join(args.viz, os.path.basename(p))
            cv2.imwrite(out, side)

    # Summary.
    n = len(rows)
    print("\n" + "=" * 60)
    print(f"A/B SUMMARY over {n} photo(s)")
    print("=" * 60)
    base_lbl = "custom"
    for lbl in labels:
        avg = totals[lbl] / n if n else 0
        ms = (times[lbl] / n * 1000) if n else 0
        delta = totals[lbl] - totals[base_lbl]
        dtxt = "" if lbl == base_lbl else f"  (vs custom: {delta:+d} total)"
        print(f"  {lbl:10} total={totals[lbl]:4d}  avg={avg:5.1f}/photo  "
              f"{ms:6.0f} ms/photo{dtxt}")
    # How often each sahi variant differs from custom (recall signal).
    for lbl in labels:
        if lbl == base_lbl:
            continue
        more = sum(1 for _, c in rows if c[lbl] > c[base_lbl])
        fewer = sum(1 for _, c in rows if c[lbl] < c[base_lbl])
        same = n - more - fewer
        print(f"  {lbl:10} vs custom: more on {more}, fewer on {fewer}, "
              f"same on {same}")
    if args.viz:
        print(f"\n  overlays written to: {args.viz}/  (left->right: {labels})")


if __name__ == "__main__":
    main()
