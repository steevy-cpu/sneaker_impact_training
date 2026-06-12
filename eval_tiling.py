"""
eval_tiling.py -- score the segmenter against the hand-counted ground truth.

Runs the CONFIGURED segmenter (config.py: backend, tiler, batch, cap, knobs --
override the model with --model) on every photo in tiling_gt.json and compares
detected single-shoe counts to the visually counted gt_shoes. This replaces
overlay eyeballing with numbers, so any tiling change (tile size, overlap,
merge thresholds, SAHI, conf) shows up as a measurable delta.

Counts-only on purpose: gt boxes don't exist yet, so "recall" here is
count-based -- detected/gt capped at 1.0 per photo. Over-detection (dupes,
phantom boxes) shows as detected > gt and is reported separately. Empty-table
photos score false positives.

Usage:
  python eval_tiling.py                          # gt + photos in default spots
  python eval_tiling.py --model yoloe-11s-seg.pt # match production weights
  python eval_tiling.py --viz /tmp/eval_out      # also dump overlay JPGs
  python eval_tiling.py --gt my_gt.json --dir /path/to/table_photos
"""
import argparse
import json
import os
import time

import config
import cv2
import segment_utils as su


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default="tiling_gt.json")
    ap.add_argument("--dir", default="../images/table_photos",
                    help="directory holding the ground-truth photos")
    ap.add_argument("--model", default=None,
                    help="override SEGMENT_MODEL (e.g. yoloe-11s-seg.pt)")
    ap.add_argument("--viz", default=None, help="dir for overlay JPGs")
    args = ap.parse_args()

    with open(args.gt) as fh:
        gt = json.load(fh)["photos"]

    if args.model:
        config.SEGMENT_MODEL = args.model
    seg = su.build_segmenter(config)
    if args.viz:
        os.makedirs(args.viz, exist_ok=True)

    rows = []
    t_total = 0.0
    for entry in gt:
        path = os.path.join(args.dir, entry["image"])
        image = cv2.imread(path)
        if image is None:
            print(f"  SKIP unreadable: {entry['image']}")
            continue
        t0 = time.time()
        segs = seg.segment(image)
        dt = time.time() - t0
        t_total += dt
        det, want = len(segs), entry["gt_shoes"]
        rows.append((entry, det, dt))
        flag = "" if entry["gt_quality"] == "confirmed" else " (est)"
        print(f"  {entry['image']:26} det={det:3d} gt={want:3d}{flag}  "
              f"{dt:5.1f}s  {entry.get('notes', '')[:48]}")
        if args.viz:
            vis = image.copy()
            for s in segs:
                cv2.rectangle(vis, s.bbox[:2], s.bbox[2:], (0, 230, 0),
                              max(2, image.shape[1] // 500))
            cv2.imwrite(os.path.join(args.viz, entry["image"]), vis)

    # Score. Count-recall per photo = min(det, gt)/gt; over-detection reported
    # apart so dupes can't masquerade as recall. Empties contribute only FPs.
    nonempty = [(e, d) for (e, d, _) in rows if e["gt_shoes"] > 0]
    empties = [(e, d) for (e, d, _) in rows if e["gt_shoes"] == 0]
    recalls = [min(d, e["gt_shoes"]) / e["gt_shoes"] for e, d in nonempty]
    overs = [max(0, d - e["gt_shoes"]) for e, d in nonempty]
    fps = sum(d for _, d in empties)

    print("\n" + "=" * 64)
    print(f"TILING EVAL -- {len(nonempty)} photos + {len(empties)} empty controls")
    print("=" * 64)
    if recalls:
        mean_r = sum(recalls) / len(recalls)
        worst = min(zip(recalls, nonempty), key=lambda z: z[0])
        print(f"  count-recall   mean={mean_r:.2f}  "
              f"min={worst[0]:.2f} ({worst[1][0]['image']})")
        print(f"  over-detection total={sum(overs)} across "
              f"{sum(1 for o in overs if o)} photo(s)")
    print(f"  false positives on empty tables: {fps}")
    print(f"  wall time: {t_total:.1f}s total, "
          f"{t_total / max(1, len(rows)):.1f}s/photo")
    print("\n  gt counts marked (est) are Claude's visual counts (+/-2, dense "
          "+/-4);\n  correct tiling_gt.json and set gt_quality='confirmed' as "
          "they're verified.")


if __name__ == "__main__":
    main()
