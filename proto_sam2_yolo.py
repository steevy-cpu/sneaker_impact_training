"""
proto_sam2_yolo.py -- PROTOTYPE of the user's idea:
  SAM2 everything (recall + clean masks)  ->  YOLO verify each crop is a shoe
  ->  drop crops with NO shoe (the reflections / box / rail)  ->  pair.

Key bet: YOLO is weak at DETECTING small shoes in a crowded table, but strong at
CONFIRMING a single, tightly-cropped object IS a shoe — an easier task. So we let
SAM2 find everything, then use YOLO as a shoe/not-shoe gate.

Shows exactly what gets DROPPED (red) vs kept (green/orange) so we can check it
only removes junk. Offline, engine-only. Output -> /tmp/proto_yolo_out.
"""
import os, sys, time
import cv2
import config
import segment_utils as su
from pair_utils import pair_shoes_visual
from embedder_utils import build_image_embedder

config.SEGMENT_MODEL = "yoloe-11m-seg.pt"
OUT = "/tmp/proto_yolo_out"; os.makedirs(OUT, exist_ok=True)
PAD = 0.06
VERIFY_CONF = 0.10            # keep a crop if YOLO finds a shoe at >= this conf


def pad(b, w, h, frac=PAD):
    x1, y1, x2, y2 = b
    px, py = int((x2 - x1) * frac), int((y2 - y1) * frac)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def sam_everything(sam, image):
    h, w = image.shape[:2]
    r = sam(image, verbose=False, device=su._resolve_device())[0]
    segs = []
    if r.masks is None:
        return segs
    for p in r.masks.xy:
        if len(p) < 3:
            continue
        x1, y1, x2, y2 = int(p[:,0].min()), int(p[:,1].min()), int(p[:,0].max()), int(p[:,1].max())
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        af = (bw * bh) / float(w * h); ar = max(bw, bh) / float(min(bw, bh))
        if af < 0.002 or af > 0.15 or ar > 6.0:    # loose pre-filter; YOLO does the real gating
            continue
        segs.append(su.Segment((x1, y1, x2, y2), 1.0, "shoe", polygon=p))
    return segs


def main():
    photos = sys.argv[1:] or ["TBL-20260626-0066", "TBL-20260626-0067", "TBL-20260626-0065"]
    tp = "../images/table_photos"
    print("[proto] loading SAM2 + YOLO verifier + embedder ...")
    from ultralytics import SAM
    sam = SAM("sam2_b.pt")
    dev = su._resolve_device()
    # YOLO verifier: plain YOLOE (no tiling), used only to confirm a crop is a shoe.
    yolo = su.YoloeSegmenter("yoloe-11m-seg.pt", getattr(config, "SEGMENT_PROMPTS", ["shoe", "sneaker"]),
                             VERIFY_CONF, dev, imgsz=640)
    embedder = build_image_embedder(config)

    print(f"\n{'photo':22} {'samMasks':>8} {'kept':>5} {'dropped':>7} {'pairs':>5}")
    print("-" * 54)
    for name in photos:
        img = cv2.imread(f"{tp}/{name}.jpg")
        if img is None:
            print(f"{name}: unreadable"); continue
        h, w = img.shape[:2]

        masks = sam_everything(sam, img)
        crops = [img[pad(s.bbox, w, h)[1]:pad(s.bbox, w, h)[3],
                     pad(s.bbox, w, h)[0]:pad(s.bbox, w, h)[2]] for s in masks]
        # YOLO gate, batched: keep a mask if YOLO finds >=1 shoe in its crop.
        verdicts = yolo.segment_batch(crops)
        kept, dropped = [], []
        for s, dets in zip(masks, verdicts):
            (kept if dets else dropped).append(s)

        pairs = pair_shoes_visual(img, kept, embedder,
                                  spatial_weight=getattr(config, "SEGMENT_PAIR_SPATIAL_WEIGHT", 0.15),
                                  min_sim=getattr(config, "SEGMENT_PAIR_MIN_SIM", 0.5), log=lambda *_: None)
        npairs = sum(1 for s in pairs if getattr(s, "label", "") == "pair")

        vis = img.copy()
        for s in dropped:                                    # DROPPED -> red (should be junk)
            cv2.rectangle(vis, s.bbox[:2], s.bbox[2:], (0, 0, 255), 3)
        for s in pairs:
            col = (0, 230, 0) if getattr(s, "label", "") == "pair" else (0, 140, 255)
            cv2.rectangle(vis, s.bbox[:2], s.bbox[2:], col, 3)
        cv2.imwrite(f"{OUT}/{name}.jpg", cv2.resize(vis, (1280, 720)))
        print(f"{name:22} {len(masks):>8} {len(kept):>5} {len(dropped):>7} {npairs:>5}")
    print(f"\noverlays -> {OUT}/  (red = dropped by YOLO gate, green = pairs, orange = singles)")
    print("PROTO DONE")


if __name__ == "__main__":
    main()
