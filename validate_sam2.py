"""
validate_sam2.py -- validate SAM2-everything + size-filter vs the current YOLOE
tiler, at scale.

Part A (RIGOROUS): detection count-recall vs the 13 hand-counted ground-truth
photos in tiling_gt.json (gt_shoes). The real recall number.

Part B (AT SCALE): on a density-spread sample of completed boxes, compare
pairs / slivers / time for CURRENT (YOLOE tiled -> pair) vs NEW (SAM2 + size
filter -> pair). Saves a few overlays to spot-check junk.

Offline, engine-only. Shares the GPU with the live engine, so it's bounded.
"""
import os, sys, time, json, sqlite3
import cv2
import config
import segment_utils as su
from pair_utils import pair_shoes_visual
from embedder_utils import build_image_embedder

config.SEGMENT_MODEL = "yoloe-11m-seg.pt"
TP = "../images/table_photos"
OUT = "/tmp/validate_out"; os.makedirs(OUT, exist_ok=True)

# Size filter for SAM masks (area fraction of image + aspect ratio).
AF_LO, AF_HI, AR_MAX = 0.004, 0.12, 4.5


SAM_MAX_SIDE = 1536          # SAM2 everything-mode is VRAM-heavy; cap input size


def sam_detect(sam, image):
    h, w = image.shape[:2]
    # Downscale big photos before SAM2 (it has no internal cap -> OOMs on the
    # 8000x6000 legacy shot). Boxes are mapped back to ORIGINAL coords.
    s = 1.0
    if max(h, w) > SAM_MAX_SIDE:
        s = SAM_MAX_SIDE / float(max(h, w))
        img_s = cv2.resize(image, (max(1, int(w*s)), max(1, int(h*s))))
    else:
        img_s = image
    inv = 1.0 / s
    r = sam(img_s, verbose=False, device=su._resolve_device())[0]
    segs = []
    if r.masks is None:
        return segs
    for p in r.masks.xy:
        if len(p) < 3:
            continue
        x1 = int(p[:,0].min()*inv); y1 = int(p[:,1].min()*inv)
        x2 = int(p[:,0].max()*inv); y2 = int(p[:,1].max()*inv)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        af = (bw*bh)/float(w*h); ar = max(bw,bh)/float(min(bw,bh))
        if af < AF_LO or af > AF_HI or ar > AR_MAX:
            continue
        segs.append(su.Segment((x1,y1,x2,y2), 1.0, "shoe", polygon=None))
    return segs


def tiny(seg, w, h):
    x1,y1,x2,y2 = seg.bbox; bw,bh = x2-x1, y2-y1
    if bw<=0 or bh<=0: return True
    return (bw*bh)/float(w*h) < 0.004 or max(bw,bh)/float(min(bw,bh)) > 6.0


def main():
    print("[val] loading YOLOE + SAM2 + DINOv2 ...")
    detector = su.build_segmenter(config)                       # YOLOE tiled (current)
    embedder = build_image_embedder(config)
    from ultralytics import SAM
    sam = SAM("sam2_b.pt")

    def vpair(image, segs):
        return pair_shoes_visual(image, segs, embedder,
            spatial_weight=getattr(config,"SEGMENT_PAIR_SPATIAL_WEIGHT",0.15),
            min_sim=getattr(config,"SEGMENT_PAIR_MIN_SIM",0.5), log=lambda *_: None)

    # ---------- Part A: detection recall vs ground truth ----------
    print("\n===== PART A: detection count-recall vs hand-counted gt (tiling_gt.json) =====")
    gt = json.load(open("tiling_gt.json"))["photos"]
    yo_rec, sm_rec = [], []
    print(f"{'photo':26} {'gt':>3} {'yoloe':>6} {'sam2':>5}")
    for e in gt:
        p = f"{TP}/{e['image']}"
        img = cv2.imread(p)
        if img is None or e["gt_shoes"] == 0:
            continue
        g = e["gt_shoes"]
        nyo = len(detector.segment(img))
        nsm = len(sam_detect(sam, img))
        yo_rec.append(min(nyo,g)/g); sm_rec.append(min(nsm,g)/g)
        print(f"{e['image']:26} {g:>3} {nyo:>6} {nsm:>5}")
    if yo_rec:
        print(f"\nMEAN count-recall:  YOLOE {sum(yo_rec)/len(yo_rec):.3f}   "
              f"SAM2+filter {sum(sm_rec)/len(sm_rec):.3f}   (n={len(yo_rec)})")

    # ---------- Part B: pairs / slivers / time at scale ----------
    conn = sqlite3.connect("file:../sneakers.db?mode=ro", uri=True)
    rows = conn.execute("""SELECT id, num_pairs FROM table_photos
        WHERE status='completed' AND image_path IS NOT NULL AND num_pairs>0
        ORDER BY num_pairs""").fetchall()
    # density-spread sample of 20
    ids = [rows[int(i*(len(rows)-1)/19)][0] for i in range(20)] if len(rows) >= 20 else [r[0] for r in rows]
    ids = list(dict.fromkeys(ids))

    print(f"\n===== PART B: pairs/slivers/time on {len(ids)} density-spread boxes =====")
    print(f"{'photo':22} {'CURpairs':>8} {'NEWpairs':>8} {'CURsl':>6} {'NEWsl':>6} {'CURt':>5} {'NEWt':>5}")
    agg = {"cp":0,"np":0,"csl":0,"nsl":0,"ct":0.0,"nt":0.0,"more":0,"n":0}
    for k, pid in enumerate(ids):
        img = cv2.imread(f"{TP}/{pid}.jpg")
        if img is None: continue
        h, w = img.shape[:2]
        t0=time.time(); cur = vpair(img, detector.segment(img)); ct=time.time()-t0
        t0=time.time(); new = vpair(img, sam_detect(sam, img)); nt=time.time()-t0
        cp = sum(1 for s in cur if getattr(s,"label","")=="pair")
        np_ = sum(1 for s in new if getattr(s,"label","")=="pair")
        csl = sum(1 for s in cur if tiny(s,w,h)); nsl = sum(1 for s in new if tiny(s,w,h))
        agg["cp"]+=cp; agg["np"]+=np_; agg["csl"]+=csl; agg["nsl"]+=nsl
        agg["ct"]+=ct; agg["nt"]+=nt; agg["n"]+=1; agg["more"] += 1 if np_>cp else 0
        print(f"{pid:22} {cp:>8} {np_:>8} {csl:>6} {nsl:>6} {ct:>4.0f}s {nt:>4.0f}s")
        if k < 6:                                                # overlays to eyeball junk
            vis=img.copy()
            for s in new:
                col=(0,230,0) if getattr(s,"label","")=="pair" else (0,140,255)
                cv2.rectangle(vis,s.bbox[:2],s.bbox[2:],col,3)
            cv2.imwrite(f"{OUT}/{pid}_NEW.jpg", cv2.resize(vis,(1280,720)))
    a=agg
    print(f"\nTOTALS over {a['n']} boxes:")
    print(f"  pairs:   CURRENT {a['cp']}   NEW {a['np']}   (NEW found more on {a['more']}/{a['n']} boxes)")
    print(f"  slivers: CURRENT {a['csl']}  NEW {a['nsl']}")
    print(f"  time/box: CURRENT {a['ct']/a['n']:.1f}s   NEW {a['nt']/a['n']:.1f}s")
    print(f"\noverlays -> {OUT}/  ; VALIDATE DONE")


if __name__ == "__main__":
    main()
