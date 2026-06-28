"""
build_gate_data.py -- auto-label a shoe/not-shoe training set for the SAM2 gate.

Run SAM2-everything on real table photos; label each (size-filtered) mask using
the DB's known shoe boxes — NO human labeling needed:
  * center inside a known pair bbox          -> POSITIVE (shoe)
  * center OUTSIDE the occupied region       -> NEGATIVE (background: reflection,
    (the bbox enclosing all shoes, padded)      the GOLDE-MATE box, the rail, glass)
  * otherwise (in the cluster, no shoe hit)  -> SKIP (could be a missed shoe)

The "empty-zone => negative" rule is the key: a shoe YOLOE missed is still
*among* the other shoes, never out in the empty table, so we never mislabel a
real shoe as background. Plus a sample of label_data crops as extra positives.

Output: gate_data/shoe/*.jpg  +  gate_data/notshoe/*.jpg   (offline, engine-only)
"""
import os, json, glob, random, sqlite3, shutil
import cv2
import config
import segment_utils as su

OUT = "dataset/gate_data"
SHOE = f"{OUT}/shoe"; NOT = f"{OUT}/notshoe"
for d in (SHOE, NOT):
    os.makedirs(d, exist_ok=True)
TP = "../images/table_photos"
SAM_MAX = 1536
AF_LO, AF_HI, AR_MAX = 0.004, 0.12, 4.5
random.seed(42)


def sam_masks(sam, image):
    h, w = image.shape[:2]
    s = SAM_MAX / float(max(h, w)) if max(h, w) > SAM_MAX else 1.0
    img_s = cv2.resize(image, (int(w*s), int(h*s))) if s != 1.0 else image
    inv = 1.0 / s
    r = sam(img_s, verbose=False, device=su._resolve_device())[0]
    out = []
    if r.masks is None:
        return out
    for p in r.masks.xy:
        if len(p) < 3:
            continue
        x1, y1 = int(p[:,0].min()*inv), int(p[:,1].min()*inv)
        x2, y2 = int(p[:,0].max()*inv), int(p[:,1].max()*inv)
        bw, bh = x2-x1, y2-y1
        if bw <= 0 or bh <= 0:
            continue
        af = (bw*bh)/float(w*h); ar = max(bw, bh)/float(min(bw, bh))
        if af < AF_LO or af > AF_HI or ar > AR_MAX:
            continue
        out.append((x1, y1, x2, y2))
    return out


def main():
    from ultralytics import SAM
    sam = SAM("sam2_b.pt")
    conn = sqlite3.connect("file:../sneakers.db?mode=ro", uri=True)
    rows = conn.execute("""SELECT id FROM table_photos WHERE status='completed'
        AND image_path IS NOT NULL AND num_pairs>0 ORDER BY num_pairs""").fetchall()
    ids = [rows[int(i*(len(rows)-1)/59)][0] for i in range(60)]   # 60-photo spread
    ids = list(dict.fromkeys(ids))

    npos = nneg = nskip = 0
    for pid in ids:
        img = cv2.imread(f"{TP}/{pid}.jpg")
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for (bb,) in conn.execute("SELECT bbox FROM pairs WHERE table_photo_id=?", (pid,)):
            try:
                boxes.append(json.loads(bb))
            except Exception:
                pass
        if not boxes:
            continue
        # occupied region = bbox enclosing all shoes, padded 8% of the image
        ox1 = min(b[0] for b in boxes); oy1 = min(b[1] for b in boxes)
        ox2 = max(b[2] for b in boxes); oy2 = max(b[3] for b in boxes)
        mx, my = int(0.08*w), int(0.08*h)
        ox1, oy1, ox2, oy2 = ox1-mx, oy1-my, ox2+mx, oy2+my

        for i, (x1, y1, x2, y2) in enumerate(sam_masks(sam, img)):
            cx, cy = (x1+x2)//2, (y1+y2)//2
            in_shoe = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)
            in_occ = (ox1 <= cx <= ox2 and oy1 <= cy <= oy2)
            crop = img[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            if in_shoe:
                cv2.imwrite(f"{SHOE}/{pid}_{i}.jpg", crop); npos += 1
            elif not in_occ:
                cv2.imwrite(f"{NOT}/{pid}_{i}.jpg", crop); nneg += 1
            else:
                nskip += 1
        print(f"  {pid}: pos={npos} neg={nneg} skip={nskip}", end="\r")

    # extra POSITIVES from the curated label_data single-shoe crops (definitely shoes)
    extra = glob.glob(str(getattr(config, "LABEL_DATA_DIR", "label_data")) + "/*.jpg")
    random.shuffle(extra)
    for j, src in enumerate(extra[:400]):
        shutil.copyfile(src, f"{SHOE}/labeldata_{j}.jpg"); npos += 1

    print(f"\n[gate-data] DONE  shoe(pos)={npos}  notshoe(neg)={nneg}  skipped={nskip}")
    print(f"  -> {OUT}/  (run train_gate.py next)")


if __name__ == "__main__":
    main()
