"""
dataset_review.py -- interactive visual dataset reviewer.

Shows each saved shoe crop one at a time. The operator can keep it, delete it,
or flip its label (Reuse <-> Recycle). All changes are applied immediately.

Usage:
    python dataset_review.py                        # all incoming* folders
    python dataset_review.py --folder incoming05292026
    python dataset_review.py --only Recycle         # one classification only
    python dataset_review.py --color brown          # one color only
    python dataset_review.py --max-sharp 60         # only shoes blurrier than 60

Controls:
    SPACE or → (right arrow)  keep and go to next
    D                         delete this shoe (jpg + json removed)
    R                         relabel (flip Reuse <-> Recycle, rename files)
    ← (left arrow)            go back to previous shoe
    Q or ESC                  quit
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

import config
from dataset_utils import find_folders, load_entries
from image_utils import sharpness as blur_score

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)

DISPLAY_W = 800     # review window width
PANEL_H = 160       # height of the info panel below the image


def apply_filters(entries, args):
    """Narrow the entry list by --only / --color / --max-sharp.

    For --max-sharp, prefer the sharpness stored in metadata; fall back to
    measuring the image only when it wasn't recorded.
    """
    out = []
    for e in entries:
        m = e["meta"]
        if args.only and m.get("classification") != args.only:
            continue
        if args.color and (m.get("detected_color") or "") != args.color:
            continue
        if args.max_sharp is not None:
            s = m.get("sharpness")
            if s is None:                          # not recorded -> measure now
                img = cv2.imread(e["jpg"])
                s = blur_score(img) if img is not None else 0.0
            if s >= args.max_sharp:
                continue
        out.append(e)
    return out


def render(entry, idx, total):
    """Build the review frame: scaled image + info panel."""
    img = cv2.imread(entry["jpg"])
    if img is None:
        img = np.zeros((200, 300, 3), np.uint8)
        cv2.putText(img, "Cannot load image", (10, 100), FONT, 0.7, RED, 2)

    # Scale image to fit display width while keeping aspect ratio.
    h, w = img.shape[:2]
    scale = DISPLAY_W / w
    img_disp = cv2.resize(img, (DISPLAY_W, int(h * scale)))

    # Info panel
    panel = np.zeros((PANEL_H, DISPLAY_W, 3), np.uint8)
    panel[:] = (30, 30, 30)

    meta = entry["meta"]
    classification = meta.get("classification", "?")
    conf = meta.get("yolo_confidence", 0.0)
    color = meta.get("detected_color") or "?"
    color_conf = meta.get("color_confidence") or 0.0
    blur = blur_score(img)
    folder_name = os.path.basename(entry["folder"])

    label_color = GREEN if classification == "Reuse" else RED

    # Line 1: shoe name + classification
    cv2.putText(panel, f"{entry['name']}  [{classification}]",
                (10, 28), FONT, 0.65, label_color, 2)

    # Line 2: stats
    stats = (f"YOLO conf: {conf:.2f}   "
             f"Color: {color} ({color_conf:.2f})   "
             f"Blur: {blur:.0f}   "
             f"Folder: {folder_name}")
    cv2.putText(panel, stats, (10, 62), FONT, 0.5, GRAY, 1)

    # Line 3: counter
    cv2.putText(panel, f"Shoe {idx + 1} of {total}",
                (10, 92), FONT, 0.5, WHITE, 1)

    # Line 4: controls
    cv2.putText(panel, "SPACE: keep   D: delete   R: relabel   B/<-: back   Q/ESC: quit",
                (10, 128), FONT, 0.45, YELLOW, 1)

    return np.vstack([img_disp, panel])


def delete_entry(entry):
    for path in (entry["jpg"], entry["json"]):
        if path and os.path.exists(path):
            os.remove(path)
            print(f"[review] deleted {path}")
    entry["_gone"] = True


def relabel_entry(entry):
    """Flip classification between Reuse and Recycle, renaming files."""
    meta = entry["meta"]
    old_cls = meta.get("classification", "Reuse")
    new_cls = "Recycle" if old_cls == "Reuse" else "Reuse"

    folder = entry["folder"]
    old_name = entry["name"]   # e.g. shoe_Reuse_black_3.jpg

    color = meta.get("detected_color") or "unknown"

    # Next available number for the new class.
    pattern = re.compile(
        rf"shoe_{re.escape(new_cls)}_\w+_(\d+)\.jpg$", re.IGNORECASE)
    max_n = 0
    for f in os.listdir(folder):
        m = pattern.match(f)
        if m:
            max_n = max(max_n, int(m.group(1)))
    new_n = max_n + 1

    new_base = f"shoe_{new_cls}_{color}_{new_n}"
    new_jpg = os.path.join(folder, new_base + ".jpg")
    new_json = os.path.join(folder, new_base + ".json")

    # Rename image.
    os.rename(entry["jpg"], new_jpg)
    entry["jpg"] = new_jpg
    entry["name"] = new_base + ".jpg"

    # Update + rename JSON.
    meta["classification"] = new_cls
    meta["filename"] = new_base + ".jpg"
    meta["shoe_number"] = new_n
    if entry["json"] and os.path.exists(entry["json"]):
        os.remove(entry["json"])
    with open(new_json, "w") as f:
        json.dump(meta, f, indent=2)
    entry["json"] = new_json
    entry["meta"] = meta

    print(f"[review] relabeled {old_name} -> {new_base}.jpg ({old_cls} -> {new_cls})")


def main():
    ap = argparse.ArgumentParser(description="Review and curate the shoe dataset.")
    ap.add_argument("--folder", default=None,
                    help="Single incoming* folder (default: all)")
    ap.add_argument("--root", default=config.OUTPUT_ROOT)
    ap.add_argument("--only", choices=["Reuse", "Recycle"], default=None,
                    help="review only this classification")
    ap.add_argument("--color", default=None,
                    help="review only shoes whose detected_color matches")
    ap.add_argument("--max-sharp", type=float, default=None,
                    help="review only shoes blurrier than this sharpness score")
    args = ap.parse_args()

    if args.folder:
        folders = [os.path.join(args.root, args.folder)]
    else:
        folders = find_folders(args.root)

    entries = load_entries(folders)
    if not entries:
        print("No shoes found. Run label_live.py first to collect data.")
        sys.exit(0)

    filtered = apply_filters(entries, args)
    if not filtered:
        print(f"No shoes match the filter (of {len(entries)} total).")
        sys.exit(0)
    entries = filtered

    print(f"Loaded {len(entries)} shoe(s). Starting review.")
    cv2.namedWindow("Dataset Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Review", DISPLAY_W, 600)

    idx = 0
    kept = deleted = relabeled = 0

    while 0 <= idx < len(entries):
        entry = entries[idx]
        if entry.get("_gone"):
            idx += 1
            continue

        frame = render(entry, idx, len(entries))
        cv2.imshow("Dataset Review", frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):           # Q or ESC -> quit
            break
        elif key in (ord(" "), 83):         # SPACE or right arrow -> keep & next
            kept += 1
            idx += 1
        elif key in (81, ord("b")):         # left arrow (Linux) or B -> go back
            idx = max(0, idx - 1)
        elif key == ord("d"):               # D -> delete
            delete_entry(entry)
            deleted += 1
            idx += 1
        elif key == ord("r"):               # R -> relabel (stay on same shoe)
            relabel_entry(entry)
            relabeled += 1

    cv2.destroyAllWindows()
    remaining = len(entries) - deleted
    print(f"\nReview complete. "
          f"Kept: {kept}  Deleted: {deleted}  Relabeled: {relabeled}  "
          f"Remaining in dataset: {remaining}")


if __name__ == "__main__":
    main()
