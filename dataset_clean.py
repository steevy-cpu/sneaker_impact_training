"""
dataset_clean.py -- batch dataset quality cleaner.

Scans one or all incoming* folders and removes low-quality entries:
  - Blurry crops (variance of Laplacian below --blur)
  - Low YOLO confidence (below --conf)
  - Near-duplicate images (perceptual hash distance <= --dedup-dist)

Deleted pairs: both the .jpg AND the matching .json sidecar are removed.

Usage:
    python dataset_clean.py                        # all incoming* folders
    python dataset_clean.py --folder incoming05292026
    python dataset_clean.py --dry-run              # preview only, no deletes
    python dataset_clean.py --blur 80 --conf 0.45
    python dataset_clean.py --no-dedup             # skip duplicate check
"""
import argparse
import os
import sys

import cv2
import numpy as np

import config
from dataset_utils import find_folders, load_entries
from image_utils import sharpness as blur_score


# ── Perceptual hash (difference hash, 8x8) ───────────────────────────────────

def dhash(image, size=8):
    """Difference hash: returns a 64-bit integer fingerprint of the image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (size + 1, size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum(1 << i for i, v in enumerate(diff.flatten()) if v)
    except Exception:
        return 0


def hamming(a, b):
    """Hamming distance between two hashes (popcount of XOR).

    int.bit_count() is a C builtin (Python 3.10+) -- far faster than a manual
    bit-shift loop, which matters as the dataset grows.
    """
    return (a ^ b).bit_count()


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_entry(entry, dry_run):
    """Remove a shoe's jpg + json. dry_run=True just prints."""
    for path in (entry["jpg"], entry["json"]):
        if path and os.path.exists(path):
            if dry_run:
                print(f"  [dry-run] would delete {path}")
            else:
                os.remove(path)
                print(f"  [deleted] {path}")


# ── Passes ────────────────────────────────────────────────────────────────────

def pass_blur(entries, threshold, dry_run):
    """Remove entries whose blur score is below threshold."""
    removed = 0
    for e in entries:
        img = cv2.imread(e["jpg"])
        if img is None:
            continue
        score = blur_score(img)
        e["blur"] = score          # cache for later passes
        if score < threshold:
            print(f"  blur {score:.1f} < {threshold}  {e['name']}")
            delete_entry(e, dry_run)
            e["_deleted"] = True
            removed += 1
    return removed


def pass_conf(entries, threshold, dry_run):
    """Remove entries whose YOLO confidence is below threshold."""
    removed = 0
    for e in entries:
        if e.get("_deleted"):
            continue
        conf = e["meta"].get("yolo_confidence", 1.0)
        if conf < threshold:
            print(f"  conf  {conf:.2f} < {threshold}  {e['name']}")
            delete_entry(e, dry_run)
            e["_deleted"] = True
            removed += 1
    return removed


def pass_dedup(entries, max_dist, dry_run):
    """Remove near-duplicate images, keeping the sharpest of each group."""
    # Build hash list (skip already-deleted entries).
    live = [e for e in entries if not e.get("_deleted")]
    hashes = []
    for e in live:
        img = cv2.imread(e["jpg"])
        if img is None:
            hashes.append(None)
            continue
        if "blur" not in e:
            e["blur"] = blur_score(img)
        hashes.append(dhash(img))

    removed = 0
    deleted_idx = set()
    for i in range(len(live)):
        if i in deleted_idx or hashes[i] is None:
            continue
        group = [i]
        for j in range(i + 1, len(live)):
            if j in deleted_idx or hashes[j] is None:
                continue
            if hamming(hashes[i], hashes[j]) <= max_dist:
                group.append(j)
        if len(group) < 2:
            continue
        # Keep the sharpest; delete the rest.
        best = max(group, key=lambda k: live[k].get("blur", 0))
        for k in group:
            if k == best:
                continue
            e = live[k]
            print(f"  dedup (dist<={max_dist})  {e['name']}")
            delete_entry(e, dry_run)
            e["_deleted"] = True
            deleted_idx.add(k)
            removed += 1
    return removed


# ── Main ──────────────────────────────────────────────────────────────────────

def clean_folder(folder, args):
    entries = load_entries(folder)
    if not entries:
        return 0, 0
    total = len(entries)
    removed = 0

    if args.blur is not None:
        r = pass_blur(entries, args.blur, args.dry_run)
        removed += r
        if r:
            print(f"  -> {r} blurry image(s) flagged")

    if args.conf is not None:
        r = pass_conf(entries, args.conf, args.dry_run)
        removed += r
        if r:
            print(f"  -> {r} low-confidence image(s) flagged")

    if not args.no_dedup:
        r = pass_dedup(entries, args.dedup_dist, args.dry_run)
        removed += r
        if r:
            print(f"  -> {r} duplicate(s) flagged")

    return total, removed


def main():
    ap = argparse.ArgumentParser(description="Clean up dataset quality issues.")
    ap.add_argument("--folder", default=None,
                    help="Single incoming* folder name (default: all)")
    ap.add_argument("--root", default=config.OUTPUT_ROOT,
                    help="Root pictures directory")
    ap.add_argument("--blur", type=float, default=50.0,
                    help="Remove images with blur score below this (default 50)")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="Remove images with YOLO conf below this (default 0.4)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Skip near-duplicate removal")
    ap.add_argument("--dedup-dist", type=int, default=8,
                    help="Max hamming distance to consider duplicate (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview what would be deleted; make no changes")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN -- no files will be deleted.\n")

    if args.folder:
        folders = [os.path.join(args.root, args.folder)]
    else:
        folders = find_folders(args.root)

    if not folders:
        print(f"No incoming* folders found under '{args.root}'.")
        sys.exit(0)

    grand_total = grand_removed = 0
    for folder in folders:
        print(f"\n=== {os.path.basename(folder)} ===")
        total, removed = clean_folder(folder, args)
        grand_total += total
        grand_removed += removed
        if removed == 0:
            print("  No issues found.")

    action = "would remove" if args.dry_run else "removed"
    print(f"\nDone. Scanned {grand_total} image(s), {action} {grand_removed}.")


if __name__ == "__main__":
    main()
