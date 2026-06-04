"""
identify_brands.py -- Phase B: fill each pair crop's `make` (brand).

Walks the pairs<MMDDYYYY> folders produced by split_table.py, runs the brand
classifier (brand_utils) on each pair crop, and writes `make` + `make_confidence`
back into that pair's JSON sidecar. Idempotent: a pair that already has a `make`
is skipped unless --force, so this is safe to re-run as new pairs arrive.

Usage:
    python identify_brands.py                       # all pairs* folders
    python identify_brands.py --folder pairs06042026
    python identify_brands.py --dry-run             # print guesses, write nothing
    python identify_brands.py --force               # re-label even if make is set

Requires the same env as Phase A (ultralytics pulled in CLIP + torch).
"""
import argparse
import glob
import json
import os

import cv2

import config
from brand_utils import build_brand_classifier


def _pairs_folders(root, only=None):
    """Return the pairs* folder(s) to process."""
    if only:
        path = only if os.path.isdir(only) else os.path.join(root, only)
        return [path] if os.path.isdir(path) else []
    return sorted(p for p in glob.glob(os.path.join(root, "pairs*"))
                  if os.path.isdir(p))


def _pair_jsons(folder):
    """JSON sidecars in a folder, skipping macOS ._ files."""
    return sorted(p for p in glob.glob(os.path.join(folder, "pair_*.json"))
                  if not os.path.basename(p).startswith("._"))


def process_folder(folder, classifier, force=False, dry_run=False):
    """Label every pair in one folder, then export confident ones to label_data.
    Returns (labeled, skipped, failed, exported)."""
    from label_export import export_if_confident
    labeled = skipped = failed = exported = 0
    for json_path in _pair_jsons(folder):
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception as exc:                     # noqa: BLE001 - skip bad file
            print(f"[brand] could not read {json_path}: {exc}")
            failed += 1
            continue

        jpg_path = os.path.join(folder, meta.get("filename", ""))

        # (Re)label unless it already has a make and we're not forcing.
        if not meta.get("make") or force:
            image = cv2.imread(jpg_path)
            if image is None:
                print(f"[brand] missing/unreadable crop for {os.path.basename(json_path)}")
                failed += 1
                continue
            make, conf = classifier.classify(image)
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "n/a"
            print(f"[brand] {meta.get('filename')}: {make} ({conf_str})")
            if dry_run:
                labeled += 1
                continue
            meta["make"] = make
            meta["make_confidence"] = round(conf, 4) if isinstance(conf, float) else None
            try:
                with open(json_path, "w") as f:
                    json.dump(meta, f, indent=2)
                labeled += 1
            except Exception as exc:                 # noqa: BLE001 - non-fatal
                print(f"[brand] could not write {json_path}: {exc}")
                failed += 1
                continue
        else:
            skipped += 1

        # Export the confident ones (idempotent); also catches already-labeled
        # pairs from earlier runs.
        if not dry_run and export_if_confident(meta, jpg_path):
            exported += 1
    return labeled, skipped, failed, exported


def main():
    ap = argparse.ArgumentParser(description="Phase B: label each pair's brand.")
    ap.add_argument("--root", default=config.TABLE_OUTPUT_ROOT)
    ap.add_argument("--folder", default=None, help="single pairs* folder")
    ap.add_argument("--force", action="store_true",
                    help="re-label even pairs that already have a make")
    ap.add_argument("--dry-run", action="store_true",
                    help="print guesses; write nothing")
    args = ap.parse_args()

    folders = _pairs_folders(args.root, args.folder)
    if not folders:
        print(f"No pairs* folders under '{args.root}'. Run split_table.py first.")
        return

    classifier = build_brand_classifier(config)

    total_l = total_s = total_f = total_e = 0
    for folder in folders:
        jsons = _pair_jsons(folder)
        if not jsons:
            continue
        print(f"\n=== {os.path.basename(folder)}: {len(jsons)} pair(s) ===")
        l, s, f, e = process_folder(folder, classifier, args.force, args.dry_run)
        total_l += l
        total_s += s
        total_f += f
        total_e += e

    print()
    verb = "would label" if args.dry_run else "labeled"
    print(f"Done. {verb} {total_l}, skipped {total_s} (already had make), "
          f"failed {total_f}.")
    if not args.dry_run:
        print(f"Exported {total_e} confident pair(s) to "
              f"{getattr(config, 'LABEL_DATA_DIR', 'label_data')}/.")


if __name__ == "__main__":
    main()
