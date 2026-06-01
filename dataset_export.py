"""
dataset_export.py -- export one manifest of the whole labeled dataset.

Walks the incoming* folders and writes a single CSV (or JSON) with one row per
labeled shoe: path, label, color, confidences, sharpness, bbox, etc. This is the
clean handoff artifact for the (separate) training step -- it does no training
itself, it just packages what's been collected.

Usage:
    python dataset_export.py                       # -> dataset_manifest.csv
    python dataset_export.py --out manifest.csv
    python dataset_export.py --format json --out manifest.json
    python dataset_export.py --folder incoming06012026
"""
import argparse
import csv
import json
import os

import config
from dataset_utils import find_folders, load_entries

_FIELDS = ["jpg", "folder", "filename", "classification", "detected_color",
           "color_confidence", "yolo_confidence", "sharpness", "tracking_id",
           "timestamp", "model_used", "frame_width", "frame_height", "bbox"]


def _row(entry):
    m = entry["meta"]
    row = {f: m.get(f) for f in _FIELDS}
    row["jpg"] = entry["jpg"]
    row["folder"] = os.path.basename(entry["folder"])
    row["filename"] = entry["name"]
    # bbox is a list; flatten to a space-separated string for CSV friendliness.
    if isinstance(row.get("bbox"), (list, tuple)):
        row["bbox"] = " ".join(str(v) for v in row["bbox"])
    return row


def main():
    ap = argparse.ArgumentParser(description="Export a manifest of the labeled dataset.")
    ap.add_argument("--folder", default=None,
                    help="Single incoming* folder (default: all)")
    ap.add_argument("--root", default=config.OUTPUT_ROOT)
    ap.add_argument("--out", default="dataset_manifest.csv", help="output file path")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    args = ap.parse_args()

    folders = ([os.path.join(args.root, args.folder)] if args.folder
               else find_folders(args.root))
    entries = load_entries(folders)
    if not entries:
        print(f"No shoes found under '{args.root}'. Nothing to export.")
        return

    rows = [_row(e) for e in entries]

    if args.format == "json":
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
    else:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.out} ({args.format}).")


if __name__ == "__main__":
    main()
