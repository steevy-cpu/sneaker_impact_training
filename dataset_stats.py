"""
dataset_stats.py -- summarize the collected shoe dataset (read-only).

Scans the incoming* folders and prints what you've collected so you can judge it
before training: total images, Reuse vs Recycle balance, color distribution,
per-day counts, and average confidence/sharpness. Flags class imbalance, which
is the usual risk here (Recycle is operator-clicked and tends to be rare).

Usage:
    python dataset_stats.py                      # all incoming* folders
    python dataset_stats.py --folder incoming06012026
    python dataset_stats.py --root other/path
"""
import argparse
import os
from collections import Counter

import config
from dataset_utils import find_folders, load_entries

# Warn if the minority class is rarer than this fraction of the dataset.
_IMBALANCE_FRAC = 0.15


def _bar(count, total, width=24):
    """A little ASCII bar for a count out of total."""
    if total <= 0:
        return ""
    filled = int(round(width * count / total))
    return "#" * filled + "-" * (width - filled)


def _pct(count, total):
    return (100.0 * count / total) if total else 0.0


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description="Summarize the collected shoe dataset.")
    ap.add_argument("--folder", default=None,
                    help="Single incoming* folder (default: all)")
    ap.add_argument("--root", default=config.OUTPUT_ROOT)
    args = ap.parse_args()

    folders = ([os.path.join(args.root, args.folder)] if args.folder
               else find_folders(args.root))
    entries = load_entries(folders)
    total = len(entries)
    if total == 0:
        print(f"No shoes found under '{args.root}'. Run label_live.py first.")
        return

    by_class = Counter(e["meta"].get("classification", "?") for e in entries)
    by_color = Counter((e["meta"].get("detected_color") or "?") for e in entries)
    by_folder = Counter(os.path.basename(e["folder"]) for e in entries)
    missing_json = sum(1 for e in entries if e["json"] is None)
    confs = [e["meta"].get("yolo_confidence") for e in entries]
    sharp_vals = [e["meta"].get("sharpness") for e in entries
                  if isinstance(e["meta"].get("sharpness"), (int, float))]

    print("\n=== Sneaker Impact dataset summary ===")
    print(f"Root: {args.root}")
    print(f"Total labeled shoes: {total}   (folders: {len(folders)})")
    if missing_json:
        print(f"  ! {missing_json} image(s) have no metadata JSON")

    print("\nClassification:")
    for name, count in by_class.most_common():
        print(f"  {name:<10} {count:>6}  {_pct(count, total):5.1f}%  {_bar(count, total)}")

    print("\nColor:")
    for name, count in by_color.most_common():
        print(f"  {name:<10} {count:>6}  {_pct(count, total):5.1f}%  {_bar(count, total)}")

    print("\nPer folder (day):")
    for name, count in sorted(by_folder.items()):
        print(f"  {name:<20} {count:>6}")

    print("\nQuality:")
    print(f"  avg YOLO confidence: {_mean(confs):.2f}")
    if sharp_vals:
        print(f"  avg sharpness:       {_mean(sharp_vals):.0f}  "
              f"({len(sharp_vals)} of {total} recorded)")
    else:
        print("  avg sharpness:       (not recorded yet)")

    # Balance check (only meaningful with the two real labels present).
    reuse = by_class.get("Reuse", 0)
    recycle = by_class.get("Recycle", 0)
    if reuse and recycle:
        minority = min(reuse, recycle)
        if minority / total < _IMBALANCE_FRAC:
            rarer = "Recycle" if recycle < reuse else "Reuse"
            print(f"\n!! Class imbalance: {rarer} is only "
                  f"{_pct(minority, total):.1f}% of the data. A model trained on "
                  f"this will struggle on {rarer} -- collect more {rarer} shoes.")
    elif reuse and not recycle:
        print("\n!! No Recycle examples yet -- click bad shoes to collect them.")
    elif recycle and not reuse:
        print("\n!! No Reuse examples yet.")

    print()


if __name__ == "__main__":
    main()
