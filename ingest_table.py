"""
ingest_table.py -- give incoming table photos a logical name (table1, table2, ...).

Copies (or moves) raw table photos into config.TABLE_INPUT_DIR, renaming each to
the next table<N>.jpg. That way every per-pair record's `source_photo` is a clean,
traceable name (table3 -> pair_7, etc.) instead of whatever the camera dumped.

Usage:
    python ingest_table.py /path/to/photo.jpg          # -> table_photos/table1.jpg
    python ingest_table.py shot1.jpg shot2.jpg          # ingest several in order
    python ingest_table.py photo.jpg --move             # move instead of copy

Then process them with:  python split_table.py --all
"""
import argparse
import os
import re
import shutil

import config


def next_table_number(folder, prefix):
    """Next N for <prefix><N>.jpg in `folder` (restart-safe scan)."""
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.jpg$", re.IGNORECASE)
    max_n = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.startswith("._"):
                continue
            m = pattern.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def main():
    ap = argparse.ArgumentParser(
        description="Rename incoming table photos to table1.jpg, table2.jpg, ...")
    ap.add_argument("photos", nargs="+", help="one or more source photos")
    ap.add_argument("--dir", default=config.TABLE_INPUT_DIR,
                    help="destination folder (default: config.TABLE_INPUT_DIR)")
    ap.add_argument("--prefix", default=getattr(config, "TABLE_PHOTO_PREFIX", "table"))
    ap.add_argument("--move", action="store_true",
                    help="move the source files instead of copying them")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    n = next_table_number(args.dir, args.prefix)
    done = 0
    for src in args.photos:
        if not os.path.isfile(src):
            print(f"[ingest] not a file, skipping: {src}")
            continue
        dst = os.path.join(args.dir, f"{args.prefix}{n}.jpg")
        try:
            if args.move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"[ingest] {os.path.basename(src)} -> {dst}")
            n += 1
            done += 1
        except Exception as exc:                       # noqa: BLE001 - report, go on
            print(f"[ingest] failed on {src}: {exc}")

    print(f"\n[ingest] {done} photo(s) ready in {args.dir}. "
          f"Next: python split_table.py --all")


if __name__ == "__main__":
    main()
