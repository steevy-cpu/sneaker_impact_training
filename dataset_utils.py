"""
dataset_utils.py -- shared helpers for the dataset tools.

`find_folders` and `load_entries` were duplicated in dataset_clean.py and
dataset_review.py; they live here now so both tools share one implementation.
"""
import json
import os


def find_folders(root):
    """Return all `incoming*` subfolders under `root`, sorted. [] if none."""
    try:
        return sorted(
            os.path.join(root, d)
            for d in os.listdir(root)
            if d.startswith("incoming") and os.path.isdir(os.path.join(root, d))
        )
    except FileNotFoundError:
        return []


def load_entries(folders):
    """Return a list of shoe entry dicts from one or more `incoming*` folders.

    `folders` may be a single folder path or a list of them. Each entry is:
        {"jpg": path, "json": path|None, "folder": path, "name": str,
         "meta": dict}
    A missing/unreadable JSON sidecar yields an empty `meta` (never raises).
    """
    if isinstance(folders, (str, bytes, os.PathLike)):
        folders = [folders]
    entries = []
    for folder in folders:
        try:
            names = sorted(os.listdir(folder))
        except FileNotFoundError:
            continue
        for name in names:
            if not name.endswith(".jpg"):
                continue
            jpg = os.path.join(folder, name)
            json_path = jpg[:-4] + ".json"
            meta = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        meta = json.load(f)
                except Exception:
                    pass
            entries.append({
                "jpg": jpg,
                "json": json_path if os.path.exists(json_path) else None,
                "folder": folder,
                "name": name,
                "meta": meta,
            })
    return entries
