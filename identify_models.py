"""
identify_models.py -- Phase C: fill each pair crop's `model`.

Walks the pairs<MMDDYYYY> folders, runs the model identifier (model_search) on
each crop using the brand Phase B already found, and writes `model`,
`model_confidence`, `model_sources` back into the pair JSON. Any matching
label_data sidecar is updated too, so the curated export carries the full label.

Idempotent: a pair that already has a `model` is skipped unless --force.

Usage:
    python identify_models.py                      # all pairs* folders
    python identify_models.py --folder pairs06042026
    python identify_models.py --dry-run            # print guesses, write nothing
    python identify_models.py --force              # re-identify even if model set

Needs the local Ollama server running with the configured vision model.
"""
import argparse
import glob
import json
import os

import cv2

import config
from model_search import build_model_identifier


def _pairs_folders(root, only=None):
    if only:
        path = only if os.path.isdir(only) else os.path.join(root, only)
        return [path] if os.path.isdir(path) else []
    return sorted(p for p in glob.glob(os.path.join(root, "pairs*"))
                  if os.path.isdir(p))


def _pair_jsons(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "pair_*.json"))
                  if not os.path.basename(p).startswith("._"))


def _update_label_data(meta, model, conf, sources):
    """Propagate the model into the matching label_data sidecar, if any."""
    folder = getattr(config, "LABEL_DATA_DIR", "label_data")
    if not os.path.isdir(folder):
        return
    key = (meta.get("source_photo"), meta.get("filename"))
    for name in os.listdir(folder):
        if name.startswith("._") or not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:                              # noqa: BLE001 - skip bad
            continue
        if (d.get("source_photo"), d.get("source_pair")) == key:
            d["model"] = model
            d["model_confidence"] = conf
            d["model_sources"] = sources
            try:
                with open(path, "w") as f:
                    json.dump(d, f, indent=2)
            except Exception as exc:                    # noqa: BLE001 - non-fatal
                print(f"[model] could not update label {path}: {exc}")
            return


def process_folder(folder, identifier, force=False, dry_run=False):
    """Identify models for one folder. Returns (done, skipped, failed)."""
    done = skipped = failed = 0
    for json_path in _pair_jsons(folder):
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception as exc:                       # noqa: BLE001 - skip bad
            print(f"[model] could not read {json_path}: {exc}")
            failed += 1
            continue

        if meta.get("model") and not force:
            skipped += 1
            continue

        jpg_path = os.path.join(folder, meta.get("filename", ""))
        image = cv2.imread(jpg_path)
        if image is None:
            print(f"[model] missing/unreadable crop for {os.path.basename(json_path)}")
            failed += 1
            continue

        brand = meta.get("make", "")
        model, conf, sources = identifier.identify(image, brand)
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "n/a"
        src_str = f"  src={sources[0]}" if sources else ""
        print(f"[model] {meta.get('filename')} [{brand}] -> {model} ({conf_str}){src_str}")

        if dry_run:
            done += 1
            continue

        meta["model"] = model
        meta["model_confidence"] = round(conf, 4) if isinstance(conf, float) else None
        meta["model_sources"] = sources
        try:
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:                        # noqa: BLE001 - non-fatal
            print(f"[model] could not write {json_path}: {exc}")
            failed += 1
            continue
        _update_label_data(meta, meta["model"], meta["model_confidence"], sources)
        done += 1
    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser(description="Phase C: identify each pair's model.")
    ap.add_argument("--root", default=config.TABLE_OUTPUT_ROOT)
    ap.add_argument("--folder", default=None, help="single pairs* folder")
    ap.add_argument("--force", action="store_true",
                    help="re-identify even pairs that already have a model")
    ap.add_argument("--dry-run", action="store_true",
                    help="print guesses; write nothing")
    args = ap.parse_args()

    folders = _pairs_folders(args.root, args.folder)
    if not folders:
        print(f"No pairs* folders under '{args.root}'. Run split_table.py first.")
        return

    identifier = build_model_identifier(config)

    total_d = total_s = total_f = 0
    for folder in folders:
        jsons = _pair_jsons(folder)
        if not jsons:
            continue
        print(f"\n=== {os.path.basename(folder)}: {len(jsons)} pair(s) ===")
        d, s, f = process_folder(folder, identifier, args.force, args.dry_run)
        total_d += d
        total_s += s
        total_f += f

    print()
    verb = "would identify" if args.dry_run else "identified"
    print(f"Done. {verb} {total_d}, skipped {total_s} (already had model), "
          f"failed {total_f}.")


if __name__ == "__main__":
    main()
