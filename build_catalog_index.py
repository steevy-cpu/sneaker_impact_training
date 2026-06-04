"""
build_catalog_index.py -- build the CLIP reverse-image index (Phase C verifier).

Embeds a catalog of known sneaker images with CLIP and saves an index the
"clip-index" model_search backend uses to verify model guesses with a REAL
similarity score + a source link. The catalog merges two sources:

  1. config.CLIP_CATALOG_DIR/<brand>/<model>/*.jpg   (drop a public dataset here)
  2. config.LABEL_DATA_DIR  shoes_<color>_<make>_N.jpg + .json  (our growing,
     human-confirmed set -- the strongest model-level source we have)

Output (next to CLIP_INDEX_PATH):
  - <stem>.npz   float32 embeddings, shape (N, D), L2-normalized
  - <stem>.json  parallel metadata list: {brand, model, source, image}

Re-run whenever the catalog or label_data changes. Needs CLIP + torch (already
pulled in by ultralytics).

Usage:
    python build_catalog_index.py
    python build_catalog_index.py --catalog /path/to/catalog --label-data label_data
"""
import argparse
import glob
import json
import os

import numpy as np

import config

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _device():
    try:
        from detector_utils import pick_device
        return pick_device()
    except Exception:                                  # noqa: BLE001 - fail safe
        return "cpu"


def collect_catalog_dir(d):
    """Entries from a <brand>/<model>/*.jpg tree (a dropped-in public dataset)."""
    entries = []
    if not os.path.isdir(d):
        return entries
    for path in glob.glob(os.path.join(d, "*", "*", "*")):
        name = os.path.basename(path)
        if name.startswith("._") or not name.lower().endswith(_IMG_EXTS):
            continue
        model = os.path.basename(os.path.dirname(path))
        brand = os.path.basename(os.path.dirname(os.path.dirname(path)))
        entries.append({"image": path, "brand": brand, "model": model,
                        "source": f"catalog:{brand}/{model}/{name}"})
    return entries


def collect_label_data(d):
    """Entries from our confirmed label_data (only ones that have a real model)."""
    entries = []
    if not os.path.isdir(d):
        return entries
    for jf in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(jf).startswith("._"):
            continue
        try:
            with open(jf) as f:
                m = json.load(f)
        except Exception:                              # noqa: BLE001 - skip bad
            continue
        model = m.get("model")
        if not model or str(model).lower() == "unknown":
            continue
        img = os.path.join(d, m.get("filename", ""))
        if not os.path.exists(img):
            continue
        entries.append({"image": img, "brand": m.get("make") or "unknown",
                        "model": model,
                        "source": f"label_data:{m.get('source_photo')}/{m.get('source_pair')}"})
    return entries


def main():
    ap = argparse.ArgumentParser(description="Build the CLIP catalog index.")
    ap.add_argument("--catalog", default=config.CLIP_CATALOG_DIR)
    ap.add_argument("--label-data", default=config.LABEL_DATA_DIR)
    ap.add_argument("--out", default=config.CLIP_INDEX_PATH)
    ap.add_argument("--model", default=config.CLIP_INDEX_MODEL)
    args = ap.parse_args()

    entries = collect_catalog_dir(args.catalog) + collect_label_data(args.label_data)
    if not entries:
        print(f"No catalog images found. Drop a dataset under '{args.catalog}' "
              f"(<brand>/<model>/*.jpg) and/or confirm models into "
              f"'{args.label_data}', then re-run.")
        return

    by_src = {}
    for e in entries:
        by_src[e["source"].split(":", 1)[0]] = by_src.get(e["source"].split(":", 1)[0], 0) + 1
    print(f"Catalog: {len(entries)} images ({by_src}). Loading CLIP {args.model}...")

    import clip
    import cv2
    import torch
    from PIL import Image

    device = _device()
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    embeddings = []
    kept = []
    for e in entries:
        img = cv2.imread(e["image"])
        if img is None:
            print(f"  skip unreadable {e['image']}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy()[0].astype("float32"))
        kept.append(e)

    if not embeddings:
        print("No images could be embedded. Nothing written.")
        return

    arr = np.vstack(embeddings).astype("float32")
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, embeddings=arr)
    with open(stem + ".json", "w") as f:
        json.dump({"model": args.model, "entries": kept}, f, indent=2)

    brands = sorted({e["brand"] for e in kept})
    print(f"\nBuilt index: {len(kept)} images, {arr.shape[1]}-d, brands={brands}")
    print(f"  embeddings -> {args.out}")
    print(f"  metadata   -> {stem}.json")


if __name__ == "__main__":
    main()
