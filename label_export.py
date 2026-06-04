"""
label_export.py -- copy confidently-labeled pairs into a clean training folder.

When a pair's color (Phase A) AND make (Phase B) are both confident -- and neither
is "unknown"/"multi" -- the crop is copied into config.LABEL_DATA_DIR named
    shoes_<color>_<make>_<N>.jpg     e.g. shoes_blue_newBalance_1.jpg
with a small .json label beside it. This is the curated, high-quality subset
ready to hand to the (out-of-scope) training step.

Idempotent: a pair already exported (same source photo + source crop) is not
copied again, so re-running identify_brands.py won't create duplicates.
"""
import json
import os
import re
import shutil

import config


def _camel_make(make):
    """'New Balance' -> 'newBalance', 'Nike' -> 'nike', 'Under Armour' ->
    'underArmour' (lower first word, capitalize the rest, no spaces)."""
    parts = make.split()
    if not parts:
        return make
    return parts[0].lower() + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _next_n(folder, color, make):
    """Next sequence number for shoes_<color>_<make>_<N>.jpg in `folder`."""
    pattern = re.compile(rf"shoes_{re.escape(color)}_{re.escape(make)}_(\d+)\.jpg$",
                         re.IGNORECASE)
    max_n = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _already_exported(folder, meta):
    """True if a label already records this (source_photo, source_pair)."""
    if not os.path.isdir(folder):
        return False
    key = (meta.get("source_photo"), meta.get("filename"))
    for name in os.listdir(folder):
        if name.startswith("._") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name)) as f:
                d = json.load(f)
            if (d.get("source_photo"), d.get("source_pair")) == key:
                return True
        except Exception:                              # noqa: BLE001 - skip bad
            continue
    return False


def export_if_confident(meta, src_jpg, cfg=None):
    """Copy `src_jpg` into LABEL_DATA_DIR if color+make are confident enough.

    Returns the destination path on export, else None (skipped). Never raises.
    """
    cfg = cfg or config
    make = meta.get("make")
    color = meta.get("detected_color")
    mconf = meta.get("make_confidence")
    cconf = meta.get("color_confidence")

    # Must have a real, confident brand and a real, confident single color.
    if not make or make.lower() == "unknown":
        return None
    if not color or color.lower() in ("unknown", "multi"):
        return None
    if not isinstance(mconf, (int, float)) or mconf < getattr(cfg, "LABEL_MAKE_MIN_CONF", 0.6):
        return None
    if not isinstance(cconf, (int, float)) or cconf < getattr(cfg, "LABEL_COLOR_MIN_CONF", 0.5):
        return None

    folder = getattr(cfg, "LABEL_DATA_DIR", "label_data")
    if _already_exported(folder, meta):
        return None

    try:
        os.makedirs(folder, exist_ok=True)
        make_slug = _camel_make(make)
        n = _next_n(folder, color, make_slug)
        base = f"shoes_{color}_{make_slug}_{n}"
        dst_jpg = os.path.join(folder, base + ".jpg")
        shutil.copy2(src_jpg, dst_jpg)
        label = {
            "filename": base + ".jpg",
            "make": make,
            "color": color,
            "model": meta.get("model"),            # filled by Phase C if present
            "make_confidence": mconf,
            "color_confidence": cconf,
            "model_confidence": meta.get("model_confidence"),
            "source_photo": meta.get("source_photo"),
            "source_pair": meta.get("filename"),
        }
        with open(os.path.join(folder, base + ".json"), "w") as f:
            json.dump(label, f, indent=2)
        print(f"[label] exported {base}.jpg")
        return dst_jpg
    except Exception as exc:                           # noqa: BLE001 - non-fatal
        print(f"[label] export failed for {src_jpg}: {exc}")
        return None
