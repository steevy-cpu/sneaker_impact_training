"""
build_dataset.py -- turn the growing label_data/ folder into a TRAINING-READY
manifest with a leak-free split and honest label-quality tiers.

This is the preprocessing step for the local brand/model classifier (Goal B).
It does NOT copy or modify images and it opens the dash DB READ-ONLY, so it can
never lock or slow the live site. It only reads label_data/*.json sidecars
(+ a read-only DB cross-reference) and writes a small manifest + stats.

Outputs (default ./dataset/):
  manifest.csv / manifest.jsonl  one row per image, with `split` and
                                 `label_quality` columns
  brand_aliases.json             the make-normalization map actually applied
  stats.json                     class balance, split sizes, quality tiers,
                                 long-tail report, leakage check

Key correctness choices:
  * SPLIT BY source_photo (group), never by image -- pairs from the same table
    photo are near-duplicates; splitting by image leaks them across train/test
    and fakes high accuracy. Whole boxes go to exactly one split.
  * Deterministic: split is a seeded hash of the group id, so re-runs are stable
    and reproducible without storing state.
  * label_quality tiers: human (a worker confirmed it in Pairs Review -> real
    ground truth) > cloud (Gemini/OpenAI pseudo-label) > local. The human tier
    is the only trustworthy TEST anchor; everything else measures
    agreement-with-the-teacher, not truth.

Usage:
  python build_dataset.py
  python build_dataset.py --out dataset --val-frac 0.1 --test-frac 0.1 --seed 42
  python build_dataset.py --min-make-count 15 --bucket-rare
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict

import config

# Canonical brand display names keyed by lowercased, space-stripped form.
# Fixes the ASICS/Asics casing split, camelCase filenames, and common aliases.
_BRAND_CANON = {
    "asics": "ASICS", "nike": "Nike", "adidas": "Adidas", "newbalance": "New Balance",
    "hoka": "Hoka", "hokaoneone": "Hoka", "brooks": "Brooks", "saucony": "Saucony",
    "on": "On", "onrunning": "On", "underarmour": "Under Armour", "reebok": "Reebok",
    "merrell": "Merrell", "puma": "Puma", "mizuno": "Mizuno", "salomon": "Salomon",
    "converse": "Converse", "vans": "Vans", "jordan": "Jordan", "altra": "Altra",
    "skechers": "Skechers", "fila": "Fila", "newbalanc": "New Balance",
    # Kept in sync with the dash's canonical map (backend/utils/brands.py
    # CANONICAL_BRANDS): official all-caps spellings that Title-Case would
    # otherwise get wrong (Oofos/Anta/…). Add new dash pins here too.
    "oofos": "OOFOS", "anta": "ANTA", "ecco": "ECCO", "peak": "PEAK",
    "nobull": "NOBULL", "akk": "AKK",
}


def _canon_make(make):
    """Return (canonical_display, raw). 'unknown'/empty stay 'unknown'."""
    raw = (make or "").strip()
    if not raw or raw.lower() == "unknown":
        return "unknown", raw
    key = "".join(raw.lower().split()).replace("-", "")
    if key in _BRAND_CANON:
        return _BRAND_CANON[key], raw
    # Fallback: Title Case the cleaned string (keeps unseen brands, normalized).
    return " ".join(w.capitalize() for w in raw.split()), raw


def _split_for(group_id, val_frac, test_frac, seed):
    """Deterministic per-GROUP split: hash -> [0,1). Same group_id always lands
    in the same split, so a whole table photo never spans train/val/test."""
    h = hashlib.sha1(f"{seed}:{group_id}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "val"
    return "train"


def _load_gold(db_path):
    """Map (source_photo, source_pair) -> the human-confirmed GOLD label, read
    from the DB's COMPLETED pairs (Quick Label / Pairs Review). source_pair is
    the crop index parsed from the pair's image_path (…_<N>.jpg), which is the
    same index the label_data sidecar stores — so gold can OVERRIDE the matching
    silver (cloud) label, and a confirmed pair that was never auto-exported to
    label_data still becomes a gold example via its crop on disk.

    Read-only DB open so we never lock the live site. Empty on any error."""
    gold = {}
    if not os.path.exists(db_path):
        print(f"[dataset] note: DB not found at {db_path}; no human gold tier.")
        return gold
    root = os.path.dirname(os.path.abspath(db_path))   # dash root (images/ lives here)
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT table_photo_id, image_path, final_make, final_color, final_model "
            "FROM pairs WHERE review_status='COMPLETED' "
            "AND final_make IS NOT NULL AND final_make NOT IN ('','unknown')").fetchall()
        conn.close()
    except Exception as exc:                              # noqa: BLE001
        print(f"[dataset] note: gold lookup failed ({exc}); skipping.")
        return gold
    for r in rows:
        ip = r["image_path"] or ""
        try:
            n = int(os.path.splitext(os.path.basename(ip))[0].split("_")[-1])
        except (ValueError, IndexError):
            continue
        gold[(r["table_photo_id"], n)] = {
            "make": r["final_make"], "color": r["final_color"] or "unknown",
            "model": r["final_model"] or "unknown",
            "disk": os.path.join(root, ip.lstrip("/")) if ip else None,
        }
    return gold


def main():
    ap = argparse.ArgumentParser(description="Build a training manifest from label_data/.")
    ap.add_argument("--label-data", default=str(getattr(config, "LABEL_DATA_DIR", "label_data")))
    ap.add_argument("--db", default="../sneakers.db", help="dash DB (read-only) for the human tier")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-make-count", type=int, default=10,
                    help="makes with fewer than this many images are flagged rare")
    ap.add_argument("--bucket-rare", action="store_true",
                    help="relabel rare makes to 'other' (default: just flag them)")
    args = ap.parse_args()

    ld = args.label_data
    jsons = [p for p in glob.glob(os.path.join(ld, "*.json"))
             if not os.path.basename(p).startswith("._")]
    if not jsons:
        raise SystemExit(f"[dataset] no label JSONs in {ld}")

    gold = _load_gold(args.db)

    # ---- SILVER: label_data sidecars (cloud/local pseudo-labels) ---------
    # Keyed by (source_photo, source_pair) so human gold can override the exact
    # crop and dedup is exact.
    rec = {}
    for jp in jsons:
        try:
            with open(jp) as fh:
                d = json.load(fh)
        except Exception:                                 # noqa: BLE001
            continue
        fn = d.get("filename") or (os.path.splitext(os.path.basename(jp))[0] + ".jpg")
        img = os.path.join(ld, fn)
        if not os.path.exists(img):
            continue
        src_photo = d.get("source_photo") or fn          # group key (fallback: file)
        src_pair = d.get("source_pair")
        make_disp, make_raw = _canon_make(d.get("make"))
        src = (d.get("prediction_source") or "").lower()
        quality = "cloud" if src.startswith("cloud") else "local" if src == "local" else "unknown"
        rec[(src_photo, src_pair if src_pair is not None else fn)] = {
            "filename": fn, "filepath": img,
            "split": _split_for(src_photo, args.val_frac, args.test_frac, args.seed),
            "label_quality": quality,
            "color": d.get("detected_color") or "unknown",
            "make": make_disp, "make_raw": make_raw,
            "model": d.get("model") or "unknown",
            "make_confidence": d.get("make_confidence"),
            "model_confidence": d.get("model_confidence"),
            "color_confidence": d.get("color_confidence"),
            "prediction_source": d.get("prediction_source") or "",
            "source_photo": src_photo, "source_pair": src_pair,
        }

    # ---- GOLD: human confirmations OVERRIDE the cloud label (or add a new
    # example for a confirmed pair that was never auto-exported) -----------
    g_over = g_add = 0
    for (tp, n), g in gold.items():
        make_disp, make_raw = _canon_make(g["make"])
        if (tp, n) in rec:
            rec[(tp, n)].update(make=make_disp, make_raw=make_raw, color=g["color"],
                                model=g["model"], label_quality="human",
                                prediction_source="human", make_confidence=1.0,
                                model_confidence=None, color_confidence=None)
            g_over += 1
        elif g["disk"] and os.path.exists(g["disk"]):
            rec[(tp, n)] = {
                "filename": os.path.basename(g["disk"]), "filepath": g["disk"],
                "split": _split_for(tp, args.val_frac, args.test_frac, args.seed),
                "label_quality": "human", "color": g["color"], "make": make_disp,
                "make_raw": make_raw, "model": g["model"], "make_confidence": 1.0,
                "model_confidence": None, "color_confidence": None,
                "prediction_source": "human", "source_photo": tp, "source_pair": n,
            }
            g_add += 1
    print(f"[dataset] gold from DB: {g_over} overrode a silver label, "
          f"{g_add} added (never auto-exported)")
    records = list(rec.values())

    # ---- long-tail handling ----------------------------------------------
    make_counts = Counter(r["make"] for r in records if r["make"] != "unknown")
    rare = {m for m, n in make_counts.items() if n < args.min_make_count}
    for r in records:
        r["rare_make"] = r["make"] in rare
        if args.bucket_rare and r["rare_make"]:
            r["make"] = "other"

    # ---- write manifest ---------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    cols = ["filename", "filepath", "split", "label_quality", "color", "make",
            "make_raw", "model", "make_confidence", "model_confidence",
            "color_confidence", "prediction_source", "source_photo",
            "source_pair", "rare_make"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "brand_aliases.json"), "w") as f:
        json.dump(_BRAND_CANON, f, indent=1)

    # ---- stats + leakage check -------------------------------------------
    by_split = Counter(r["split"] for r in records)
    by_quality = Counter(r["label_quality"] for r in records)
    # leakage: a source_photo must live in exactly one split
    photo_splits = defaultdict(set)
    for r in records:
        photo_splits[r["source_photo"]].add(r["split"])
    leaks = [p for p, s in photo_splits.items() if len(s) > 1]
    known_make = sum(1 for r in records if r["make"] not in ("unknown", "other"))
    unknown_color = sum(1 for r in records if (r["color"] or "unknown") == "unknown")
    stats = {
        "total_images": len(records),
        "groups_source_photos": len(photo_splits),
        "by_split": dict(by_split),
        "by_label_quality": dict(by_quality),
        "distinct_makes": len(make_counts),
        "rare_makes_below_min": sorted(rare),
        "top_makes": make_counts.most_common(15),
        "known_make_pct": round(100 * known_make / len(records), 1),
        "unknown_color_pct": round(100 * unknown_color / len(records), 1),
        "leakage_photos_in_multiple_splits": len(leaks),
        "params": {"val_frac": args.val_frac, "test_frac": args.test_frac,
                   "seed": args.seed, "min_make_count": args.min_make_count,
                   "bucket_rare": args.bucket_rare},
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- console summary --------------------------------------------------
    print(f"\n=== DATASET BUILT -> {args.out}/ ===")
    print(f"images: {len(records)}  groups(source_photos): {len(photo_splits)}")
    print(f"split: {dict(by_split)}")
    print(f"label_quality: {dict(by_quality)}")
    if by_quality.get("human", 0) == 0:
        print("  ⚠ NO human-verified labels yet — the TEST split measures "
              "agreement-with-Gemini, not ground truth. Confirm pairs in Pairs "
              "Review to build a real gold test set.")
    print(f"distinct makes: {len(make_counts)}  rare(<{args.min_make_count}): "
          f"{len(rare)}{' -> bucketed to other' if args.bucket_rare else ' (flagged)'}")
    print(f"known-make: {stats['known_make_pct']}%  unknown-color: "
          f"{stats['unknown_color_pct']}%")
    print(f"leakage check: {len(leaks)} photos span >1 split "
          f"({'OK' if not leaks else 'FAIL'})")
    print("top makes:", ", ".join(f"{m}={n}" for m, n in make_counts.most_common(8)))


if __name__ == "__main__":
    main()
