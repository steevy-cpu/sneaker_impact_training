"""
eval_index.py -- measure how good the reverse-image index actually is.

The whole point of the "clip-index" verifier is a TRUSTWORTHY confidence: it
should (1) retrieve the right model and (2) score correct matches higher than
wrong ones, so a single threshold (config.CLIP_INDEX_MIN_SIM) can separate them.
CLIP failed #2 (scores ~0.78-0.84 regardless of correctness). This tool tells you
whether a given embedder (config.EMBED_BACKEND -- e.g. DINOv2) does better.

What it does, mirroring the real backend (brand filter, cosine, top-1):
  - loads the built index (config.CLIP_INDEX_PATH) and the SAME embedder it was
    built with (refuses a mismatch, like the live backend),
  - embeds each labeled query image and ranks the catalog by cosine similarity,
  - LEAVE-ONE-OUT: excludes any catalog entry that IS the query (same source/
    image), so an image that's in the index can't trivially match itself -- the
    self-match leak HANDOFF.md warns about,
  - scores top-1 / top-5 accuracy (model match), AND the score separation:
    mean similarity of correct vs wrong top-1s, a manual ROC-AUC, and the best
    threshold (Youden's J) -- i.e. a data-driven CLIP_INDEX_MIN_SIM.

Ground truth comes from the query set's folder/label structure (same collectors
build_catalog_index.py uses). Default query set = label_data (human-confirmed).
For a clean read, also point --query-catalog/--query-dataset at images that are
NOT in the index.

Usage:
    python eval_index.py                              # queries = label_data
    python eval_index.py --query-dataset path/to/flat_dataset
    python eval_index.py --query-catalog path/to/brand/model/tree
    python eval_index.py --no-brand-filter            # ignore the brand gate
    python eval_index.py --csv results.csv            # per-query dump
"""
import argparse
import json
import os

import numpy as np

import config
from build_catalog_index import (collect_catalog_dir, collect_flat_dataset,
                                  collect_label_data)


def _norm(s):
    """Normalize a model/brand string for comparison (case + spacing)."""
    return " ".join(str(s or "").lower().replace("-", " ").replace("_", " ").split())


def load_index(index_path):
    """Load embeddings + metadata written by build_catalog_index.py."""
    stem = index_path[:-4] if index_path.endswith(".npz") else index_path
    emb = np.load(index_path)["embeddings"].astype("float32")
    with open(stem + ".json") as f:
        meta = json.load(f)
    return emb, meta["entries"], (meta.get("embedder") or {})


def collect_queries(args):
    """Gather labeled query entries (image + brand + model ground truth)."""
    entries = []
    if args.query_label_data:
        entries += collect_label_data(args.query_label_data)
    if args.query_catalog:
        entries += collect_catalog_dir(args.query_catalog)
    for d in args.query_dataset:
        entries += collect_flat_dataset(d)
    return entries


def roc_auc(scores, labels):
    """Manual ROC-AUC (probability a correct match outranks a wrong one).
    No sklearn dependency. Returns None if one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(scores, labels):
    """Threshold maximizing Youden's J (tpr - fpr). Returns (thr, tpr, fpr)."""
    pos_n = sum(labels)
    neg_n = len(labels) - pos_n
    if pos_n == 0 or neg_n == 0:
        return None
    best = (None, -1.0, 0.0, 0.0)
    for thr in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= thr)
        fp = sum(1 for s, y in zip(scores, labels) if not y and s >= thr)
        tpr, fpr = tp / pos_n, fp / neg_n
        j = tpr - fpr
        if j > best[1]:
            best = (thr, j, tpr, fpr)
    return best[0], best[2], best[3]


def main():
    ap = argparse.ArgumentParser(description="Evaluate the reverse-image index.")
    ap.add_argument("--index", default=config.CLIP_INDEX_PATH)
    ap.add_argument("--query-label-data", default=None,
                    help="label_data dir as the query set (default if no other "
                         "query source is given)")
    ap.add_argument("--query-catalog", default=None,
                    help="<brand>/<model>/*.jpg tree to use as queries")
    ap.add_argument("--query-dataset", action="append", default=[],
                    help="flat <brand>_<model>/*.jpg query dir (repeatable)")
    ap.add_argument("--no-brand-filter", action="store_true",
                    help="don't restrict candidates to the query's brand")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not (args.query_label_data or args.query_catalog or args.query_dataset):
        args.query_label_data = config.LABEL_DATA_DIR      # sensible default

    if not os.path.exists(args.index):
        print(f"No index at '{args.index}'. Build it: python build_catalog_index.py")
        return

    emb, entries, idx_embedder = load_index(args.index)
    queries = collect_queries(args)
    if not queries:
        print("No labeled query images found (try --query-dataset / --query-catalog).")
        return

    from embedder_utils import build_image_embedder
    import cv2
    embedder = build_image_embedder(config)
    if not embedder.ok:
        print("Embedder failed to load; cannot evaluate.")
        return
    # Same guard as the live backend: the index must match the embedder.
    built = idx_embedder.get("name")
    if built and built != embedder.name:
        print(f"Index built with '{built}' but config uses '{embedder.name}'. "
              f"Rebuild it (build_catalog_index.py) or switch EMBED_BACKEND.")
        return
    if emb.shape[1] != embedder.dim:
        print(f"Index is {emb.shape[1]}-d but embedder is {embedder.dim}-d. Rebuild.")
        return

    cat_brand = [_norm(e.get("brand")) for e in entries]
    cat_model = [_norm(e.get("model")) for e in entries]
    cat_src = [e.get("source") for e in entries]
    cat_img = [e.get("image") for e in entries]

    rows = []                       # (query_img, gt_model, top1_sim, top1_correct, in_top5)
    skipped = 0
    for q in queries:
        img = cv2.imread(q["image"])
        if img is None:
            skipped += 1
            continue
        vec = embedder.embed(img)
        sims = emb @ vec
        gt_model = _norm(q.get("model"))
        gt_brand = _norm(q.get("brand"))

        # Candidate pool: leave-one-out (drop the query itself) + optional brand gate.
        cand = []
        for i in range(len(entries)):
            if cat_src[i] == q.get("source") or cat_img[i] == q.get("image"):
                continue                                   # leave-one-out
            if not args.no_brand_filter and gt_brand and gt_brand != "unknown":
                if cat_brand[i] != gt_brand:
                    continue
            cand.append(i)
        if not cand:
            skipped += 1
            continue

        cand.sort(key=lambda i: sims[i], reverse=True)
        top1 = cand[0]
        topk = cand[:args.topk]
        top1_correct = bool(gt_model) and cat_model[top1] == gt_model
        in_topk = bool(gt_model) and any(cat_model[i] == gt_model for i in topk)
        rows.append((q["image"], q.get("model"), float(sims[top1]),
                     top1_correct, in_topk))

    if not rows:
        print(f"No queries could be evaluated (skipped {skipped}).")
        return

    n = len(rows)
    top1_acc = sum(r[3] for r in rows) / n
    topk_acc = sum(r[4] for r in rows) / n
    scores = [r[2] for r in rows]
    labels = [r[3] for r in rows]
    corr = [s for s, y in zip(scores, labels) if y]
    wrong = [s for s, y in zip(scores, labels) if not y]
    auc = roc_auc(scores, labels)
    thr = best_threshold(scores, labels)

    print(f"\n=== Index eval: {embedder.name}  |  {len(entries)} catalog imgs  |  "
          f"{n} queries evaluated ({skipped} skipped)  |  "
          f"brand_filter={'off' if args.no_brand_filter else 'on'} ===")
    print(f"top-1 accuracy : {top1_acc:.1%}")
    print(f"top-{args.topk} accuracy : {topk_acc:.1%}")
    print("\n-- score separation (the real question: do correct matches score higher?) --")
    if corr:
        print(f"correct top-1 sim: mean {np.mean(corr):.3f}  median {np.median(corr):.3f}  "
              f"min {np.min(corr):.3f}  (n={len(corr)})")
    if wrong:
        print(f"wrong   top-1 sim: mean {np.mean(wrong):.3f}  median {np.median(wrong):.3f}  "
              f"max {np.max(wrong):.3f}  (n={len(wrong)})")
    if auc is not None:
        print(f"ROC-AUC (correct vs wrong by sim): {auc:.3f}  "
              f"(0.5 = useless, 1.0 = perfectly separable)")
    if thr is not None:
        t, tpr, fpr = thr
        print(f"best threshold (Youden's J): CLIP_INDEX_MIN_SIM ~= {t:.3f}  "
              f"-> keeps {tpr:.0%} of correct, {fpr:.0%} of wrong")
    else:
        print("threshold: need both correct AND wrong top-1s to compute (none of one class).")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_image", "gt_model", "top1_sim", "top1_correct", "in_topk"])
            w.writerows(rows)
        print(f"\nper-query results -> {args.csv}")


if __name__ == "__main__":
    main()
