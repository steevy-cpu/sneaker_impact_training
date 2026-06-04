# Sneaker Impact — Project Summary

## What we're building (the big picture)

We're building a system that takes a **photo of a whole table of shoes** and
automatically turns it into an organized **dataset** — where every pair of shoes
is identified by its **make** (brand, like Nike) and **model** (like Air Force 1).

This is a change of direction. The old version watched a live camera and a
person clicked each shoe as "Reuse" or "Recycle." The new goal from the CEO:
snap one picture of the table, and let the computer do the sorting **in the
background** — no clicking, no Reuse/Recycle for now. Just: *what shoe is this?*

The full plan, start to finish:

```
PHOTO OF TABLE  ->  find each pair of shoes  ->  cut them into separate pictures
                ->  identify the BRAND  ->  look up the MODEL  ->  save to a dataset
```

Three phases of "smarts":

- **Phase A (done):** find and cut out each pair of shoes.
- **Phase B (next):** recognize the **brand**.
- **Phase C (later):** figure out the exact **model**, by having the computer
  search a sneaker database.

The finished data also feeds your **dashboard** (the website you built) for
viewing, while the company's **Airtable** stays as the separate "shoes arrived"
intake list.

---

## What we just built — Phase A

**Goal:** take one table photo and produce one clean cropped picture (plus a data
file) for **each pair of shoes**.

### The tools we used (plain words + the technical terms)

- **YOLO26** — the brand-new AI vision model (released January 2026). We use
  **YOLOE-26**, the "open-vocabulary" version, which means you can just *type
  words* like `"shoe"` and it finds those things in a picture — no training
  needed. This is called **segmentation**: the model outlines each object it
  finds.
- It runs **locally on your Mac**, using **MPS** (your Apple Silicon GPU).
  Nothing goes to the cloud.

### The two real problems we hit and solved

1. **The model missed almost everything.** On a crowded table, one wide photo
   made each shoe tiny, and the model only found **6 out of ~80 shoes**.
   **Fix — tiling (technical name: SAHI):** we slice the big photo into smaller
   overlapping squares, look at each square up close, then stitch the results
   back together. Recall (how many it catches) jumped from **6 -> 32** clean
   shoes.

2. **We needed pairs, but it found single shoes.** After tiling, it found every
   *individual* shoe (32 of them) — but you want **one record per pair**.
   **Fix — pairing:** since tied shoes sit right next to each other, we group the
   **two nearest shoes** into one pair. Result: **32 shoes -> 16 pairs.** Exactly
   right for that table.

### What you get out of it

Run one command:

```bash
python split_table.py table.jpg --viz
```

and you get, for each pair:

- a cropped **`.jpg`** of just that pair, and
- a **`.json`** data file with the color, location, confidence, and **empty slots
  for `make` and `model`** — ready for Phase B to fill in.

### Key decisions we locked in

- **License:** YOLO26 is free under **AGPL-3.0** while this stays internal. The
  code is built so we can swap to **SAM 2** (a free-for-commercial model) or buy
  a license before it becomes a product — a decision to make with the CEO later.
- **Mac vs. supercomputer:** the Mac is plenty for *developing* and running the
  *small* model on a few photos. The supercomputer is for the heavy stuff
  later — **training** your own brand recognizer and running the **biggest**
  models fast at volume.

### The files Phase A added

| File | What it does |
|------|--------------|
| `segment_utils.py` | Finds the shoes (YOLOE-26) + tiling for crowded tables. Backend is swappable (YOLOE-26 or SAM 2). |
| `pair_utils.py` | Groups the single shoes into tied pairs (one record per pair). |
| `split_table.py` | The command you run: photo -> find -> pair -> save crops + data files. |
| `config.py` | New "Table segmentation" settings block (all the knobs live here). |

### Status

Working end-to-end on the Mac (MPS, `yoloe-26s-seg.pt`), committed and pushed to
`main`. A realistic 16-pair table produces 16 clean pair crops.

**Realistic batch size:** ~16–20 pairs per table, not 70+.

---

## What's next — Phase B (brand recognition)

Fill in the `make` field for each pair: start with a vision model that names the
brand, and later train our own brand classifier on the supercomputer. Then
Phase C looks up the exact model using a sneaker database/API.
