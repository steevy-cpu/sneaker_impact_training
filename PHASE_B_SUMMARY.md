# Phase B Summary — Brand (Make) Recognition

## What Phase B does

Phase A gave us a clean cropped picture of each **pair** of shoes. Phase B looks
at each crop and **fills in the brand** (the `make` field) — Nike, Adidas,
Jordan, New Balance, and so on. It does **not** figure out the exact model yet
(that's Phase C).

So after Phase B, each pair's data file goes from this:

```json
{ "make": null, "model": null, "detected_color": "black" }
```

to this:

```json
{ "make": "Jordan", "make_confidence": 0.95, "model": null, "detected_color": "black" }
```

---

## The process, part by part

### Part 1 — The make classifier (how it actually works)

We use a model called **CLIP**. The trick CLIP uses is called **zero-shot
classification** — "zero-shot" means *we never trained it on shoe brands*; it
uses general knowledge it already learned.

Here's the idea in plain words:

1. CLIP was trained on hundreds of millions of **(image, caption)** pairs from
   the internet. From that, it learned to turn **any image** and **any sentence**
   into a list of numbers (a **vector**, also called an **embedding**) in the
   same "meaning space." Pictures and the words that describe them land close
   together.
2. We write one sentence per brand: `"a photo of Nike shoes"`,
   `"a photo of Adidas shoes"`, … one for each brand in our list. We turn each
   sentence into a vector **once** (it never changes).
3. For a shoe crop, we turn the **image** into a vector.
4. We measure which **brand sentence** is closest to the **image** (this
   closeness is called **cosine similarity**).
5. We convert those closeness scores into percentages that add up to 100% across
   all brands (a step called **softmax**). The winning brand's percentage is our
   **confidence**.

So "Jordan 0.95" means: *of all the brands in our list, the crop looks most like
"a photo of Jordan shoes," and the model is 95% sure compared to the others.*

This all runs **locally on the Mac** using the GPU (MPS). No internet, no API
key, no training, no cost.

### Part 2 — Confidence: what counts as high vs. low

The confidence is a **probability across our brand list** (we have 17 brands, so
a pure random guess would be about 1 ÷ 17 ≈ **6%**). We use three bands:

| Band | Confidence | What we do | Verified reality |
|------|-----------|------------|------------------|
| **High** | above ~0.8 | Trust it | Jordan 0.95, Adidas 0.91, New Balance 0.89 — **all correct** |
| **Mid** | 0.35 – 0.8 | Keep as a guess, but flag for human review | A New Balance pair came back "Saucony" at 0.47 — **wrong** |
| **Low** | below **0.35** (`BRAND_MIN_CONF`) | Don't trust it → label `"unknown"` | weak guesses on plain shoes |

The key honesty point: **confidence is a hint, not a guarantee.** High scores are
reliable; mid scores are sometimes wrong; low scores we throw out. That's exactly
why the pipeline keeps a **human-confirm** step (in the dashboard) and why the
long-term plan is to **train our own brand model** on the supercomputer once we
have enough confirmed examples.

### Part 3 — Writing the answer back

`identify_brands.py` runs the classifier on every pair crop and writes `make` +
`make_confidence` into that pair's JSON. It is **idempotent**: a pair that
already has a brand is skipped on the next run (unless you pass `--force`), so
it's safe to re-run as new batches come in.

### Part 4 — The curated `label_data/` folder (the clean subset)

Not every result is good enough to keep as training data. So we copy **only the
best ones** into a separate folder called `label_data/`. A pair qualifies only
if **both**:

- its **make** is confident (≥ `0.60`) and not `"unknown"`, **and**
- its **color** is confident (≥ `0.50`) and a real single color (not `"unknown"`
  or `"multi"`).

Qualifying crops are copied with a clear, self-describing name:

```
shoes_<color>_<make>_<N>.jpg      e.g.  shoes_white_adidas_1.jpg
                                        shoes_blue_newBalance_1.jpg
```

(Multi-word brands like "New Balance" become `newBalance`.) Each gets a small
label file beside it that traces back to the original photo and pair. This is
**idempotent too** — re-running won't make duplicates.

> Example from our test: of 16 pairs, **5** made it into `label_data/`
> (Jordan, Adidas, Converse, Yeezy, Vans). The very-confident **New Balance**
> pair did **not** — its color was `"multi"`, so it failed the color rule. That's
> the "confident in *both*" rule doing its job.

### Part 5 — Logical table-photo names

`ingest_table.py` renames incoming table photos to `table1.jpg`, `table2.jpg`, …
before processing, so every pair record's `source_photo` is clean and traceable
(you can always tell which table photo a pair came from).

```bash
python ingest_table.py raw_camera_shot.jpg   # -> table_photos/table1.jpg
python split_table.py --all                   # process everything in that folder
```

---

## The files Phase B added

| File | What it does |
|------|--------------|
| `brand_utils.py` | The brand classifier (CLIP zero-shot). Backend is swappable for a trained model or a vision-LLM later. |
| `identify_brands.py` | Runs the classifier over all pairs and writes `make` + `make_confidence`. |
| `label_export.py` | Copies the high-confidence pairs into `label_data/`. |
| `ingest_table.py` | Gives incoming table photos logical names (`table1`, `table2`, …). |
| `config.py` | New settings: brand list, confidence floors, export thresholds, photo prefix. |

## The settings you can tune (in `config.py`)

- `BRAND_CLASSES` — the list of brands it can choose from.
- `BRAND_MIN_CONF` (0.35) — below this, a guess becomes `"unknown"`.
- `LABEL_MAKE_MIN_CONF` (0.60) / `LABEL_COLOR_MIN_CONF` (0.50) — how confident a
  pair must be to be copied into `label_data/`.
- `BRAND_MODEL` (`ViT-B/32`) — the CLIP size; a bigger one is more accurate but
  slower.

## Status & what's next

✅ Phase B works end-to-end on the Mac, committed and pushed to `main`.

**Phase C (next):** take the crop + the confirmed brand and look up the exact
**model** (e.g. "Air Jordan 1 Mid") using a sneaker database/API, filling
`model` / `model_confidence` / `model_sources`. That step needs internet access
and likely an API key, so we'll choose the service first.
