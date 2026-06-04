# Phase C Summary — Model Identification

## What Phase C does

By now each pair has a **color** (Phase A) and a **brand/make** (Phase B). Phase C
fills the last piece: the **specific model** — e.g. `"Air Jordan 1"`,
`"Adidas Superstar"`, `"Nike Dunk Low"`. After Phase C a record is complete:

```json
{ "detected_color": "black", "make": "Jordan", "model": "Air Jordan 1",
  "make_confidence": 0.95, "model_confidence": 0.95, "model_sources": [] }
```

## Why this is the hardest phase

Brand was ~17 choices. **Model is tens of thousands** of silhouettes and
colorways, with new ones every week — there is no fixed "model classifier" you
can train once and be done. So instead of classifying, we **ask a vision AI to
recognize it**, the same way a knowledgeable person would.

## The process, part by part

### Part 1 — The model identifier (a local vision AI via Ollama)

We run a **vision language model (VLM)** — an AI that understands images *and*
text — **locally** on the Mac using **Ollama**. The model is `qwen2.5vl:7b`.

For each pair we:
1. Take the cropped photo of the pair.
2. Tell the model the **brand we already found** (from Phase B) — this is a big
   hint that narrows things down.
3. Ask: *"What specific model is this `<brand>` shoe? If unsure, say unknown.
   Answer as JSON: {model, confidence}."*
4. Read back the model name.

**Key points:**
- It runs **100% locally** — nothing is sent to the internet, no API key, no
  per-image cost. The model lives on the T7 drive and Ollama serves it on the
  Mac's GPU.
- "VLM" = a model trained on huge amounts of images + text, so it has **seen**
  thousands of sneakers and can often name them from memory.
- We give it the **brand** as context, which improves accuracy and keeps the
  answer consistent with Phase B.

### Part 2 — How good is it, and what about confidence

On our 16-pair test: **12 got a model name, 4 honestly returned "unknown."**
Verified-correct examples: Air Jordan 1, Adidas Superstar, Nike Dunk Low, Vans
Old Skool. A nice bonus: even some pairs whose *brand* came back "unknown" still
got a model (e.g. Nike Air Force 1), because the model AI can recognize the whole
shoe at once.

**The honest catch — confidence is NOT reliable here.** Unlike the brand step
(where the number was a real probability), this AI just *says* it's "0.95" for
almost everything it answers. So:
- The only confidence signal we trust is when it says **"unknown."**
- For everything else, the **human-confirm step** (in the dashboard) is the real
  check — especially for exact versions like "990v5" or "350 V2," which are
  plausible guesses but need a human to confirm.
- A future upgrade (Part 5) gives us a *real* confidence + a source link.

### Part 3 — Writing the answer back

`identify_models.py` runs the identifier on every pair crop and writes three
fields into the pair's JSON: `model`, `model_confidence`, and `model_sources`
(a list of source links — empty for now, since the AI doesn't cite anything). It
**also updates the matching `label_data` label**, so the curated training set
carries the full color + make + model. It's **idempotent** — pairs that already
have a model are skipped unless you pass `--force`.

### Part 4 — Where Ollama runs (the T7, on purpose)

The Mac's internal drive is small and was full, so **all the AI models live on
the T7 drive**. One quirk: the Ollama desktop app insists on using the internal
drive, so we run Ollama's server from the terminal pointed at the T7:

```bash
OLLAMA_MODELS=/Volumes/T7/ARIA/models/ollama ollama serve
```

(Or set the model location once in the Ollama app's Settings.)

### Part 5 — What comes next (and why)

Two known limitations have the same fix on the roadmap:
- the confidence isn't real, and
- we have no source link to prove the answer.

The plan is a **CLIP reverse-image index**: build a catalog of known sneaker
photos, and match each crop to it. That gives a **real similarity score** (true
confidence) and a **source link** for every answer — a fully-local "second
opinion" that verifies the VLM. And on the company **supercomputer**, we can run
a much larger vision model (e.g. Qwen3-VL 32B/72B) for a big accuracy jump on
obscure models.

---

## The files Phase C added

| File | What it does |
|------|--------------|
| `model_search.py` | The model identifier (local Ollama VLM). Backend is swappable for a CLIP index or a bigger model later. |
| `identify_models.py` | Runs it over all pairs and writes `model` + confidence + sources (and updates `label_data`). |
| `config.py` | New settings: which Ollama model, server URL, timeout. |
| `label_export.py` | Curated labels now include the model. |

## The settings you can tune (in `config.py`)

- `MODEL_OLLAMA_MODEL` (`qwen2.5vl:7b`) — which local vision model to use; swap a
  bigger one on the supercomputer.
- `MODEL_OLLAMA_URL` — where the Ollama server is (local by default).
- `MODEL_MIN_CONF` (0.0) — below this the answer becomes "unknown" (kept at 0 for
  now because the VLM's confidence is unreliable anyway).

## Status

✅ Phase C works end-to-end, locally, committed and pushed to `main`. The pipeline
now goes **photo → pairs → color → make → model**, all on-device and free.
