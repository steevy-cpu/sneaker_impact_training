# Session Handoff — Sneaker Impact (2026 pivot)

> **Read this first** if you're a new Claude session, especially on the
> **supercomputer**. It captures where we left off. Then read `CLAUDE.md` (full
> module/phase docs) and `PHASE_A_SUMMARY.md` / `PHASE_B_SUMMARY.md` /
> `PHASE_C_SUMMARY.md` (plain-language explanations). Last commit on `main` when
> this was written: see `git log` (was `be5e870`, branch `Steeve` == `main`).

## Why we're switching machines
All work so far was on a **MacBook Air (M2, 16 GB, macOS)** with the repo on an
**exFAT T7 external drive**. We're now moving to the company **supercomputer**
(big CUDA GPUs) to run heavy models and train. The pipeline runs *correctly* on
the Mac but is capped by small models + a tiny GPU; the supercomputer unlocks the
accuracy/scale steps.

## The project in one paragraph
CEO pivot: photograph a **whole table of shoes** → in the background, **segment**
it into individual **pairs** (shoes arrive tied in pairs → one record per pair) →
identify each pair's **color**, **make** (brand), and **model** → store a labeled
dataset (no Reuse/Recycle anymore). Realistic batch ≈ 16–20 pairs per table. The
old live click-to-label app (`label_live.py`, tracking) is preserved but OFF the
main path. A separate FastAPI/SQLite **dashboard** visualizes data; the company's
**Airtable** is intake-only (no pictures) and not yet wired to this new flow.

## Pipeline status (all on `main`)

| Phase | What | Status | Entry point |
|------|------|--------|-------------|
| A | table photo → segment (YOLOE-26 + **tiling**) → **pair** → per-pair crop + JSON | ✅ works | `python split_table.py table.jpg --viz` |
| B | fill `make` (brand) — CLIP zero-shot | ✅ works (baseline) | `python identify_brands.py` |
| B+ | export confident pairs → `label_data/shoes_<color>_<make>_N.jpg` | ✅ works | (runs inside identify_brands) |
| C | fill `model` — **local Ollama VLM** (qwen2.5vl:7b) | ✅ works (primary) | `python identify_models.py` |
| C-verify | CLIP reverse-image index | ⚠️ infra works, **embedder insufficient** | `python build_catalog_index.py` |
| — | color naming = **CIELAB**, single dominant color (no "multi") | ✅ | `color_utils.py` |

Per-pair JSON accumulates: `detected_color`, `make`/`make_confidence`,
`model`/`model_confidence`/`model_sources`, plus `bbox`, `segment_*`. Helpers:
`ingest_table.py` (names incoming photos `table1.jpg`…), `dataset_*` tools.

## What works vs. what doesn't (honest)
- **Segmentation:** a single wide photo missed ~95% of a crowded table; **tiling**
  fixed recall, then we detect single shoes and **pair them geometrically**
  (32 shoes → 16 pairs). Good.
- **Brand (CLIP zero-shot):** nails iconic brands (Jordan/Adidas/New Balance) but
  is uncalibrated mid-range (a New Balance once read "Saucony" 0.47). Floor at
  `BRAND_MIN_CONF=0.35` → weak → "unknown".
- **Model (Ollama VLM):** **the primary, working model-ID.** Named Air Jordan 1,
  Adidas Superstar, Nike Dunk Low, Vans Old Skool correctly. BUT its self-
  confidence is **uncalibrated (flat 0.95)** — only "unknown" is trustworthy;
  human-confirm is the real check.
- **CLIP index verifier:** built a real 5,959-image index (Kaggle dataset +
  label_data). **Off-the-shelf CLIP ViT-B/32 is NOT good enough** — on cross-
  photo pairs top-1 was wrong and crossed brands (Air Force 1 → Reebok/Vans;
  Dunk → Cortez), scores ~0.78–0.84 with no separating threshold. Infra is
  correct/reusable; **the embedder is the bottleneck** (+ domain gap: our top-
  down used-shoe table crops vs. clean product shots).

## >>> IMMEDIATE NEXT TASK <<<
**Swap the index embedder from CLIP to DINOv2** (or a shoe-fine-tuned embedder)
and re-run the verifier test. Keep ALL the index infrastructure
(`build_catalog_index.py`, the `clip-index`/index backend in `model_search.py`,
the catalog + label_data ingestion, brand filter) — only change the function
that turns an image into a vector. DINOv2 is self-supervised for instance/fine-
grained retrieval (much better than CLIP at "same model, different photo"),
local, free. Re-test with the same comparison we ran (CLIP-match vs VLM guess on
the 16 pairs; focus on pairs NOT in label_data). If DINOv2 still struggles, the
real fix is a **fine-tuned embedder trained on our growing labels** — a
supercomputer job.

## Bigger supercomputer jobs (the reason we moved)
1. **Heavy VLM for model-ID:** swap `config.MODEL_OLLAMA_MODEL` to `qwen3-vl:32b`
   or `qwen2.5vl:72b` (needs lots of VRAM) for a big accuracy jump on obscure
   models. Same `model_search.py` ollama backend — just the model name.
2. **Train a brand classifier** to replace CLIP zero-shot (Phase B), on the
   accumulating human-confirmed `label_data`.
3. **Fine-tune / train an embedder** for the index (the real Phase-C-verify fix).
4. **Bigger YOLOE-26 (l/x) at higher res** for segmentation on dense tables.

## Environment setup on the supercomputer (fresh clone)
The repo is on GitHub (`steevy-cpu`/this repo… confirm remote with `git remote -v`).
After cloning:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # ultralytics pulls torch + CLIP
pip install kaggle                        # for the catalog dataset
# Ollama: install, then `ollama pull qwen2.5vl:7b` (or a heavy qwen3-vl on big GPUs)
```
- `detector_utils.pick_device()` auto-picks **CUDA** on the supercomputer (it was
  MPS on the Mac) — no code change needed.
- Nothing should hardcode paths/thresholds — it's all in `config.py`.

## IMPORTANT: data & models do NOT come with the repo
These are git-ignored, so a fresh clone will NOT have them — regenerate or copy
from the Mac/T7:
- `sneaker_impact/pairs/` — per-pair crops + JSON (regenerate: `split_table.py`)
- `label_data/` — curated confirmed labels (regenerate via the pipeline, or copy)
- `sneaker_impact/catalog/`, `downloads/`, `sneaker_impact/clip_index.*` — catalog
  + built index (re-download the Kaggle set + `build_catalog_index.py`)
- **Table photos** themselves (e.g. `table1.jpg`) live on the Mac/T7 — copy the
  ones you want to process.
- **Ollama models** lived on the Mac's T7 (`/Volumes/T7/ARIA/models/ollama`); the
  supercomputer pulls its own with `ollama pull`.
- **Kaggle token** was at `~/.kaggle/access_token` on the Mac; set up Kaggle
  creds on the supercomputer to re-download `nikolasgegenava/sneakers-classification`.

## Mac-only gotchas (do NOT apply on the supercomputer)
- The T7 is **exFAT**, so macOS scattered `._` AppleDouble files that broke
  matplotlib/ultralytics, git, and dataset counts — we cleaned with
  `find <dir> -name '._*' -delete`. A normal Linux filesystem on the
  supercomputer won't have this problem.
- The Mac's internal disk was full, so Ollama models were forced onto the T7 and
  we ran `OLLAMA_MODELS=/Volumes/T7/ARIA/models/ollama ollama serve` from the
  shell (the GUI app overrode the env). On the supercomputer just use Ollama
  normally.

## Key config knobs (`config.py`)
Segmentation: `SEGMENT_BACKEND/MODEL/PROMPTS/IMGSZ/TILE*/PAIR*`. Brand:
`BRAND_BACKEND/MODEL/CLASSES/MIN_CONF`. Label export: `LABEL_*_MIN_CONF`. Model:
`MODEL_BACKEND` (`ollama`/`clip-index`), `MODEL_OLLAMA_MODEL`. Index:
`CLIP_INDEX_*`, `CLIP_DATASET_DIRS`. License note: YOLO26/YOLOE-26 is **AGPL-3.0**
(fine internal; SAM 2 / Enterprise before shipping a product).

## Workflow conventions
Commit + push only when asked; branch `Steeve` is fast-forwarded to `main` then
both pushed. End commit messages with the Co-Authored-By line. Each phase has a
`PHASE_*_SUMMARY.md` written in plain language for the user to explain to others.
