"""
model_search.py -- identify the specific MODEL of a shoe (Phase C).

Pluggable like segment_utils/brand_utils (config.MODEL_BACKEND):
  "ollama"     -- ask a LOCAL Ollama vision model (e.g. qwen2.5vl) "what model is
                  this <brand> shoe?" and parse {model, confidence}. Free, private.
  "clip-index" -- match the crop against a reverse-image index for a REAL
                  similarity + a source link (see build_catalog_index.py).
  "hybrid"     -- the VLM proposes the name and the index verifies it, attaching
                  the index's real similarity + source when they agree.

The VLM's self-reported confidence is NOT calibrated (often a flat 0.95), so
treat "unknown" as the real signal; true verification will come from the future
CLIP-index backend and the dashboard human-confirm step.

Fail-safe: any error -> ("unknown", None, []). The third value is a list of
source URLs (empty for the VLM, which cites nothing; populated later by a
sneaker-DB / CLIP-index backend).
"""
import base64
import json
import re
import urllib.request

import config


def _parse_json(text):
    """Pull a JSON object out of an LLM reply (it may wrap it in ``` fences)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:                                  # noqa: BLE001
        return None


def _norm_model(s):
    """Normalize a model name for comparison (case + spacing/punctuation)."""
    return " ".join(str(s or "").lower().replace("-", " ").replace("_", " ").split())


class ModelIdentifier:
    """Common interface. identify(image_bgr, brand) -> (model, confidence, sources)."""

    def identify(self, image_bgr, brand):
        raise NotImplementedError


class OllamaModelIdentifier(ModelIdentifier):
    """Identify the model with a local Ollama vision model."""

    def __init__(self, model, url, timeout, min_conf=0.0):
        self.model = model
        self.url = url.rstrip("/") + "/api/generate"
        self.timeout = timeout
        self.min_conf = min_conf

    def _encode(self, image_bgr):
        import cv2
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ValueError("could not encode image")
        return base64.b64encode(buf.tobytes()).decode()

    def identify(self, image_bgr, brand):
        try:
            b64 = self._encode(image_bgr)
            brand_txt = brand if brand and brand.lower() != "unknown" else ""
            prompt = (
                f"This is a photo of a pair of {brand_txt} shoes. Identify the "
                f"specific model / silhouette name (for example 'Air Jordan 1', "
                f"'New Balance 990v5', 'Adidas Samba'). If you are not sure, use "
                f'"unknown". Reply ONLY as JSON: '
                f'{{"model": "<name or unknown>", "confidence": <0.0-1.0>}}.'
            )
            payload = {
                "model": self.model, "prompt": prompt, "images": [b64],
                "stream": False, "format": "json",
                "options": {"temperature": 0},
            }
            req = urllib.request.Request(
                self.url, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())

            parsed = _parse_json(body.get("response", ""))
            if not parsed:
                return "unknown", None, []
            name = (parsed.get("model") or "unknown").strip()
            conf = parsed.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else None
            if not name or name.lower() == "unknown":
                return "unknown", conf, []
            if conf is not None and conf < self.min_conf:
                return "unknown", conf, []
            return name, conf, []                      # VLM has no source links
        except Exception as exc:                       # noqa: BLE001 - fail safe
            print(f"[model] identify failed: {exc}")
            return "unknown", None, []


class IndexModelIdentifier(ModelIdentifier):
    """Identify the model by matching the crop against a reverse-image index
    (built by build_catalog_index.py). The embedder is config-driven
    (config.EMBED_BACKEND: "clip" or "dinov2") via embedder_utils, and must be
    the SAME one the index was built with -- this checks the index's stored
    identity + vector size and disables itself (with a "rebuild" hint) on a
    mismatch. Returns the nearest catalog model, the cosine similarity as a REAL
    confidence, and that entry's source link."""

    def __init__(self, cfg, index_path, min_sim, brand_filter):
        self.min_sim = min_sim
        self.brand_filter = brand_filter
        self.ok = False
        try:
            import numpy as np
            from embedder_utils import build_image_embedder
            self.np = np
            stem = index_path[:-4] if index_path.endswith(".npz") else index_path
            self.emb = np.load(index_path)["embeddings"].astype("float32")
            with open(stem + ".json") as f:
                meta = json.load(f)
            self.entries = meta["entries"]
            self.embedder = build_image_embedder(cfg)
            if not self.embedder.ok:
                print("[model] embedder failed to load; clip-index disabled.")
                return
            # The index is only valid for the embedder that built it. Refuse a
            # stale index (e.g. a CLIP index after switching to DINOv2) instead
            # of producing garbage cosine scores.
            built = (meta.get("embedder") or {}).get("name")
            if built and built != self.embedder.name:
                print(f"[model] index built with '{built}' but config uses "
                      f"'{self.embedder.name}'. Rebuild: build_catalog_index.py")
                return
            if self.emb.shape[1] != self.embedder.dim:
                print(f"[model] index is {self.emb.shape[1]}-d but embedder is "
                      f"{self.embedder.dim}-d. Rebuild: build_catalog_index.py")
                return
            self.ok = len(self.entries) > 0 and len(self.entries) == self.emb.shape[0]
            print(f"[model] index ready: {len(self.entries)} catalog images, "
                  f"{self.embedder.name}")
        except Exception as exc:                       # noqa: BLE001 - fail safe
            print(f"[model] ERROR loading index '{index_path}': {exc} "
                  f"(run build_catalog_index.py)")

    def identify(self, image_bgr, brand):
        if not self.ok:
            return "unknown", None, []
        try:
            q = self.embedder.embed(image_bgr)
            sims = self.emb @ q                        # cosine (both normalized)

            # Restrict to catalog entries of the known brand. If the brand isn't
            # in the catalog at all, we can't verify -> "unknown" (do NOT fall
            # back to other brands, which produced false matches).
            idxs = list(range(len(self.entries)))
            if self.brand_filter and brand and brand.lower() != "unknown":
                idxs = [i for i in idxs
                        if (self.entries[i].get("brand") or "").lower() == brand.lower()]
                if not idxs:
                    return "unknown", None, []
            best_i = max(idxs, key=lambda i: sims[i])
            best_sim = float(sims[best_i])
            if best_sim < self.min_sim:
                return "unknown", best_sim, []
            e = self.entries[best_i]
            return e["model"], best_sim, [e["source"]]
        except Exception as exc:                       # noqa: BLE001 - fail safe
            print(f"[model] index identify failed: {exc}")
            return "unknown", None, []


# Back-compat alias: this backend is no longer CLIP-only (see config.EMBED_BACKEND).
ClipIndexModelIdentifier = IndexModelIdentifier


class HybridModelIdentifier(ModelIdentifier):
    """VLM proposes the model; the reverse-image index verifies it.

    The VLM (Ollama) names the model well but its confidence is uncalibrated
    (flat ~0.95). The index gives a REAL cosine similarity + a source link. We
    keep the VLM's name and use the index to attach a trustworthy confidence:

      - index's nearest match AGREES (same model) and sim >= min_sim
            -> verified: return (vlm_model, sim, [catalog source]).
      - index is confident (sim >= min_sim) but DISAGREES
            -> return (vlm_model, None, ["index-disagrees:<model> (sim ..)", ...])
               so the conflict is visible to the human; VLM stays primary.
      - index can't verify (no verifier, brand absent, or below min_sim)
            -> return (vlm_model, None, []): unverified, lean on human-confirm.

    So a NON-NULL confidence here means "index-verified", with a source to prove
    it -- exactly the trustworthy signal the VLM alone never gave.
    """

    def __init__(self, proposer, verifier, min_sim):
        self.proposer = proposer
        self.verifier = verifier
        self.min_sim = min_sim

    def identify(self, image_bgr, brand):
        model, _vlm_conf, _ = self.proposer.identify(image_bgr, brand)
        if not model or model.lower() == "unknown":
            return "unknown", None, []                 # nothing to verify
        if not getattr(self.verifier, "ok", False):
            return model, None, []                     # no verifier -> unverified

        idx_model, idx_sim, idx_sources = self.verifier.identify(image_bgr, brand)
        verifiable = (idx_model and idx_model.lower() != "unknown"
                      and idx_sim is not None and idx_sim >= self.min_sim)
        if not verifiable:
            return model, None, []                     # could not verify
        if _norm_model(idx_model) == _norm_model(model):
            return model, idx_sim, idx_sources         # VERIFIED: real conf + source
        note = f"index-disagrees:{idx_model} (sim {idx_sim:.2f})"
        return model, None, [note] + list(idx_sources)  # conflict flagged for human


def _build_ollama(cfg):
    return OllamaModelIdentifier(
        getattr(cfg, "MODEL_OLLAMA_MODEL", "qwen2.5vl:7b"),
        getattr(cfg, "MODEL_OLLAMA_URL", "http://localhost:11434"),
        getattr(cfg, "MODEL_OLLAMA_TIMEOUT", 180),
        getattr(cfg, "MODEL_MIN_CONF", 0.0))


def _build_index(cfg):
    return IndexModelIdentifier(
        cfg,
        getattr(cfg, "CLIP_INDEX_PATH", "sneaker_impact/clip_index.npz"),
        getattr(cfg, "CLIP_INDEX_MIN_SIM", 0.75),
        getattr(cfg, "CLIP_INDEX_BRAND_FILTER", True))


def build_model_identifier(cfg=None):
    """Construct the configured model identifier (never raises; failures yield
    ('unknown', None, []) on every identify)."""
    cfg = cfg or config
    backend = getattr(cfg, "MODEL_BACKEND", "ollama").lower()
    if backend == "clip-index":
        return _build_index(cfg)
    if backend == "hybrid":                            # VLM proposes, index verifies
        return HybridModelIdentifier(
            _build_ollama(cfg), _build_index(cfg),
            getattr(cfg, "CLIP_INDEX_MIN_SIM", 0.75))
    if backend != "ollama":
        print(f"[model] unknown MODEL_BACKEND '{backend}', using ollama.")
    return _build_ollama(cfg)
