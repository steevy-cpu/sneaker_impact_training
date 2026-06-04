"""
model_search.py -- identify the specific MODEL of a shoe (Phase C).

Pluggable like segment_utils/brand_utils. Backend "ollama" asks a LOCAL Ollama
vision model (e.g. qwen2.5vl) "what model is this <brand> shoe?" and parses
{model, confidence}. Free, private, on-device.

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


class ClipIndexModelIdentifier(ModelIdentifier):
    """Identify the model by matching the crop against a CLIP catalog index
    (built by build_catalog_index.py). Returns the nearest catalog model, the
    cosine similarity as a REAL confidence, and that entry's source link."""

    def __init__(self, index_path, model_name, device, min_sim, brand_filter):
        self.min_sim = min_sim
        self.brand_filter = brand_filter
        self.ok = False
        try:
            import clip
            import numpy as np
            import torch
            self.np = np
            self.torch = torch
            stem = index_path[:-4] if index_path.endswith(".npz") else index_path
            self.emb = np.load(index_path)["embeddings"].astype("float32")
            with open(stem + ".json") as f:
                meta = json.load(f)
            self.entries = meta["entries"]
            self.device = device
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            self.ok = len(self.entries) > 0 and len(self.entries) == self.emb.shape[0]
            print(f"[model] CLIP-index ready: {len(self.entries)} catalog images "
                  f"on {device}")
        except Exception as exc:                       # noqa: BLE001 - fail safe
            print(f"[model] ERROR loading CLIP index '{index_path}': {exc} "
                  f"(run build_catalog_index.py)")

    def identify(self, image_bgr, brand):
        if not self.ok:
            return "unknown", None, []
        try:
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            q = feat.cpu().numpy()[0].astype("float32")

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
            print(f"[model] clip-index identify failed: {exc}")
            return "unknown", None, []


def build_model_identifier(cfg=None):
    """Construct the configured model identifier (never raises; failures yield
    ('unknown', None, []) on every identify)."""
    cfg = cfg or config
    backend = getattr(cfg, "MODEL_BACKEND", "ollama").lower()
    if backend == "clip-index":
        return ClipIndexModelIdentifier(
            getattr(cfg, "CLIP_INDEX_PATH", "sneaker_impact/clip_index.npz"),
            getattr(cfg, "CLIP_INDEX_MODEL", "ViT-B/32"),
            _clip_device(cfg),
            getattr(cfg, "CLIP_INDEX_MIN_SIM", 0.75),
            getattr(cfg, "CLIP_INDEX_BRAND_FILTER", True))
    if backend != "ollama":
        print(f"[model] unknown MODEL_BACKEND '{backend}', using ollama.")
    return OllamaModelIdentifier(
        getattr(cfg, "MODEL_OLLAMA_MODEL", "qwen2.5vl:7b"),
        getattr(cfg, "MODEL_OLLAMA_URL", "http://localhost:11434"),
        getattr(cfg, "MODEL_OLLAMA_TIMEOUT", 180),
        getattr(cfg, "MODEL_MIN_CONF", 0.0))


def _clip_device(cfg):
    pref = getattr(cfg, "BRAND_DEVICE", "auto")        # reuse the same probe
    if pref and pref != "auto":
        return pref
    try:
        from detector_utils import pick_device
        return pick_device()
    except Exception:                                  # noqa: BLE001 - fail safe
        return "cpu"
