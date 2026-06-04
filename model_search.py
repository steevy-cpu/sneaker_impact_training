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


def build_model_identifier(cfg=None):
    """Construct the configured model identifier (never raises; failures yield
    ('unknown', None, []) on every identify)."""
    cfg = cfg or config
    backend = getattr(cfg, "MODEL_BACKEND", "ollama").lower()
    if backend != "ollama":
        print(f"[model] unknown MODEL_BACKEND '{backend}', using ollama.")
    return OllamaModelIdentifier(
        getattr(cfg, "MODEL_OLLAMA_MODEL", "qwen2.5vl:7b"),
        getattr(cfg, "MODEL_OLLAMA_URL", "http://localhost:11434"),
        getattr(cfg, "MODEL_OLLAMA_TIMEOUT", 180),
        getattr(cfg, "MODEL_MIN_CONF", 0.0))
