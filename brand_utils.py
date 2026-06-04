"""
brand_utils.py -- recognize the BRAND (make) of a shoe crop.

Phase B of the 2026 pivot. Pluggable like segment_utils so the backend can be
upgraded without touching callers (config.BRAND_BACKEND):
  "clip" -- local zero-shot CLIP. Embeds the crop and each brand prompt
            ("a photo of Nike shoes", ...) and picks the closest. No training,
            no API key, no cost. A baseline -- small logos limit accuracy, so the
            dashboard human-confirm stays the safety net.

Later backends (not built yet): a trained brand classifier (supercomputer) or a
vision-LLM API for higher accuracy.

Fail-safe: a load/inference error logs and returns ("unknown", None) so a caller
never crashes on one bad crop.
"""
import config


def _resolve_device():
    """Pick the inference device, honoring config.BRAND_DEVICE."""
    pref = getattr(config, "BRAND_DEVICE", "auto")
    if pref and pref != "auto":
        return pref
    try:
        from detector_utils import pick_device      # CUDA -> MPS -> CPU probe
        return pick_device()
    except Exception:                                # noqa: BLE001 - fail safe
        return "cpu"


class BrandClassifier:
    """Common interface. classify(image_bgr) -> (make:str, confidence:float|None)."""

    def classify(self, image_bgr):
        raise NotImplementedError


class ClipBrandClassifier(BrandClassifier):
    """Zero-shot brand recognition with OpenAI CLIP (local, no training)."""

    def __init__(self, model_name, classes, device, prompt, min_conf=0.0):
        self.classes = list(classes)
        self.device = device
        self.min_conf = min_conf
        self.ok = False
        try:
            import clip                              # OpenAI CLIP
            import torch
            self.torch = torch
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            # Pre-encode the brand text prompts once (they never change).
            prompts = [prompt.format(b) for b in self.classes]
            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self.text_features = feats
            self.ok = True
            print(f"[brand] CLIP backend ready: {model_name} on {device}, "
                  f"{len(self.classes)} brands")
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[brand] ERROR loading CLIP '{model_name}': {exc}")

    def classify(self, image_bgr):
        if not self.ok:
            return "unknown", None
        try:
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                feats = self.model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                probs = (100.0 * feats @ self.text_features.T).softmax(dim=-1)
            conf, idx = probs[0].max(0)
            conf = float(conf)
            make = self.classes[int(idx)]
            if conf < self.min_conf:
                return "unknown", conf
            return make, conf
        except Exception as exc:                     # noqa: BLE001 - fail safe
            print(f"[brand] classify failed: {exc}")
            return "unknown", None


def build_brand_classifier(cfg=None):
    """Construct the configured brand classifier (never raises; a failed load
    just yields ('unknown', None) on every classify)."""
    cfg = cfg or config
    backend = getattr(cfg, "BRAND_BACKEND", "clip").lower()
    classes = getattr(cfg, "BRAND_CLASSES", ["Nike", "Adidas"])
    device = _resolve_device()
    if backend != "clip":
        print(f"[brand] unknown BRAND_BACKEND '{backend}', using clip.")
    return ClipBrandClassifier(
        getattr(cfg, "BRAND_MODEL", "ViT-B/32"), classes, device,
        getattr(cfg, "BRAND_PROMPT", "a photo of {} shoes"),
        getattr(cfg, "BRAND_MIN_CONF", 0.0))
