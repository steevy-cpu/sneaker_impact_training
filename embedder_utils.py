"""
embedder_utils.py -- turn an image into a vector for the reverse-image index.

ONE place owns "image -> embedding" so build_catalog_index.py and the
"clip-index" model backend (model_search.py) always build AND query the index
with the *same* embedder. Pick the backend with config.EMBED_BACKEND:

  "clip"   -- OpenAI CLIP (e.g. ViT-B/32, 512-d). The original baseline. Fast,
              but weak at fine-grained sneaker retrieval (crosses brands; the
              similarity scores don't separate right matches from wrong).
  "dinov2" -- Meta DINOv2 (self-supervised, built for instance / fine-grained
              retrieval). Much better at "same model, different photo". Local
              and free -- weights download once via torch.hub. Recommended.

Every embedder returns an L2-normalized float32 vector, so cosine similarity is
just a dot product (exactly how the index already compares).

Fail-safe: a load error leaves the embedder `.ok == False`; callers check it and
degrade to "unknown" instead of crashing.

NOTE: CLIP (512-d) and DINOv2 (384/768/1024/1536-d) produce DIFFERENT-sized
vectors, so switching EMBED_BACKEND means the index must be rebuilt. The query
side stores the embedder identity in the index and refuses a mismatch.
"""
import config

# ImageNet normalization for the DINOv2 preprocessing transform.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_embed_device(cfg=None):
    """Pick the embedder device, honoring config.EMBED_DEVICE ("auto" probes
    CUDA -> MPS -> CPU via detector_utils.pick_device)."""
    cfg = cfg or config
    pref = getattr(cfg, "EMBED_DEVICE", "auto")
    if pref and pref != "auto":
        return pref
    try:
        from detector_utils import pick_device          # CUDA -> MPS -> CPU
        return pick_device()
    except Exception:                                    # noqa: BLE001 - fail safe
        return "cpu"


class ImageEmbedder:
    """Common interface. embed(image_bgr) -> np.ndarray (D,) L2-normalized.

    Attributes: .ok (loaded?), .name (identity string), .dim (vector size)."""
    ok = False
    name = "none"
    dim = 0

    def embed(self, image_bgr):
        raise NotImplementedError


class ClipImageEmbedder(ImageEmbedder):
    """OpenAI CLIP image embedder (local, no API). The original baseline."""

    def __init__(self, model_name, device):
        self.device = device
        try:
            import clip                                  # OpenAI CLIP
            import torch
            self.torch = torch
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            self.dim = int(self.model.visual.output_dim)
            self.name = f"clip:{model_name}"
            self.ok = True
            print(f"[embed] CLIP ready: {model_name} ({self.dim}-d) on {device}")
        except Exception as exc:                         # noqa: BLE001 - fail safe
            print(f"[embed] ERROR loading CLIP '{model_name}': {exc}")

    def embed(self, image_bgr):
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype("float32")


class Dinov2ImageEmbedder(ImageEmbedder):
    """Meta DINOv2 image embedder. Loaded once via torch.hub; the CLS-token
    feature is L2-normalized and returned.

    Self-supervised and built for instance / fine-grained retrieval, so it is
    far better than CLIP at matching the SAME sneaker model across different
    photos. Variants: dinov2_vits14 (384-d), _vitb14 (768-d), _vitl14 (1024-d),
    _vitg14 (1536-d); the "_reg" register variants retrieve a little better.
    """

    def __init__(self, model_name, device):
        self.device = device
        try:
            import torch
            from torchvision import transforms
            self.torch = torch
            # Standard DINOv2 inference transform: resize -> center-crop 224 ->
            # ImageNet-normalize. (DINOv2 uses a patch size of 14; 224 is a
            # multiple of 14, so the crop tiles cleanly with no padding.)
            self.transform = transforms.Compose([
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
            # Downloads weights to ~/.cache/torch/hub on first use (needs net).
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.model.eval().to(device)
            self.dim = int(getattr(self.model, "embed_dim", 0)) or self._probe_dim()
            self.name = f"dinov2:{model_name}"
            self.ok = True
            print(f"[embed] DINOv2 ready: {model_name} ({self.dim}-d) on {device}")
        except Exception as exc:                         # noqa: BLE001 - fail safe
            print(f"[embed] ERROR loading DINOv2 '{model_name}': {exc} "
                  f"(first load needs internet; requires torch + torchvision)")

    def _probe_dim(self):
        """Fallback: measure the output width with one dummy forward pass."""
        x = self.torch.zeros(1, 3, 224, 224, device=self.device)
        with self.torch.no_grad():
            out = self.model(x)
        return int(out.shape[-1])

    def embed(self, image_bgr):
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model(tensor)                    # CLS-token feature (1, D)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype("float32")


def build_image_embedder(cfg=None):
    """Construct the configured image embedder (never raises; a failed load just
    yields .ok == False, and callers degrade to "unknown")."""
    cfg = cfg or config
    backend = getattr(cfg, "EMBED_BACKEND", "dinov2").lower()
    device = resolve_embed_device(cfg)
    if backend == "clip":
        return ClipImageEmbedder(
            getattr(cfg, "CLIP_INDEX_MODEL", "ViT-B/32"), device)
    if backend != "dinov2":
        print(f"[embed] unknown EMBED_BACKEND '{backend}', using dinov2.")
    return Dinov2ImageEmbedder(
        getattr(cfg, "EMBED_DINOV2_MODEL", "dinov2_vitl14_reg"), device)
