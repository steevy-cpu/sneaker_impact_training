"""
clip_shoeness.py -- zero-shot "is this crop footwear?" using OpenAI CLIP.

Replaces the custom-trained resnet18 shoe/not-shoe gate: CLIP already knows what
a shoe / fan / box / bag is (trained on 400M internet images), needs NO training,
and is ALREADY loaded in the pipeline for brand classification -- so this is
nearly free and far more robust on shoe-shaped junk (e.g. a white plastic blower)
that the trained gate waved through.

P(footwear) = softmax over CLIP image-vs-text similarity, summed over the
footwear text classes. Validated 2026-07-01: blower -> 0.000, real shoes -> >=0.96.

Fail-safe: any load/inference error -> ok=False / score None (callers treat "no
score" as "don't block", so the live flow never breaks).
"""
# Footwear vocabulary is deliberately broad: running-shoe-only prompts under-score
# FLAT casual sneakers (Dunk / Air Force 1 / Blazer / Vans / Superstar), which are
# real shoes. With these, junk (blower/box/bag) scores ~0.00 while every real shoe
# -- chunky or flat -- scores >=0.39, so a LOW floor (~0.10) separates them cleanly.
FOOTWEAR = ["a shoe", "a sneaker", "a running shoe", "a sandal", "a boot",
            "a low-top sneaker", "a skate shoe", "a canvas sneaker",
            "a basketball shoe", "a leather dress shoe", "a slipper", "a clog"]
OTHER = ["a fan or blower", "an electric appliance", "a cardboard box",
         "a plastic container", "a bag", "a bottle", "a can",
         "a random object", "a piece of trash"]


class ClipShoeness:
    def __init__(self, model=None, preprocess=None, device=None, model_name="ViT-B/32"):
        """Pass an already-loaded CLIP (model, preprocess, device) to reuse the
        brand classifier's model with zero extra GPU load; otherwise it loads its
        own ViT-B/32."""
        self.ok = False
        try:
            import torch
            import clip
            if model is None or preprocess is None:
                device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                model, preprocess = clip.load(model_name, device=device)
            elif device is None:
                # reusing the brand classifier's CLIP -> derive its device
                device = next(model.parameters()).device
            self.model, self.preprocess, self.device = model, preprocess, device
            self._torch = torch
            # PAD-TO-SQUARE preprocessing (NOT CLIP's default center-crop, which
            # throws away the shoes at the edges of a WIDE two-shoe crop and
            # scored real pairs as non-footwear). Letterbox keeps the whole crop
            # + aspect ratio -> real shoes ~0.95, junk ~0.00. CLIP's own norm.
            from torchvision import transforms
            self._img_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
            self._classes = FOOTWEAR + OTHER
            self._n_foot = len(FOOTWEAR)
            tokens = clip.tokenize(self._classes).to(device)
            with torch.no_grad():
                tf = model.encode_text(tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
            self._tf = tf
            self.ok = True
        except Exception as exc:                          # noqa: BLE001 - fail safe
            print(f"[shoeness] CLIP load failed ({exc}); shoeness disabled.")

    def _pad_square(self, rgb):
        import numpy as np
        h, w = rgb.shape[:2]
        s = max(h, w)
        canvas = np.full((s, s, 3), 128, dtype=np.uint8)   # neutral gray letterbox
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = rgb
        return canvas

    def _embed(self, crop_bgr):
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = self._img_tf(Image.fromarray(self._pad_square(rgb))).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f

    def score(self, crop_bgr):
        """P(footwear) in [0,1] for one BGR crop, or None on failure."""
        if not self.ok or crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return None
        try:
            f = self._embed(crop_bgr)
            p = (100.0 * f @ self._tf.T).softmax(-1)[0]
            return float(p[:self._n_foot].sum().item())
        except Exception:                                 # noqa: BLE001
            return None


def _validate(prefix, pairs_dir, floor):
    import glob, os, sqlite3, cv2
    sh = ClipShoeness()
    if not sh.ok:
        print("CLIP not loaded -> abort"); return
    # DB context (make/conf) keyed by crop basename
    meta = {}
    try:
        c = sqlite3.connect("file:../sneakers.db?mode=ro", uri=True)
        for ipath, mk, mkc, md, mdc in c.execute(
                "SELECT image_path,make,make_confidence,model,model_confidence "
                "FROM pairs WHERE table_photo_id LIKE ?", (prefix + "%",)):
            meta[os.path.basename(ipath or "")] = (mk, mkc, md, mdc)
    except Exception as exc:
        print("(db lookup failed:", exc, ")")
    crops = glob.glob(f"{pairs_dir}/{prefix}*.jpg")
    print(f"scoring {len(crops)} pair crops from tables like {prefix}* ...")
    flagged = []
    n = 0
    for path in crops:
        img = cv2.imread(path)
        p = sh.score(img)
        if p is None:
            continue
        n += 1
        if p < floor:
            flagged.append((p, os.path.basename(path)))
    flagged.sort()
    print(f"\nscored {n} crops   floor={floor}")
    print(f"FLAGGED non-footwear: {len(flagged)}  ({100.0*len(flagged)/max(1,n):.1f}%)")
    print("(inspect each -- should be genuine non-shoes, NOT real shoes dropped)")
    print(f"\n{'P(footwear)':>11}  {'crop':<28} gemini make(conf)  model(conf)")
    for p, base in flagged:
        mk, mkc, md, mdc = meta.get(base, ("?", None, "?", None))
        print(f"{p:>11.3f}  {base:<28} {mk} ({mkc})  {md} ({mdc})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="TBL-20260701", help="table id prefix (default today)")
    ap.add_argument("--pairs-dir", default="../images/pairs")
    ap.add_argument("--floor", type=float, default=0.10)
    a = ap.parse_args()
    _validate(a.prefix, a.pairs_dir, a.floor)
