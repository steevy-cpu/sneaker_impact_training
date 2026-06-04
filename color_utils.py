"""
color_utils.py -- dominant shoe color estimation (CIELAB).

`classify_color(image, mask=None)` returns `(name, confidence)` for the dominant
broad color in the image. Pipeline:

  1. Convert to CIELAB -- a perceptual color space where Euclidean distance
     matches how different two colors LOOK to the human eye (better than HSV for
     naming, and much better than raw RGB, which mixes brightness into every
     channel).
  2. Per pixel:
       - chroma C* = sqrt(a*^2 + b*^2). If C* is low the pixel is NEUTRAL, named
         by lightness L*: black (dark) / white (light) / gray (in between).
       - otherwise the pixel has a real hue: name it by the NEAREST colored
         anchor in Lab (red/orange/yellow/green/blue/purple/pink/brown).
  3. A polygon mask (or, without one, a centered fraction COLOR_CENTER_FRAC)
     restricts which pixels count, so background doesn't bias the answer.
  4. Return the most-common color and its fraction of counted pixels as the
     confidence. There is NO "multi" -- we always keep the single dominant color.

Failure modes (empty image, broken mask, unexpected exception) return
("unknown", 0.0) -- wrapped in try/except so it can never crash the save path.
"""
import cv2
import numpy as np

import config

COLOR_NAMES = [
    "unknown",   # 0
    "black",     # 1
    "white",     # 2
    "gray",      # 3
    "brown",     # 4
    "red",       # 5
    "orange",    # 6
    "yellow",    # 7
    "green",     # 8
    "blue",      # 9
    "purple",    # 10
    "pink",      # 11
]
_CODE = {name: i for i, name in enumerate(COLOR_NAMES)}

# Colored (chromatic) reference anchors, given as (name, R, G, B). Several
# anchors may share a name (e.g. navy + blue) to cover a color's natural spread.
# Neutral colors (black/gray/white) are handled separately by lightness, so they
# are NOT anchors here.
_CHROMA_ANCHORS = [
    ("red",    (200, 30, 30)),
    ("brown",  (115, 75, 45)),
    ("brown",  (165, 120, 80)),    # tan / light brown
    ("orange", (225, 120, 20)),
    ("yellow", (220, 210, 40)),
    ("green",  (40, 150, 60)),
    ("blue",   (40, 80, 190)),
    ("blue",   (25, 35, 90)),      # navy
    ("purple", (120, 50, 170)),
    ("pink",   (235, 130, 175)),
]


def _anchors_lab():
    """Precompute the chromatic anchors in standard Lab (L*0-100, a*/b* approx
    -128..127). Returns (codes[K], L[K], a[K], b[K])."""
    rgb = np.array([c for _, c in _CHROMA_ANCHORS], dtype=np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    codes = np.array([_CODE[name] for name, _ in _CHROMA_ANCHORS], dtype=np.uint8)
    return codes, lab[:, 0] * (100.0 / 255.0), lab[:, 1] - 128.0, lab[:, 2] - 128.0


_A_CODES, _A_L, _A_A, _A_B = _anchors_lab()


def _pixel_mask(image, mask):
    """Boolean (H, W) of which pixels to count: inside the polygon if given,
    else a centered fraction of the crop (config.COLOR_CENTER_FRAC)."""
    h, w = image.shape[:2]
    if mask is not None and len(mask) >= 3:
        poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly, [np.asarray(mask, dtype=np.int32)], 255)
        return poly > 0
    frac = getattr(config, "COLOR_CENTER_FRAC", 1.0)
    pm = np.zeros((h, w), dtype=bool)
    if frac >= 1.0:
        pm[:] = True
    else:
        my = int(h * (1.0 - frac) / 2.0)
        mx = int(w * (1.0 - frac) / 2.0)
        pm[my:h - my, mx:w - mx] = True
        if not pm.any():
            pm[:] = True
    return pm


def classify_color(image, mask=None):
    """Return `(name, confidence)` for the dominant color in a BGR `image`.

    Confidence is the fraction of counted pixels that share the winning color
    (a solid shoe approaches 1.0). On any failure returns ("unknown", 0.0).
    """
    try:
        if image is None or image.size == 0:
            return "unknown", 0.0

        pm = _pixel_mask(image, mask)
        if not pm.any():
            return "unknown", 0.0

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0] * (100.0 / 255.0)
        A = lab[..., 1] - 128.0
        B = lab[..., 2] - 128.0

        # Nearest chromatic anchor for every pixel (loop over the few anchors).
        best_idx = np.zeros(L.shape, dtype=np.int32)
        best_d = np.full(L.shape, np.inf, dtype=np.float32)
        for k in range(len(_A_CODES)):
            d = (L - _A_L[k]) ** 2 + (A - _A_A[k]) ** 2 + (B - _A_B[k]) ** 2
            closer = d < best_d
            best_d[closer] = d[closer]
            best_idx[closer] = k
        labels = _A_CODES[best_idx]                    # chromatic guess everywhere

        # Override neutral (low-chroma) pixels by lightness.
        chroma_min = getattr(config, "COLOR_LAB_CHROMA_MIN", 12)
        l_black = getattr(config, "COLOR_LAB_L_BLACK", 30)
        l_white = getattr(config, "COLOR_LAB_L_WHITE", 80)
        chroma = np.sqrt(A * A + B * B)
        neutral = chroma < chroma_min
        labels[neutral & (L < l_black)] = _CODE["black"]
        labels[neutral & (L > l_white)] = _CODE["white"]
        labels[neutral & (L >= l_black) & (L <= l_white)] = _CODE["gray"]

        # Histogram over counted pixels; pick the most common color.
        counts = np.bincount(labels[pm].ravel(), minlength=len(COLOR_NAMES))
        counts[0] = 0                                  # never report "unknown"
        total = int(counts.sum())
        if total == 0:
            return "unknown", 0.0
        winner = int(np.argmax(counts))
        return COLOR_NAMES[winner], float(counts[winner]) / float(total)

    except Exception:                                  # noqa: BLE001 - fail safe
        return "unknown", 0.0
