"""
color_utils.py -- dominant shoe color estimation.

`classify_color(image, mask=None)` returns `(name, confidence)` for the
dominant broad color in the image. Pipeline:

  1. Convert to HSV (OpenCV: H in [0,180), S/V in [0,256)).
  2. Vectorized per-pixel bucketing:
       - low V                              -> "black"
       - low S + high V                     -> "white"
       - low S + mid V                      -> "gray"
       - mid V + reddish hue + medium S     -> "brown"
       - otherwise, hue ranges -> one of:
         red / orange / yellow / green / blue / purple / pink
  3. If a polygon mask is supplied, only pixels INSIDE the polygon count;
     this avoids letting background pixels dominate. Without a mask, a
     centered fraction (config.COLOR_CENTER_FRAC) is sampled instead.
  4. Return the name of the most-populated bucket; confidence = fraction
     of valid pixels that fell in that bucket (0.0-1.0).

Failure modes (empty image, broken mask, unexpected exception) return
("unknown", 0.0) -- this function is wrapped in try/except so it can
never crash the app's save path.
"""
import cv2
import numpy as np

import config

# Bucket-name -> small integer code. Order matters: index 0 is "unknown"
# so np.zeros initialization defaults to unknown.
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

# Tunable thresholds (HSV, OpenCV convention). Canonical values + docs live in
# config.py so they can be tuned without editing code; the fallbacks here keep
# color detection working if a key is missing from config.
_V_BLACK = getattr(config, "COLOR_V_BLACK", 50)
_V_WHITE = getattr(config, "COLOR_V_WHITE", 180)
_S_GRAY = getattr(config, "COLOR_S_GRAY", 50)
_V_BROWN = getattr(config, "COLOR_V_BROWN", 200)


def classify_color(image, mask=None):
    """Return `(name, confidence)` for the dominant color in `image`.

    `image` is a BGR numpy array (OpenCV order).
    `mask` is optional: a polygon contour of shape (N, 1, 2). Only pixels
    inside the polygon are counted. If None, a centered fraction of the image
    (config.COLOR_CENTER_FRAC) is sampled instead.

    Confidence is the fraction of counted pixels that fell in the winning
    bucket (so a single-color shoe approaches 1.0; a multi-color shoe is
    lower, e.g. 0.4). On any failure returns `("unknown", 0.0)`.
    """
    try:
        if image is None or image.size == 0:
            return "unknown", 0.0

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]

        # Build pixel-selection mask (True for pixels we count).
        if mask is not None and len(mask) >= 3:
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [np.asarray(mask, dtype=np.int32)], 255)
            pixel_mask = poly_mask > 0
        else:
            # No polygon: sample a centered region so background near the bbox
            # edges doesn't bias the color (config.COLOR_CENTER_FRAC controls it).
            frac = getattr(config, "COLOR_CENTER_FRAC", 1.0)
            pixel_mask = np.zeros((h, w), dtype=bool)
            if frac >= 1.0:
                pixel_mask[:] = True
            else:
                my = int(h * (1.0 - frac) / 2.0)
                mx = int(w * (1.0 - frac) / 2.0)
                pixel_mask[my:h - my, mx:w - mx] = True
                if not pixel_mask.any():       # crop too small -> use all of it
                    pixel_mask[:] = True

        if not pixel_mask.any():
            return "unknown", 0.0

        # Vectorized per-pixel classification, written into `labels`
        # (small integer codes from COLOR_NAMES).
        labels = np.zeros((h, w), dtype=np.uint8)  # 0 = unknown by default

        is_black = val < _V_BLACK
        labels[is_black] = _CODE["black"]

        non_black = ~is_black
        low_sat = non_black & (sat < _S_GRAY)
        labels[low_sat & (val >= _V_WHITE)] = _CODE["white"]
        labels[low_sat & (val < _V_WHITE)] = _CODE["gray"]

        sat_pix = non_black & ~low_sat

        # Brown = dark + reddish (red wraps around hue 0 / 180).
        is_red_hue = (hue <= 25) | (hue >= 170)
        brown = sat_pix & (val < _V_BROWN) & is_red_hue
        labels[brown] = _CODE["brown"]

        rem = sat_pix & ~brown
        labels[rem & is_red_hue & ((hue <= 10) | (hue >= 170))] = _CODE["red"]
        labels[rem & (hue > 10) & (hue <= 25)] = _CODE["orange"]
        labels[rem & (hue > 25) & (hue <= 35)] = _CODE["yellow"]
        labels[rem & (hue > 35) & (hue <= 85)] = _CODE["green"]
        labels[rem & (hue > 85) & (hue <= 130)] = _CODE["blue"]
        labels[rem & (hue > 130) & (hue <= 155)] = _CODE["purple"]
        labels[rem & (hue > 155) & (hue < 170)] = _CODE["pink"]

        # Histogram over only the in-mask pixels.
        counts = np.bincount(labels[pixel_mask].ravel(),
                             minlength=len(COLOR_NAMES))
        # Skip "unknown" (index 0) when picking the winner -- the unknown
        # bucket is just a fallback for pixels none of the rules matched.
        named_counts = counts.copy()
        named_counts[0] = 0
        total = int(named_counts.sum())
        if total == 0:
            return "unknown", 0.0

        # Rank buckets; flag "multi" when the top two are within a small margin
        # so a genuinely two-tone shoe isn't forced into one color.
        order = np.argsort(named_counts)[::-1]
        w1 = int(order[0])
        c1 = int(named_counts[w1])
        c2 = int(named_counts[int(order[1])]) if named_counts.size > 1 else 0
        top1_frac = c1 / float(total)
        margin = getattr(config, "COLOR_AMBIGUOUS_MARGIN", 0.0)
        if margin > 0 and c2 > 0 and (top1_frac - c2 / float(total)) < margin:
            return "multi", float(top1_frac)
        return COLOR_NAMES[w1], float(top1_frac)

    except Exception:                                  # noqa: BLE001 - fail safe
        return "unknown", 0.0
