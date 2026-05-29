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
     this avoids letting background pixels dominate. Without a mask, the
     whole image is sampled.
  4. Return the name of the most-populated bucket; confidence = fraction
     of valid pixels that fell in that bucket (0.0-1.0).

Failure modes (empty image, broken mask, unexpected exception) return
("unknown", 0.0) -- this function is wrapped in try/except so it can
never crash the app's save path.
"""
import cv2
import numpy as np

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

# Tunable thresholds (HSV in OpenCV's convention).
_V_BLACK = 50         # value below this -> black, regardless of hue
_V_WHITE = 180        # value above this AND low saturation -> white
_S_GRAY = 50          # saturation below this -> gray / white / black axis
                      # (raised from 30 so warm-tinted neutral backgrounds
                      #  don't leak into orange)
_V_BROWN = 200        # value below this AND reddish hue -> brown
                      # (raised from 140: tan/beige/light leather are brown,
                      #  not orange; true vivid orange still lands in orange)


def classify_color(image, mask=None):
    """Return `(name, confidence)` for the dominant color in `image`.

    `image` is a BGR numpy array (OpenCV order).
    `mask` is optional: a polygon contour of shape (N, 1, 2). Only pixels
    inside the polygon are counted. If None, the whole image is sampled.

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
            pixel_mask = np.ones((h, w), dtype=bool)

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

        winner = int(named_counts.argmax())
        return COLOR_NAMES[winner], float(named_counts[winner]) / float(total)

    except Exception:                                  # noqa: BLE001 - fail safe
        return "unknown", 0.0
