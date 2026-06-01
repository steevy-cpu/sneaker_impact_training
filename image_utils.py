"""
image_utils.py -- small shared image helpers.

Centralizes utilities that were duplicated across modules so there's a single
definition to maintain.
"""
import cv2


def sharpness(image):
    """Variance of the Laplacian -- higher = sharper. Returns 0.0 on failure.

    A cheap focus/blur metric: used to pick the sharpest frame per track
    (tracking_utils) and to filter/sort blurry crops (the dataset tools). Never
    raises -- a bad/empty image just scores 0.0.
    """
    try:
        if image is None or getattr(image, "size", 0) == 0:
            return 0.0
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else image)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:                              # noqa: BLE001 - fail safe
        return 0.0
