"""
ui_utils.py -- live UI / overlay helpers.

Small drawing helpers for the live detection window. Phase 2 draws a
translucent green mask over each detected shoe (filling its bounding-box
extents) plus a confidence caption, FPS, and a status line. Mouse handling
and Reuse/Recycle color-coding arrive in Phase 3 -- not implemented here.
"""
import cv2
import numpy as np

import config

# Colors are BGR (OpenCV order).
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


MASK_ALPHA = 0.4   # 0 = invisible, 1 = solid green; 0.4 lets the shoe show through


def grabcut_polygon(image, bbox, iters=None, min_area_frac=0.10, simplify=0.005):
    """Run GrabCut on a padded crop around `bbox` and return a polygon.

    Returns a numpy contour of shape (N, 1, 2) in absolute frame coordinates,
    suitable for cv2.fillPoly / cv2.polylines. Returns None if GrabCut
    failed, produced no foreground, or the foreground was too small to be
    a real shoe.

    Speed note: running GrabCut on the full 1920x1080 frame is slow because
    its cost scales with the image size. We crop to bbox + ~10% padding so
    each call works on a small region (typically ~5-30 ms on CPU).
    """
    try:
        if image is None or image.size == 0:
            return None
        if iters is None:
            iters = int(getattr(config, "GRABCUT_ITERS", 1))
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            return None

        # Pad the work region so GrabCut has nearby background to learn from.
        pad_x = max(10, bw // 10)
        pad_y = max(10, bh // 10)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # The "definitely foreground rectangle" is the bbox, in crop coords.
        rect = (x1 - cx1, y1 - cy1, bw, bh)
        mask = np.zeros(crop.shape[:2], np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)

        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype("uint8")
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area_frac * bw * bh:
            return None

        # Simplify so we don't store hundreds of points per shoe.
        epsilon = simplify * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) < 3:
            return None

        # Translate from crop coords back to full-frame coords.
        approx = approx.copy()
        approx[:, :, 0] += cx1
        approx[:, :, 1] += cy1
        return approx
    except Exception as exc:                       # noqa: BLE001 - never crash live
        print(f"[grabcut] failed: {exc}")
        return None


def draw_detection_mask(frame, bbox, label, confidence, color=GREEN,
                        polygon=None):
    """Draw a solid rectangle around one detected shoe.

    Green = Reuse (default), Red = Recycle. The label and confidence score
    are printed just above the top-left corner of the box.
    Click hit-testing uses the same full bbox, so clicking anywhere
    inside the rectangle flags the shoe as Recycle.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(frame, text, (x1, max(y1 - 8, 14)), FONT, 0.6, color, 2)
    return frame


def draw_fps(frame, fps, det_fps=None):
    """Draw the FPS counter in the top-right corner.

    `fps` is the display/main-loop rate. If `det_fps` is given (the detector
    thread's inference rate), it's shown alongside so you can see whether YOLO
    is keeping up with the camera.
    """
    text = f"FPS: {fps:.1f}" if det_fps is None else f"FPS {fps:.1f} | det {det_fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x = max(frame.shape[1] - text_w - 10, 10)
    cv2.putText(frame, text, (x, 24), FONT, 0.6, YELLOW, 2)
    return frame


def draw_status_text(frame, text):
    """Draw a status line in the top-left corner."""
    cv2.putText(frame, text, (10, 24), FONT, 0.6, WHITE, 2)
    return frame


def draw_toast(frame, text, color=GREEN):
    """Draw a short-lived confirmation banner near the bottom-left.

    Used to confirm saves / undo so the operator gets feedback without watching
    the console. Drawn with a black shadow so it stays readable on any shoe.
    """
    y = frame.shape[0] - 16
    cv2.putText(frame, text, (11, y + 1), FONT, 0.7, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, y), FONT, 0.7, color, 2)
    return frame
