"""
ui_utils.py -- live UI / overlay helpers.

Small drawing helpers for the live detection window. Phase 2 draws a
translucent green mask over each detected shoe (filling its bounding-box
extents) plus a confidence caption, FPS, and a status line. Mouse handling
and Reuse/Recycle color-coding arrive in Phase 3 -- not implemented here.
"""
import cv2

# Colors are BGR (OpenCV order).
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


MASK_ALPHA = 0.4   # 0 = invisible, 1 = solid green; 0.4 lets the shoe show through


def draw_detection_mask(frame, bbox, label, confidence):
    """Overlay one detection as a translucent green mask filling its bbox.

    bbox is (x1, y1, x2, y2) in pixels -- i.e. the min/max x and y of the
    detected shoe. We fill that whole rectangle with green and alpha-blend it
    onto the frame, so the shoe stays visible underneath. A small caption
    "<label> <confidence>" (e.g. "Shoe 0.87") sits just above the mask.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Alpha-blend a filled green rectangle over only the bbox region.
    region = frame[y1:y2, x1:x2]
    if region.size:                                # skip degenerate (0-area) boxes
        green_layer = region.copy()
        green_layer[:] = GREEN
        cv2.addWeighted(green_layer, MASK_ALPHA, region, 1 - MASK_ALPHA, 0, region)

    text = f"{label} {confidence:.2f}"
    cv2.putText(frame, text, (x1, max(y1 - 8, 14)), FONT, 0.6, GREEN, 2)
    return frame


def draw_fps(frame, fps):
    """Draw the FPS counter in the top-right corner."""
    text = f"FPS: {fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x = max(frame.shape[1] - text_w - 10, 10)
    cv2.putText(frame, text, (x, 24), FONT, 0.6, YELLOW, 2)
    return frame


def draw_status_text(frame, text):
    """Draw a status line in the top-left corner."""
    cv2.putText(frame, text, (10, 24), FONT, 0.6, WHITE, 2)
    return frame
