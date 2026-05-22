"""
ui_utils.py -- live UI / overlay helpers.

Small drawing helpers for the live detection window. Phase 2 only draws
bounding boxes, confidence, FPS, and a status line. Mouse handling and
Reuse/Recycle color-coding arrive in Phase 3 -- not implemented here.
"""
import cv2

# Colors are BGR (OpenCV order).
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection_box(frame, bbox, label, confidence):
    """Draw one detection: a green box with a "<label> <confidence>" caption.

    bbox is (x1, y1, x2, y2) in pixels. Example caption: "Shoe 0.87".
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
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
