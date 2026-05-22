"""
save_utils.py -- dataset storage (PLACEHOLDER).

FUTURE PURPOSE
--------------
Persist a finalized shoe as an image crop plus a metadata JSON sidecar, into
a dated incoming folder under config.OUTPUT_ROOT, with safe shoe numbering.

Planned layout:
    sneaker_impact/pictures/incoming_YYYY-MM-DD/
        shoe_Reuse_1.jpg
        shoe_Reuse_1.json
        shoe_Recycle_2.jpg
        shoe_Recycle_2.json

Planned metadata fields:
    filename, classification, shoe_number, timestamp, detected_color,
    color_confidence, yolo_confidence, bbox, tracking_id,
    frame_width, frame_height, model_used

Optionally also saves the full frame when config.SAVE_FULL_FRAME is True.

Implementation arrives in Phase 4. Nothing is implemented yet.
"""
