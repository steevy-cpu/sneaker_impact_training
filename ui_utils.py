"""
ui_utils.py -- live UI / overlay helpers (PLACEHOLDER).

FUTURE PURPOSE
--------------
Draw the operator-facing overlays on the live feed and handle mouse input:

    - Draw bounding boxes around detected shoes.
    - Show YOLO confidence and (optionally) detected color per box.
    - Color-code the current classification (Reuse vs. Recycle).
    - Handle double-click inside a box to flip Reuse -> Recycle.
    - Ignore clicks outside any box.
    - Flash a brief confirmation after a recycle selection.
    - Optionally draw an FPS counter (config.DISPLAY_FPS).

Priority is a fast, clear labeling workflow -- not a fancy UI.

Implementation arrives in Phase 2-3. Nothing is implemented yet.
"""
