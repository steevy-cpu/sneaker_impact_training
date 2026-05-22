"""
color_utils.py -- dominant shoe color estimation (PLACEHOLDER).

FUTURE PURPOSE
--------------
Estimate a single broad color label for a cropped shoe. Must be lightweight
and must NEVER crash the app -- if color estimation fails it returns
"unknown" and the pipeline continues.

Allowed broad categories only:
    black, white, gray, red, blue, green, yellow,
    brown, orange, purple, pink, unknown

Possible implementations (kept simple, not over-engineered):
    - OpenCV HSV rules
    - KMeans clustering
    - ColorThief-style dominant color extraction

Gated by config.ENABLE_COLOR_DETECTION.

Implementation arrives in Phase 5. Nothing is implemented yet.
"""
