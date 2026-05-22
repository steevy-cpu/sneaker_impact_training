"""
camera_utils.py -- cross-platform camera helpers (PLACEHOLDER).

FUTURE PURPOSE
--------------
Open and configure the live camera feed in a way that works on every OS.
The original scripts hardcoded cv2.CAP_DSHOW, which only works on Windows.
This module will pick the correct OpenCV backend per platform:

    macOS   -> cv2.CAP_AVFOUNDATION
    Windows -> cv2.CAP_DSHOW
    Linux   -> default backend

It will also support external USB-C cameras and never hardcode the camera
index (that comes from config.CAMERA_INDEX).

Implementation arrives in Phase 1. Nothing is implemented yet.
"""
