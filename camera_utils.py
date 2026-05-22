"""
camera_utils.py -- cross-platform camera helpers.

The original scripts hardcoded cv2.CAP_DSHOW, which only works on Windows.
This module picks the correct OpenCV backend for the current operating system
so the camera works on macOS (AVFoundation), Windows (DirectShow), and Linux
(default) -- including external USB-C cameras.

Public functions:
    get_camera_backend()        -> the OpenCV backend flag for this OS
    open_camera(camera_index)   -> a working, opened cv2.VideoCapture (or None)
    release_camera(cap)         -> safely release a camera

Camera index is never hardcoded: open_camera() defaults to config.CAMERA_INDEX.
Run `python list_cameras.py` to find which index your camera is on.
"""
import sys

import cv2

import config


def get_camera_backend():
    """Return the OpenCV VideoCapture backend best suited to this platform.

    macOS   -> cv2.CAP_AVFOUNDATION
    Windows -> cv2.CAP_DSHOW
    Linux / anything else -> cv2.CAP_ANY (let OpenCV choose; no forced backend)
    """
    if sys.platform.startswith("darwin"):
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(camera_index=None):
    """Open a camera and confirm it actually produces frames.

    If `camera_index` is None, falls back to config.CAMERA_INDEX so the index
    is never hardcoded. Prints clear messages on failure and returns the opened
    cv2.VideoCapture on success, or None if the camera could not be opened or
    did not deliver a frame.
    """
    if camera_index is None:
        camera_index = config.CAMERA_INDEX

    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    # Some machines don't support the preferred backend; fall back to default.
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera index {camera_index}.")
        print("[camera] Try a different index (run: python list_cameras.py),")
        print("[camera] check the USB-C connection, and confirm camera permissions.")
        return None

    # Confirm the camera really delivers frames -- "opened" alone isn't enough.
    ok, _ = cap.read()
    if not ok:
        print(f"[camera] ERROR: camera index {camera_index} opened but returned no frame.")
        print("[camera] Try a different index or check that no other app is using it.")
        cap.release()
        return None

    print(f"[camera] OK: using camera index {camera_index}.")
    return cap


def release_camera(cap):
    """Safely release a camera capture object (None-safe)."""
    if cap is not None:
        cap.release()
