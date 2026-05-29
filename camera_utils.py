"""
camera_utils.py -- cross-platform camera helpers.

The original scripts hardcoded cv2.CAP_DSHOW, which only works on Windows.
This module picks the correct OpenCV backend for the current operating system
so the camera works on macOS (AVFoundation), Windows (DirectShow), and Linux
(default) -- including external USB-C cameras.

Selecting the RIGHT camera:
    Camera index numbers can shift between reboots or when you replug a USB-C
    camera, so pinning an index isn't always reliable. If config.CAMERA_NAME is
    set, open_camera() finds the camera whose device name contains that text
    (e.g. "USB") and uses it explicitly -- otherwise it falls back to
    config.CAMERA_INDEX. open_camera() prints the chosen camera's name so you
    can confirm it's the external camera and not the built-in webcam.

Public functions:
    get_camera_backend()              -> the OpenCV backend flag for this OS
    list_camera_names()               -> camera device names, in index order
    find_camera_index_by_name(text)   -> index whose name contains text, or None
    open_camera(camera_index=None)    -> a working, opened cv2.VideoCapture (or None)
    release_camera(cap)               -> safely release a camera
"""
import json
import subprocess
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


def list_camera_names():
    """Return camera device names in index order (macOS only).

    Uses macOS `system_profiler` so cameras can be chosen by name. On this list
    position 0 is the first camera, position 1 the second, etc., which matches
    OpenCV's AVFoundation index order in typical setups. On Windows/Linux this
    returns [] (name lookup isn't supported there yet) and callers fall back to
    a numeric index.
    """
    if not sys.platform.startswith("darwin"):
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(out.stdout)
        cameras = data.get("SPCameraDataType", [])
        return [cam.get("_name", "") for cam in cameras]
    except Exception:
        # Never let camera-name probing crash the app; just skip name lookup.
        return []


def find_camera_index_by_name(text):
    """Return the index of the first camera whose name contains `text`
    (case-insensitive), or None if there's no match / no text given."""
    if not text:
        return None
    target = text.strip().lower()
    for index, name in enumerate(list_camera_names()):
        if target in name.lower():
            return index
    return None


def open_camera(camera_index=None):
    """Open a camera and confirm it actually produces frames.

    Selection order when `camera_index` is None:
      1. config.CAMERA_NAME (explicit by name -- recommended for USB-C cameras)
      2. config.CAMERA_INDEX (numeric fallback)
    Passing `camera_index` explicitly overrides both.

    Prints clear messages and the chosen camera's name. Returns the opened
    cv2.VideoCapture on success, or None on failure.
    """
    # --- GigE Vision path -------------------------------------------------
    # When enabled, the camera is an industrial GigE camera that
    # cv2.VideoCapture can't open. Hand off to the Aravis backend, which
    # returns an object with the same .read()/.release() interface. The import
    # is lazy so the normal USB path never needs Aravis/gi installed.
    if getattr(config, "USE_GIGE_CAMERA", False):
        from gige_camera import open_gige_camera
        if camera_index is not None:
            print("[camera] note: USE_GIGE_CAMERA is on; ignoring camera index "
                  f"{camera_index} and using the GigE camera.")
        return open_gige_camera()

    # --- Decide which camera to use --------------------------------------
    if camera_index is None:
        by_name = find_camera_index_by_name(config.CAMERA_NAME)
        if config.CAMERA_NAME and by_name is None:
            print(f"[camera] WARNING: no camera name contains "
                  f"'{config.CAMERA_NAME}'; falling back to "
                  f"CAMERA_INDEX={config.CAMERA_INDEX}.")
        camera_index = by_name if by_name is not None else config.CAMERA_INDEX

    # NOTE: we deliberately don't print a device name here. On macOS the order
    # returned by `system_profiler` can differ from OpenCV's AVFoundation index
    # order, so showing a name would be misleading. Verify visually via
    # `list_cameras.py`, which saves a preview JPG from each working index.
    label = f"index {camera_index}"

    # --- Open it ----------------------------------------------------------
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    # Some machines don't support the preferred backend; fall back to default.
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera {label}.")
        print("[camera] Try a different index/name (run: python list_cameras.py),")
        print("[camera] check the USB-C connection, and confirm camera permissions.")
        return None

    # Confirm the camera really delivers frames -- "opened" alone isn't enough.
    ok, _ = cap.read()
    if not ok:
        print(f"[camera] ERROR: camera {label} opened but returned no frame.")
        print("[camera] Try a different index or check that no other app is using it.")
        cap.release()
        return None

    print(f"[camera] OK: using camera {label}.")
    return cap


def release_camera(cap):
    """Safely release a camera capture object (None-safe)."""
    if cap is not None:
        cap.release()
